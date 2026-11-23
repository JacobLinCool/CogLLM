"""Sentence-level first-pass rating fits on a cached Qwen last MLP layer."""

import math

import torch

from .polish import newton_a_step, preconditioned_newton_step


RANK = 4
ALPHA = 4.0
HEAD_L2 = 1.0
ADAPTER_L2 = 1.0
TARGETS = 2
LEVELS = 5
FROZEN_STEP_LIMIT = 720
LORA_STEP_LIMIT = 12000
FROZEN_STEPS_PER_CHECK = 40
LORA_STEPS_PER_CHECK = 80
GRADIENT_INF_LIMIT = 3e-7
GRADIENT_L2_LIMIT = 1e-5
RELATIVE_OBJECTIVE_LIMIT = 1e-9
NEWTON_START_GRADIENT_L2 = 3e-3
NEWTON_MIN_LBFGS_STEPS = 1200
LBFGS_STEPS_BETWEEN_POLISH = 400
POLISH_EPISODE_LIMIT = 8
POLISH_GRADIENT_GROWTH = 3.0
NEWTON_STEP_LIMIT = 64
TARGETED_A_STEP_LIMIT = 24
TARGETED_A_MIN_GRADIENT_FRACTION = 0.9
ADAPTER_SEEDS = (12, 29)
POLISHING_MODES = ('joint', 'guarded')


class LoRANotConverged(ArithmeticError):
    """The adapter finished an optimization attempt without passing its gates."""


def _unit_hidden(pre_norm, norm_weight, norm_eps):
    inverse_rms = torch.rsqrt(pre_norm.square().mean(dim=1, keepdim=True) + norm_eps)
    hidden = pre_norm * inverse_rms * (1.0 + norm_weight)
    return hidden / torch.linalg.vector_norm(hidden, dim=1, keepdim=True)


def _log_ratings(hidden, length, head):
    # Head rows are intercept, L2-normalized Qwen coordinates, then length features.
    design = torch.cat((torch.ones_like(length[:, :1]), hidden, length), dim=1)
    logits = design @ head.reshape(head.shape[0], TARGETS * LEVELS)
    return logits.reshape(-1, TARGETS, LEVELS).log_softmax(dim=-1)


@torch.no_grad()
def project(normalized_states, lengths, head, length_mean, length_scale):
    """Return log probabilities from a saved head and Qwen final RMSNorm output."""
    hidden = torch.as_tensor(normalized_states, dtype=torch.float64, device='cpu')
    lengths = torch.as_tensor(lengths, dtype=hidden.dtype, device='cpu')
    center = torch.as_tensor(length_mean, dtype=hidden.dtype, device='cpu')
    scale = torch.as_tensor(length_scale, dtype=hidden.dtype, device='cpu')
    head = torch.as_tensor(head, dtype=hidden.dtype, device='cpu')
    if (hidden.ndim != 2 or lengths.shape != (len(hidden), 2) or center.shape != (2,) or scale.shape != (2,)
            or head.shape != (TARGETS, hidden.shape[1] + 3, LEVELS) or not (scale > 0).all()):
        raise ValueError('Incompatible shapes or nonpositive scale in the saved projection')
    hidden = hidden / torch.linalg.vector_norm(hidden, dim=1, keepdim=True)
    log_p = _log_ratings(hidden, (lengths - center) / (scale * math.sqrt(2)), head.permute(1, 0, 2))
    if not torch.isfinite(log_p).all():
        raise ValueError('Saved projection yielded non-finite log probabilities')
    return log_p.numpy()


def _fit(objective, parameters, *, step_limit, steps_per_check, attempt, progress, raw_a_gradient=None,
         joint_polish_step=None, targeted_a_step=None, polishing='joint'):
    def make_lbfgs():
        return torch.optim.LBFGS(
            parameters,
            max_iter=steps_per_check,
            max_eval=2 * steps_per_check,
            tolerance_grad=0.0,
            tolerance_change=1e-15,
            history_size=40,
            line_search_fn='strong_wolfe',
        )

    optimizer = make_lbfgs()
    evaluations = 0

    def closure():
        nonlocal evaluations
        for parameter in parameters:
            parameter.grad = None
        loss = objective()
        if not torch.isfinite(loss):
            raise ArithmeticError('Non-finite optimization objective')
        loss.backward()
        evaluations += 1
        return loss

    initial = previous = float(closure().detach())
    history = []
    polish_rejections = []
    guarded = polishing == 'guarded'
    convergence_error = LoRANotConverged if raw_a_gradient is not None else ArithmeticError
    lbfgs_steps = joint_attempts = joint_steps = targeted_attempts = targeted_steps = 0

    def can_target_a(checkpoint):
        if targeted_a_step is None or targeted_attempts == TARGETED_A_STEP_LIMIT:
            return False
        a_l2 = checkpoint['raw_lora_a_gradient_l2']
        non_a_eligible = guarded or (joint_steps > 0 and checkpoint['non_a_gradient_inf'] <= GRADIENT_INF_LIMIT)
        return ((checkpoint['raw_lora_a_gradient_inf'] > GRADIENT_INF_LIMIT or a_l2 > GRADIENT_L2_LIMIT)
                and non_a_eligible and checkpoint['non_a_gradient_l2'] <= GRADIENT_L2_LIMIT
                and a_l2 >= TARGETED_A_MIN_GRADIENT_FRACTION * checkpoint['gradient_l2'])

    def record(current, previous, phase, **step_details):
        gradients = [parameter.grad for parameter in parameters]
        if raw_a_gradient is None:
            gradient = torch.cat([value.flatten() for value in gradients])
            grad_inf = float(gradient.abs().max())
            grad_l2 = float(torch.linalg.vector_norm(gradient))
            adapter_gradient = {}
        else:
            a_gradient = raw_a_gradient(gradients[1])
            rest = torch.cat((gradients[0].flatten(), gradients[2].flatten()))
            a_inf = float(a_gradient.abs().max())
            a_l2 = float(torch.linalg.vector_norm(a_gradient))
            rest_inf = float(rest.abs().max())
            rest_l2 = float(torch.linalg.vector_norm(rest))
            grad_inf = max(a_inf, rest_inf)
            grad_l2 = math.hypot(a_l2, rest_l2)
            adapter_gradient = {
                'raw_lora_a_gradient_inf': a_inf, 'raw_lora_a_gradient_l2': a_l2,
                'non_a_gradient_inf': rest_inf, 'non_a_gradient_l2': rest_l2,
            }
        relative_change = abs(previous - current) / max(1.0, abs(previous))
        history.append({
            'steps': lbfgs_steps, 'objective': current,
            'gradient_inf': grad_inf, 'gradient_l2': grad_l2,
            'relative_objective_change': relative_change, 'phase': phase,
            'newton_steps': joint_steps, 'newton_attempts': joint_attempts, 'targeted_a_attempts': targeted_attempts,
            'targeted_a_steps': targeted_steps,
            **adapter_gradient, **step_details,
        })
        converged = (grad_inf <= GRADIENT_INF_LIMIT and grad_l2 <= GRADIENT_L2_LIMIT
                     and relative_change <= RELATIVE_OBJECTIVE_LIMIT)
        if progress is not None:
            progress({**attempt, 'event': 'converged' if converged else 'check', **history[-1]})
        return converged

    if progress is not None:
        progress({**attempt, 'event': 'started', 'objective': initial, 'step_limit': step_limit})
    converged = False
    next_polish = NEWTON_MIN_LBFGS_STEPS
    lbfgs_checks = 0
    check_limit = math.ceil(step_limit / steps_per_check)
    while lbfgs_steps < step_limit and (guarded or lbfgs_checks < check_limit):
        state = optimizer.state[parameters[0]]
        settings = optimizer.param_groups[0]
        remaining = step_limit - lbfgs_steps
        settings['max_iter'] = min(steps_per_check, remaining)
        settings['max_eval'] = 2 * settings['max_iter']
        before = state.get('n_iter', 0)
        optimizer.step(closure)
        advanced = state['n_iter'] - before
        lbfgs_steps += advanced
        lbfgs_checks += 1
        current = float(closure().detach())
        converged = record(current, previous, 'lbfgs')
        if converged:
            break
        if not advanced:
            raise convergence_error(f'L-BFGS made no iteration and did not meet convergence criteria: {history[-1]}')
        previous = current
        terminal = lbfgs_steps == step_limit
        last = history[-1]
        has_polish = joint_polish_step is not None and (joint_attempts < NEWTON_STEP_LIMIT or can_target_a(last))
        can_polish = (has_polish and lbfgs_steps >= NEWTON_MIN_LBFGS_STEPS
                      and last['gradient_l2'] <= NEWTON_START_GRADIENT_L2
                      and ((guarded and terminal) or (lbfgs_steps >= next_polish
                                                     and last['relative_objective_change'] <= RELATIVE_OBJECTIVE_LIMIT)))
        if not can_polish:
            continue
        episode_limit = POLISH_EPISODE_LIMIT if guarded and not terminal else NEWTON_STEP_LIMIT + TARGETED_A_STEP_LIMIT
        if progress is not None:
            progress({
                **attempt, **last, 'event': 'polish_started', 'newton_step_limit': NEWTON_STEP_LIMIT,
                'min_lbfgs_steps': NEWTON_MIN_LBFGS_STEPS, 'episode_step_limit': episode_limit,
                'targeted_a_step_limit': 0 if targeted_a_step is None else TARGETED_A_STEP_LIMIT,
            })
        targeted_allowed = targeted_a_step is not None
        best_gradient = last['gradient_l2']
        accepted = 0
        resume_reason = 'polishing episode reached its step limit'
        for _ in range(episode_limit):
            last = history[-1]
            target_a = targeted_allowed and can_target_a(last)
            if target_a:
                targeted_attempts += 1
                phase = 'newton_a'
                take_step = targeted_a_step
            else:
                if joint_attempts == NEWTON_STEP_LIMIT:
                    resume_reason = 'joint polishing reached its attempt limit'
                    break
                joint_attempts += 1
                phase = 'newton_pcg'
                take_step = joint_polish_step
            if progress is not None:
                progress({
                    **attempt, 'event': 'polish_step_started', 'phase': phase, 'steps': lbfgs_steps,
                    'newton_attempts': joint_attempts, 'targeted_a_attempts': targeted_attempts,
                })
            try:
                details = take_step()
            except ArithmeticError as error:
                # Both line searches restore parameters before reporting no step.
                rejection = {'phase': phase, 'steps': lbfgs_steps, 'objective': current, 'reason': str(error)}
                polish_rejections.append(rejection)
                if progress is not None:
                    progress({**attempt, 'event': 'polish_step_rejected', **rejection})
                if target_a:
                    targeted_allowed = False
                    continue
                resume_reason = 'joint polishing rejected its candidate steps'
                break
            accepted += 1
            targeted_steps += int(target_a)
            joint_steps += int(not target_a)
            targeted_allowed = targeted_a_step is not None
            evaluations += details['objective_evaluations']
            previous = current
            current = float(closure().detach())
            converged = record(current, previous, phase, **details)
            if converged:
                break
            raw_gradient = history[-1]['gradient_l2']
            if (guarded and not terminal
                    and raw_gradient > max(2 * GRADIENT_L2_LIMIT, POLISH_GRADIENT_GROWTH * best_gradient)):
                resume_reason = 'raw gradient increased during polishing'
                break
            best_gradient = min(best_gradient, raw_gradient)
        if converged or not guarded:
            break
        if not terminal:
            # A curvature history captured before an accepted external step is stale.
            if accepted:
                optimizer = make_lbfgs()
            next_polish = lbfgs_steps + LBFGS_STEPS_BETWEEN_POLISH
            previous = current
            if progress is not None:
                progress({
                    **attempt, **history[-1], 'event': 'lbfgs_resumed', 'reason': resume_reason,
                    'next_polish_lbfgs_step': next_polish, 'reset_lbfgs_history': bool(accepted),
                })

    if not converged:
        raise convergence_error(f'Fit did not meet gradient and plateau criteria: {history[-1]}')
    return {
        'converged': True, 'initial_objective': initial, 'evaluations': evaluations,
        'objective_decrease': initial - current, 'step_limit': step_limit,
        'criteria': {
            'gradient_inf': GRADIENT_INF_LIMIT, 'gradient_l2': GRADIENT_L2_LIMIT,
            'relative_objective_change': RELATIVE_OBJECTIVE_LIMIT,
        },
        'history': history, 'polish_rejections': polish_rejections,
        **history[-1],
    }


def fit_fold(pre_norm, down_input, norm_weight, norm_eps, lengths, observed, train, contrast_weights,
             *, predict=None, device='cpu', progress=None, polishing='joint'):
    """Fit matched frozen and LoRA heads and return requested log probabilities.

    ``pre_norm`` is the frozen last-block residual after the down projection.
    ``down_input`` is the gated input of that projection at the same positions.
    ``lengths`` stores the log1p character and word counts for the scored sentences.
    ``observed`` stores empirical probabilities for two targets and five bins.
    Signed ``contrast_weights`` are the train-only vector from ``fit_contrast``.
    Outputs follow the input order, using ``predict`` or the held-out mask by default.
    ``progress`` receives a dictionary on each attempt start and convergence check.
    ``polishing`` selects a numerical route to the same training objective and convergence gates.
    """
    if polishing not in POLISHING_MODES:
        raise ValueError(f'Unknown LoRA polishing method: {polishing!r}')
    pre_norm = torch.as_tensor(pre_norm, device=device, dtype=torch.float64).detach()
    down_input = torch.as_tensor(down_input, device=device, dtype=torch.float64).detach()
    norm_weight = torch.as_tensor(norm_weight, device=device, dtype=torch.float64).detach()
    lengths = torch.as_tensor(lengths, device=device, dtype=torch.float64).detach()
    observed = torch.as_tensor(observed, device=device, dtype=torch.float64).detach()
    train = torch.as_tensor(train, device=device)
    contrast = torch.as_tensor(contrast_weights, device=device, dtype=torch.float64).detach()
    norm_eps = float(norm_eps)

    if pre_norm.ndim != 2 or down_input.ndim != 2:
        raise ValueError('Expected last residual and down projection activations as row matrices')
    count, width = pre_norm.shape
    if down_input.shape[0] != count or norm_weight.shape != (width,) or lengths.shape != (count, 2):
        raise ValueError('Inconsistent activation, final norm or length shapes')
    if observed.shape != (count, TARGETS, LEVELS) or train.shape != (count,) or train.dtype != torch.bool:
        raise ValueError('Expected two five-level rating distributions per sentence and a Boolean fold mask')
    predict = ~train if predict is None else torch.as_tensor(predict, device=device)
    if predict.shape != (count,) or predict.dtype != torch.bool or not predict.any():
        raise ValueError('Expected a nonempty Boolean prediction mask')
    training_count = int(train.sum())
    if training_count == 0 or training_count == count or contrast.shape != (training_count,):
        raise ValueError('Expected nonempty training and held-out rows and one contrast coefficient per training row')
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError('RMSNorm epsilon must be finite and strictly positive')
    ratings = observed[train]
    if (not torch.isfinite(ratings).all() or (ratings < 0).any()
            or not torch.allclose(ratings.sum(dim=-1), torch.ones_like(ratings[..., 0]), atol=1e-7, rtol=0)
            or (ratings.sum(dim=0) <= 0).any()):
        raise ValueError('Training ratings must be probability distributions with support in every bin')
    if (not torch.isfinite(contrast).all() or abs(float(contrast.sum())) > 1e-8
            or not (contrast < 0).any() or not (contrast > 0).any()):
        raise ValueError('Expected finite, balanced signed coefficients for a training rewrite contrast')
    if not all(torch.isfinite(array).all() for array in (pre_norm, down_input, norm_weight, lengths)):
        raise ValueError('Activation, final norm and length inputs must be finite')

    length_mean = lengths[train].mean(dim=0)
    length_scale = lengths[train].std(dim=0, correction=0)
    length_scale = torch.where(length_scale > 0, length_scale, torch.ones_like(length_scale))
    lengths = (lengths - length_mean) / (length_scale * math.sqrt(lengths.shape[1]))
    train_length = lengths[train]
    levels = torch.arange(1, LEVELS + 1, device=device, dtype=pre_norm.dtype)
    effect_target = contrast @ (ratings @ levels)
    original_features = _unit_hidden(pre_norm, norm_weight, norm_eps)
    if not torch.isfinite(original_features).all():
        raise ValueError('Expected a nonzero, finite final hidden norm for each sentence')
    original_features_train = original_features[train]
    train_residual = pre_norm[train]
    train_down = down_input[train]
    gain = ALPHA / RANK

    def head_loss(features, head):
        log_p = _log_ratings(features, train_length, head)
        nll = -(ratings * log_p).sum() / training_count
        effect_error = contrast @ (log_p.exp() @ levels) - effect_target
        return nll + 0.5 * effect_error.square().sum() + 0.5 * HEAD_L2 / training_count * head[1:].square().sum()

    frozen_head = torch.nn.Parameter(pre_norm.new_zeros((width + 3, TARGETS, LEVELS)))
    frozen_diagnostics = _fit(
        lambda: head_loss(original_features_train, frozen_head), [frozen_head],
        step_limit=FROZEN_STEP_LIMIT, steps_per_check=FROZEN_STEPS_PER_CHECK,
        attempt={'method': 'frozen'}, progress=progress, polishing=polishing,
    )

    # Optimize A in the row basis of the training down-projection input. The
    # complementary component contributes only to the existing L2 penalty.
    # Attenuate large singular directions in the optimization coordinates while
    # evaluating the original adapter penalty and convergence in raw coordinates.
    left, singular, right = torch.linalg.svd(train_down.cpu(), full_matrices=False)
    ridge = singular.median().clamp_min(torch.finfo(singular.dtype).eps)
    coordinate_scale = torch.sqrt(1 + (singular / ridge).square()).to(device)
    right = right.to(device)
    reduced_down = (left * singular).to(device) / coordinate_scale

    def raw_a_gradient(gradient):
        return (gradient * coordinate_scale) @ right

    def adapter_run(seed):
        head = torch.nn.Parameter(frozen_head.detach().clone())
        generator = torch.Generator().manual_seed(seed)
        a_shape = (RANK, down_input.shape[1])
        a_start = torch.empty(a_shape, dtype=pre_norm.dtype)
        a_start.uniform_(-1 / math.sqrt(a_shape[1]), 1 / math.sqrt(a_shape[1]), generator=generator)
        a = torch.nn.Parameter((a_start.to(device) @ right.T) * coordinate_scale)
        b = torch.nn.Parameter(pre_norm.new_zeros((width, RANK)))

        def objective():
            adjusted = train_residual + (reduced_down @ a.T @ b.T) * gain
            adapted = _unit_hidden(adjusted, norm_weight, norm_eps)
            squared_norms = (a / coordinate_scale).square().sum() + b.square().sum()
            penalty = 0.5 * ADAPTER_L2 / training_count * squared_norms
            return head_loss(adapted, head) + penalty

        geometry = {
            'reduced_down': reduced_down, 'residual': train_residual, 'norm_weight': norm_weight,
            'gain': gain, 'lengths': train_length, 'head': head, 'ratings': ratings, 'contrast': contrast,
            'effect_target': effect_target, 'b': b, 'coordinate_scale': coordinate_scale, 'adapter_l2': ADAPTER_L2,
        }

        def joint_step():
            return preconditioned_newton_step(
                objective, [head, a, b], **geometry, head_l2=HEAD_L2, raw_a_gradient=raw_a_gradient,
                guarded=polishing == 'guarded',
            )

        def targeted_step():
            return newton_a_step(objective, a, **geometry)

        try:
            diagnostics = _fit(
                objective, [head, a, b], step_limit=LORA_STEP_LIMIT, steps_per_check=LORA_STEPS_PER_CHECK,
                attempt={'method': 'lora', 'seed': seed, 'polishing': polishing}, progress=progress,
                raw_a_gradient=raw_a_gradient, joint_polish_step=joint_step, targeted_a_step=targeted_step,
                polishing=polishing,
            )
        except LoRANotConverged as error:
            raise LoRANotConverged(f'LoRA optimizer seed {seed} ({polishing}): {error}') from error
        raw_a = (a.detach() / coordinate_scale) @ right
        return head.detach(), raw_a, b.detach(), {'seed': seed, **diagnostics}

    attempts = [adapter_run(seed) for seed in ADAPTER_SEEDS]
    head, a, b, best = min(attempts, key=lambda attempt: attempt[-1]['objective'])
    with torch.no_grad():
        frozen_log_p = _log_ratings(original_features[predict], lengths[predict], frozen_head)
        adjusted = pre_norm[predict] + (down_input[predict] @ a.T @ b.T) * gain
        adapted = _unit_hidden(adjusted, norm_weight, norm_eps)
        lora_log_p = _log_ratings(adapted, lengths[predict], head)
    if not torch.isfinite(frozen_log_p).all() or not torch.isfinite(lora_log_p).all():
        raise ArithmeticError('Fit yielded non-finite log probabilities for requested prediction rows')

    return {
        'frozen_log_probabilities': frozen_log_p.cpu().numpy(),
        'lora_log_probabilities': lora_log_p.cpu().numpy(),
        'frozen_head': frozen_head.detach().permute(1, 0, 2).cpu().numpy(),
        'lora_head': head.permute(1, 0, 2).cpu().numpy(),
        'lora_a': a.cpu().numpy(),
        'lora_b': b.cpu().numpy(),
        'length_mean': length_mean.cpu().numpy(),
        'length_scale': length_scale.cpu().numpy(),
        'diagnostics': {
            'frozen': frozen_diagnostics, 'lora_restarts': [attempt[-1] for attempt in attempts],
            'selected_seed': best['seed'], 'head_l2': HEAD_L2, 'adapter_l2': ADAPTER_L2, 'polishing': polishing,
            'restart_objective_spread': max(attempt[-1]['objective'] for attempt in attempts) - best['objective'],
            'first_start_objective_decreased': attempts[0][-1]['objective_decrease'] > 0,
            'lora_a_conditioning': {
                'basis_rows': len(singular), 'singular_ridge': float(ridge),
                'max_coordinate_scale': float(coordinate_scale.max()),
            },
            'newton_polish': {
                'raw_gradient_l2_trigger': NEWTON_START_GRADIENT_L2,
                'relative_objective_change_trigger': RELATIVE_OBJECTIVE_LIMIT, 'step_limit': NEWTON_STEP_LIMIT,
                'min_lbfgs_steps': NEWTON_MIN_LBFGS_STEPS,
                'lbfgs_steps_between_episodes': LBFGS_STEPS_BETWEEN_POLISH,
                'episode_step_limit': POLISH_EPISODE_LIMIT,
                'episode_gradient_growth_limit': POLISH_GRADIENT_GROWTH,
                'targeted_a_step_limit': TARGETED_A_STEP_LIMIT,
                'targeted_a_min_gradient_fraction': TARGETED_A_MIN_GRADIENT_FRACTION,
            },
            'rank': RANK, 'alpha': ALPHA,
        },
    }
