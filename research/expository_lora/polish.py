"""Polish a near-stationary fit using the unchanged objective's exact curvature."""

import math

import torch


MAX_CG_STEPS = 256
JOINT_DAMPINGS = (0.0, 1e-8, 1e-6, 1e-4)
GUARDED_DAMPINGS = tuple(reversed(JOINT_DAMPINGS))
ARMIJO = 1e-4
MAX_BACKTRACKS = 18


def _newton_step(objective, parameters, precondition, raw_a_gradient, guarded):
    """Take one damped truncated Newton step with an objective-decrease check."""
    shape = [parameter.shape for parameter in parameters]
    size = [parameter.numel() for parameter in parameters]

    def flatten(vectors):
        return torch.cat([value.detach().flatten() for value in vectors])

    def blocks(vector):
        return tuple(value.reshape(dims) for value, dims in zip(vector.split(size), shape, strict=True))

    def raw_norm(vectors):
        head, a, b = vectors
        return float(torch.linalg.vector_norm(flatten((head, raw_a_gradient(a.detach()), b))))

    current = objective()
    differentiable_gradient = torch.autograd.grad(current, parameters, create_graph=True)
    gradient = flatten(differentiable_gradient)
    initial = float(current.detach())
    origin = [parameter.detach().clone() for parameter in parameters]
    squared_norm = torch.dot(gradient, gradient)
    initial_raw_norm = raw_norm(differentiable_gradient) if guarded else 0.0
    maximum_raw_norm = max(3e-6, 1.25 * initial_raw_norm) if guarded else None
    if not math.isfinite(initial) or not math.isfinite(initial_raw_norm) or not float(squared_norm) < math.inf:
        raise ValueError('Joint Newton received a nonfinite objective or gradient')
    if not float(squared_norm) > 0:
        raise ArithmeticError('Joint Newton needs a nonzero gradient to form a step')

    evaluations = 1
    total_cg_steps = 0
    for damping in GUARDED_DAMPINGS if guarded else JOINT_DAMPINGS:
        apply_preconditioner = precondition(damping)
        direction = torch.zeros_like(gradient)
        residual = -gradient
        search = flatten(apply_preconditioner(blocks(residual)))
        squared_residual = torch.dot(residual, search)
        threshold = float(squared_residual) * 1e-12
        for step in range(MAX_CG_STEPS):
            hessian_vector = torch.autograd.grad(
                differentiable_gradient, parameters, blocks(search), retain_graph=True)
            curvature = flatten(hessian_vector) + damping * search
            denominator = torch.dot(search, curvature)
            total_cg_steps += 1
            if not torch.isfinite(denominator) or denominator <= 0:
                if step == 0:
                    direction = search
                break
            size_along_search = squared_residual / denominator
            direction = direction + size_along_search * search
            residual = residual - size_along_search * curvature
            preconditioned = flatten(apply_preconditioner(blocks(residual)))
            next_squared = torch.dot(residual, preconditioned)
            if next_squared <= threshold:
                break
            search = preconditioned + (next_squared / squared_residual) * search
            squared_residual = next_squared
        slope = float(torch.dot(gradient, direction))
        if not math.isfinite(slope) or slope >= 0:
            continue
        direction_blocks = blocks(direction)
        for backtrack in range(MAX_BACKTRACKS):
            scale = 0.5 ** backtrack
            with torch.no_grad():
                for parameter, start, update in zip(parameters, origin, direction_blocks, strict=True):
                    parameter.copy_(start + scale * update)
            with torch.set_grad_enabled(guarded):
                candidate_loss = objective()
            candidate = float(candidate_loss.detach())
            evaluations += 1
            decreased = math.isfinite(candidate) and candidate <= initial + scale * ARMIJO * slope
            if decreased and (maximum_raw_norm is None
                              or raw_norm(torch.autograd.grad(candidate_loss, parameters)) <= maximum_raw_norm):
                return {'hessian_vector_products': total_cg_steps, 'damping': damping, 'step_scale': scale,
                        'objective_evaluations': evaluations, 'allowed_candidate_raw_gradient_l2': maximum_raw_norm}
            with torch.no_grad():
                for parameter, start in zip(parameters, origin, strict=True):
                    parameter.copy_(start)
        # In-place candidate evaluations increment tensor versions. Build a fresh
        # graph before another HVP solve at the restored parameters.
        current = objective()
        differentiable_gradient = torch.autograd.grad(current, parameters, create_graph=True)
        evaluations += 1

    raise ArithmeticError(f'Newton polishing found no decreasing step at objective {initial}')


def preconditioned_newton_step(objective, parameters, *, reduced_down, residual, norm_weight, gain, lengths,
                               head, ratings, contrast, effect_target, b, coordinate_scale, adapter_l2, head_l2,
                               raw_a_gradient, guarded):
    """Use an exact A-block preconditioner for the joint unchanged Hessian."""
    _, a, _ = parameters
    dense_a = _a_hessian(a, reduced_down, residual, norm_weight, gain, lengths, head, ratings, contrast,
                         effect_target, b, coordinate_scale, adapter_l2)
    count = len(ratings)
    with torch.no_grad():
        adjusted = (residual + gain * (reduced_down @ a.T @ b.T)) * (1 + norm_weight)
        unit = adjusted / torch.linalg.vector_norm(adjusted, dim=1, keepdim=True)
        design = torch.cat((torch.ones_like(lengths[:, :1]), unit, lengths), dim=1)
        logits = (design @ head.flatten(1)).reshape_as(ratings)
        probabilities = logits.softmax(dim=-1)
        levels = torch.arange(1, probabilities.shape[-1] + 1, dtype=head.dtype, device=head.device)
        centered = levels - (probabilities @ levels).unsqueeze(-1)
        effect_jacobian = design.T @ (contrast[:, None, None] * probabilities * centered).flatten(1)
        log_loss = (design.square().T @ (probabilities * (1 - probabilities)).flatten(1)) / count
        head_diagonal = (log_loss + effect_jacobian.square()).reshape_as(head)
        head_diagonal[1:] += head_l2 / count
        b_diagonal = adapter_l2 / count
        a_diagonal = dense_a.diagonal().abs().clamp_min(torch.finfo(dense_a.dtype).eps)

    def make_preconditioner(damping):
        system = dense_a.clone()
        system.diagonal().add_(damping)
        lower, info = torch.linalg.cholesky_ex(system)
        if int(info) != 0:
            # A diagonal positive preconditioner remains valid when the
            # exact local block has negative curvature.
            def solve_a(value):
                return value / (a_diagonal.reshape_as(a) + damping)
        else:
            def solve_a(value):
                return torch.cholesky_solve(value.reshape(-1, 1), lower).reshape_as(a)

        def apply(values):
            h, c, t = values
            safe_head = (head_diagonal + damping).clamp_min(torch.finfo(head_diagonal.dtype).eps)
            return h / safe_head, solve_a(c), t / (b_diagonal + damping)

        return apply

    return _newton_step(objective, parameters, make_preconditioner, raw_a_gradient, guarded)


def _a_hessian(a, reduced_down, residual, norm_weight, gain, lengths, head, ratings, contrast, effect_target, b,
               coordinate_scale, adapter_l2):
    """Assemble the exact A-block Hessian from rank-sized sentence derivatives."""
    with torch.no_grad():
        width = residual.shape[1]
        hidden_head = head[1:1 + width].flatten(1)
        lexical_head = head[1 + width:].flatten(1)
        center = residual * (1 + norm_weight)
        basis = gain * (1 + norm_weight[:, None]) * b
        center_sq = center.square().sum(dim=1)
        cross = center @ basis
        gram = basis.T @ basis
        base_logits = center @ hidden_head
        variable_logits = basis.T @ hidden_head
        lexical_logits = head[0].flatten() + lengths @ lexical_head
        coordinates = reduced_down @ a.T
    levels = torch.arange(1, ratings.shape[-1] + 1, dtype=a.dtype, device=a.device)
    chunk = len(ratings)

    def log_prob(z, sq, dot, base, lexical):
        squared_length = sq + 2 * torch.dot(dot, z) + torch.dot(z, gram @ z)
        return ((base + z @ variable_logits) * torch.rsqrt(squared_length) + lexical).reshape(
            ratings.shape[1:]).log_softmax(dim=-1)

    def expectation(*args):
        return log_prob(*args).exp() @ levels

    local_args = (coordinates, center_sq, cross, base_logits, lexical_logits)
    expectations = torch.vmap(expectation)(*local_args)
    effect_error = contrast @ expectations - effect_target

    def independent_loss(z, sq, dot, base, lexical, rating, sign):
        logp = log_prob(z, sq, dot, base, lexical)
        nll = -(rating * logp).sum() / chunk
        return nll + sign * torch.dot(logp.exp() @ levels, effect_error)

    blocks = torch.vmap(torch.func.hessian(independent_loss))(*local_args, ratings, contrast)
    jacobian = torch.vmap(torch.func.jacrev(expectation))(*local_args)
    # Outer product across all positions of each rewrite effect's Jacobian.
    global_jacobian = torch.einsum('ik,itr,i->trk', reduced_down, jacobian, contrast).flatten(1)
    weighted = reduced_down.T * blocks.permute(1, 2, 0).unsqueeze(-2)
    dense = (weighted @ reduced_down).permute(0, 2, 1, 3).reshape(a.numel(), a.numel())
    dense = dense + global_jacobian.T @ global_jacobian
    dense.diagonal().add_((adapter_l2 / chunk / coordinate_scale.square()).repeat(a.shape[0]))
    return (dense + dense.T) * 0.5


def newton_a_step(objective, a, *, reduced_down, residual, norm_weight, gain, lengths, head, ratings, contrast,
                  effect_target, b, coordinate_scale, adapter_l2):
    """Solve the local A Newton system directly, then test the unchanged objective."""
    current = objective()
    gradient = torch.autograd.grad(current, [a])[0].flatten()
    initial = float(current.detach())
    origin = a.detach().clone()
    hessian = _a_hessian(a, reduced_down, residual, norm_weight, gain, lengths, head, ratings, contrast, effect_target,
                        b, coordinate_scale, adapter_l2)
    if not math.isfinite(initial) or not torch.isfinite(gradient).all() or not torch.isfinite(hessian).all():
        raise ValueError('A-block Newton received nonfinite objective curvature')

    diagonal_scale = hessian.diagonal().abs().clamp_min(torch.finfo(hessian.dtype).eps)
    evaluations = 1
    for factorization, damping in enumerate((0.0, 1e-8, 1e-6, 1e-4, 0.01, 1.0), 1):
        trial = hessian.clone()
        trial.diagonal().add_(damping * diagonal_scale)
        lower, success = torch.linalg.cholesky_ex(trial)
        if int(success) != 0:
            continue
        direction = torch.cholesky_solve(-gradient[:, None], lower).reshape_as(a)
        slope = float(torch.dot(gradient, direction.flatten()))
        if not math.isfinite(slope) or slope >= 0:
            continue
        for backtrack in range(MAX_BACKTRACKS):
            scale = 0.5 ** backtrack
            with torch.no_grad():
                a.copy_(origin + scale * direction)
                candidate = float(objective())
                evaluations += 1
                if math.isfinite(candidate) and candidate <= initial + scale * ARMIJO * slope:
                    return {'hessian_vector_products': 0, 'dense_factorizations': factorization,
                            'damping': damping, 'step_scale': scale, 'objective_evaluations': evaluations}
                a.copy_(origin)

    raise ArithmeticError(f'A-block Newton polishing found no decreasing step at objective {initial}')
