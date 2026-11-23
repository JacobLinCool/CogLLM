"""Marginalize anchored question states after a single causal text encoding."""

import torch
from torch import Tensor


TRACE_STATES = ('not_open', 'unresolved', 'set_aside', 'resolved')
ALLOWED_EDGES = (
    (True, True, False, False),
    (False, True, True, True),
    (False, True, True, True),
    (False, True, False, True),
)
STATE_COUNT = len(TRACE_STATES)


def transition_log_probs(logits: Tensor, visible: Tensor) -> Tensor:
    """Row-normalize each visible transition and use identity before an anchor is seen.

    The final three axes of logits are token, previous state, next state.
    The boolean visible mask has the same batch and token axes.
    A visible row may forbid some legal edges with negative infinity.
    """
    if (logits.ndim < 3 or logits.shape[-2:] != (STATE_COUNT, STATE_COUNT) or not logits.is_floating_point()
            or visible.shape != logits.shape[:-2] or visible.dtype != torch.bool or visible.device != logits.device):
        raise ValueError('Expected floating-point state logits and a boolean visibility mask at each token')
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    allowed = torch.tensor(ALLOWED_EDGES, device=logits.device)
    diagonal = torch.eye(STATE_COUNT, dtype=torch.bool, device=logits.device)
    active = visible[..., None, None]
    permitted = torch.where(active, allowed, diagonal)
    masked = torch.where(active, logits, 0).masked_fill(~permitted, -torch.inf)
    if torch.isnan(masked).any() or torch.isposinf(masked).any() or not torch.isfinite(masked).any(dim=-1).all():
        raise ValueError('Every visible state must have a legal finite logit, with no NaN or positive infinity')
    return torch.log_softmax(masked, dim=-1)


def _validate_trace(transitions: Tensor, initial: Tensor) -> Tensor:
    if (transitions.ndim < 3 or transitions.shape[-2:] != (STATE_COUNT, STATE_COUNT)
            or not transitions.is_floating_point()
            or initial.shape != transitions.shape[:-3] + (STATE_COUNT,)
            or not initial.is_floating_point() or initial.device != transitions.device):
        raise ValueError('Expected [..., T, 4, 4] transitions and [..., 4] initial log probabilities')
    initial = initial.to(transitions.dtype)
    initial_mass = initial.logsumexp(dim=-1)
    if not torch.allclose(initial_mass, torch.zeros_like(initial_mass), rtol=0, atol=1e-6):
        raise ValueError('Initial log probabilities must sum to one in probability space')
    return initial


def _logsumexp(values: Tensor, dim: int) -> Tensor:
    # Impossible paths are valid. A raw all-minus-inf logsumexp has undefined gradients.
    reachable = ~torch.isneginf(values).all(dim=dim, keepdim=True)
    safe = torch.where(reachable, values, 0)
    return torch.logsumexp(safe, dim=dim).masked_fill(~reachable.squeeze(dim), -torch.inf)


def _compose(first: Tensor, second: Tensor) -> Tensor:
    return _logsumexp(first.unsqueeze(-1) + second.unsqueeze(-3), dim=-2)


def _prefix_products(matrices: Tensor) -> Tensor:
    """Inclusive log-semiring scan with a linear number of small matrix products."""
    length = matrices.shape[-3]
    if length < 2:
        return matrices
    pairs = length // 2
    paired = 2 * pairs
    pair_products = _compose(matrices[..., :paired:2, :, :], matrices[..., 1:paired:2, :, :])
    odd = _prefix_products(pair_products)
    following = matrices[..., 2::2, :, :]
    even_tail = _compose(odd[..., :following.shape[-3], :, :], following)
    even = torch.cat((matrices[..., :1, :, :], even_tail), dim=-3)
    interleaved = torch.stack((even[..., :pairs, :, :], odd), dim=-3).flatten(-4, -3)
    return torch.cat((interleaved, even[..., pairs:, :, :]), dim=-3)


def _final_product(matrices: Tensor) -> Tensor:
    while matrices.shape[-3] > 1:
        paired = matrices.shape[-3] // 2 * 2
        products = _compose(matrices[..., :paired:2, :, :], matrices[..., 1:paired:2, :, :])
        matrices = torch.cat((products, matrices[..., paired:, :, :]), dim=-3)
    return matrices.squeeze(-3)


def _marginals(transitions: Tensor, initial: Tensor) -> Tensor:
    products = _prefix_products(transitions)
    return _logsumexp(initial[..., None, :, None] + products, dim=-2)


def trace_log_marginals(log_transition: Tensor, initial_log_probs: Tensor) -> Tensor:
    """Return unconditioned state occupancy by anchored question and token."""
    initial = _validate_trace(log_transition, initial_log_probs)
    return _marginals(log_transition, initial)


def trace_log_events(log_transition: Tensor, initial_log_probs: Tensor) -> Tensor:
    """Return transition-event log masses, with previous and next state axes."""
    initial = _validate_trace(log_transition, initial_log_probs)
    if log_transition.shape[-3] == 0:
        return log_transition
    earlier = _marginals(log_transition[..., :-1, :, :], initial)
    previous = torch.cat((initial.unsqueeze(-2), earlier), dim=-2)
    return previous.unsqueeze(-1) + log_transition


def trace_nll(log_transition: Tensor, initial_log_probs: Tensor, observation_log_likelihood: Tensor,
              observed: Tensor) -> Tensor:
    """Return one negative log likelihood per trace for its actual observations.

    Log emissions can encode exact states with zero for compatibility and negative
    infinity for impossible states. Missing stops multiply by identity in the
    observation model, so the hidden state still evolves on visible tokens.
    """
    initial = _validate_trace(log_transition, initial_log_probs)
    if (observation_log_likelihood.shape != log_transition.shape[:-1]
            or not observation_log_likelihood.is_floating_point()
            or observation_log_likelihood.device != log_transition.device
            or observed.shape != log_transition.shape[:-2] or observed.dtype != torch.bool
            or observed.device != log_transition.device):
        raise ValueError('Expected one vector of log emissions and one observation flag per token')
    emissions = torch.where(observed.unsqueeze(-1), observation_log_likelihood.to(log_transition.dtype), 0)
    if torch.isnan(emissions).any() or (emissions > 0).any():
        raise ValueError('Observed emissions must be log probabilities no greater than zero')
    if log_transition.shape[-3] == 0:
        return -_logsumexp(initial, dim=-1)
    product = _final_product(log_transition + emissions.unsqueeze(-2))
    terminal = _logsumexp(initial.unsqueeze(-1) + product, dim=-2)
    loss = -_logsumexp(terminal, dim=-1)
    return torch.where(observed.any(dim=-1), loss, 0)


def first_hit_log_masses(log_transition: Tensor, initial_log_probs: Tensor, target: int = 3) -> Tensor:
    """Return first entry mass at each token and a final no-entry mass.

    This probability uses paths that have never reached the target, even if the
    state graph allows subsequent reopening. Time zero must be outside the target.
    """
    initial = _validate_trace(log_transition, initial_log_probs)
    if not 0 <= target < STATE_COUNT or not torch.isneginf(initial[..., target]).all():
        raise ValueError('Initial mass must be outside the designated first-hit state')
    if log_transition.shape[-3] == 0:
        return _logsumexp(initial, dim=-1).unsqueeze(-1)
    entered = torch.arange(STATE_COUNT, device=log_transition.device) == target
    alive = _marginals(log_transition.masked_fill(entered, -torch.inf), initial)
    before = torch.cat((initial.unsqueeze(-2), alive[..., :-1, :]), dim=-2)
    hit = _logsumexp(before + log_transition[..., :, :, target], dim=-1)
    unhit = _logsumexp(alive[..., -1, :], dim=-1).unsqueeze(-1)
    return torch.cat((hit, unhit), dim=-1)
