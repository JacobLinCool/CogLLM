"""Differentiable observation likelihoods for joint reading models."""

import torch
from torch.nn import functional as F


def ordered_cutpoints(raw_gaps):
    """Map K-2 unconstrained gaps to K-1 centered, ordered cutpoints."""
    gaps = F.softplus(raw_gaps) + torch.finfo(raw_gaps.dtype).eps
    origin = raw_gaps.new_zeros((*raw_gaps.shape[:-1], 1))
    cuts = torch.cat((origin, gaps.cumsum(dim=-1)), dim=-1)
    return cuts - cuts.mean(dim=-1, keepdim=True)


def ordered_log_probs(location, cutpoints):
    """Return all category log probabilities for a cumulative logistic model."""
    offsets = cutpoints - location.unsqueeze(-1)
    # sigmoid(b) - sigmoid(a) = sigmoid(b) * sigmoid(-a) * (1 - exp(a-b)).
    # Compute a-b from the cutpoints before shifting them by the location.
    log_gap = (-torch.expm1(cutpoints[..., :-1] - cutpoints[..., 1:])).log()
    middle = F.logsigmoid(offsets[..., 1:]) + F.logsigmoid(-offsets[..., :-1]) + log_gap
    return torch.cat((F.logsigmoid(offsets[..., :1]), middle,
                      F.logsigmoid(-offsets[..., -1:])), dim=-1)


def categorical_nll(log_probabilities, counts):
    """One loss per count row, without multiplying unobserved categories by -inf."""
    observed = torch.where(counts > 0, log_probabilities, 0.0)
    return -(counts * observed).sum(dim=-1)


def coarsened_categorical_nll(log_probabilities, groups, counts):
    """Score observed groups of alternatives using a boolean [..., group, category] mask."""
    # An empty, unobserved group must not introduce a logsumexp(-inf, ...) gradient.
    allowed = groups | (counts == 0).unsqueeze(-1)
    log_groups = log_probabilities.unsqueeze(-2).masked_fill(~allowed, -torch.inf)
    return categorical_nll(log_groups.logsumexp(dim=-1), counts)


def first_event_log_probs(logits, risk=None):
    """Return log first-event mass at each stop and log survival after 0 through T stops."""
    log_no_event = F.logsigmoid(-logits)
    log_event = F.logsigmoid(logits)
    if risk is not None:
        log_no_event = log_no_event.masked_fill(~risk, 0.0)
        log_event = log_event.masked_fill(~risk, -torch.inf)
    initial = logits.new_zeros((*logits.shape[:-1], 1))
    log_survival = torch.cat((initial, log_no_event.cumsum(dim=-1)), dim=-1)
    return log_survival[..., :-1] + log_event, log_survival


def first_event_interval_nll(logits, lower, upper, risk=None):
    """Score a first event in (lower, upper], or censor at lower when upper is -1.

    Bounds count presented stops, so zero is before the first stop. The optional
    risk mask excludes padded or ineligible stops. A bounded event requires at
    least one eligible stop in its interval.
    """
    censored = upper == -1
    invalid = (lower < 0) | (lower > logits.shape[-1])
    invalid = invalid | (~censored & ((upper <= lower) | (upper > logits.shape[-1])))
    if logits.shape[-1] == 0 or torch.any(invalid):
        raise ValueError('Invalid first-event interval or censoring stop')

    log_event, log_survival = first_event_log_probs(logits, risk)
    stops = torch.arange(1, logits.shape[-1] + 1, device=logits.device)
    inside = (stops > lower.unsqueeze(-1)) & (stops <= upper.unsqueeze(-1))
    if risk is not None and torch.any(~censored & ~(inside & risk).any(dim=-1)):
        raise ValueError('First-event interval has no eligible stop')
    log_inside = log_event.masked_fill(~inside, -torch.inf)
    # Keep the unused interval branch differentiable even on right-censored rows.
    log_inside = log_inside.masked_fill(censored.unsqueeze(-1) & (stops == 1), 0.0)
    log_censor = log_survival.gather(-1, lower.unsqueeze(-1)).squeeze(-1)
    return -torch.where(censored, log_censor, log_inside.logsumexp(dim=-1))
