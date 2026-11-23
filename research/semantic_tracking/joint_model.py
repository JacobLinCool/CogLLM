"""Shared causal readouts for position, prior source spans, and closure intervals."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


TASKS = ('onset', 'source', 'closure')
PROJECTION_SEED = 71
PROJECTED_SIZE = 48
GEOMETRY_SIZE = 8
INPUT_SIZE = GEOMETRY_SIZE + 3 * PROJECTED_SIZE
ADAPTER_SIZE = 40


def projection():
    rng = np.random.default_rng(PROJECTION_SEED)
    matrix = rng.integers(0, 2, size=(2048, PROJECTED_SIZE), dtype=np.int8)
    return (matrix.astype(np.float32) * 2 - 1) / np.sqrt(np.float32(PROJECTED_SIZE))


def readout_features(now, anchor, now_width, anchor_width, position, lag, relative, text_available,
                     present_state=None, anchor_state=None):
    """Each row uses its current location and one already known anchor."""
    values = np.broadcast_arrays(now, anchor, now_width, anchor_width, position, lag, relative, text_available)
    geometry = np.stack((
        np.log1p(values[0]) / 7,
        np.log1p(values[1]) / 7,
        np.log1p(values[2]) / 4,
        np.log1p(values[3]) / 4,
        np.log1p(values[4]) / 7,
        np.log1p(values[5]) / 6,
        values[6],
        values[7],
    ), axis=-1).astype(np.float32)
    shape = geometry.shape[:-1] + (PROJECTED_SIZE,)
    current = np.zeros(shape, np.float32) if present_state is None else np.broadcast_to(present_state, shape)
    earlier = np.zeros(shape, np.float32) if anchor_state is None else np.broadcast_to(anchor_state, shape)
    interaction = current * earlier * np.sqrt(np.float32(PROJECTED_SIZE))
    return np.concatenate((geometry, current, earlier, interaction), axis=-1)


class SemanticHeads(nn.Module):
    def __init__(self, tasks):
        super().__init__()
        self.adapter = nn.Sequential(nn.Linear(INPUT_SIZE, ADAPTER_SIZE), nn.SiLU(),
                                     nn.Linear(ADAPTER_SIZE, ADAPTER_SIZE), nn.SiLU())
        heads = {name: nn.Linear(ADAPTER_SIZE, 1) for name in TASKS}
        self.head = nn.ModuleDict({name: heads[name] for name in tasks})
        for name, bias in (('onset', -3.0), ('closure', -4.0), ('source', 0.0)):
            if name in self.head:
                nn.init.constant_(self.head[name].bias, bias)

    def forward(self, task, features):
        return self.head[task](self.adapter(features)).squeeze(-1)


def interval_nll(logits, lower, upper):
    """Log survival to lower plus event mass on (lower, upper]; upper=-1 censors."""
    # Wide intervals can have tiny remaining survival. Keep this subtraction in FP64.
    follow = F.logsigmoid(-logits.double())
    survival = F.pad(torch.cumsum(follow, dim=-1), (1, 0))
    first = survival.gather(-1, lower[..., None]).squeeze(-1)
    closed = upper >= 0
    # For censoring, gather a legal fictitious interval only for eager evaluation.
    safe_upper = torch.where(closed, upper, torch.ones_like(upper))
    safe_lower = torch.where(closed, lower, torch.zeros_like(lower))
    positions = torch.arange(1, follow.shape[-1] + 1, device=follow.device)
    interval = (positions > safe_lower[..., None]) & (positions <= safe_upper[..., None])
    log_no_event = follow.masked_fill(~interval, 0.0).sum(dim=-1)
    event = torch.log(-torch.expm1(log_no_event))
    return -(first + torch.where(closed, event, 0.0))


def objective(task, logits, mask, labels):
    if task == 'onset':
        marked, total = labels.unbind(-1)
        logits = logits[:, 0]
        return -(marked * F.logsigmoid(logits) + (total - marked) * F.logsigmoid(-logits)).sum() / total.sum()
    if task == 'source':
        all_words = logits.masked_fill(~mask, -torch.inf)
        gold_words = logits.masked_fill(~labels, -torch.inf)
        return (torch.logsumexp(all_words, -1) - torch.logsumexp(gold_words, -1)).mean()
    if task == 'closure':
        return interval_nll(logits, labels[:, 0], labels[:, 1]).mean()
    raise ValueError(task)


def parallel_trace(model, backbone_forward, char_ends, anchor_tokens, line_end_tokens, device):
    """Run a supplied frozen backbone once, then score fixed anchor threads in parallel over time.

    The callback must return the causal last hidden states as a float array, with one
    row per article token. Closing risk is calibrated at provided line boundaries
    with coordinates only. An anchor must coincide with one such boundary to
    receive that closing risk. Anchors are externally specified in this pilot.
    """
    char_ends = np.asarray(char_ends)
    lengths = np.diff(char_ends, prepend=0)
    if not len(char_ends) or np.any(lengths < 0) or char_ends[0] <= 0:
        raise ValueError('Expected monotone exclusive token ends in the original text')
    if (sorted(set(anchor_tokens)) != list(anchor_tokens) or sorted(set(line_end_tokens)) != list(line_end_tokens)
            or any(not 0 <= index < len(char_ends) for index in (*anchor_tokens, *line_end_tokens))):
        raise ValueError('Thread anchors and line boundaries must be sorted legal token indices')
    states = np.asarray(backbone_forward(), dtype=np.float32)
    norms = np.linalg.norm(states, axis=-1, keepdims=True)
    if states.shape != (len(char_ends), 2048) or not np.all(np.isfinite(norms) & (norms > 0)):
        raise ValueError('Expected one nonzero causal Qwen state per article token')
    features = (states / norms) @ projection()
    positions = np.arange(1, len(char_ends) + 1)

    def score(name, x):
        return model(name, torch.as_tensor(x, device=device)).double()

    with torch.inference_mode():
        event = readout_features(char_ends, 0, lengths, 0, positions, positions, 0, 1, features)
        opening = torch.sigmoid(score('onset', event)).cpu().numpy()
        line_lookup = {token: i + 1 for i, token in enumerate(line_end_tokens)}
        tracks = []
        for anchor in anchor_tokens:
            prior = readout_features(char_ends[anchor], char_ends[:anchor], lengths[anchor], lengths[:anchor],
                                     positions[:anchor], anchor - positions[:anchor] + 1, 0, 1,
                                     features[anchor], features[:anchor])
            source = torch.softmax(score('source', prior), dim=-1).cpu().numpy() if anchor else np.array([])
            closure_candidates = []
            survival = np.array([])
            hazards = np.array([])
            if anchor in line_lookup:
                closure_candidates = [index for index in line_end_tokens if index > anchor]
                earlier_line = line_lookup[anchor]
                later_lines = np.arange(earlier_line + 1, earlier_line + len(closure_candidates) + 1)
                x = readout_features(later_lines, earlier_line, 0, 0, later_lines, later_lines - earlier_line, 1, 0)
                logits = score('closure', x)
                hazards = torch.sigmoid(logits).cpu().numpy()
                survival = F.logsigmoid(-logits).cumsum(-1).exp().cpu().numpy()
            tracks.append({'anchor_token': anchor, 'source_tokens': list(range(anchor)), 'source_probability': source,
                           'closure_tokens': closure_candidates, 'closure_hazard': hazards, 'survival': survival})
    return {'onset': opening, 'tracks': tracks}
