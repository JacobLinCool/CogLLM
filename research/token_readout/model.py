"""Causal token readout and observation likelihoods for one Qwen pass."""

from dataclasses import dataclass
from math import log, sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from research.token_readout.extraction import causal_attention_blocks


@dataclass(frozen=True)
class Scores:
    opening: Tensor
    resolution: Tensor
    continuity: Tensor


class TokenReadout(nn.Module):
    def __init__(self, input_width=2048, readout_width=48, attention_heads=48):
        super().__init__()
        self.final = nn.Linear(input_width, readout_width, bias=False)
        self.middle = nn.Linear(input_width, readout_width, bias=False)
        self.attention_mix = nn.Linear(readout_width, attention_heads)
        self.observation = nn.Linear(readout_width * 3, 1)
        self.query = nn.Linear(readout_width * 3, readout_width, bias=False)
        self.key = nn.Linear(readout_width, readout_width, bias=False)
        self.resolve_bias = nn.Parameter(torch.zeros(()))
        self.attention_weight = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.observation.weight)
        nn.init.zeros_(self.observation.bias)
        nn.init.zeros_(self.attention_mix.weight)
        nn.init.zeros_(self.attention_mix.bias)
        nn.init.normal_(self.query.weight, std=0.015)

    def forward(self, features) -> Scores:
        hidden = features.hidden.float()
        middle = features.mid.float()
        state = self.final(F.layer_norm(hidden, hidden.shape[-1:]))
        state = state + self.middle(F.layer_norm(middle, middle.shape[-1:]))
        state = F.layer_norm(state, state.shape[-1:])
        previous = torch.cat((state[:1], state[:-1]))
        change = state - previous
        mix = self.attention_mix(state).softmax(-1)
        source = self.key(state)
        event, resolution, continuity = [], [], []
        positions = torch.arange(len(state), device=state.device)
        for start, block in causal_attention_blocks(features.queries, features.keys):
            evidence = block.flatten(0, 1).transpose(0, 1).float()
            length = len(evidence)
            end = start + length
            native = torch.einsum('rh,rht->rt', mix[start:end], evidence)
            earlier = positions[None, :] < positions[start:end, None]
            # Only strictly earlier positions contribute to the context summary.
            previous_mass = native * earlier
            previous_context = (previous_mass @ state) / previous_mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            representation = torch.cat((state[start:end], change[start:end], previous_context), dim=-1)
            event.append(self.observation(representation).squeeze(-1))
            semantic = self.query(representation) @ source.T / sqrt(source.shape[-1])
            # Attend over cues at any prior distance. The uniform reference removes the basic prefix-length scale.
            relative_mass = (native * (positions[start:end, None] + 1)).clamp_min(1e-8).log()
            resolution.append(semantic + self.attention_weight * relative_mass + self.resolve_bias)
            continuity.append(F.cosine_similarity(state[start:end], previous_context, dim=-1))
        return Scores(torch.cat(event), torch.cat(resolution), torch.cat(continuity))


def intervals(offsets, start, end) -> tuple[int, int]:
    matching = [index for index, (left, right) in enumerate(offsets)
                if right > left and left < end and right > start]
    if (not matching or offsets[matching[-1]][1] != end
            or matching != list(range(matching[0], matching[-1] + 1))
            or offsets[matching[0]][0] > start):
        raise ValueError(f'Observation [{start}, {end}) does not end on a certified token boundary')
    return matching[0], matching[-1] + 1


def phase_first_event(sentence_logits: Tensor) -> Tensor:
    no_report = F.logsigmoid(-sentence_logits).cumsum(0)
    survival = torch.cat((no_report.new_zeros(1), no_report))
    return torch.cat((survival[:-1] + F.logsigmoid(sentence_logits), survival[-1:]))


def first_report_logits(scores, spans, prior) -> Tensor:
    residual = torch.stack([torch.logsumexp(scores[start:end], dim=0) - log(end - start) for start, end in spans])
    return residual + prior


def first_report_nll(logits, counts) -> Tensor:
    mass = phase_first_event(logits)
    return -(counts * mass).sum() / counts.sum()


def interval_nll(logits: Tensor, risk_start: int, start: int, end: int) -> Tensor:
    """Negative log mass of first activation in [start, end), conditional on risk start."""
    if not 0 <= risk_start <= start < end <= len(logits):
        raise ValueError('Invalid first-hit time window')
    before = F.softplus(logits[risk_start:start]).sum()
    inside = F.softplus(logits[start:end]).sum().clamp_min(torch.finfo(logits.dtype).tiny)
    return before - torch.log(-torch.expm1(-inside))


def censored_nll(logits: Tensor, risk_start: int, end: int) -> Tensor:
    if not 0 <= risk_start < end <= len(logits):
        raise ValueError('Invalid right-censoring time window')
    return F.softplus(logits[risk_start:end]).sum()


def matched_effect(logits, chances) -> Tensor:
    """Calibrate the contrast of observed reports at the identical target sentence."""
    return F.smooth_l1_loss(logits[1].sigmoid() - logits[0].sigmoid(), chances[1] - chances[0], beta=0.1)


def question_mass(opening_logits, resolution_logits, candidates=None):
    """Expected unresolved candidates and causal resolution mass by token and anchor."""
    n = len(opening_logits)
    positions = torch.arange(n, device=opening_logits.device)
    visible = positions[:, None] >= positions
    earlier = positions[:, None] > positions
    opening = opening_logits.sigmoid()
    if candidates is not None:
        opening = opening * candidates
    keep = F.logsigmoid(-resolution_logits).masked_fill(~earlier, 0.0)
    survival = keep.cumsum(0)
    unresolved = survival.exp() * opening[None, :] * visible
    resolved = (survival - keep).exp() * opening[None, :] * resolution_logits.sigmoid() * earlier
    return opening, unresolved, resolved
