"""Causal point and span readouts over frozen token features."""

from math import log, sqrt

import torch
from torch import nn
from torch.nn import functional as F

from research.token_readout.model import Scores


PROJECTION_BUDGET = 196_608
MATCH_WIDTH = 32
SPAN_TOKENS = 4


class PortableReadout(nn.Module):
    """Score opening and link resolution for all token positions at once.

    Each column corresponds to an anchor ending at that token. The span
    variant reads the preceding tokens as part of the anchor and can match it
    to a cue anywhere later in the document. Native relation inputs contain
    mean and maximum global attention as log1p(relative to uniform) - log(2).
    """

    def __init__(self, input_width, variant='point', relations=True):
        super().__init__()
        if type(input_width) is not int or input_width <= 0:
            raise ValueError('The backbone must provide a positive integer hidden width')
        if variant not in ('point', 'span') or type(relations) is not bool:
            raise ValueError('Specify a point or span readout and whether to use native relations')
        self.input_width = input_width
        self.variant = variant
        self.use_relations = relations
        self.anchor_tokens = SPAN_TOKENS if variant == 'span' else 1

        # Keep the input projection budget approximately constant across the
        # tested hidden widths. The pairwise match width is identical.
        width = max(8, min(64, PROJECTION_BUDGET // (2 * input_width)))
        self.final = nn.Linear(input_width, width, bias=False)
        self.middle = nn.Linear(input_width, width, bias=False)
        self.opening = nn.Linear(3 * width, 1)
        self.query = nn.Linear(3 * width, MATCH_WIDTH, bias=False)
        self.key = nn.Linear(width, MATCH_WIDTH, bias=False)
        self.resolve_bias = nn.Parameter(torch.zeros(()))
        if relations:
            self.relation_weight = nn.Parameter(torch.zeros(2))
        nn.init.zeros_(self.opening.weight)
        nn.init.zeros_(self.opening.bias)

        if variant == 'span':
            self.span_weight = nn.Linear(width, 1, bias=False)
            self.span_position_bias = nn.Parameter(torch.zeros(SPAN_TOKENS))
            self.interaction_query = nn.Linear(3 * width, MATCH_WIDTH, bias=False)
            self.interaction_key = nn.Linear(width, MATCH_WIDTH, bias=False)
            self.interaction_gain = nn.Parameter(torch.zeros(()))

    def _anchor_spans(self, state, positions):
        """Represent a short preceding window at each anchor endpoint."""
        if self.variant == 'point':
            return state
        span_states = F.pad(state, (0, 0, SPAN_TOKENS - 1, 0)).unfold(0, SPAN_TOKENS, 1).transpose(1, 2)
        span_indices = positions[:, None] + positions.new_tensor(range(1 - SPAN_TOKENS, 1))
        logits = self.span_weight(span_states).squeeze(-1) + self.span_position_bias
        weights = logits.masked_fill(span_indices < 0, -torch.inf).softmax(-1)
        pooled = (span_states * weights.unsqueeze(-1)).sum(1)
        return F.layer_norm(state + pooled, state.shape[-1:])

    def _history(self, state, relations, earlier, positions):
        if self.use_relations:
            weights = (relations[..., 0] + log(2)).expm1().clamp_min(0).masked_fill(~earlier, 0)
            return (weights @ state) / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        before = torch.cat((torch.zeros_like(state[:1]), state[:-1]), dim=0).cumsum(dim=0)
        return before / positions[:, None].clamp_min(1)

    def project(self, features):
        hidden, middle = features.hidden, features.mid
        if (hidden.ndim != 2 or len(hidden) == 0 or hidden.shape[1] != self.input_width
                or middle.shape != hidden.shape or middle.device != hidden.device
                or len(features.offsets) != len(hidden)):
            raise ValueError('The token-aligned backbone features do not match the readout width')

        return (self.final(F.layer_norm(hidden.float(), hidden.shape[-1:])),
                self.middle(F.layer_norm(middle.float(), middle.shape[-1:])))

    def score_projected(self, final, middle, native) -> Scores:
        if (final.ndim != 2 or len(final) == 0 or final.shape[1] != self.final.out_features
                or middle.shape != final.shape or middle.device != final.device):
            raise ValueError('Projected final and middle states must follow the same token axis')
        if self.use_relations and (native is None or native.shape != (len(final), len(final), 2)
                                   or native.device != final.device or not native.is_floating_point()):
            raise ValueError('Native relations must have two channels on the aligned token axes')
        state = F.layer_norm(final + middle, final.shape[-1:])
        previous = torch.cat((state[:1], state[:-1]), dim=0)
        positions = torch.arange(len(state), device=state.device)
        earlier = positions[:, None] > positions[None, :]
        relations = native.float() if self.use_relations else None
        history = self._history(state, relations, earlier, positions)
        cue = torch.cat((state, state - previous, history), dim=-1)
        anchors = self._anchor_spans(state, positions)

        # Two matrix products avoid an all-pairs tensor with a hidden axis.
        scale = 1 / sqrt(MATCH_WIDTH)
        resolution = (self.query(cue) @ self.key(anchors).T) * scale
        relation_shift = relations @ self.relation_weight if self.use_relations else 0.0
        resolution = resolution + relation_shift + self.resolve_bias
        if self.variant == 'span':
            interaction = (self.interaction_query(cue) @ self.interaction_key(anchors).T) * scale
            resolution = resolution + self.interaction_gain * F.silu(interaction + relation_shift)
        resolution = resolution.masked_fill(~earlier, torch.finfo(resolution.dtype).min)

        return Scores(self.opening(cue).squeeze(-1), resolution, F.cosine_similarity(state, history, dim=-1))

    def forward(self, features) -> Scores:
        return self.score_projected(*self.project(features), features.relations)
