"""Apply the CompLex target-end head to source spans from an existing forward."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from research.semantic_tracking.iirc_anchor_fit import WORD
from research.stage5.complex_tokenization import token_span

from .causal_plaba import CausalEncoding, HIDDEN_WIDTH, digest


CHECKPOINT = Path(__file__).resolve().parents[1] / 'results' / 'stage5-complex' / 'fit'


@dataclass(frozen=True)
class LexicalRating:
    start: int
    end: int
    token: int
    difficulty: float


class LexicalReadout:
    """Estimate a designated lexical rating on the published 0–1 scale."""

    def __init__(self, checkpoint: Path = CHECKPOINT):
        head_path = checkpoint / 'heads.npz'
        manifest = json.loads((checkpoint / 'fitting.json').read_text(encoding='utf-8'))
        self.head_sha256 = digest(head_path)
        if self.head_sha256 != manifest['artifacts']['heads.npz']['sha256']:
            raise ValueError('The CompLex head differs from its frozen fitting manifest')
        with np.load(head_path, allow_pickle=False) as heads:
            method_index = heads['methods'].tolist().index('target_end')
            fraction_index = heads['fractions'].tolist().index(100)
            weights = heads['weights_target_end'][fraction_index]
            biases = heads['intercepts'][method_index, fraction_index]
            if (weights.ndim != 2 or weights.shape[1] != HIDDEN_WIDTH or len(weights) == 0
                    or biases.shape != (len(weights),) or len(weights) != len(heads['seeds'])
                    or not np.isfinite(weights).all() or not np.isfinite(biases).all()
                    or not np.all(weights == weights[0]) or not np.all(biases == biases[0])):
                raise ValueError('Expected the common full-training CompLex target-end head')
            self.weights = weights[0]
            self.intercept = float(biases[0])

    def score_states(self, states: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != HIDDEN_WIDTH:
            raise ValueError('Expected one frozen target-end vector per lexical span')
        norms = np.linalg.norm(states, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms) & (norms > 0)):
            raise ValueError('Lexical target states must have finite positive norms')
        return np.clip((states / norms) @ self.weights + self.intercept, 0, 1)

    def rate(self, encoding: CausalEncoding, tokenizer) -> tuple[tuple[LexicalRating, ...], tuple[tuple[int, int], ...]]:
        certified, unavailable = [], []
        text, offsets = encoding.text, encoding.offsets
        for word in WORD.finditer(text):
            start, end = word.span()
            _, tokens = token_span(text, offsets, start, end)
            if tokens is not None:
                last = tokens[1]
                reread = tokenizer.encode(text[:end], add_special_tokens=False).ids
                if tuple(reread) == encoding.tokens[:last + 1]:
                    certified.append((start, end, last))
                    continue
            unavailable.append((start, end))
        if not certified:
            return (), tuple(unavailable)
        states = encoding.states[[last for _, _, last in certified]]
        ratings = tuple(LexicalRating(start, end, last, float(value))
                        for (start, end, last), value in zip(certified, self.score_states(states), strict=True))
        return ratings, tuple(unavailable)
