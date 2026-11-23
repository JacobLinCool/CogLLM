"""Apply the experimental PLABA distribution readout at certified sentence ends."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from research.stage3.probe import TARGETS
from research.stage3.run import MODEL

from .causal_plaba import HIDDEN_WIDTH, digest
from .plaba import BASE_REVISION


LEVELS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class SentenceDistribution:
    ease: tuple[float, ...]
    understanding: tuple[float, ...]


class OrdinalReadout:
    def __init__(self, directory: Path):
        head = directory / 'head.npz'
        self.head_sha256 = digest(head)
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if (self.head_sha256 != manifest['head_sha256'] or manifest['model'] != MODEL
                or manifest['revision'] != BASE_REVISION or tuple(manifest['target_order']) != TARGETS
                or tuple(manifest['ratings']) != LEVELS or not manifest['all_data_head_not_scored_out_of_fold']):
            raise ValueError('The ordinal PLABA weights differ from their development manifest')
        with np.load(head, allow_pickle=False) as weights:
            if tuple(weights['ratings']) != LEVELS or tuple(weights['targets']) != TARGETS:
                raise ValueError('Unexpected ordinal PLABA targets')
            self.weights = weights['effect_calibrated_weights']
            self.length_mean, self.length_scale = weights['length_mean'], weights['length_scale']
        if (self.weights.shape != (len(TARGETS), HIDDEN_WIDTH + 3, len(LEVELS))
                or self.length_mean.shape != (2,) or self.length_scale.shape != (2,)
                or any(array.dtype != np.float64 or not np.isfinite(array).all()
                       for array in (self.weights, self.length_mean, self.length_scale))
                or np.any(self.length_scale <= 0)):
            raise ValueError('Unexpected ordinal PLABA parameters')

    def score(self, hidden: np.ndarray, sentences: list[str]) -> tuple[SentenceDistribution, ...]:
        states = np.asarray(hidden, dtype=np.float64)
        if states.shape != (len(sentences), HIDDEN_WIDTH):
            raise ValueError('Expected one ordinal state per source sentence')
        if not sentences:
            return ()
        norms = np.linalg.norm(states, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms) & (norms > 0)):
            raise ValueError('Expected nonzero finite ordinal states')
        lengths = np.log1p([(len(text), len(text.split())) for text in sentences])
        lengths = (lengths - self.length_mean) / self.length_scale / np.sqrt(2)
        features = np.column_stack((np.ones(len(states)), states / norms, lengths))
        logits = np.einsum('nf,tfr->ntr', features, self.weights)
        unnormalized = np.exp(logits - logits.max(axis=-1, keepdims=True))
        p = unnormalized / unnormalized.sum(axis=-1, keepdims=True)
        return tuple(SentenceDistribution(tuple(ease), tuple(understanding)) for ease, understanding in p.tolist())
