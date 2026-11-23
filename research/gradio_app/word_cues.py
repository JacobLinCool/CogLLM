"""Score candidate lexical disambiguation cues from an existing causal forward."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from research.semantic_tracking.blott_format_prepare import TOKEN
from research.semantic_tracking.blott_word_fit import RankFit
from research.semantic_tracking.blott_word_states import word_updates

from .causal_plaba import CausalEncoding, HIDDEN_WIDTH, digest
from .checkpoint import BACKBONE, Checkpoint
from .lexical import LexicalRating


RELEASE = Path(__file__).resolve().parents[1] / 'results' / 'blott-word-l4' / 'word-cue-release.json'
RELEASE_SHA256 = '26b581e2754ed76540bb3dc337a427ba17aa556590d2ba2067b7eeb9f9dd65bd'
BLOCKS = ('position', 'surface', 'change_norm', 'delta')
WIDTHS = (2, 3, 1, HIDDEN_WIDTH)


@dataclass(frozen=True)
class CueCandidate:
    start: int
    end: int
    token: int
    sentence_number: int
    rank: int
    score: float


class CueReadout:
    def __init__(self, checkpoint: Checkpoint, *, release: Path = RELEASE):
        if digest(release) != RELEASE_SHA256:
            raise ValueError('The cue readout differs from the verified release')
        package = json.loads(release.read_text(encoding='utf-8'))
        identity = {'candidate_id': checkpoint.identifier, 'model': BACKBONE['model'],
                    'revision': BACKBONE['revision'], 'tokenizer_sha256': BACKBONE['snapshot_sha256']['tokenizer.json']}
        if (package['schema'] != 1 or package['identity'] != identity
                or package['evidence']['localization_2021_primary_pass'] is not True):
            raise ValueError('The cue readout requires the verified Qwen snapshot and candidate')
        fit = package['fit']
        centers = tuple(np.asarray(row, dtype=np.float64) for row in fit['centers'])
        scales = tuple(np.asarray(row, dtype=np.float64) for row in fit['scales'])
        coefficients = np.asarray(fit['coefficients'], dtype=np.float64)
        if (tuple(fit['blocks']) != BLOCKS or tuple(row.shape for row in centers) != tuple((w,) for w in WIDTHS)
                or tuple(row.shape for row in scales) != ((2,), (3,), (1,), ())
                or coefficients.shape != (sum(WIDTHS),)
                or any(not np.isfinite(row).all() for row in (*centers, *scales, coefficients))
                or any(np.any(row <= 0) for row in scales)):
            raise ValueError('Unexpected cue readout parameters')
        self.ranker = RankFit(BLOCKS, centers, scales, coefficients, fit['diagnostics'])
        self.identifier = RELEASE_SHA256

    def rate(self, encoding: CausalEncoding, tokenizer, sentences: list[tuple[int, int]],
             lexical: tuple[LexicalRating, ...]) -> tuple[CueCandidate, ...]:
        text = encoding.text
        positions = {right: i for i, (left, right) in enumerate(encoding.offsets) if left < right}
        certified = {word.end: word.token for word in lexical}
        result = []
        last_sentence_end = 0
        for number, (start, end) in enumerate(sentences, 1):
            if not last_sentence_end <= start < end <= len(text):
                raise ValueError('Cue scoring requires ordered, disjoint source sentences')
            last_sentence_end = end
            words = [(match.start() - start, match.end() - start) for match in TOKEN.finditer(text, start, end)]
            token_by_end = {}
            for _, stop in words:
                absolute = start + stop
                token = positions.get(absolute)
                if token is None:
                    continue
                if certified.get(absolute) != token:
                    prefix = tokenizer.encode(text[:absolute], add_special_tokens=False).ids
                    if tuple(prefix) != encoding.tokens[:token + 1]:
                        continue
                token_by_end[stop] = token
            if not any(a[1] in token_by_end and b[1] in token_by_end for a, b in zip(words, words[1:])):
                continue
            vectors = encoding.states[list(token_by_end.values())].astype(np.float64)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.all(np.isfinite(norms) & (norms > 0)):
                raise ValueError('Expected finite nonzero states at certified cue boundaries')
            unit_states = dict(zip(token_by_end, vectors / norms, strict=True))
            candidates = word_updates(text[start:end], words, token_by_end, unit_states)
            scores = self.ranker.score(candidates.features)
            ranks = np.empty(len(scores), dtype=np.int64)
            ranks[np.argsort(-scores, kind='stable')] = np.arange(1, len(scores) + 1)
            result.extend(CueCandidate(start + left, start + right, token, number, int(rank), float(score))
                          for (left, right), token, rank, score in
                          zip(candidates.spans, candidates.token_indices, ranks, scores, strict=True))
        return tuple(result)
