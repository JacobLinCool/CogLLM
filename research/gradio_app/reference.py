"""Rank prior word locations from certified states in an existing analysis."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from math import log1p
from pathlib import Path
from unicodedata import normalize

import numpy as np

from research.semantic_tracking.iirc_anchor_fit import WORD

from .causal_plaba import digest
from .checkpoint import BACKBONE


DIRECTORY = Path(__file__).resolve().parents[1] / 'results' / 'arrau-bridge-l4'
RELEASE_SHA256 = '0c770d52cdae40bfd66ec45745c29cf3448886afb2f01fe87ba5ca7b70bbd0fa'
FEATURES = ('log_word_distance', 'log_word_length', 'capitalized', 'log_visible_frequency',
            'trigger_overlap', 'function_word', 'has_digit', 'following_punctuation',
            'preceding_preposition', 'qwen_cosine')


@dataclass(frozen=True)
class ReferenceStates:
    row_by_end: dict[int, int]
    vectors: np.ndarray


@dataclass(frozen=True)
class ReferenceLocation:
    start: int
    end: int
    rank: int


@dataclass(frozen=True)
class ReferenceResult:
    start: int
    end: int
    candidates: tuple[ReferenceLocation, ...]


def phrase_locations(text: str, phrase: str, states: ReferenceStates) -> tuple[tuple[int, int], ...]:
    """Find exact full-word occurrences whose final token has a certified prefix."""
    literal = phrase.strip()
    if not literal:
        return ()
    return tuple((word.start(), word.start() + len(literal)) for word in WORD.finditer(text)
                 if word.start() + len(literal) in states.row_by_end and text.startswith(literal, word.start()))


def form(word: str) -> str:
    return normalize('NFKC', word).casefold().replace('’', "'")


class ReferenceReadout:
    def __init__(self, checkpoint, projection, *, directory: Path = DIRECTORY):
        manifest_file = directory / 'reference-release.json'
        if digest(manifest_file) != RELEASE_SHA256:
            raise ValueError('The reference release differs from its verified fit and holdout bindings')
        manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
        fit_path, scores_path = (directory / filename for filename in ('fit.json', 'scores.json'))
        if digest(fit_path) != manifest['fit_sha256'] or digest(scores_path) != manifest['scores_sha256']:
            raise ValueError('The reference fit or held-out results changed')
        fit, scores = (json.loads(path.read_text(encoding='utf-8')) for path in (fit_path, scores_path))
        identity = {
            'candidate_id': checkpoint.identifier, 'model': BACKBONE['model'], 'revision': BACKBONE['revision'],
            'tokenizer_sha256': BACKBONE['snapshot_sha256']['tokenizer.json'],
            'projection_sha256': sha256(projection.tobytes(order='C')).hexdigest(),
            'projection_width': projection.shape[1],
            'anchor_heads_sha256': digest(checkpoint.directory / 'anchor' / 'heads.npz'),
        }
        if (manifest['schema'] != 1 or identity != manifest['projection_identity']
                or fit['projection_identity'] != identity or tuple(manifest['features']) != FEATURES
                or tuple(fit['features']) != FEATURES or scores['fit_sha256'] != manifest['fit_sha256']
                or not scores['pear_primary']['pass']):
            raise ValueError('The reference readout requires the verified Qwen, tokenizer and token projection')
        head = fit['fitted']['cosine']
        center, scale, coefficients = (np.asarray(head[name], dtype=np.float64)
                                       for name in ('center', 'scale', 'coefficients'))
        if (head['input_features'] != len(FEATURES) or head['columns'] != list(range(len(FEATURES)))
                or any(row.shape != (len(FEATURES),) or not np.isfinite(row).all()
                       for row in (center, scale, coefficients)) or np.any(scale <= 0)):
            raise ValueError('Invalid reference coefficients or feature scaling')
        self.effective = coefficients / scale
        self.intercept = float(-center @ self.effective)
        self.function_words = frozenset(manifest['function_words'])
        self.prepositions = frozenset(manifest['prepositions'])
        self.punctuation = manifest['following_punctuation']
        self.identifier = RELEASE_SHA256

    def cache(self, encoding, projection, tokenizer, lexical) -> ReferenceStates:
        # Lexical scoring already verified these prefixes. Reuse a proof only when
        # both readouts select the same last token at the same source endpoint.
        known = {word.end: word.token for word in lexical}
        positions = {right: index for index, (left, right) in enumerate(encoding.offsets) if left < right}
        row_by_end, indices = {}, []
        for word in WORD.finditer(encoding.text):
            end = word.end()
            index = positions.get(end)
            if index is None:
                continue
            if known.get(end) != index:
                prefix = tokenizer.encode(encoding.text[:end], add_special_tokens=False).ids
                if tuple(prefix) != encoding.tokens[:index + 1]:
                    continue
            row_by_end[end] = len(indices)
            indices.append(index)
        states = projection[indices].astype(np.float16)
        states.setflags(write=False)
        return ReferenceStates(row_by_end, states)

    def logits(self, text: str, states: ReferenceStates, start: int, end: int):
        if not 0 <= start < end <= len(text) or end not in states.row_by_end:
            raise ValueError('片語末端無法對齊。請使用完整且可對齊的原文詞。')
        words = list(WORD.finditer(text, 0, end))
        terms = [form(word.group()) for word in words]
        count = sum(word.end() <= start for word in words)
        indices = [index for index in range(count) if words[index].end() in states.row_by_end]
        if not indices:
            return (), np.empty(0, dtype=np.float64)
        frequency = Counter(terms)
        trigger = {form(word.group()) for word in WORD.finditer(text, start, end)}
        rows, spans, positions = [], [], []
        for index in indices:
            word, term = words[index], terms[index]
            stop = word.end()
            rows.append((
                log1p(count - index), log1p(len(term)), word.group()[0].isupper(),
                log1p(frequency[term]), len(term) >= 3 and term in trigger,
                term in self.function_words, any(character.isdigit() for character in term),
                stop < start and text[stop] in self.punctuation,
                bool(index and terms[index - 1] in self.prepositions),
            ))
            spans.append(word.span())
            positions.append(states.row_by_end[stop])
        anchor = states.vectors[states.row_by_end[end]].astype(np.float32)
        prior = states.vectors[positions].astype(np.float32)
        norms = np.linalg.norm(prior, axis=1) * np.linalg.norm(anchor)
        if not np.all(np.isfinite(norms) & (norms > 0)):
            raise RuntimeError('Invalid cached state at a certified reference endpoint')
        cosine = np.clip((prior @ anchor) / norms, -1.0, 1.0)
        design = np.column_stack((np.asarray(rows, dtype=np.float32), cosine))
        return spans, design @ self.effective + self.intercept

    def rank(self, text: str, states: ReferenceStates, start: int, end: int) -> ReferenceResult:
        spans, scores = self.logits(text, states, start, end)
        order = np.argsort(-scores, kind='stable')[:5]
        candidates = tuple(ReferenceLocation(*spans[index], rank) for rank, index in enumerate(order, 1))
        return ReferenceResult(start, end, candidates)
