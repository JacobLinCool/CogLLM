"""Build target-blind candidate features for earlier sentences in one Frick article."""

from collections import Counter
from math import log, log1p
import re
from unicodedata import normalize

import numpy as np


AGE_FEATURES = ('log_lag', 'log_lag_squared', 'lag_7', 'lag_8', 'lag_9', 'lag_10',
                'log_candidate_ordinal', 'log_candidate_ordinal_squared',
                'candidate_ordinal_5', 'candidate_ordinal_6')
TEXT_FEATURES = ('term_jaccard', 'current_prefix_idf_overlap', 'log1p_shared_prefix_singletons')
STATE_FEATURES = ('sentence_state_cosine', 'state_update_cosine', 'current_state_update_dot_earlier_state')
FEATURES = AGE_FEATURES + TEXT_FEATURES + STATE_FEATURES
TERM = re.compile(r'[^\W_]+')


def lexical_terms(sentence):
    """Keep normalized words of length four or more and exact numeral tokens."""
    return frozenset(word for word in TERM.findall(normalize('NFKC', sentence).casefold())
                     if len(word) >= 4 or word.isdecimal())


def unit_sentence_states(states, sentence_count):
    values = np.asarray(states, dtype=np.float64)
    if (values.ndim != 2 or values.shape[0] != sentence_count or values.shape[1] == 0
            or not np.isfinite(values).all()):
        raise ValueError('Expected one finite contextual state per displayed sentence')
    scales = np.max(np.abs(values), axis=1, keepdims=True)
    if np.any(scales <= 0):
        raise ValueError('A contextual sentence state has zero norm')
    scaled = values / scales
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


def text_overlap(current, earlier, prefix_counts, idf, current_weight):
    shared = current & earlier
    if not shared:
        return 0., 0., 0.
    weight = sum(idf[word] for word in sorted(shared))
    exclusive = sum(prefix_counts[word] == 1 for word in shared)
    return len(shared) / len(current | earlier), weight / current_weight, log1p(exclusive)


def build_features(sentences, states):
    """Return one feature matrix per visible t, with row j denoting prior sentence j."""
    if not all(isinstance(sentence, str) for sentence in sentences):
        raise ValueError('Expected original sentence strings in display order')
    contextual = unit_sentence_states(states, len(sentences))
    changes = np.zeros_like(contextual)
    changes[1:] = contextual[1:] - contextual[:-1]
    magnitudes = np.linalg.norm(changes, axis=1, keepdims=True)
    unit_changes = np.divide(changes, magnitudes, out=np.zeros_like(changes), where=magnitudes > 0)

    terms = [lexical_terms(sentence) for sentence in sentences]
    prefix_counts = Counter()
    matrices = []
    for t, current in enumerate(terms):
        matrix = np.empty((t, len(FEATURES)), dtype=np.float64)
        if t:
            lags = np.arange(t, 0, -1)
            log_lags = np.log(lags)
            ordinals = np.arange(1, t + 1)
            log_ordinals = np.log(ordinals)
            matrix[:, :len(AGE_FEATURES)] = np.column_stack((
                log_lags, np.square(log_lags), lags[:, None] == np.arange(7, 11),
                log_ordinals, np.square(log_ordinals), ordinals[:, None] == np.array([5, 6]),
            ))
            idf = {word: 1 + log((t + 1) / (1 + prefix_counts[word])) for word in sorted(current)}
            current_weight = sum(idf.values())
            matrix[:, len(AGE_FEATURES):-len(STATE_FEATURES)] = [
                text_overlap(current, earlier, prefix_counts, idf, current_weight) for earlier in terms[:t]
            ]
            matrix[:, -len(STATE_FEATURES):] = np.column_stack((
                contextual[:t] @ contextual[t],
                unit_changes[:t] @ unit_changes[t],
                contextual[:t] @ changes[t],
            ))
        matrices.append(matrix)
        prefix_counts.update(current)
    return tuple(matrices)
