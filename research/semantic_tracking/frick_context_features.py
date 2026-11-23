"""Causal lexical retrieval and state-change features for Frick sentence ends."""

from collections import Counter

import numpy as np

from research.semantic_tracking.frick_consistency_score import word_types


FEATURES = (
    "maximum_lexical_overlap",
    "mean_lexical_overlap",
    "lexical_weighted_relative_lag",
    "maximum_state_update_similarity",
    "lexical_weighted_update_similarity_difference",
    "lexical_weighted_state_similarity_difference",
    "state_update_norm",
)
LEXICAL_WEIGHT = 4.0


def contextual_features(sentences, normalized_states):
    """Use precisely the current visible sentence and all earlier sentence ends."""
    states = np.asarray(normalized_states, dtype=np.float64)
    if states.ndim != 2 or len(states) != len(sentences) or not len(states):
        raise ValueError("Each present sentence requires its causal sentence-end state")
    if not np.allclose(np.linalg.norm(states, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("Expected L2-normalized causal sentence-end states")
    changes = np.empty_like(states)
    changes[0] = states[0]
    changes[1:] = states[1:] - states[:-1]
    widths = np.linalg.norm(changes, axis=1)
    normalized_changes = np.divide(changes, widths[:, None], out=np.zeros_like(changes), where=widths[:, None] > 0)
    vocabulary = [word_types(sentence) for sentence in sentences]
    previous_documents = Counter(vocabulary[0])
    values = np.empty((len(sentences) - 1, len(FEATURES)), dtype=np.float64)
    for i, current in enumerate(vocabulary[1:], 1):
        relevance = np.zeros(i, dtype=np.float64)
        if current:
            idf = {word: 1 + np.log((i + 1) / (1 + previous_documents[word])) for word in sorted(current)}
            scale = sum(idf.values())
            for j, earlier in enumerate(vocabulary[:i]):
                relevance[j] = sum(idf[word] for word in sorted(current & earlier)) / scale
        logits = LEXICAL_WEIGHT * relevance
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        excess = weights - 1 / i
        lag = np.arange(i, 0, -1, dtype=np.float64) / i
        update_similarity = normalized_changes[:i] @ normalized_changes[i]
        state_similarity = states[:i] @ states[i]
        values[i - 1] = (
            relevance.max(),
            relevance.mean(),
            excess @ lag,
            update_similarity.max(),
            excess @ update_similarity,
            excess @ state_similarity,
            widths[i],
        )
        previous_documents.update(current)
    if not np.isfinite(values).all():
        raise ValueError("All causal contextual features must be finite")
    return values


def verify_prefix_features(sentences, states, features):
    """Check that rebuilding features instead from each available prefix gives the same rows."""
    maximum = 0.0
    tested = 0
    for stop in range(2, len(sentences)):
        prefix_rows = contextual_features(sentences[:stop], states[:stop])
        maximum = max(maximum, float(np.max(np.abs(prefix_rows - features[:stop - 1]))))
        tested += 1
    if maximum > 1e-12:
        raise ValueError("Contextual features changed when removing an unseen suffix")
    return tested, maximum
