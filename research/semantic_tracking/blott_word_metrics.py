"""Score unmarked Blott word localization with family-paired comparisons."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


METHODS = ('uniform', 'position', 'surface', 'magnitude', 'state')
SEED = 735972
DRAWS = 10_000
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Sentence:
    identifier: str
    family: str
    fold: int | str
    ordinals: tuple[int, ...]
    gold: int
    logits: np.ndarray


def validated_sentences(rows):
    sentences = []
    seen = set()
    family_folds = {}
    for row in rows:
        identifier, family, fold = (row[name] for name in ('id', 'family', 'fold'))
        ordinals, gold, methods = (row[name] for name in ('ordinals', 'gold', 'logits'))
        if (type(identifier) is not str or not identifier or identifier in seen
                or type(family) is not str or not family or type(fold) not in (int, str)
                or family_folds.setdefault(family, fold) != fold):
            raise ValueError('Expected distinct sentence IDs and a single fold for every family')
        if (type(ordinals) is not list or not ordinals
                or any(type(position) is not int or position < 2 for position in ordinals)
                or ordinals != sorted(set(ordinals))
                or type(gold) is not int or not 0 <= gold < len(ordinals)):
            raise ValueError('Expected ordered eligible word positions and one candidate index as gold')
        if set(methods) != set(METHODS):
            raise ValueError('Every sentence must carry logits for all registered methods')
        logits = np.asarray([methods[name] for name in METHODS], dtype=np.float64)
        if logits.shape != (len(METHODS), len(ordinals)) or not np.isfinite(logits).all():
            raise ValueError('Expected one finite logit per method and candidate')
        if not np.all(logits[0] == logits[0, 0]):
            raise ValueError('The uniform model must give every candidate an equal score')
        seen.add(identifier)
        sentences.append(Sentence(identifier, family, fold, tuple(ordinals), gold, logits))
    if not sentences:
        raise ValueError('Expected at least one scorable sentence')
    return sorted(sentences, key=lambda sentence: sentence.identifier)


def rank_scores(sentence):
    logits, gold = sentence.logits, sentence.gold
    shifted = logits - logits.max(axis=1, keepdims=True)
    nll = logsumexp(shifted, axis=1) - shifted[:, gold]
    if not np.isfinite(nll).all():
        raise ValueError('The conditional negative log likelihood must be finite')
    difference = logits - logits[:, gold, None]
    ahead = (difference > TIE_TOLERANCE).sum(axis=1)
    tied = (np.abs(difference) <= TIE_TOLERANCE).sum(axis=1)
    top1 = (ahead == 0) / tied
    mrr = [np.mean(1 / np.arange(a + 1, a + n + 1)) for a, n in zip(ahead, tied, strict=True)]
    return np.column_stack((nll, top1, mrr))


def same_ordinal(sentences):
    negatives = defaultdict(list)
    for sentence in sentences:
        for index, ordinal in enumerate(sentence.ordinals):
            if index != sentence.gold:
                negatives[sentence.fold, ordinal].append((sentence.family, sentence.logits[:, index]))
    wins = np.zeros(len(METHODS), dtype=np.float64)
    pairs = matched = 0
    for sentence in sentences:
        scope = sentence.fold, sentence.ordinals[sentence.gold]
        controls = [logits for family, logits in negatives[scope] if family != sentence.family]
        if not controls:
            continue
        difference = sentence.logits[:, sentence.gold] - np.asarray(controls)
        wins += (difference > TIE_TOLERANCE).sum(axis=0)
        wins += 0.5 * (np.abs(difference) <= TIE_TOLERANCE).sum(axis=0)
        matched += 1
        pairs += len(controls)
    return {'matched_sentences': matched, 'total_sentences': len(sentences), 'pairs': pairs,
            'methods': {name: float(value / pairs) if pairs else None
                        for name, value in zip(METHODS, wins, strict=True)}}


def evaluate(rows, *, primary):
    """Summarize conditional ranking and cross-family ordinal-matched comparisons."""
    if type(primary) is not bool:
        raise ValueError('The primary flag must be boolean')
    sentences = validated_sentences(rows)
    families = defaultdict(list)
    for sentence in sentences:
        families[sentence.family].append(sentence)
    ordered = [families[key] for key in sorted(families)]
    metrics = np.stack([np.mean([rank_scores(sentence) for sentence in group], axis=0) for group in ordered])
    methods = {name: dict(zip(('nll', 'top1', 'mrr'), score.tolist(), strict=True))
               for name, score in zip(METHODS, metrics.mean(axis=0), strict=True)}

    samples = np.random.default_rng(SEED).integers(0, len(ordered), size=(DRAWS, len(ordered)))
    gains = {}
    for index, name in enumerate(METHODS[:-1]):
        difference = metrics[:, index, 0] - metrics[:, -1, 0]
        ci = np.quantile(difference[samples].mean(axis=1), (0.025, 0.975)).tolist()
        gains[name] = {'mean': float(difference.mean()), 'ci95': ci, 'pass': ci[0] > 0}

    last_word = np.mean([np.mean([sentence.gold == len(sentence.ordinals) - 1 for sentence in group])
                         for group in ordered])
    return {'sentences': len(sentences), 'families': len(ordered),
            'candidates': sum(len(sentence.ordinals) for sentence in sentences),
            'bootstrap_draws': DRAWS, 'methods': methods, 'qwen_nll_gain': gains,
            'last_word_oracle_top1': float(last_word), 'same_ordinal': same_ordinal(sentences),
            'primary_pass': all(gains[name]['pass'] for name in ('surface', 'magnitude')) if primary else None}
