"""Score earlier source links on held-out retrospective human annotations."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from research.stage3.probe import _jsonl
from research.stage3.run import save_json


METHODS = ('uniform', 'distance_word', 'qwen')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(outcomes, predictions, seal, output):
    document = json.loads(seal.read_text(encoding='utf-8'))
    if digest(predictions) != document['predictions_sha256']:
        raise ValueError('Source predictions differ from the pre-scoring hash')
    gold = {row['row_id']: row for row in _jsonl(outcomes)}
    forecasts = {row['row_id']: row for row in _jsonl(predictions)}
    rejected = {row['row_id'] for row in document['excluded_test']}
    if rejected & forecasts.keys() or rejected | forecasts.keys() != gold.keys() or not forecasts:
        raise ValueError('Held-out labels do not match the certified prediction inventory')
    rows = defaultdict(list)
    candidate_counts = []
    for key, pred in forecasts.items():
        truth = gold[key]
        fields = ('id', 'group_id', 'stop', 'source_candidates')
        if any(truth[field] != pred[field] for field in fields):
            raise ValueError(f'Candidate set changed after model execution: {key}')
        candidate = np.asarray(pred['source_candidates'])
        marked = (truth['source_start'] < candidate) & (candidate <= truth['source_stop'])
        if not marked.any():
            raise ValueError(f'No earlier candidate aligned to the annotated source: {key}')
        candidate_counts.append((len(candidate), int(marked.sum())))
        result = []
        for method in METHODS:
            p = np.asarray(pred[method], dtype=np.float64)
            if p.shape != marked.shape or not np.all(np.isfinite(p) & (p > 0)) or not np.isclose(p.sum(), 1):
                raise ValueError(f'Invalid {method} pointer distribution')
            tied_top = np.isclose(p, p.max(), rtol=0, atol=1e-12)
            rank = 1 + np.sum(p > p[marked].max() + 1e-12)
            tied_first_hit = np.isclose(p, p[marked].max(), rtol=0, atol=1e-12)
            num_correct = (marked & tied_first_hit).sum()
            positions = np.arange(rank, rank + tied_first_hit.sum())
            # Expected reciprocal rank of the first correct candidate under uniform tie breaks.
            survival = 1.0
            reciprocal = 0.0
            for place, remaining in zip(positions, range(len(positions), 0, -1), strict=True):
                hit = survival * num_correct / remaining
                reciprocal += hit / place
                survival -= hit
                if remaining <= num_correct:
                    break
            result.append([-np.log(min(1.0, p[marked].sum())), (marked & tied_top).sum() / tied_top.sum(), reciprocal])
        rows[pred['group_id']].append(result)
    if len(rows) < 2:
        raise ValueError('Source reporting needs at least two held-out text families')
    metrics = ('source_span_nll', 'top_word_in_source', 'reciprocal_rank')
    family_sizes = np.array([len(examples) for examples in rows.values()])
    family_sums = np.stack([np.sum(examples, axis=0) for examples in rows.values()])
    means = family_sums.sum(axis=0) / family_sizes.sum()
    rng = np.random.default_rng(43)
    groups = len(family_sizes)
    weights = rng.multinomial(groups, np.full(groups, 1 / groups), size=10_000)
    paired = (weights @ (family_sums[:, 2, :] - family_sums[:, 1, :])) / (weights @ family_sizes)[:, None]
    counts = np.array(candidate_counts)
    save_json(output, {
        'annotation_rows': len(forecasts), 'excluded_alignment_rows': len(rejected),
        'held_out_families': groups, 'unique_texts': len({row['id'] for row in forecasts.values()}),
        'candidate_words_median': float(np.median(counts[:, 0])),
        'gold_source_candidate_words_median': float(np.median(counts[:, 1])),
        'methods': {method: dict(zip(metrics, row.tolist(), strict=True))
                    for method, row in zip(METHODS, means, strict=True)},
        'qwen_minus_distance_word': {metric: {'difference': float(means[2, i] - means[1, i]),
                                             'paired_family_95_interval': np.quantile(paired[:, i], [0.025, 0.975]).tolist()}
                                     for i, metric in enumerate(metrics)},
        'bootstrap': {'seed': 43, 'draws': len(weights), 'unit': 'original shared prompt family'},
        'input_sha256': {'outcomes': digest(outcomes), 'forecasts': digest(predictions), 'seal': digest(seal)},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('outcomes', 'predictions', 'seal', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    run(**vars(parser.parse_args()))
