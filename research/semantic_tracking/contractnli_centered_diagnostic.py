"""Post-hoc replay of contract evidence rankings after removing shared location scores."""

import argparse
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from research.semantic_tracking import contractnli_forecast as forecast
from research.semantic_tracking import contractnli_score as score
from research.semantic_tracking.contractnli_report import SOURCE_AUDIT_SHA, hypothesis_rows, lexical_baseline
from research.semantic_tracking.expmrc_forecast import digest


MODES = ('token_rank', 'mean_hypothesis', 'centered_token_rank', 'bm25')


def analyze(layout, gold, predictions, maps, lexical):
    documents, certified, positions = score.read_layout(layout, maps)
    claims = score.read_gold(gold, documents)
    by_doc = defaultdict(dict)
    for row in predictions:
        identifier, query = row['id'], row['query_id']
        indices = [stop for _, stop in certified[identifier]]
        by_doc[identifier][query] = np.asarray(row['methods']['token_rank'])[indices]
    scores = {}
    for identifier, from_row in by_doc.items():
        if len(from_row) != 17:
            raise ValueError('Shared location scores require every public hypothesis')
        names = sorted(from_row)
        raw = np.stack([from_row[key] for key in names])
        logits = logit(raw)
        center = logits.mean(axis=0)
        for name, r, l in zip(names, raw, logits, strict=True):
            bm25 = [lexical[identifier, name][i] for i, _ in certified[identifier]]
            scores[identifier, name] = np.stack((r, expit(center), expit(l - center), bm25))
    if scores.keys() != claims.keys():
        raise ValueError('Centered score replay needs the complete frozen holdout')

    ap, pairs = defaultdict(list), defaultdict(list)
    for doc, queries in by_doc.items():
        eligible = {label: [] for label in score.POSITIVE_LABELS}
        for query in queries:
            label, evidence = claims[doc, query]
            if label not in eligible or any(i not in positions[doc] for i in evidence):
                continue
            aligned = tuple(positions[doc][i] for i in evidence)
            positive = np.zeros(len(certified[doc]), dtype=np.bool_)
            positive[list(aligned)] = True
            ap[doc].append([score.average_precision(values, positive) for values in scores[doc, query]])
            eligible[label].append((query, set(evidence), aligned))
        for (a, left, i), (b, right, j) in product(*(eligible[label] for label in score.POSITIVE_LABELS)):
            if left.isdisjoint(right):
                pairs[doc].append((score.evidence_order(scores[doc, a], i, j)
                                   + score.evidence_order(scores[doc, b], j, i)) / 2)

    families = sorted({row['group_id'] for row in layout})
    rng = np.random.default_rng(score.SEED)
    draws = rng.multinomial(len(families), np.full(len(families), 1 / len(families)), size=score.DRAWS)
    result = {}
    for name, stats in (('average_precision', ap), ('claim_pair_ordering', pairs)):
        present, values = score.family_means(stats, documents, families)
        weights = draws[:, present]
        totals = weights.sum(axis=1)
        valid = totals > 0
        sampled = (weights[valid] @ values) / totals[valid, None]
        points = values.mean(axis=0)
        result[name] = {
            'documents': len(stats), 'count': sum(map(len, stats.values())),
            'methods': {mode: score.estimate(points[i], sampled[:, i]) for i, mode in enumerate(MODES)},
            'centered_minus_raw': score.estimate(points[2] - points[0], sampled[:, 2] - sampled[:, 0]),
            'centered_minus_bm25': score.estimate(points[2] - points[3], sampled[:, 2] - sampled[:, 3]),
        }
    return result


def run(prepared, packet, forecasts_dir, output):
    source = prepared / 'source-audit.json'
    audit = forecast.read_json(source)
    seal_path = forecasts_dir / 'seal.json'
    seal = forecast.read_json(seal_path)
    if not digest(source) == SOURCE_AUDIT_SHA == seal['source_audit_sha256']:
        raise ValueError('The replay requires the previously sealed public test outcome inventory')
    for name in ('layout.jsonl', 'gold.jsonl'):
        if digest(prepared / name) != audit['output_sha256'][name]:
            raise ValueError(f'Unregistered replay input: {name}')
    for path, field in ((packet / 'articles.jsonl', 'articles_sha256'),
                        (forecasts_dir / 'predictions.jsonl', 'predictions_sha256')):
        if digest(path) != seal[field]:
            raise ValueError(f'{path.name} changed after the forecast was frozen')
    layout = list(forecast.read_rows(
        prepared / 'layout.jsonl', {'id', 'official_id', 'group_id', 'document_type', 'spans'},
    ).values())
    gold = hypothesis_rows(prepared / 'gold.jsonl', {'id', 'group_id', 'official_id', 'query_id', 'label', 'evidence_indices'})
    predictions = hypothesis_rows(forecasts_dir / 'predictions.jsonl', {'id', 'query_id', 'stops', 'methods'})
    mapping = forecast.read_rows(packet / 'articles.jsonl', {'id', 'candidate_stops'})
    maps = {key: row['candidate_stops'] for key, row in mapping.items()}
    stimuli, _, hypotheses, _ = forecast.load_packet(packet)
    if digest(packet / 'stimuli.jsonl') != seal['stimuli_sha256']:
        raise ValueError('The lexical control needs the original certified source text')
    lexical = lexical_baseline(stimuli, layout, maps, hypotheses)
    forecast.write_json(output, {'posthoc': True, 'forecast_seal_sha256': digest(seal_path),
                                 'diagnostic_source_sha256': digest(Path(__file__)),
                                 **analyze(layout, gold, predictions, maps, lexical)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'packet', 'forecasts-dir', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
