"""Score the fixed Gradio lexical display on professional-news CWI votes."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from research.gradio_app.forecast import read_stimuli
from research.semantic_tracking.contractnli_forecast import read_json, read_rows, write_json
from research.semantic_tracking.iirc_anchor_fit import WORD
from research.semantic_tracking.iirc_prepare import digest
from research.stage6 import medreadme_lexical_forecast
from research.stage6.medreadme_lexical_score import unique_predictions


PROTOCOL = Path(__file__).with_name('cwi-news-holdout.md')
METHODS = ('served', 'length', 'inverse_frequency', 'combined_surface')
GROUPS = ('pooled', 'native', 'nonnative')
SEED, DRAWS = 735972, 10_000


def sentence_score(text, words, forecast):
    all_offsets = {match.span() for match in WORD.finditer(text)}
    scores = unique_predictions(forecast, all_offsets)
    offsets, votes, frequencies = [], [], []
    for row in words:
        if row.keys() != {'start', 'end', 'native_votes', 'nonnative_votes', 'form_count'}:
            raise ValueError('CWI outcomes require exact word positions, vote counts and frozen frequencies')
        offset = row['start'], row['end']
        observed = row['native_votes'], row['nonnative_votes']
        if (any(type(point) is not int for point in offset) or offset not in all_offsets
                or any(type(hits) is not int or not 0 <= hits <= 10 for hits in observed)
                or type(row['form_count']) is not int or row['form_count'] < 0):
            raise ValueError('Unexpected source span, group votes or lexical frequency')
        offsets.append(offset)
        votes.append(observed)
        frequencies.append(row['form_count'])
    if offsets != sorted(set(offsets)) or not offsets:
        raise ValueError('Eligible word positions must be distinct and ordered')
    aligned = sum(offset in scores for offset in offsets)
    served = np.asarray([scores.get(offset, -1.0) for offset in offsets])
    lengths = np.asarray([end - start for start, end in offsets])
    inverse = -np.asarray(frequencies)
    combined = rankdata(lengths) + rankdata(inverse)
    left, right = np.triu_indices(len(offsets), k=1)
    methods = np.asarray((served, lengths, inverse, combined))
    ordering = np.sign(methods[:, right] - methods[:, left])
    counts = np.asarray(votes)
    observed = np.vstack((counts.sum(axis=1), counts.T))
    summaries = {}
    for group, tally in zip(GROUPS, observed, strict=True):
        direction = np.sign(tally[right] - tally[left])
        present = direction != 0
        if present.any():
            comparison = ordering[:, present] * direction[present]
            summaries[group] = {'scores': np.mean((comparison + 1) / 2, axis=1), 'pairs': int(present.sum()),
                                'served_ties': int((comparison[0] == 0).sum())}
    return aligned, summaries


def aggregate(samples, *, primary):
    if len(samples) < 2:
        if primary:
            raise ValueError('Paired news inference requires at least two independently drawn HIT groups')
        return {'hits': len(samples), 'sentences': sum(map(len, samples.values())), 'methods': None}
    source_means = np.array([np.mean([row['scores'] for row in samples[key]], axis=0) for key in sorted(samples)])
    estimates = source_means.mean(axis=0)
    result = {'hits': len(samples), 'sentences': sum(map(len, samples.values())),
              'pairs': sum(row['pairs'] for rows in samples.values() for row in rows),
              'served_ties': sum(row['served_ties'] for rows in samples.values() for row in rows),
              'methods': dict(zip(METHODS, estimates.tolist(), strict=True))}
    if primary:
        rng = np.random.default_rng(SEED)
        draws = rng.multinomial(len(samples), np.full(len(samples), 1 / len(samples)), size=DRAWS)
        paired = draws @ source_means / len(samples)
        result['served_minus_baseline'] = {
            method: {'difference': float(estimates[0] - estimates[index]),
                     'ci95': np.quantile(paired[:, 0] - paired[:, index], [0.025, 0.975]).tolist()}
            for index, method in enumerate(METHODS[1:], 1)
        }
    return result


def evaluate(documents, truth, predictions):
    if documents.keys() != truth.keys() or documents.keys() != predictions.keys():
        raise ValueError('Expected exact agreement among CWI source, outcome and forecast IDs')
    samples = {group: defaultdict(list) for group in GROUPS}
    known = aligned = 0
    for source, text in documents.items():
        row = truth[source]
        if type(row['hit_id']) is not str or not row['hit_id']:
            raise ValueError('Every sentence requires a published HIT group')
        covered, summaries = sentence_score(text, row['words'], predictions[source])
        aligned += covered
        known += len(row['words'])
        for group, summary in summaries.items():
            samples[group][row['hit_id']].append(summary)
    if not known:
        raise ValueError('Expected published singleton word candidates')
    results = {group: aggregate(rows, primary=group == 'pooled') for group, rows in samples.items()}
    gates = {
        'candidate_alignment_at_least_95_percent': aligned / known >= 0.95,
        **{f'served_over_{baseline}': row['ci95'][0] > 0
           for baseline, row in results['pooled']['served_minus_baseline'].items()},
    }
    return {'eligible_words': known, 'aligned_words': aligned, 'aligned_fraction': aligned / known,
            'sentences': len(documents), 'hits': len({row['hit_id'] for row in truth.values()}),
            'within_sentence_ordering': results, 'bootstrap': {'draws': DRAWS, 'seed': SEED, 'unit': 'HIT'},
            'gates': gates, 'pass': all(gates.values())}


def score(assay, stimuli, outcomes, forecasts, seal, output):
    plan, receipt = read_json(assay), read_json(seal)
    if (plan['protocol_sha256'] != digest(PROTOCOL) or plan['outcomes_sha256'] != digest(outcomes)
            or receipt['forecasts_sha256'] != digest(forecasts)
            or receipt['forecast_code_sha256'] != digest(Path(medreadme_lexical_forecast.__file__))
            or any(receipt[key] != plan[key] for key in ('candidate_id', 'lexical_weights_sha256', 'stimuli_sha256'))
            or receipt['model_forwards'] != plan['retained_sentences']):
        raise ValueError('Expected exact sealed news word forecasts and the prospectively selected outcomes')
    documents = dict(read_stimuli(stimuli, plan['stimuli_sha256']).documents)
    source = read_rows(outcomes, {'id', 'hit_id', 'words'})
    predicted = read_rows(forecasts, {'id', 'words'})
    if len(documents) != plan['retained_sentences']:
        raise ValueError('News stimulus count differs from the day-of-run assay')
    result = evaluate(documents, source, predicted)
    write_json(output, {**result, 'protocol_sha256': digest(PROTOCOL), 'assay_sha256': digest(assay),
                        'candidate_id': receipt['candidate_id'], 'forecasts_seal_sha256': digest(seal),
                        'scoring_code_sha256': digest(Path(__file__))})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assay', 'stimuli', 'outcomes', 'forecasts', 'seal', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    score(**vars(parser.parse_args()))
