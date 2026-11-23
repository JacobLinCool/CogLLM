"""Score frozen Gradio word forecasts against MedReadMe human span annotations."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.core_features.readback_score import average_precision
from research.gradio_app.forecast import read_stimuli
from research.semantic_tracking.iirc_anchor_fit import WORD
from research.semantic_tracking.iirc_prepare import digest
from research.semantic_tracking.joint_score import rows_by_key
from research.stage3.run import save_json


PROTOCOL = Path(__file__).with_name('medreadme-lexical-holdout.md')
METHODS = ('served', 'length', 'inverse_frequency')
DRAWS, SEED = 10_000, 43


def unique_predictions(row, intended):
    if set(row) != {'id', 'words'}:
        raise ValueError('Unexpected MedReadMe forecast fields')
    scores = {}
    for start, end, score in row['words']:
        point = start, end
        if (point in scores or point not in intended or not np.isfinite(score)
                or not 0 <= score <= 1 or score != round(score, 2)):
            raise ValueError('Duplicate, unaligned or invalid MedReadMe forecast score')
        scores[point] = score
    return scores


def summarize(rows):
    groups = defaultdict(list)
    for source, values in rows:
        groups[source].append(values)
    samples = [np.asarray(groups[source]) for source in sorted(groups)]
    if len(samples) < 2:
        raise ValueError('Need at least two publication sources for the paired intervals')
    center = np.concatenate(samples).mean(axis=0)
    rng = np.random.default_rng(SEED)
    picked = rng.integers(len(samples), size=(DRAWS, len(samples)))
    totals = np.zeros((DRAWS, len(METHODS)))
    counts = np.zeros(DRAWS)
    for index, sample in enumerate(samples):
        destination = np.nonzero(picked == index)[0]
        for block in range(0, len(destination), 256):
            batch = destination[block:block + 256]
            choice = rng.integers(len(sample), size=(len(batch), len(sample)))
            np.add.at(totals, batch, sample[choice].sum(axis=1))
            np.add.at(counts, batch, len(sample))
    results = totals / counts[:, None]
    contrasts = {method: {'difference': float(center[0] - center[index]),
                          'paired_95_ci': np.quantile(results[:, 0] - results[:, index], [0.025, 0.975]).tolist()}
                 for index, method in enumerate(METHODS[1:], 1)}
    return dict(zip(METHODS, center.tolist(), strict=True)), contrasts, len(samples)


def score(assay, stimuli, outcomes, forecasts, seal, output):
    plan = json.loads(assay.read_bytes())
    receipt = json.loads(seal.read_bytes())
    if (plan['protocol_sha256'] != digest(PROTOCOL) or plan['outcomes_sha256'] != digest(outcomes)
            or receipt['forecasts_sha256'] != digest(forecasts)
            or any(receipt[key] != plan[key] for key in ('candidate_id', 'lexical_weights_sha256', 'stimuli_sha256'))
            or receipt['model_forwards'] != plan['retained_sentences']):
        raise ValueError('Expected the frozen candidate and exact sealed lexical holdout forecasts')
    documents = dict(read_stimuli(stimuli, plan['stimuli_sha256']).documents)
    truth = rows_by_key(outcomes, lambda row: row['id'])
    predicted = rows_by_key(forecasts, lambda row: row['id'])
    if truth.keys() != documents.keys() or predicted.keys() != documents.keys():
        raise ValueError('MedReadMe stimuli, annotations and predictions differ in sentence coverage')
    rows, planned, aligned, span_counts = [], 0, 0, defaultdict(lambda: [0, 0])
    for case, text in documents.items():
        positions = [word.span() for word in WORD.finditer(text)]
        row = truth[case]
        if [(word['start'], word['end']) for word in row['words']] != positions:
            raise ValueError('Held-out words no longer match current app candidates')
        ratings = unique_predictions(predicted[case], set(positions))
        words = [word for word in row['words'] if word['target'] is not None]
        planned += len(words)
        aligned += sum((word['start'], word['end']) in ratings for word in words)
        for span in row['spans']:
            contained = [(left, right) for left, right in positions if span['start'] <= left < right <= span['end']]
            kind = 'multiword' if len(contained) > 1 else 'one_word' if contained else 'no_complete_word'
            tally = span_counts[(span['category'], kind)]
            tally[0] += 1
            tally[1] += bool(contained and all(point in ratings for point in contained))
        gold = np.asarray([word['target'] for word in words], dtype=bool)
        if not gold.any() or gold.all():
            continue
        scores = [[ratings.get((word['start'], word['end']), -1.0) for word in words],
                  [word['end'] - word['start'] for word in words], [-word['form_count'] for word in words]]
        rows.append((row['source'], [average_precision(np.asarray(method), gold) for method in scores]))
    scores, contrasts, sources = summarize(rows)
    save_json(output, {'candidate_id': plan['candidate_id'], 'forecast_seal_sha256': digest(seal),
              'assay_sha256': digest(assay), 'scoring_code_sha256': digest(Path(__file__)),
              'sentences': len(documents), 'sentences_with_word_comparisons': len(rows), 'sources': sources,
              'mean_within_sentence_average_precision': scores, 'paired_served_minus_baseline': contrasts,
              'eligible_words': planned, 'aligned_eligible_words': aligned, 'aligned_fraction': aligned / planned,
              'span_alignment_by_category_and_length': [{'category': category, 'type': kind, 'span_count': total,
                                                         'all_complete_words_aligned': mapped}
                                                        for (category, kind), (total, mapped) in sorted(span_counts.items())],
              'passed': aligned / planned >= 0.95 and all(row['paired_95_ci'][0] > 0 for row in contrasts.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assay', 'stimuli', 'outcomes', 'forecasts', 'seal', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    score(**vars(parser.parse_args()))
