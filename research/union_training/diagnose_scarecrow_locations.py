"""Replay word localization on the observation fit's SCARECROW development families."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, xlogy

from research.core_features.readback_score import average_precision, paired_material_interval
from research.semantic_tracking.scarecrow_detection_score import counts_row, fit_binary
from research.union_training.verify_observations import digest, read_json, records, text_digest, verified_member


TASK = 'scarecrow_retrospective_contradiction_mark_ordinal'
SOURCE = Path('semantic-tracking/scarecrow-detection/prepared')
METHODS = ('round_12', 'round_31', 'length_position', 'constant')


def load_positions(cache, packet, receipt):
    pins = receipt['source_files_sha256']['scarecrow']
    materials = {}
    for name in ('stimuli', 'layout', 'outcomes'):
        relative = SOURCE / f'{name}.jsonl'
        source = verified_member(cache, relative, pins[relative.as_posix()])
        rows = list(records(source))
        materials[name] = {row['id']: row for row in rows}
        if len(rows) != len(materials[name]):
            raise ValueError(f'Repeated SCARECROW record in {name}')
    texts, layouts, outcomes = (materials[key] for key in ('stimuli', 'layout', 'outcomes'))
    if not texts.keys() == layouts.keys() == outcomes.keys():
        raise ValueError('SCARECROW study documents do not align across materials and votes')

    source_positions, exposure = {}, {}
    for identifier, row in texts.items():
        text = row['text']
        key = text_digest(text)
        layout = layouts[identifier]
        exposure[key] = text
        for index, ((start, stop), votes) in enumerate(
            zip(layout['words'], outcomes[identifier]['counts'], strict=True)
        ):
            if not votes['total']:
                continue
            if (not 0 <= start < stop <= len(text) or (key, stop) in source_positions
                    or stop not in row['stops']):
                raise ValueError('Invalid or repeated SCARECROW word boundary')
            source_positions[key, stop] = {
                'id': identifier, 'group_id': f"scarecrow:{layout['group_id']}",
                'start': start, 'stop': stop, 'word_index': index, 'counts': counts_row(votes),
                'text': text,
            }

    relative = Path(f'{TASK}.jsonl')
    observations = verified_member(packet, relative, receipt['packet_sha256'][relative.as_posix()])
    roles, positions = {}, []
    for observation in records(observations):
        (key, stop), (end_key, end) = observation['coordinates']
        source = source_positions.pop((key, stop))
        family, role = observation['family'], observation['role']
        if (end_key != key or end != len(exposure[key]) or observation['outcome'] != source['counts']
                or family != source['group_id'] or role not in (0, 1) or roles.setdefault(family, role) != role):
            raise ValueError('Observation coordinate, vote or prompt-family division differs from SCARECROW source')
        positions.append({**source, 'role': role})
    if source_positions:
        raise ValueError('Observed SCARECROW source words missing from the joint study packet')
    return positions


def frozen_mark_probabilities(directory, epoch, packet_sha256, dev_indices):
    seal = read_json(directory / 'seal.json')
    if seal['packet_receipt_sha256'] != packet_sha256:
        raise ValueError('Forecast archive refers to a different observation packet')
    fit = verified_member(directory, Path('fitting.json'), seal['files_sha256']['fitting.json'])
    if read_json(fit)['selected_epoch'] != epoch:
        raise ValueError(f'Expected sealed forecasts from round {epoch}')
    forecast = verified_member(directory, Path('predictions.npz'), seal['files_sha256']['predictions.npz'])
    with np.load(forecast, allow_pickle=False) as archive:
        indices = archive[f'{TASK}.development.rows']
        probabilities = archive[f'{TASK}.development.probabilities']
    if (not np.array_equal(indices, dev_indices) or probabilities.shape != (len(indices), 4)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError('The sealed SCARECROW forecast does not cover the registered development words')
    return probabilities[:, 1:].sum(axis=1, dtype=np.float64), digest(directory / 'seal.json')


def surface_baselines(positions, votes, marked, train):
    features = np.log1p(np.asarray([(row['stop'] - row['start'], row['word_index']) for row in positions]))
    center = features[train].mean(axis=0)
    scale = features[train].std(axis=0)
    if np.any(scale <= 0):
        raise ValueError('SCARECROW training words lack variation in length or index')
    features = (features - center) / scale
    coefficients, diagnostics = fit_binary(features[train], marked[train], votes[train])
    constant = marked[train].sum() / votes[train].sum()
    return {
        'length_position': expit(coefficients[0] + features[~train] @ coefficients[1:]),
        'constant': np.full(np.count_nonzero(~train), constant),
    }, {'mean': center.tolist(), 'scale': scale.tolist(), 'coefficients': coefficients.tolist(),
        'optimizer': diagnostics, 'training_votes': int(votes[train].sum()),
        'training_marked_votes': int(marked[train].sum())}


def document_scores(rows, votes, marked, forecasts):
    documents = defaultdict(list)
    for index, row in enumerate(rows):
        documents[row['id']].append(index)
    mixed = [(rows[indices[0]], np.asarray(indices)) for indices in documents.values()
             if np.any(marked[indices] > 0) and np.any(marked[indices] == 0)]
    mixed_rows = [row for row, _ in mixed]
    scores = {
        method: np.asarray([average_precision(p[ix], marked[ix] > 0) for _, ix in mixed])
        for method, p in forecasts.items()
    }
    for probability in forecasts.values():
        if not np.all(np.isfinite(probability) & (probability > 0) & (probability < 1)):
            raise ValueError('Word-mark predictive probabilities must be finite and strictly between zero and one')

    results = {}
    for method in METHODS:
        probability = forecasts[method]
        results[method] = {
            'mean_within_document_ap': float(scores[method].mean()),
            'binary_nll_nats_per_eligible_vote': float(-np.sum(
                xlogy(marked, probability) + xlogy(votes - marked, 1 - probability)
            ) / votes.sum()),
            'calibration_bias_percentage_points': float(100 * (votes @ probability - marked.sum()) / votes.sum()),
        }
    comparisons = {}
    for baseline in ('round_12', 'length_position'):
        comparisons[f'round_31_minus_{baseline}'] = paired_material_interval(
            scores['round_31'] - scores[baseline], mixed_rows,
        )
    examples = []
    for top_marked in (True, False):
        for row, indices in sorted(mixed, key=lambda item: int(item[0]['id'].split('-')[-1])):
            peak = indices[np.argmax(forecasts['round_31'][indices])]
            if bool(marked[peak]) == top_marked:
                most_marked = indices[np.argmax(marked[indices])]
                examples.append({
                    'id': row['id'], 'round_31_top_word_is_marked': top_marked,
                    'round_31_top_word': position_example(rows[peak], forecasts['round_31'][peak]),
                    'word_with_most_positive_votes': position_example(rows[most_marked], forecasts['round_31'][most_marked]),
                })
                break
    return {
        'development_documents': len(documents), 'development_prompt_families': len({row['group_id'] for row in rows}),
        'documents_with_both_marked_and_unmarked_words': len(mixed),
        'development_words': len(rows), 'eligible_annotation_votes': int(votes.sum()),
        'positive_annotation_votes': int(marked.sum()), 'metrics': results,
        'within_document_ap_comparisons': comparisons, 'examples': examples,
    }


def position_example(row, probability):
    return {
        'text': row['text'][row['start']:row['stop']],
        'unicode_span': [row['start'], row['stop']],
        'marks': sum(row['counts'][1:]), 'eligible_votes': sum(row['counts']),
        'round_31_mark_probability': float(probability),
    }


def run(cache, packet, round_12, round_31, output):
    receipt_path = packet / 'receipt.json'
    positions = load_positions(cache, packet, read_json(receipt_path))
    counts = np.asarray([row['counts'] for row in positions], dtype=np.int64)
    votes, marked = counts.sum(axis=1), counts[:, 1:].sum(axis=1)
    train = np.asarray([row['role'] == 0 for row in positions])
    dev_indices = np.flatnonzero(~train)
    forecasts, baseline = surface_baselines(positions, votes, marked, train)
    packet_hash = digest(receipt_path)
    seals = {}
    for epoch, path in ((12, round_12), (31, round_31)):
        key = f'round_{epoch}'
        forecasts[key], seals[key] = frozen_mark_probabilities(path, epoch, packet_hash, dev_indices)
    report = {
        'scope': 'Post-selection diagnostic on the joint model development split',
        'outcome': 'Retrospective self-contradiction span coverage of each continuation word',
        'ranking': 'AP uses any eligible mark as the binary label and per-annotator mark probability as the score',
        'exposure': 'Both joint heads use the word-end and full-continuation states; annotators also saw the news prompt',
        'packet_receipt_sha256': packet_hash, 'forecast_seals_sha256': seals,
        'baseline_fit_on_registered_training_families': baseline,
        **document_scores([positions[index] for index in dev_indices], votes[~train], marked[~train], forecasts),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'packet', 'round-12', 'round-31', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
