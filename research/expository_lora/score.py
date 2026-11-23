"""Score sealed PLABA and Kerwer forecasts using local human observations."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json


METHODS = ('frozen', 'lora')
TARGETS = ('ease', 'understanding')
ARMS = ('plain_without_headings', 'plain_with_headings')
LEVELS = np.arange(1, 6)
DRAWS = 10_000
SEED = 735972


def table(path):
    result = {}
    for row in stream_rows(path):
        if row['id'] in result:
            raise ValueError(f'Duplicate identity in {path.name}')
        result[row['id']] = row
    return result


def audit_inputs(assay, protocol, predictions, seal_path, kerwer_ratings):
    audit_path = assay / 'source-audit.json'
    audit = read_json(audit_path)
    seal = read_json(seal_path)
    keys = {'assay_sha256': digest(audit_path), 'protocol_sha256': digest(protocol),
            'predictions_sha256': digest(predictions)}
    if any(seal[key] != expected for key, expected in keys.items()):
        raise ValueError('Forecasts, source audit or protocol differ from the sealed inputs')
    required = ('plaba/stimuli.jsonl', 'kerwer/stimuli.jsonl', 'plaba/ratings.jsonl', 'plaba/pairs.jsonl',
                'kerwer/layout.jsonl')
    if any(digest(assay / name) != audit['files_sha256'][name] for name in required):
        raise ValueError('A scoring input differs from the prepared receipt')
    stimulus_hashes = {name: audit['files_sha256'][f'{name}/stimuli.jsonl'] for name in ('plaba', 'kerwer')}
    if seal['stimulus_file_sha256'] != stimulus_hashes:
        raise ValueError('The prediction seal does not bind both exact stimulus inventories')
    if (seal['fit_converged'] is not True or seal['gpu'] != 'NVIDIA L4'
            or seal['adapter_placement'] != 'layers[-1].mlp.down_proj'):
        raise ValueError('The sealed outputs do not match the registered converged L4 comparison')
    registry = Path(__file__).resolve().parents[1] / 'backbone_transfer/models.json'
    qwen = read_json(registry)['models']['qwen']
    if (seal['qwen_revision'] != qwen['revision'] or seal['model'] != qwen['repo']
            or seal['model_files_sha256'] != qwen['sha256']):
        raise ValueError('The sealed language checkpoint differs from the experiment registry')
    for name in ('documents', 'stops', 'input_tokens'):
        if seal[name] != sum(audit['inventory'][item][name] for item in stimulus_hashes):
            raise ValueError(f'Unexpected sealed inventory: {name}')
    sealed_artifacts = {'predictions.jsonl': keys['predictions_sha256']}
    sealed_artifacts.update({name: digest(seal_path.with_name(name))
                             for name in ('causality.json', 'convergence.json', 'checkpoints.npz')})
    if seal['file_sha256'] != sealed_artifacts:
        raise ValueError('The complete model, convergence, causality and forecast artifacts must match their seal')
    causality = read_json(seal_path.with_name('causality.json'))
    validate_causality(causality)
    ratings_hash = digest(kerwer_ratings)
    if ratings_hash != audit['sources']['kerwer']['source_ratings_sha256']:
        raise ValueError('Kerwer local reader observations differ from the registered author release')
    return {**keys, 'forecast_seal_sha256': digest(seal_path), 'kerwer_ratings_sha256': ratings_hash,
            'score_code_sha256': digest(Path(__file__))}


def validate_causality(causality):
    expected = {(fold, method) for fold in (*range(5), 'full') for method in METHODS}
    comparisons = causality['cached_live_equivalence']
    actual = {(row['fold'], row['model']) for row in comparisons}
    if (causality['inference_forwards_per_document_per_model'] != 1 or actual != expected
            or len(comparisons) != len(expected)
            or not 0 < causality['probability_tolerance'] <= 5e-5
            or any(not 0 <= row['max_absolute_probability_error'] <= causality['probability_tolerance']
                   for row in comparisons)):
        raise ValueError('Live single-forward forecasts differ from the sealed matched fit')
    for model in METHODS:
        result = causality[model]
        checks = result['checks']
        if (len(checks) < 6 or not 0 < result['relative_tolerance'] <= 0.001
                or {row['variant'] for row in checks} != {'truncation', 'replacement'}
                or any(row['passed'] is not True or not 0 <= row['max_relative_rms'] <= result['relative_tolerance']
                       for row in checks)):
            raise ValueError(f'The {model} prefix causality validation failed')


def load_predictions(path, stimuli_by_dataset):
    predictions = {}
    for row in stream_rows(path):
        if set(row) != {'model', 'dataset', 'fold', 'id', 'stops', 'probabilities'}:
            raise ValueError('Unexpected sealed forecast schema')
        model, dataset, fold, key = (row[name] for name in ('model', 'dataset', 'fold', 'id'))
        if model not in METHODS or dataset not in stimuli_by_dataset or key not in stimuli_by_dataset[dataset]:
            raise ValueError('Unexpected dataset, model or document identity')
        if ((dataset == 'plaba' and (type(fold) is not int or fold not in range(5)))
                or (dataset == 'kerwer' and fold is not None)):
            raise ValueError('Forecast uses an invalid fold identity')
        registered = stimuli_by_dataset[dataset][key]['stops']
        values = np.asarray(row['probabilities'], dtype=np.float64)
        if (row['stops'] != registered or any(type(stop) is not int for stop in row['stops'])
                or values.shape != (len(registered), len(TARGETS), len(LEVELS))
                or not np.isfinite(values).all() or np.any((values < 0) | (values > 1))
                or not np.allclose(values.sum(axis=-1), 1.0, atol=1e-6, rtol=0)):
            raise ValueError('Probability or registered observation-stop inventory is invalid')
        identity = model, dataset, key
        if identity in predictions:
            raise ValueError('Repeated sealed forecast')
        predictions[identity] = fold, dict(zip(registered, values, strict=True))
    expected = {(model, dataset, key) for dataset, items in stimuli_by_dataset.items() for key in items
                for model in METHODS}
    if predictions.keys() != expected:
        raise ValueError('Sealed forecasts do not cover both methods on the entire registered inventory')
    return predictions


def interval(point, draws, level=0.95):
    tail = (1 - level) / 2
    return {'estimate': float(point), f'ci{round(level * 100)}': np.quantile(draws, (tail, 1 - tail)).tolist()}


def group_average(values, groups, weights=None):
    values = np.asarray(values, dtype=np.float64)
    unique, index = np.unique(groups, return_inverse=True)
    if len(unique) < 2 or len(index) != len(values):
        raise ValueError('Grouped scoring needs at least two complete material groups')
    item_weight = np.ones(len(values)) if weights is None else np.asarray(weights, dtype=np.float64)
    if item_weight.shape != (len(values),) or not np.isfinite(item_weight).all() or np.any(item_weight <= 0):
        raise ValueError('Invalid positive observation weights')
    flat = values.reshape(len(values), -1)
    sums = np.zeros((len(unique), flat.shape[1]))
    np.add.at(sums, index, item_weight[:, None] * flat)
    denominators = np.bincount(index, weights=item_weight, minlength=len(unique))
    resampled = np.random.default_rng(SEED).multinomial(len(unique), np.full(len(unique), 1 / len(unique)), DRAWS)
    point = sums.sum(axis=0) / denominators.sum()
    draws = (resampled @ sums) / (resampled @ denominators)[:, None]
    return point.reshape(values.shape[1:]), draws.reshape(DRAWS, *values.shape[1:])


def ordinal_losses(probabilities, counts):
    """Return per-sentence expected per-vote NLL and the four-cutpoint RPS."""
    observed = counts / counts.sum(axis=-1, keepdims=True)
    if np.any((probabilities == 0) & (observed[None] > 0)):
        raise ValueError('A frozen forecast assigns zero probability to an observed rating')
    log_probabilities = np.log(np.where(probabilities > 0, probabilities, 1.0))
    nll = -np.sum(observed[None] * log_probabilities, axis=-1)
    prediction_cdf = probabilities.cumsum(axis=-1)[..., :-1]
    human_cdf = observed.cumsum(axis=-1)[None, ..., :-1]
    rps = (prediction_cdf ** 2 - 2 * prediction_cdf * human_cdf + human_cdf).mean(axis=-1)
    return np.stack((nll, rps), axis=-1).transpose(1, 2, 0, 3)


def metric_comparison(point, draws):
    return {**{name: interval(point[index], draws[:, index]) for index, name in enumerate(METHODS)},
            'lora_minus_frozen': interval(point[1] - point[0], draws[:, 1] - draws[:, 0])}


def plaba_ordinal(probabilities, counts, groups):
    losses = ordinal_losses(probabilities, counts)
    metrics = ('nll_nats', 'rps')
    point, draws = group_average(losses, groups)
    result = {}
    for index, target in enumerate(TARGETS):
        votes = counts[:, index, :].sum(axis=-1)
        vote_point, vote_draws = group_average(losses[:, index], groups, votes)
        result[target] = {
            'votes': int(votes.sum()),
            'sentence_weighted': {metric: metric_comparison(point[index, :, column], draws[:, index, :, column])
                                  for column, metric in enumerate(metrics)},
            'vote_weighted': {metric: metric_comparison(vote_point[:, column], vote_draws[:, :, column])
                              for column, metric in enumerate(metrics)},
        }
    return result


def effect_metrics(human, frozen, lora):
    base_error, update_error = frozen - human, lora - human
    return np.stack((human, frozen, lora, lora - frozen, base_error, update_error,
                     np.abs(base_error), np.abs(update_error), np.abs(update_error) - np.abs(base_error)), axis=-1)


EFFECT_NAMES = ('human', 'frozen', 'lora', 'lora_minus_frozen', 'frozen_minus_human', 'lora_minus_human',
                'frozen_absolute_item_error', 'lora_absolute_item_error', 'lora_minus_frozen_absolute_item_error')


def effect_report(point, draws):
    report = {name: interval(point[column], draws[:, column]) for column, name in enumerate(EFFECT_NAMES)}
    for index, method in enumerate(METHODS, 4):
        report[method + '_minus_human'].update(interval(point[index], draws[:, index], 0.90))
    report['lora_minus_frozen_absolute_macro_error'] = interval(
        np.abs(point[5]) - np.abs(point[4]), np.abs(draws[:, 5]) - np.abs(draws[:, 4]))
    return report


def load_plaba(assay, stimuli, predictions):
    records = table(assay / 'ratings.jsonl')
    if records.keys() != stimuli.keys():
        raise ValueError('PLABA outcomes and registered article IDs differ')
    indexes, groups, counts, probability = {}, [], [], []
    fold_by_group, fold_by_family = {}, {}
    for key, row in records.items():
        rated = row['ratings']
        stop_list = [sentence['stop'] for sentence in rated]
        if stop_list != stimuli[key]['stops'] or type(row['fold']) is not int or row['fold'] not in range(5):
            raise ValueError('PLABA ratings are not aligned to the exact registered sentence stops')
        if (fold_by_group.setdefault(row['group_id'], row['fold']) != row['fold']
                or fold_by_family.setdefault(row['family_id'], row['fold']) != row['fold']):
            raise ValueError('PLABA reader group or article family spans held folds')
        forecasts = [predictions[model, 'plaba', key] for model in METHODS]
        if any(fold != row['fold'] for fold, _ in forecasts):
            raise ValueError('PLABA prediction was not generated for the registered out-of-fold model')
        for sentence in rated:
            stop = sentence['stop']
            raw = [sentence[target] for target in TARGETS]
            if any(len(votes) != len(LEVELS) or any(type(n) is not int or n < 0 for n in votes)
                   or not sum(votes) for votes in raw):
                raise ValueError('Invalid PLABA human grade histogram')
            indexes[key, stop] = len(counts)
            groups.append(row['group_id'])
            counts.append(raw)
            probability.append([values[stop] for _, values in forecasts])
    if set(fold_by_group.values()) != set(range(5)):
        raise ValueError('Expected all five registered reader-connected PLABA folds')
    return records, indexes, groups, np.asarray(counts), np.asarray(probability).transpose(1, 0, 2, 3)


def plaba_pairs(path, records, indexes):
    pairs, groups, used = [], [], set()
    for row in stream_rows(path):
        group = row['group_id']
        key = row['alignment_id']
        if key in used:
            raise ValueError('Repeated PLABA source alignment')
        used.add(key)
        pair = []
        for arm in ('original', 'plain'):
            indices = []
            for identifier, stop in row[arm]:
                document = records[identifier]
                if (document['family_id'], document['group_id'], document['fold']) != (
                        row['family_id'], group, row['fold']):
                    raise ValueError('A rewrite pair crosses a family, reader group or fold boundary')
                indices.append(indexes[identifier, stop])
            if not indices or len(set(indices)) != len(indices):
                raise ValueError('A PLABA caption pair has an empty or duplicated arm')
            pair.append(indices)
        if set(pair[0]) & set(pair[1]):
            raise ValueError('A sentence identifies both arms of a rewrite pair')
        pairs.append(pair)
        groups.append(group)
    return pairs, groups


def score_plaba(path, stimuli, predictions):
    records, indexes, groups, counts, probability = load_plaba(path, stimuli, predictions)
    pairs, pair_groups = plaba_pairs(path / 'pairs.jsonl', records, indexes)
    observed = (counts @ LEVELS) / counts.sum(axis=-1)
    expected_scores = probability @ LEVELS
    readings = np.concatenate((observed[None], expected_scores), axis=0)
    effects = np.asarray([readings[:, plain].mean(axis=1) - readings[:, original].mean(axis=1)
                          for original, plain in pairs])
    metrics = effect_metrics(effects[:, 0], effects[:, 1], effects[:, 2])
    point, draws = group_average(metrics, pair_groups)
    return {'documents': len(records), 'sentences': len(counts), 'reader_connected_groups': len(set(groups)),
            'ordinal': plaba_ordinal(probability, counts, groups),
            'rewrites': {'alignments': len(pairs), 'reader_connected_groups': len(set(pair_groups)),
                         'scale': [1, 5], 'contrast': 'plain minus original',
                         'targets': {target: effect_report(point[i], draws[:, i]) for i, target in enumerate(TARGETS)}}}


def kerwer_layout(path, stimuli, predictions):
    layout = table(path)
    if layout.keys() != stimuli.keys():
        raise ValueError('Kerwer layout and registered article IDs differ')
    families = defaultdict(dict)
    expected = {}
    for key, row in layout.items():
        doi, arm, stops = row['doi'], row['arm'], row['matched_stops']
        if arm not in ARMS or arm in families[doi] or not stops or len(set(stops)) != len(stops):
            raise ValueError('Invalid Kerwer matched heading arm')
        if not set(stops) <= set(stimuli[key]['stops']):
            raise ValueError('A Kerwer matched body stop is absent from the input register')
        families[doi][arm] = key
        expected[key] = [float(np.stack([predictions[method, 'kerwer', key][1][stop][1] for stop in stops])
                               .mean(axis=0) @ LEVELS) for method in METHODS]
    ordered = sorted(families)
    if any(families[doi].keys() != set(ARMS) for doi in ordered):
        raise ValueError('A Kerwer family needs both registered heading conditions')
    cases = [[families[doi][arm] for arm in ARMS] for doi in ordered]
    if any(len(layout[left]['matched_stops']) != len(layout[right]['matched_stops']) for left, right in cases):
        raise ValueError('Heading conditions do not have equal matched body-sentence inventories')
    return ordered, cases, np.asarray([[expected[key] for key in pair] for pair in cases])


def kerwer_readers(path, cases):
    columns = {key: (i, j) for i, pair in enumerate(cases) for j, key in enumerate(pair)}
    rows = list(stream_rows(path))
    readers = {reader: i for i, reader in enumerate(sorted({row['reader'] for row in rows}))}
    observed = np.zeros((len(readers), len(cases), 2))
    for row in rows:
        key = row['id']
        if key in columns:
            fam, arm = columns[key]
            subject, value = readers[row['reader']], row['comprehensibility']
            if observed[subject, fam].any() or type(value) is not int or not 1 <= value <= 8:
                raise ValueError('Unexpected Kerwer response or duplicate exposure within source DOI')
            observed[subject, fam, arm] = value
    if np.any((observed > 0).sum(axis=0) == 0):
        raise ValueError('A Kerwer DOI/heading arm has no observations')
    return observed


def kerwer_effects(observed, modeled):
    readers, families, _ = observed.shape
    flattened = observed.reshape(readers, -1)
    recorded = flattened > 0

    def by_reader(weight):
        counts = weight @ recorded
        if np.any(counts == 0):
            raise ValueError('A Kerwer bootstrap draw omitted an entire DOI/heading arm')
        means = ((weight @ flattened) / counts).reshape(len(weight), families, 2)
        return effect_metrics(means[..., 1] - means[..., 0],
                              np.broadcast_to(modeled[:, 0], means.shape[:2]),
                              np.broadcast_to(modeled[:, 1], means.shape[:2]))

    points = by_reader(np.ones((1, readers)))[0]
    reader_rng, family_rng = (np.random.default_rng(seed) for seed in np.random.SeedSequence(SEED).spawn(2))
    crossed, fixed = [], []
    for start in range(0, DRAWS, 128):
        batch = min(128, DRAWS - start)
        weights = reader_rng.multinomial(readers, np.full(readers, 1 / readers), batch)
        source_weights = family_rng.multinomial(families, np.full(families, 1 / families), batch)
        summaries = by_reader(weights)
        crossed.append(np.einsum('bf,bfm->bm', source_weights, summaries) / families)
        fixed.append(summaries.mean(axis=1))
    return points, np.concatenate(crossed), np.concatenate(fixed)


def score_kerwer(layout_path, ratings, stimuli, predictions):
    sources, cases, expected = kerwer_layout(layout_path, stimuli, predictions)
    observed = kerwer_readers(ratings, cases)
    modeled = (expected[:, 1] - expected[:, 0]) * 7 / 4
    points, draws, fixed = kerwer_effects(observed, modeled)
    point = points.mean(axis=0)
    return {'heading_contrast': 'plain with headings minus plain without headings', 'source_scale': [1, 8],
            'model_scale': [1, 5], 'effect_conversion': 'model difference × 7 / 4',
            'source_dois': len(sources), 'reader_pool': len(observed), 'heading_arm_ratings': int((observed > 0).sum()),
            'crossed_doi_reader': effect_report(point, draws), 'fixed_doi_reader': effect_report(point, fixed),
            'doi_effects': {doi: dict(zip(EFFECT_NAMES, row.tolist(), strict=True))
                            for doi, row in zip(sources, points, strict=True)}}


def score(assay, protocol, predictions, seal, kerwer_ratings, output):
    provenance = audit_inputs(assay, protocol, predictions, seal, kerwer_ratings)
    stimuli = {dataset: table(assay / dataset / 'stimuli.jsonl') for dataset in ('plaba', 'kerwer')}
    forecasts = load_predictions(predictions, stimuli)
    result = {'inputs': provenance, 'bootstrap': {'draws': DRAWS, 'seed': SEED},
              'plaba': score_plaba(assay / 'plaba', stimuli['plaba'], forecasts),
              'kerwer': score_kerwer(assay / 'kerwer/layout.jsonl', kerwer_ratings, stimuli['kerwer'], forecasts)}
    write_json(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('assay', 'protocol', 'predictions', 'seal', 'kerwer-ratings', 'output'):
        parser.add_argument(f'--{option}', type=Path, required=True)
    args = parser.parse_args()
    score(args.assay, args.protocol, args.predictions, args.seal, args.kerwer_ratings, args.output)


if __name__ == '__main__':
    main()
