"""Independently check the observation packet against original source records."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
import re
import unicodedata

from research.union_training.observation_registry import REPLACED, contracts


ROOT = Path(__file__).resolve().parents[2]
HOLDOUT = Path(__file__).with_name('holdout.json')
COUNT_MODELS = {
    'categorical', 'choice', 'hurdle_ordinal', 'ordinal', 'ordinal_binary', 'ordered_choice', 'first_event',
}
WORDS = re.compile(r'[^\W_]+')


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def digest(path):
    with path.open('rb') as source:
        return file_digest(source, 'sha256').hexdigest()


def text_digest(text):
    return sha256(text.encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def records(path):
    with path.open(encoding='utf-8') as source:
        for line in source:
            yield json.loads(line)


def verified_member(root, relative, expected):
    path = Path(relative)
    require(not path.is_absolute() and '..' not in path.parts, f'Invalid relative member: {relative}')
    member = root / path
    require(member.is_file() and not member.is_symlink() and digest(member) == expected,
            f'Missing, altered or nonregular member: {relative}')
    return member


def roles_from_base(base, receipt):
    design = read_json(base / 'index.json')['tasks']
    registered = {'index.json', 'stimuli.jsonl'} | {task['file'] for task in design}
    require(set(receipt['packet_sha256']) == registered, 'Base packet inventory changed')
    for relative, expected in receipt['packet_sha256'].items():
        verified_member(base, relative, expected)
    roles = {}
    families = {}
    for task in design:
        for row in records(base / task['file']):
            key, role, family = row['doc'], row['split'], row['family']
            require(type(role) is int and role in (0, 1) and bool(family)
                    and roles.setdefault(key, (role, family)) == (role, family)
                    and families.setdefault(family, role) == role, 'Base material family spans multiple roles')
    return roles, {row['id']: row['text'] for row in records(base / 'stimuli.jsonl')}


def read_formal_registry(holdout):
    design = read_json(holdout)
    require(design['schema_version'] == 1, 'Formal registry schema changed')
    names, fingerprints, pinned_bytes = set(), set(), set()
    for family in design['families']:
        names.update(name.casefold() for name in (family['id'], *family['aliases']))
        for pin in family.get('source_pins', []):
            if 'path' not in pin:
                continue
            path = verified_member(ROOT, pin['path'], pin['sha256'])
            pinned_bytes.add(pin['sha256'])
            if pin.get('kind') == 'normalized_text_hashes':
                fingerprints.update(row['sha256'] for row in records(path))
    return names, fingerprints, pinned_bytes


def read_geometry(packet, receipt, base_text, base_roles, formal, tokenizer_path):
    tokenizer = None
    if tokenizer_path is not None:
        from tokenizers import Tokenizer

        require(digest(tokenizer_path) == receipt['tokenizer_sha256'], 'Tokenizer hash changed')
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    token_count = 0
    documents = {}
    for row in records(packet / 'stimuli.jsonl'):
        key, text, stops = row['id'], row['text'], row['stops']
        normalized = ' '.join(WORDS.findall(unicodedata.normalize('NFKC', text).casefold()))
        require(row.keys() == {'id', 'text', 'stops'} and key not in documents
                and key == text_digest(text) and base_text.get(key) == text and key in base_roles
                and normalized and text_digest(normalized) not in formal and stops
                and stops == sorted(set(stops)) and all(type(stop) is int and 0 < stop <= len(text) for stop in stops),
                f'Unregistered, duplicate, formally held-out or invalid text: {key}')
        documents[key] = len(text), set(stops)
        if tokenizer is None:
            continue
        full = tokenizer.encode(text, add_special_tokens=False)
        last = {end: offset for offset, (start, end) in enumerate(full.offsets) if start < end}
        require(not set(stops) - last.keys(), f'Observation does not fall at a token boundary: {key}')
        prefixes = tokenizer.encode_batch([text[:stop] for stop in stops], add_special_tokens=False)
        require(all(prefix.ids == full.ids[:last[stop] + 1]
                    and max(end for _, end in full.offsets[:last[stop] + 1]) == stop
                    for stop, prefix in zip(stops, prefixes, strict=True)),
                f'Observation does not match the independently tokenized prefix: {key}')
        token_count += len(full.ids)
    require(len(documents) == receipt['documents']
            and sum(len(stops) for _, stops in documents.values()) == receipt['positions']
            and (tokenizer is None or token_count == receipt['input_tokens']), 'Text or token totals changed')
    return documents


def validate_outcome(task, row):
    name = task['name']
    kind, values = task['likelihood'], row['outcome']
    require(isinstance(values, list) and values
            and all(type(value) in (int, float) and math.isfinite(value) for value in values),
            f'Invalid numeric observation for {name}')
    if kind in COUNT_MODELS:
        width = len(row['coordinates']) + 1 if kind == 'first_event' else len(task['labels'])
        require(len(values) == width and sum(values) > 0
                and all(value == int(value) and value >= 0 for value in values),
                f'Invalid categorical count or event support for {name}')
    elif kind == 'gaussian':
        require(len(values) == 3 and values[0] == int(values[0]) and values[0] > 0 and values[2] >= 0,
                f'Invalid continuous sufficient statistics for {name}')
    elif kind == 'known_error_gaussian':
        require(len(values) == 2 and values[1] >= 0, f'Invalid published estimate and standard error for {name}')
    elif kind == 'paired_gaussian':
        require(len(values) == 2, f'Invalid paired response for {name}')
    else:
        raise ValueError(f'Unrecognized likelihood for {name}: {kind}')
    if kind == 'choice':
        correct = row['correct_index']
        require(type(correct) is int and 0 <= correct < len(values), f'Invalid correct option for {name}')


def verify_tasks(packet, tasks, receipt, documents, base_roles):
    registry = contracts()
    names = {task['name'] for task in tasks}
    require(len(names) == len(tasks) and names == set(registry) == set(receipt['endpoints'])
            and not names & REPLACED, 'Unregistered, duplicate or superseded observation blocks')
    require({task['source'] for task in tasks} == set(receipt['source_files_sha256']),
            'Observation blocks differ from the 24 registered sources')
    touched = defaultdict(set)
    inventory = {}
    for task in tasks:
        name = task['name']
        contract = asdict(registry[name])
        contract['states'] = list(contract['states'])
        require(all(task[key] == value for key, value in contract.items()) and task['file'] == f'{name}.jsonl',
                f'Observation contract changed: {name}')
        roles = Counter()
        families = {}
        expected_keys = {'coordinates', 'role', 'family', 'outcome'}
        if task['likelihood'] == 'choice':
            expected_keys.add('correct_index')
        for row in records(packet / task['file']):
            role, family, points = row['role'], row['family'], row['coordinates']
            require(row.keys() == expected_keys and type(role) is int and role in (0, 1)
                    and points and families.setdefault(family, role) == role, f'Invalid row or material role: {name}')
            for key, stop in points:
                require(base_roles.get(key) == (role, family) and key in documents
                        and type(stop) is int and stop in documents[key][1], f'Position or family changed: {name}')
                touched[key].add(stop)
            if task['coordinates'] == 'point':
                require(len(points) == 1, f'Point observation contains another position: {name}')
            elif task['coordinates'] in {'pair', 'target_and_document_end'}:
                require(len(points) == 2, f'Paired observation is missing a position: {name}')
                if task['coordinates'] == 'target_and_document_end':
                    require(points[0][0] == points[1][0] and points[1][1] == documents[points[0][0]][0],
                            f'Retrospective observation lacks a full-document state: {name}')
            elif task['coordinates'] == 'trajectory':
                stops = [stop for _, stop in points]
                require(len({key for key, _ in points}) == 1 and stops == sorted(set(stops)),
                        f'Noncausal or repeated trajectory position: {name}')
            else:
                raise ValueError(f'Unknown coordinate contract: {name}')
            validate_outcome(task, row)
            roles[role] += 1
        entry = {'source': task['source'], 'train_rows': roles[0], 'development_rows': roles[1], **contract}
        require(roles[0] and entry == receipt['endpoints'][name], f'Observation inventory changed: {name}')
        inventory[name] = entry
    require(dict(touched) == {key: stops for key, (_, stops) in documents.items()},
            'Observation packet contains unused or missing extraction positions')
    return inventory


def check_frick(packet, cache):
    base = cache / 'semantic-tracking/frick-consistency/prepared'
    materials = {row['id']: row['text'] for row in records(base / 'stimuli.jsonl')}
    points = {
        row['id']: tuple((text_digest(materials[row['id']]), entry['stop']) for entry in row['sentences'][1:])
        for row in records(base / 'metadata.jsonl')
    }
    first_counts = defaultdict(Counter)
    after_counts = defaultdict(Counter)
    readers = set()
    for raw in records(base / 'outcomes.jsonl'):
        key = raw['id']
        reader_trial = key, raw['reader_id']
        require(reader_trial not in readers, 'Frick reader-story pair belongs to two raw paths')
        readers.add(reader_trial)
        path = points[key]
        keys = raw['keys'][1:]
        require(len(keys) == len(path) and set(keys) <= {'k', 'd'}, 'Frick raw path and displayed text differ')
        reports = [i for i, keypress in enumerate(keys) if keypress == 'd']
        first = min(reports, default=len(path))
        first_counts[path][first] += 1
        for point, keypress in zip(path[first + 1:], keys[first + 1:], strict=True):
            after_counts[point][keypress] += 1
    require(len(readers) == 1800 and len(first_counts) == 40, 'Original Frick development cohort changed')
    first_packet = {}
    for row in records(packet / 'frick_first_report.jsonl'):
        path = tuple(map(tuple, row['coordinates']))
        require(path not in first_packet, 'Frick first events appear in duplicate packet rows')
        first_packet[path] = row['outcome']
    after_packet = {}
    for row in records(packet / 'frick_after_first_report.jsonl'):
        [point] = row['coordinates']
        point = tuple(point)
        require(point not in after_packet, 'Frick post-first decisions appear in duplicate packet rows')
        after_packet[point] = row['outcome']
    require(first_packet == {path: [count[i] for i in range(len(path) + 1)] for path, count in first_counts.items()}
            and after_packet == {point: [count['k'], count['d']] for point, count in after_counts.items()},
            'Frick event or censoring distribution differs from the original linked human responses')
    return len(readers), sum(sum(count.values()) for count in after_counts.values())


def verify_packet(packet, base, cache, holdout, tokenizer_path=None):
    receipt = read_json(packet / 'receipt.json')
    prior = read_json(base / 'receipt.json')
    index = read_json(packet / 'index.json')
    tasks = index['tasks']
    require(index['schema_version'] == 2 and receipt['base_receipt_sha256'] == digest(base / 'receipt.json')
            and receipt['formal_holdout_sha256'] == prior['formal_holdout_sha256'] == digest(holdout)
            and receipt['tokenizer_sha256'] == prior['tokenizer_sha256'], 'Base corpus or observations schema changed')
    sources = receipt['source_files_sha256']
    require(len(sources) == 24 and sources == prior['source_files_sha256'], 'Registered source inventory changed')
    names, fingerprints, pinned_bytes = read_formal_registry(holdout)
    require(not {source.casefold() for source in sources} & names
            and len(fingerprints) == prior['normalized_formal_text_hashes_checked'],
            'Source roles intersect the formal registry')
    for members in sources.values():
        for name, expected in members.items():
            require(expected not in pinned_bytes, f'Source digest intersects the formal registry: {name}')
            verified_member(cache, name, expected)
    manifest = receipt['packet_sha256']
    expected = {'index.json', 'stimuli.jsonl'} | {task['file'] for task in tasks}
    require(set(manifest) == expected and {path.name for path in packet.iterdir()} == expected | {'receipt.json'},
            'Observation packet contains undeclared members')
    for name, checksum in manifest.items():
        require(checksum not in pinned_bytes, f'Packet member intersects the formal registry: {name}')
        verified_member(packet, name, checksum)
    base_roles, base_text = roles_from_base(base, prior)
    geometry = read_geometry(packet, receipt, base_text, base_roles, fingerprints, tokenizer_path)
    inventory = verify_tasks(packet, tasks, receipt, geometry, base_roles)
    first, later = check_frick(packet, cache)
    return {
        'schema_version': 2,
        'observation_blocks': len(inventory),
        'sources': len(sources),
        'documents': len(geometry),
        'positions': receipt['positions'],
        'input_tokens_verified': receipt['input_tokens'] if tokenizer_path is not None else None,
        'packet_receipt_sha256': digest(packet / 'receipt.json'),
        'formal_holdout_sha256': digest(holdout),
        'formal_fingerprints_checked': len(fingerprints),
        'frick_first_events_or_censors_verified': first,
        'frick_later_responses_verified': later,
    }


def load_forecast(archive, expected, prefix, name, shape):
    import numpy as np

    key = f'{prefix}.{name}'
    expected.add(key)
    forecast = archive[key]
    require(forecast.shape == shape and forecast.dtype.kind in 'fiu'
            and np.isfinite(forecast).all(), f'Unexpected or invalid forecast array: {key}')
    return forecast


def count_responses(task, rows):
    import numpy as np

    if task['likelihood'] != 'first_event':
        return np.array([row['outcome'] for row in rows], dtype=np.float64)
    width = max(len(row['coordinates']) for row in rows) + 1
    counts = np.zeros((len(rows), width))
    for i, row in enumerate(rows):
        counts[i, :len(row['coordinates'])] = row['outcome'][:-1]
        counts[i, -1] = row['outcome'][-1]
    return counts


def count_nll(task, rows, side, fetch):
    import numpy as np

    counts = count_responses(task, rows)[side]
    probability = fetch('probabilities', counts.shape).astype(np.float64)
    require(np.all((probability >= 0) & (probability <= 1))
            and np.allclose(probability.sum(axis=1), 1.0, rtol=0, atol=2e-6),
            f"Unnormalized forecast distribution: {task['name']}")
    if task['likelihood'] == 'first_event':
        for array_index, row_index in enumerate(side):
            stop_count = len(rows[row_index]['coordinates'])
            require(not np.any(probability[array_index, stop_count:-1]),
                    f"First-event forecast has mass outside the observed path: {task['name']}")
    if task['likelihood'] == 'choice':
        key = np.array([rows[i]['correct_index'] for i in side])
        correct = fetch('correct_probability', (len(side),))
        require(np.array_equal(correct, probability[np.arange(len(side)), key]),
                f"Correctness probability differs from the observed option key: {task['name']}")
    if task['likelihood'] == 'hurdle_ordinal':
        family_totals = Counter()
        for i, total in zip(side, counts.sum(axis=1), strict=True):
            family_totals[rows[i]['family']] += total
        counts = counts / np.array([family_totals[rows[i]['family']] for i in side])[:, None]
    seen = counts > 0
    require(np.all(probability[seen] > 0), f"Observed event has zero forecast mass: {task['name']}")
    return float(-np.dot(counts[seen], np.log(probability[seen])) / counts.sum())


def continuous_nll(task, rows, side, fetch):
    import numpy as np

    kind = task['likelihood']
    raw = np.array([row['outcome'] for row in rows], dtype=np.float64)
    train = raw[[row['role'] == 0 for row in rows]]
    observed = raw[side]
    row_count = len(side)
    if kind == 'paired_gaussian':
        means = fetch('means', (row_count, 2)).astype(np.float64)
        scales = fetch('scales', (row_count, 2)).astype(np.float64)
        difference = fetch('difference', (row_count,))
        correlation = fetch('correlation', (row_count,)).astype(np.float64)
        require(np.all(scales > 0) and np.all(np.abs(correlation) < 1)
                and np.allclose(difference, means[:, 1] - means[:, 0], rtol=1e-6, atol=1e-6),
                f"Invalid paired means, scales, correlation or signed change: {task['name']}")
        standardized = (observed - means) / scales
        determinant = 1 - correlation ** 2
        product = standardized[:, 0] * standardized[:, 1]
        quadratic = (np.square(standardized).sum(axis=1) - 2 * correlation * product) / determinant
        standardization = float(np.std(train)) or 1.0
        return float(np.mean(math.log(2 * math.pi) + np.log(scales).sum(axis=1)
                             + 0.5 * (np.log(determinant) + quadratic)) - 2 * math.log(standardization))
    means = fetch('mean', (row_count,)).astype(np.float64)
    scales = fetch('scale', (row_count,)).astype(np.float64)
    require(np.all(scales > 0), f"Nonpositive continuous forecast scale: {task['name']}")
    if kind == 'known_error_gaussian':
        variance = scales ** 2 + observed[:, 1] ** 2
        squared_error = (observed[:, 0] - means) ** 2
        weight = np.ones(row_count)
        standardization = float(np.std(train[:, 0])) or 1.0
    else:
        weight, target, disagreement = observed.T
        variance = scales ** 2
        squared_error = (target - means) ** 2 + disagreement
        counts, population, spread = train.T
        center = np.average(population, weights=counts)
        standardization = float(np.sqrt(np.average(spread + (population - center) ** 2, weights=counts))) or 1.0
    return float(np.average(0.5 * (np.log(2 * math.pi * variance) + squared_error / variance), weights=weight)
                 - math.log(standardization))


def verify_continuation(results, fitting, seal):
    if 'continuation_policy_sha256' not in seal:
        return {}
    path = results / 'policy.json'
    policy = read_json(path)
    require(digest(path) == seal['continuation_policy_sha256']
            and fitting['continuation'] == policy
            and fitting['continuation_policy_sha256'] == seal['continuation_policy_sha256']
            and seal['origin_heads_sha256'] == policy['origin_heads_sha256'],
            'Continued fit and prespecified learning rule disagree')
    history = fitting['history']
    origin = history[0]
    require(origin['epoch'] == policy['origin_epoch'] and origin['updates'] == 0
            and abs(origin['source_balanced_development_nll'] - policy['origin_development_nll']) <= 2e-5,
            'The archived starting checkpoint was not replayed')
    anchor = origin['source_balanced_development_nll']
    anchor_round = origin['epoch']
    stage = 0
    stopping_round = None
    for offset, row in enumerate(history[1:], start=1):
        epoch = origin['epoch'] + offset
        require(stopping_round is None and row['epoch'] == epoch
                and row['learning_rate'] == policy['learning_rates'][stage],
                'Continuation history contains a skipped epoch or incorrect learning rate')
        score = row['source_balanced_development_nll']
        if score <= anchor - policy['minimum_gain']:
            anchor = score
            anchor_round = epoch
        action = 'continue'
        if epoch - anchor_round >= policy['patience']:
            if stage + 1 < len(policy['learning_rates']):
                stage += 1
                anchor_round = epoch
                action = 'reduce_rate'
            else:
                action = 'plateau'
                stopping_round = epoch
        require(row['plateau_action'] == action, 'The observed development stopping decision does not replay')
    reason = 'development_plateau' if stopping_round is not None else 'epoch_budget'
    require(fitting['stop_reason'] == reason
            and (stopping_round is not None or len(history) - 1 == policy['max_additional_epochs']),
            'Continuation ended before its declared stopping rule allowed it')
    return {'continuation_stop_reason_replayed': reason, 'continuation_epochs': len(history) - 1,
            'development_plateau_epoch_verified': stopping_round, 'continuation_policy_sha256': digest(path)}


def verify_results(packet, results, report):
    import numpy as np

    seal = read_json(results / 'seal.json')
    hashes = seal['files_sha256']
    fit_files = {'fitting.json', 'heads.npz', 'predictions.npz'}
    extraction_files = {'extraction.json', 'causality.json'}
    require(set(hashes) in (fit_files, fit_files | extraction_files)
            and seal['packet_receipt_sha256'] == report['packet_receipt_sha256']
            and seal['backbone_updates'] == 0, 'Trained artifacts do not refer to the verified frozen-input packet')
    for name, expected in hashes.items():
        verified_member(results, name, expected)
    if extraction_files <= set(hashes):
        extraction = read_json(results / 'extraction.json')
        receipt = read_json(packet / 'receipt.json')
        require(seal['formal_holdout_sha256'] == report['formal_holdout_sha256']
                and seal['formal_data_in_worker'] is False
                and extraction['content_forward_per_document'] == 1
                and extraction['tokenizer_sha256'] == receipt['tokenizer_sha256']
                and extraction['stimuli_sha256'] == receipt['packet_sha256']['stimuli.jsonl']
                and extraction['certified_stops'] == report['positions']
                and extraction['documents'] == report['documents'],
                'Returned Qwen extraction differs from the verified observations')
    fitting = read_json(results / 'fitting.json')
    metrics = fitting['development']
    tasks = read_json(packet / 'index.json')['tasks']
    expected_arrays = set()
    actual = {}
    checked_rows = 0
    with np.load(results / 'predictions.npz', allow_pickle=False) as archive:
        for task in tasks:
            name = task['name']
            rows = list(records(packet / task['file']))
            for role, label in enumerate(('train', 'development')):
                indices = np.array([i for i, row in enumerate(rows) if row['role'] == role], dtype=np.int64)
                if not len(indices):
                    continue
                prefix = f'{name}.{label}'

                def fetch(name, shape):
                    return load_forecast(archive, expected_arrays, prefix, name, shape)

                packaged = fetch('rows', indices.shape)
                require(packaged.dtype.kind in 'iu' and np.array_equal(packaged, indices),
                        f'Forecast row mapping disagrees with the observation role: {prefix}')
                if task['likelihood'] in COUNT_MODELS:
                    actual_nll = count_nll(task, rows, indices, fetch)
                else:
                    actual_nll = continuous_nll(task, rows, indices, fetch)
                require(math.isfinite(actual_nll), f'Invalid replayed observation loss: {prefix}')
                checked_rows += len(indices)
                if role == 1:
                    metric = metrics[name]
                    require(metric['source'] == task['source'] and metric['rows'] == len(indices)
                            and math.isclose(actual_nll, metric['nll'], rel_tol=2e-5, abs_tol=2e-5),
                            f'Development observation likelihood does not replay: {name}')
                    actual[name] = metric, actual_nll
        require(len(archive.files) == len(expected_arrays) and set(archive.files) == expected_arrays,
                'Forecast archive contains undeclared arrays')
    require(set(actual) == set(metrics), 'Replayed development observation inventory differs')
    by_source = defaultdict(list)
    for metric, loss in actual.values():
        by_source[metric['source']].append(loss)
    average = sum(sum(values) / len(values) for values in by_source.values()) / len(by_source)
    history = fitting['history']
    require(all(math.isfinite(row['source_balanced_development_nll']) for row in history),
            'Nonfinite development optimization trace')
    selected = min(history, key=lambda row: row['source_balanced_development_nll'])
    require(selected['epoch'] == fitting['selected_epoch']
            and math.isclose(average, selected['source_balanced_development_nll'], rel_tol=2e-5, abs_tol=2e-5),
            'The selected checkpoint score does not replay')
    return {
        'forecast_archive_sha256': hashes['predictions.npz'],
        'forecasts_verified': checked_rows,
        'development_blocks_replayed': len(actual),
        'source_balanced_development_nll_replayed': average,
        'selected_epoch_verified': selected['epoch'],
        **verify_continuation(results, fitting, seal),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'base'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('.cache/cogllm'))
    parser.add_argument('--holdout', type=Path, default=HOLDOUT)
    parser.add_argument('--tokenizer-path', type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--output', type=Path)
    args = vars(parser.parse_args())
    output = args.pop('output')
    results = args.pop('results')
    result = verify_packet(**args)
    if results is not None:
        result.update(verify_results(args['packet'], results, result))
    data = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if output is not None:
        with output.open('x', encoding='utf-8') as stream:
            stream.write(data)
    print(data, end='')


if __name__ == '__main__':
    main()
