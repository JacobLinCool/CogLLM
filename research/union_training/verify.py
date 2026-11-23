"""Replay the sealed union development forecasts using NumPy and original-scale observations."""

import argparse
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
import re
import tarfile

import numpy as np


METHODS = ('independent', 'shared', 'shared_effect')
CODE_FILES = {
    'research/backbone_transfer/models.json',
    'research/backbone_transfer/extract.py',
    'research/joint/train.py',
    'research/union_training/fit.py',
    'research/union_training/run_l4.py',
}
ARTIFACT_FILES = {
    'causality.json', 'development-effects.json', 'extraction.json', 'fitting.json',
    'heads.npz', 'predictions.npz', 'seal.json',
}
ROOT = Path(__file__).resolve().parents[2]
HOLDOUT = Path(__file__).with_name('holdout.json')
WORDS = re.compile(r'[^\W_]+')
NLL_RTOL = 2e-5
NLL_ATOL = 2e-5
EFFECT_TOL = 1e-9


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def digest(path):
    with path.open('rb') as source:
        return file_digest(source, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def records(path):
    with path.open(encoding='utf-8') as source:
        for line in source:
            yield json.loads(line)


def verify_hashes(root, expected):
    for name, checksum in expected.items():
        require(Path(name).name == name and digest(root / name) == checksum, f'Bad local hash for {name}')


def verify_payload(run, manifest, packet):
    receipt_file = packet / 'receipt.json'
    receipt = read_json(receipt_file)
    input_files = receipt['packet_sha256']
    design = read_json(packet / 'index.json')
    tasks = design['tasks']
    require(
        set(input_files) == {'index.json', 'stimuli.jsonl'} | {task['file'] for task in tasks},
        'Development packet file inventory is inconsistent',
    )
    verify_hashes(packet, input_files)
    require(
        manifest['formal_holdout_sha256'] == receipt['formal_holdout_sha256'] == digest(HOLDOUT),
        'Formal seal changed between packet preparation and the L4 run',
    )
    expected = manifest['payload_members_sha256']
    require(
        set(expected) == set(input_files) | {'receipt.json'} | CODE_FILES
        and all(expected[name] == checksum for name, checksum in input_files.items())
        and expected['receipt.json'] == digest(receipt_file),
        'The uploaded files must be the listed development packet and registered source code',
    )
    package = run / 'payload.tar.gz'
    require(digest(package) == manifest['payload_sha256'], 'Uploaded tarball hash changed')
    with tarfile.open(package, 'r:gz') as archive:
        members = archive.getmembers()
        require(
            len(members) == len(expected) and {member.name for member in members} == set(expected)
            and all(member.isfile() for member in members),
            'Uploaded tarball contains unregistered or nonregular files',
        )
        for member in members:
            with archive.extractfile(member) as stream:
                require(file_digest(stream, 'sha256').hexdigest() == expected[member.name],
                        f'Tarball member hash changed for {member.name}')
    require(
        digest(run / 'worker.py') == manifest['worker_sha256'],
        'Uploaded Colab worker hash changed',
    )
    return receipt, tasks


def verify_stimuli(packet, receipt):
    holdout = read_json(HOLDOUT)
    pins = set()
    reserved = set()
    for family in holdout['families']:
        reserved.update(name.casefold() for name in (family['id'], *family['aliases']))
        for pin in family.get('source_pins', []):
            if pin.get('kind') != 'normalized_text_hashes':
                continue
            relative = Path(pin['path'])
            require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid reserved fingerprint path')
            local_pin = ROOT / relative
            require(digest(local_pin) == pin['sha256'], f'Reserved fingerprints changed in {relative}')
            pins.update(row['sha256'] for row in records(local_pin))
    require(
        len(pins) == receipt['normalized_formal_text_hashes_checked']
        and not reserved & {source.casefold() for source in receipt['source_files_sha256']},
        'Formal aliases or normalized fingerprints differ from the development receipt',
    )
    documents = {}
    for row in records(packet / 'stimuli.jsonl'):
        identifier, text, stops = (row[key] for key in ('id', 'text', 'stops'))
        normalized = ' '.join(WORDS.findall(text.casefold()))
        require(
            row.keys() == {'id', 'text', 'stops'} and identifier not in documents and bool(normalized)
            and identifier == sha256(text.encode()).hexdigest()
            and sha256(normalized.encode()).hexdigest() not in pins and bool(stops)
            and stops == sorted(set(stops)) and all(type(stop) is int and 0 < stop <= len(text) for stop in stops),
            'Malformed or formally held-out text appeared in the development packet',
        )
        documents[identifier] = set(stops)
    require(len(documents) == receipt['development_documents'], 'Development text count changed')
    return documents


def verify_artifacts(run, manifest, packet, receipt, documents):
    output = run / 'extraction'
    actual = {path.name for path in output.iterdir()}
    require(actual == ARTIFACT_FILES and all((output / name).is_file() for name in actual),
            'Unexpected model outputs or extra files in verified extraction')
    artifact = manifest['artifact']
    members = {f'extraction/{name}': digest(output / name) for name in actual}
    package = run / 'results.tar.gz'
    require(
        members == artifact['members_sha256'] and artifact['archive_bytes'] == package.stat().st_size
        and digest(package) == artifact['archive_sha256'],
        'Returned compressed archive or artifact hash changed',
    )
    seal = read_json(output / 'seal.json')
    require(
        seal['files_sha256'] == {name: members[f'extraction/{name}'] for name in actual - {'seal.json'}}
        and seal['fit_concluded'] is True and seal['backbone_updates'] == 0
        and seal['device'] == seal['encoder_gpu'] == 'NVIDIA L4'
        and seal['encoder_revision'] == manifest['revision']
        and seal['formal_data_in_worker'] is False and seal['one_forward_per_stimulus'] is True
        and seal['packet_receipt_sha256'] == digest(packet / 'receipt.json')
        and seal['formal_holdout_sha256'] == manifest['formal_holdout_sha256']
        and seal['protocol_sha256'] == manifest['protocol_sha256']
        and seal['fit_code_sha256'] == manifest['payload_members_sha256']['research/union_training/fit.py'],
        'Final seal does not identify the frozen backbone, input packet and development-only run',
    )
    frozen = read_json(output / 'extraction.json')
    prefix_checks = read_json(output / 'causality.json')
    stop_count = sum(len(stops) for stops in documents.values())
    tested = {}
    for check in prefix_checks['checks']:
        variants = tested.setdefault((check['id'], check['character_end']), set())
        require(check['id'] in documents and check['variant'] not in variants
                and 0 <= check['max_relative_rms'] <= prefix_checks['relative_tolerance'],
                'An invalid or duplicate independent prefix check was collected')
        variants.add(check['variant'])
    require(
        frozen['candidate'] == manifest['candidate'] == 'qwen'
        and frozen['model'] == manifest['model'] and frozen['revision'] == manifest['revision']
        and frozen['source_code_sha256'] == manifest['payload_members_sha256']['research/backbone_transfer/extract.py']
        and frozen['protocol_sha256'] == manifest['protocol_sha256']
        and frozen['stimuli_sha256'] == receipt['packet_sha256']['stimuli.jsonl']
        and frozen['tokenizer_sha256'] == receipt['tokenizer_sha256']
        and frozen['documents'] == manifest['input_documents'] == len(documents)
        and frozen['certified_stops'] == stop_count and frozen['states_shape'] == [stop_count, 2048]
        and frozen['gpu'] == 'NVIDIA L4' and frozen['content_forward_per_document'] == 1
        and frozen['file_sha256']['causality.json'] == seal['files_sha256']['causality.json']
        and len(prefix_checks['checks']) == frozen['independent_causal_checks']
        and len(tested) >= min(3, len(documents))
        and all(variants == {'truncation', 'replacement'} for variants in tested.values())
        and all(check['passed'] is True for check in prefix_checks['checks']),
        'Frozen-state or causal-prefix verification failed',
    )
    return seal, frozen, prefix_checks


def read_outcomes(path, task, documents, roles, receipt):
    ids, groups, split, observed = [], [], [], []
    count = len(task['labels']) if task['kind'] == 'categorical' else 3
    for row in records(path):
        identifier, doc, stop, family, role, outcome = (
            row[k] for k in ('id', 'doc', 'stop', 'family', 'split', 'outcome')
        )
        require(
            row.keys() == {'id', 'doc', 'stop', 'family', 'split', 'outcome'}
            and doc in documents and type(stop) is int and stop in documents[doc] and identifier == f'{doc}:{stop}'
            and type(family) is str and bool(family) and type(role) is int and role in (0, 1)
            and roles.setdefault(('family', family), role) == role and roles.setdefault(('document', doc), role) == role
            and len(outcome) == count and all(math.isfinite(value) for value in outcome),
            f'An unregistered coordinate, observation, or split crossed roles for {task["name"]}',
        )
        ids.append(identifier)
        groups.append(family)
        split.append(role)
        observed.append(outcome)
    require(len(ids) > 0 and len(set(ids)) == len(ids), f'Invalid row inventory for {task["name"]}')
    split = np.asarray(split)
    outcomes = np.asarray(observed, dtype=np.float64)
    if task['kind'] == 'categorical':
        require(outcomes.shape[1] > 1 and np.all((outcomes >= 0) & (outcomes == np.floor(outcomes))),
                f'Invalid vote counts for {task["name"]}')
        counts = outcomes.sum(axis=1)
    else:
        require(task['kind'] == 'gaussian' and len(task['labels']) == 1
                and np.all(outcomes[:, 2] >= 0) and np.all(outcomes[:, 0] == np.floor(outcomes[:, 0])),
                f'Invalid Gaussian moments for {task["name"]}')
        counts = outcomes[:, 0]
    require(np.all(counts > 0) and np.any(split == 0), 'A packed observation has no denominator or no fitting rows')
    for value, name in ((0, 'fit'), (1, 'dev')):
        selected = split == value
        require(
            (receipt[f'{name}_rows'], receipt[f'{name}_supervision'], receipt[f'{name}_families'])
            == (int(selected.sum()), int(counts[selected].sum()), len(np.unique(np.asarray(groups)[selected]))),
            f'Row, family or observation inventory changed for {task["name"]}/{name}',
        )
    return np.asarray(ids), np.asarray(groups), split, outcomes, counts


def verify_forecasts(archive, task, method, observed, split, counts, expected_metrics):
    rows = split == 1
    values = archive[f'{method}.{task["name"]}.predictions'].astype(np.float64)
    labels = observed[rows]
    predictions, total = values[rows], counts[rows]
    observations = float(total.sum())
    require(
        values.shape == (len(split), observed.shape[1] if task['kind'] == 'categorical' else 2)
        and np.isfinite(values).all(), f'Invalid prediction shape or values for {method}/{task["name"]}',
    )
    if task['kind'] == 'categorical':
        require(np.all((values >= 0) & (values <= 1)) and np.allclose(values.sum(axis=1), 1, rtol=0, atol=2e-6),
                f'Unnormalized category forecasts for {method}/{task["name"]}')
    else:
        require(np.all(values[:, 1] > 0), f'Nonpositive Gaussian scale for {method}/{task["name"]}')
    if not observations:
        require(expected_metrics is None, f'Unobserved endpoint was evaluated: {method}/{task["name"]}')
        return values, None
    if task['kind'] == 'categorical':
        positive = labels > 0
        require(np.all(predictions[positive] > 0), 'Observed category assigned exactly zero probability')
        nll = float(-(labels[positive] * np.log(predictions[positive])).sum() / observations)
        row_brier = total * (1 + (predictions ** 2).sum(axis=1)) - 2 * (labels * predictions).sum(axis=1)
        brier = float(row_brier.sum() / observations)
        require(math.isclose(brier, expected_metrics['brier'], rel_tol=2e-5, abs_tol=2e-5),
                f'Brier replay differs for {method}/{task["name"]}')
    else:
        mean, std = predictions.T
        mse = (mean - labels[:, 1]) ** 2 + labels[:, 2]
        nll = float((total * (0.5 * (math.log(2 * math.pi) + mse / std ** 2) + np.log(std))).sum() / observations)
        rmse = float(math.sqrt((mse * total).sum() / observations))
        require(math.isclose(rmse, expected_metrics['rmse'], rel_tol=2e-5, abs_tol=2e-5),
                f'Gaussian original-scale RMSE differs for {method}/{task["name"]}')
    require(
        expected_metrics['observations'] == observations and expected_metrics['rows'] == int(rows.sum())
        and math.isclose(nll, expected_metrics['nll_nats'], rel_tol=NLL_RTOL, abs_tol=NLL_ATOL),
        f'Independent original-scale dev NLL differs for {method}/{task["name"]}',
    )
    return values, nll


def verify_effect_records(task, split, outcomes, counts, predictions, exposure_role, development_report):
    output = []
    effects = task['effects' if exposure_role == 0 else 'dev_effects']
    require(exposure_role == 0 or len(development_report) == len(effects),
            f'Missing held-out development contrasts for {task["name"]}')
    for row, effect in enumerate(effects):
        positions = np.asarray(effect['indices'])
        weights = np.asarray(effect['coefficients'], dtype=np.float64)
        declared = effect['target']
        require(
            positions.ndim == 1 and positions.dtype.kind in 'iu' and len(positions) > 0
            and weights.shape == positions.shape and np.isfinite(weights).all()
            and len(np.unique(positions)) == len(positions) and np.all((positions >= 0) & (positions < len(split)))
            and np.all(split[positions] == exposure_role) and math.isfinite(declared),
            f'Contrast indices cross development/training for {task["name"]}',
        )
        if task['kind'] == 'gaussian':
            require(effect['metric'] == 'mean', f'Gaussian category contrast in {task["name"]}')
            actual = outcomes[positions, 1]
            model_score = lambda values: values[positions, 0]
        elif effect['metric'] == 'category':
            category = effect['category']
            require(type(category) is int and 0 <= category < outcomes.shape[1], 'Invalid comparison category')
            actual = outcomes[positions, category] / counts[positions]
            model_score = lambda values: values[positions, category]
        else:
            require(effect['metric'] == 'mean', 'Invalid categorical contrast metric')
            levels = np.arange(outcomes.shape[1], dtype=np.float64)
            actual = (outcomes[positions] @ levels) / counts[positions]
            model_score = lambda values: values[positions] @ levels
        actual_difference = float(weights @ actual)
        require(math.isclose(actual_difference, declared, rel_tol=1e-10, abs_tol=1e-10),
                f'Contrast ground truth does not replay in {task["name"]}')
        if exposure_role == 1:
            results = development_report[row]
            require(
                results['label'] == effect['label'] and results['target'] == declared
                and results['matched_pairs'] == effect['matched_pairs'] and results['scale'] == effect['scale']
                and set(results['methods']) == set(METHODS),
                f'Development effect declaration changed in {task["name"]}',
            )
            differences = {}
            for method, values in predictions.items():
                difference = float(weights @ model_score(values))
                estimate = results['methods'][method]
                require(
                    math.isclose(estimate['prediction'], difference, rel_tol=EFFECT_TOL, abs_tol=EFFECT_TOL)
                    and math.isclose(estimate['error'], difference - declared, rel_tol=EFFECT_TOL, abs_tol=EFFECT_TOL),
                    f'Independent replay disagrees on {task["name"]}/{effect["label"]}/{method}',
                )
                differences[method] = {'prediction': difference, 'error': difference - declared}
            output.append({'label': effect['label'], 'human_difference': actual_difference,
                           'matched_material_pairs': effect['matched_pairs'], 'methods': differences})
    return output


def verify_checkpoints(path, tasks):
    width = 2048
    rank = 32
    expected = {}
    for method in METHODS:
        for i in range(len(tasks) if method == 'independent' else 1):
            prefix = f'{method}.adapters.{i}'
            expected.update({
                f'{prefix}.down.weight': (rank, width), f'{prefix}.down.bias': (rank,),
                f'{prefix}.up.weight': (width, rank),
            })
        for i, task in enumerate(tasks):
            classes = len(task['labels']) if task['kind'] == 'categorical' else 2
            expected[f'{method}.heads.{i}.weight'] = (classes, width)
            expected[f'{method}.heads.{i}.bias'] = (classes,)
    with np.load(path, allow_pickle=False) as weights:
        require(set(weights.files) == set(expected), 'Incomplete or unregistered model parameter arrays')
        for name, shape in expected.items():
            value = weights[name]
            require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                    f'Unexpected checkpoint array on {name}')


def verify(run, packet, output):
    manifest = read_json(run / 'manifest.json')
    require(
        manifest['run_id'] == run.name and manifest['phase'] == 'finished'
        and manifest['extraction_kind'] == 'union-training'
        and manifest['verified_outputs'] is True and manifest['release_verified'] is True
        and 'failure' not in manifest,
        'Expected a completed, collected union-training run with verified L4 release',
    )
    receipt, tasks = verify_payload(run, manifest, packet)
    require(bool(tasks) and len(tasks) == len({task['name'] for task in tasks})
            and {task['name'] for task in tasks} == set(receipt['endpoints']), 'Duplicate or missing packet endpoints')
    documents = verify_stimuli(packet, receipt)
    seal, collected, prefix_checks = verify_artifacts(run, manifest, packet, receipt, documents)
    extraction = run / 'extraction'
    fit = read_json(extraction / 'fitting.json')
    dev_report = read_json(extraction / 'development-effects.json')
    require(fit['backbone_updates'] == 0 and fit['device'] == 'NVIDIA L4'
            and fit['index_sha256'] == seal['index_sha256'] and set(fit['methods']) == set(METHODS)
            and set(fit['tasks']) == {task['name'] for task in tasks}
            and set(dev_report) == {task['name'] for task in tasks if task['dev_effects']},
            'Incomplete development reports for the three frozen-model training arms')
    verify_checkpoints(extraction / 'heads.npz', tasks)
    steps = {record['updates'] for record in fit['methods'].values()}
    require(len(steps) == 1 and next(iter(steps)) > 0, 'Matched methods have different update budgets')
    for record in fit['methods'].values():
        history = record['epochs']
        selected = history[record['selected_epoch'] - 1]['development']
        replayed = record['selected_development']
        same_scores = selected.keys() == replayed.keys() and all(
            original.keys() == replayed[name].keys() and all(
                original[metric] == replayed[name][metric] if metric in {'observations', 'rows'} else
                math.isclose(original[metric], replayed[name][metric], rel_tol=1e-6, abs_tol=1e-7)
                for metric in original
            )
            for name, original in selected.items()
        )
        require(
            len(history) == fit['epochs'] and all(row['epoch'] == i and row['cumulative_updates'] == i * record['steps_per_epoch']
                for i, row in enumerate(history, 1))
            and history[-1]['cumulative_updates'] == record['updates']
            and min(history, key=lambda row: row['development_objective'])['epoch'] == record['selected_epoch']
            and record['observed_training_row_coverage_each_epoch'] == 'exactly_once'
            and same_scores,
            'Checkpoint choice or training coverage does not match the recorded development schedule',
        )
    report, split_roles = {}, {}
    with np.load(extraction / 'predictions.npz', allow_pickle=False) as forecasts:
        expected = set()
        for task in tasks:
            name = task['name']
            meta = receipt['endpoints'][name]
            require(task['source'] == meta['source'] and task['kind'] == fit['tasks'][name]['kind']
                    and task['labels'] == fit['tasks'][name]['labels'], f'Endpoint changed: {name}')
            ids, groups, split, outcomes, counts = read_outcomes(packet / task['file'], task, documents, split_roles, meta)
            for suffix, values in (('ids', ids), ('group', groups), ('split', split)):
                field = f'{name}.{suffix}'
                require(np.array_equal(forecasts[field], values), f'Sealed forecast {field} does not match raw packet')
                expected.add(field)
            nlls, predictions = {}, {}
            dev_nll = {}
            for method in METHODS:
                prediction, nll = verify_forecasts(
                    forecasts, task, method, outcomes, split, counts,
                    fit['methods'][method]['selected_development'].get(name),
                )
                expected.add(f'{method}.{name}.predictions')
                predictions[method] = prediction
                if nll is not None:
                    nlls[method] = nll
                    dev_nll[method] = nll - (math.log(fit['tasks'][name]['spread']) if task['kind'] == 'gaussian' else 0.0)
            verify_effect_records(task, split, outcomes, counts, predictions, 0, [])
            comparisons = verify_effect_records(task, split, outcomes, counts, predictions, 1, dev_report.get(name, []))
            report[name] = {'source': task['source'], 'kind': task['kind'],
                            'development_rows': meta['dev_rows'], 'development_observations': meta['dev_supervision'],
                            'development_nll_original_scale': nlls, 'checkpoint_objective_contribution': dev_nll,
                            'paired_development_effects': comparisons}
        require(forecasts.files and set(forecasts.files) == expected,
                'Forecast archive contains unregistered, incomplete or formal-test arrays')

    aggregate = {}
    for method in METHODS:
        all_scores = [row['checkpoint_objective_contribution'][method] for row in report.values()
                      if row['development_rows'] > 0]
        result = float(np.mean(all_scores))
        fitted = fit['methods'][method]
        reported = fitted['epochs'][fitted['selected_epoch'] - 1]['development_objective']
        require(math.isclose(result, reported, rel_tol=NLL_RTOL, abs_tol=NLL_ATOL),
                'Checkpoint-wide held-out development objective does not replay')
        aggregate[method] = result
    with output.open('x', encoding='utf-8') as target:
        json.dump({
            'status': 'independently_verified', 'run_id': manifest['run_id'], 'training_updates_per_method': next(iter(steps)),
            'endpoint_count': len(tasks), 'development_endpoints_evaluated': len(all_scores),
            'paired_development_effects_replayed': sum(len(row['paired_development_effects']) for row in report.values()),
            'original_scale_nll_absolute_tolerance': NLL_ATOL, 'original_scale_nll_relative_tolerance': NLL_RTOL,
            'paired_effect_tolerance': EFFECT_TOL, 'equal_endpoint_checkpoint_objective': aggregate,
            'documents': len(documents), 'verified_causal_prefix_checks': len(prefix_checks['checks']),
            'forward_and_readout_seconds': collected['forward_and_readout_s'],
            'formal_data_in_worker': False, 'formal_data_scored': False, 'release_verified': True,
            'registered_formal_seal_sha256': seal['formal_holdout_sha256'],
            'packet_receipt_sha256': seal['packet_receipt_sha256'],
            'forecast_sha256': seal['files_sha256']['predictions.npz'],
            'forecast_seal_sha256': digest(extraction / 'seal.json'),
            'verification_code_sha256': digest(Path(__file__)), 'endpoints': report,
        }, target, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        target.write('\n')
    print(f'Independently verified {len(all_scores)} held-out development endpoints across {len(METHODS)} methods.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'packet', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    verify(**vars(parser.parse_args()))
