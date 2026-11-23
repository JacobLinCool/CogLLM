"""Prepare word-aligned OneStop development text for the reserved Timkey assay."""

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
from io import TextIOWrapper
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from research.backbone_transfer.extract import MODELS, digest, geometry, read_json, stream_rows, write_json, write_rows
from research.stage6.onestop_rereading_audit import SOURCES
from research.union_training import formal_sap as sap


PROTOCOL = Path(__file__).with_name('formal-timkey-eye.md')
SOURCE = 'ia_Paragraph_ordinary.csv.zip'
FIELDS = ('participant_id', 'trial_index', 'practice_trial', 'question_preview', 'repeated_reading_trial',
          'article_batch', 'article_id', 'paragraph_id', 'difficulty_level', 'paragraph', 'IA_ID', 'IA_LABEL')
MODE = ('False', 'False', 'False')
OUTCOMES = ('regressions_out', 'forward_duration', 'regressive_gaze', 'regressive_go_past')
PRIMARY = dict(zip(OUTCOMES, (1, 0, 0, 1)))
MODEL_WIDTHS = {'intercept': 0, 'lexical_position': 2, 'cogllm': 14}
PENALTY = 0.01
READING_FIELDS = ('IA_FIRST_FIX_PROGRESSIVE', 'IA_SKIP', 'IA_REGRESSION_OUT', 'IA_FIRST_RUN_DWELL_TIME',
                  'IA_REGRESSION_PATH_DURATION')
SOURCE_PINS = {
    'timkey-valid-N368.rds': 'adb5cdca5ba11ec27ee478a6d4459ee5be159cd6b34139e27421782e9006839f',
    'timkey-process.Rmd': 'b8cbf73d1cfa4216b18e709f0a8323380d6be92f2de834d8d5a168699b9c3ca9',
}
WORD_INDEX = 'timkey-items-byword-ClassicGP.csv'
WORD_INDEX_SHA256 = 'df7d2a78b5f77da92fa122af8beaa6edc2646fee8d193e1b69987860efd8192d'
JOINT_SEAL_SHA256 = '32965693755a21862888eb9ab5f1580fa1d61600dbcf38e010d24d4859fb81d0'


def word_geometry(path):
    if path.name != SOURCE or digest(path) != SOURCES[SOURCE]:
        raise ValueError('Expected the pinned ordinary OneStop interest-area release')
    texts = {}
    families = defaultdict(set)
    trials, readers = set(), set()
    current, words, index, rows = None, (), 0, 0
    with ZipFile(path) as archive, archive.open(path.stem) as raw, TextIOWrapper(
            raw, encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader)
        if len(set(header)) != len(header):
            raise ValueError('Repeated OneStop interest-area column')
        columns = [header.index(field) for field in FIELDS]
        for record in reader:
            person, trial, practice, preview, repeated, batch, article, paragraph, level, text, ia, word = (
                record[column] for column in columns)
            if (practice, preview, repeated) != MODE:
                continue
            key = person, trial
            material = int(batch), int(article), int(paragraph), level
            if key != current:
                if index != len(words) or key in trials:
                    raise ValueError('OneStop trial is incomplete or has nonconsecutive rows')
                current, index = key, 0
                trials.add(key)
                readers.add(person)
                words = tuple(re.finditer(r'\S+', text))
                if not words or material[0] not in (1, 2, 3) or level not in {'Adv', 'Ele'}:
                    raise ValueError('OneStop trial has invalid paragraph metadata')
                identity = sha256(text.encode('utf-8')).hexdigest()
                stimulus = {'id': identity, 'text': text, 'stops': [word.end() for word in words]}
                if texts.setdefault(identity, stimulus) != stimulus:
                    raise ValueError('Unexpected OneStop text hash collision')
                families[identity].add(material)
            if (text != stimulus['text'] or material not in families[identity] or int(ia) != index + 1
                    or index >= len(words) or words[index][0] != word):
                raise ValueError('OneStop interest area does not identify the original displayed word')
            index += 1
            rows += 1
    if (index != len(words) or (len(texts), len(trials), len(readers), rows) != (339, 9718, 180, 1_055_203)
            or any(len(materials) != 1 for materials in families.values())):
        raise ValueError('OneStop ordinary cohort differs from the verified text audit')
    design = [{'id': key, 'article_batch': batch, 'article': article, 'paragraph': paragraph, 'level': level}
              for key, materials in sorted(families.items()) for batch, article, paragraph, level in sorted(materials)]
    if len({(row['article_batch'], row['article'], row['paragraph'], row['level']) for row in design}) != 324:
        raise ValueError('OneStop material census differs from the verified text audit')
    return texts, design


def prepare(source, model, output, max_positions=10_000):
    texts, targets = word_geometry(source)
    if output.exists():
        raise FileExistsError(output)
    model_hash = read_json(MODELS)['models']['qwen']['sha256']['tokenizer.json']
    if digest(model / 'tokenizer.json') != model_hash:
        raise ValueError('Expected the frozen joint-model tokenizer')
    protocol_hash = digest(PROTOCOL)
    with TemporaryDirectory(dir=output.parent, prefix='.onestop-eye-geometry-') as temporary:
        stage = Path(temporary) / 'packet'
        stage.mkdir()
        all_documents = stage / 'all-stimuli.jsonl'
        write_rows(all_documents, texts.values())
        _, _, preflight = geometry('qwen', model, all_documents, None, protocol_hash)
        if (preflight['input_tokens'], preflight['certified_stops'], preflight['requested_stops']) != (
                47_763, 36_983, 36_983):
            raise ValueError('OneStop position certification differs from the verified text audit')
        batches, documents, positions = [], [], 0
        for item in texts.values():
            needed = len(item['stops'])
            if needed > max_positions:
                raise ValueError('An entire OneStop text does not fit the declared shard size')
            if documents and positions + needed > max_positions:
                batches.append((documents, positions))
                documents, positions = [], 0
            documents.append(item)
            positions += needed
        batches.append((documents, positions))
        inventory = {}
        for number, (documents, positions) in enumerate(batches):
            name = f'timkey-onestop-{number:03d}'
            shard = stage / 'formal' / name
            shard.mkdir(parents=True)
            stimuli = shard / 'stimuli.jsonl'
            write_rows(stimuli, documents)
            inventory[name] = {'protocol_sha256': protocol_hash, 'stimuli_sha256': digest(stimuli),
                               'documents': len(documents), 'positions': positions}
        write_rows(stage / 'material-index.jsonl', targets)
        write_json(stage / 'formal' / 'inventory.json', inventory)
        (stage / 'stimuli.jsonl').touch()
        write_json(stage / 'geometry.json', {
            'protocol_sha256': protocol_hash, 'source_sha256': SOURCES[SOURCE],
            'model_tokenizer_sha256': model_hash, 'stimuli_sha256': digest(all_documents),
            'material_index_sha256': digest(stage / 'material-index.jsonl'),
            'documents': len(texts), 'input_tokens': preflight['input_tokens'],
            'certified_stops': preflight['certified_stops'], 'shards': len(inventory),
            'human_response_columns_read': [], 'formal_response_files_read': [],
        })
        stage.rename(output)


def feature_columns(rows, states):
    import numpy as np

    result = {}
    for row in rows:
        for number, word in enumerate(re.finditer(r'\S+', row['text']), start=1):
            point = row['id'], word.end()
            if word.end() in row['stops']:
                result[point] = [sum(character.isalpha() for character in word[0]), math.log1p(number),
                                 *states[point]]
    if set(result) != set(states):
        raise ValueError('Text does not account for all certified feature positions')
    return np.asarray(list(result.values()), dtype=np.float64), list(result)


def presealed_projection(directory, stimuli, protocol_hash, heads, required):
    import numpy as np

    seal = read_json(directory / 'seal.json')
    model = read_json(MODELS)['models']['qwen']
    checks = {'model': model['repo'], 'revision': model['revision'], 'hidden_width': model['hidden_width'],
              'tokenizer_sha256': model['sha256']['tokenizer.json'], 'protocol_sha256': protocol_hash,
              'stimuli_sha256': digest(stimuli), 'content_forward_per_document': 1, 'gpu': 'NVIDIA L4'}
    if any(seal.get(key) != value for key, value in checks.items()):
        raise ValueError('Eye features do not match the exact frozen extraction')
    for name in ('states.npy', 'coordinates.jsonl', 'causality.json'):
        sap.check_digest(directory / name, seal['file_sha256'][name])
    expected = [(row['id'], stop) for row in stream_rows(stimuli) for stop in row['stops']]
    coordinates = [(row['id'], stop) for row in stream_rows(directory / 'coordinates.jsonl') for stop in row['stops']]
    x = np.load(directory / 'states.npy', mmap_mode='r', allow_pickle=False)
    if expected != coordinates or x.shape != (len(expected), model['hidden_width']):
        raise ValueError('Eye projection does not contain the certified source order and states')
    rows = [index for index, point in enumerate(coordinates) if point in required]
    with np.load(heads, allow_pickle=False) as parameters:
        weight, bias = (parameters[f'shared.processing.{name}'] for name in ('weight', 'bias'))
    if weight.shape != (12, x.shape[1]) or bias.shape != (12,):
        raise ValueError('Wrong processing state dimensionality')
    result = {}
    for block in range(0, len(rows), 512):
        indices = rows[block:block + 512]
        hidden = np.asarray(x[indices], dtype=np.float32)
        norm = np.linalg.norm(hidden, axis=-1, keepdims=True)
        shared = np.tanh((hidden / np.maximum(norm, 1e-12) * x.shape[1] ** 0.5) @ weight.T + bias)
        if not np.isfinite(shared).all():
            raise ValueError('Nonfinite processing states in eye projection')
        result.update((coordinates[index], state) for index, state in zip(indices, shared, strict=True))
    return result


def accept_gaze(gaze):
    return math.isfinite(gaze) and 0 < gaze <= 2000


def record_measures(statistics, index, regression, gaze, path):
    gt = math.log(gaze)
    statistics[0, index] += (1, regression, regression)
    statistics[1 + regression, index] += (1, gt, gt * gt)
    if regression and math.isfinite(path) and path >= gaze:
        gpt = math.log(path)
        statistics[3, index] += (1, gpt, gpt * gpt)


def measurements(path, documents, point_index):
    import numpy as np

    if path.name != SOURCE or digest(path) != SOURCES[SOURCE]:
        raise ValueError('OneStop eye release is not pinned')
    stats = np.zeros((len(OUTCOMES), len(point_index), 3), dtype=np.float64)
    with ZipFile(path) as archive, archive.open(path.stem) as raw, TextIOWrapper(
            raw, encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader)
        fields = ('participant_id', 'trial_index', 'practice_trial', 'question_preview',
                  'repeated_reading_trial', 'paragraph', 'IA_ID', *READING_FIELDS)
        indices = [header.index(field) for field in fields]
        current = None
        verified_rows = 0
        for record in reader:
            person, trial, practice, preview, repeated, text, ia, progressive, skipped, regression, gaze, path_time = (
                record[column] for column in indices)
            if (practice, preview, repeated) != MODE:
                continue
            verified_rows += 1
            if (person, trial) != current:
                current = person, trial
                identity = sha256(text.encode('utf-8')).hexdigest()
                stops = documents[identity]['stops']
            position = int(ia)
            if (progressive, skipped) != ('1', '0') or regression not in {'0', '1'} or position == len(stops):
                continue
            gaze = float(gaze) if gaze not in {'', '.'} else math.nan
            if not accept_gaze(gaze):
                continue
            path_time = float(path_time) if path_time not in {'', '.'} else math.nan
            record_measures(stats, point_index[identity, stops[position - 1]], int(regression), gaze, path_time)
    if verified_rows != 1_055_203 or (stats[:, :, 0].sum(axis=1) == 0).any():
        raise ValueError('The development measurement cohort is incomplete')
    return stats


def fit_measurement(x, stats, binary):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit

    weight, target, second = stats.T
    measure_count = weight.sum()
    if not measure_count:
        raise ValueError('Cannot fit eye readout with no eligible response')
    weights = weight / measure_count
    mean = np.sum(x * weights[:, None], axis=0)
    deviation = np.sqrt(np.sum((x - mean) ** 2 * weights[:, None], axis=0))
    safe_dev = np.where(deviation > 0, deviation, 1)
    cols = np.column_stack((np.ones(len(x)), (x - mean) / safe_dev))
    centered = target / measure_count
    target_second = second.sum() / measure_count
    cross = cols.T @ centered
    covariance = (cols.T * weights) @ cols
    initial = np.zeros(cols.shape[1])
    grand = centered.sum()
    initial[0] = math.log(grand / (1 - grand)) if binary else grand
    if x.shape[1] == 0:
        variance = None if binary else float(target_second - grand ** 2)
        return {'mean': [], 'scale': [], 'beta': initial.tolist(), 'variance': variance, 'gradient_max': 0.0}

    def loss(beta):
        pred = cols @ beta
        if binary:
            value = np.dot(weights, np.logaddexp(0, pred)) - np.dot(centered, pred)
            gradient = cols.T @ (weights * expit(pred) - centered)
        else:
            variance = target_second - 2 * cross @ beta + beta @ covariance @ beta
            if variance <= 0:
                raise ValueError('Eye timing residuals have zero estimated variance')
            value = 0.5 * math.log(variance)
            gradient = (covariance @ beta - cross) / variance
        penalty = PENALTY * np.dot(beta[1:], beta[1:])
        gradient[1:] += 2 * PENALTY * beta[1:]
        return value + penalty, gradient

    fit = minimize(loss, initial, jac=True, method='L-BFGS-B',
                   options={'gtol': 1e-9, 'ftol': 1e-14, 'maxiter': 300})
    gradient = float(np.max(np.abs(fit.jac)))
    if not math.isfinite(gradient) or gradient > 1e-6:
        raise RuntimeError(f'Eye readout gradient {gradient:g} did not converge: {fit.message}')
    beta = fit.x
    variance = None if binary else float(target_second - 2 * cross @ beta + beta @ covariance @ beta)
    return {'mean': mean.tolist(), 'scale': safe_dev.tolist(), 'beta': beta.tolist(),
            'variance': variance, 'gradient_max': gradient}


def outputs(x, fit):
    import numpy as np
    from scipy.special import expit

    beta = np.asarray(fit['beta'])
    values = ((x - fit['mean']) / fit['scale']) @ beta[1:] + beta[0]
    variance = fit['variance']
    if variance is None:
        return values, expit(values)
    return values, np.exp(values + 0.5 * variance)


def wloss(stats, values, variance):
    import numpy as np

    n, target, second = stats.T
    if variance is None:
        return n * np.logaddexp(0, values) - target * values
    return 0.5 * (n * np.log(2 * np.pi * variance) + (second - 2 * target * values + n * values ** 2) / variance)


def timkey_pin(raw):
    for name, checksum in SOURCE_PINS.items():
        sap.check_digest(raw / name, checksum)
    return raw / 'timkey-valid-N368.rds'


def raw_geometry(design, raw):
    index = raw / WORD_INDEX
    sap.check_digest(index, WORD_INDEX_SHA256)
    words = {}
    for item, condition, sentence, critical, offset, word in sap.projected_csv(
            index, ('item', 'condition', 'Sentence', 'disambPosition_0idx', 'word_pos', 'word')):
        key = int(item), condition, int(offset) + 1
        if key in words:
            raise ValueError('Nonunique exact word in the Timkey author index')
        words[key] = sentence, int(critical) + 1, word
    doc = {row['id']: row for row in stream_rows(design / 'stimuli.jsonl') if row['id'].startswith('sap-')}
    targets = [row for row in stream_rows(design / 'sap-targets.jsonl') if row['roi'] in (0, 1)]
    if {(item, condition) for item, condition, _ in words} != {(r['item'], r['condition']) for r in targets}:
        raise ValueError('Timkey and SAP share a different set of classic constructions')
    for row in targets:
        text = doc[row['id']]['text']
        slot, = (i for i, m in enumerate(re.finditer(r'\S+', text), start=1) if m.end() == row['stop'])
        if words[row['item'], row['condition'], slot] != (text, slot - row['roi'], row['word']):
            raise ValueError('Timkey has a different classic-garden-path word coordinate')
        row['slot'] = slot
    return doc, targets


def forecast(assay, development_projection, sap_assay, sap_projection, heads, one_stop, raw, output):
    import numpy as np

    joint_seal = heads.with_name('seal.json')
    sap.check_digest(joint_seal, JOINT_SEAL_SHA256)
    sap.check_digest(heads, read_json(joint_seal)['files_sha256']['heads.npz'])
    design = read_json(assay / 'geometry.json')
    protocol_hash = digest(PROTOCOL)
    if protocol_hash != design['protocol_sha256']:
        raise ValueError('Eye protocol has changed')
    documents = {row['id']: row for row in stream_rows(assay / 'all-stimuli.jsonl')}
    material_index = assay / 'material-index.jsonl'
    sap.check_digest(assay / 'all-stimuli.jsonl', design['stimuli_sha256'])
    sap.check_digest(material_index, design['material_index_sha256'])
    groups = {row['id']: row['article_batch'] for row in stream_rows(material_index)}
    required = {(row['id'], stop) for row in documents.values() for stop in row['stops']}
    states = {}
    seals = {}
    for name in read_json(assay / 'formal' / 'inventory.json'):
        source = assay / 'formal' / name / 'stimuli.jsonl'
        proj = development_projection / name
        states.update(presealed_projection(proj, source, protocol_hash, heads, required))
        seals[name] = digest(proj / 'seal.json')
    dev_x, points = feature_columns(documents.values(), states)
    if set(points) != required:
        raise ValueError('Missing OneStop eye state')
    del states
    data = measurements(one_stop, documents, {key: idx for idx, key in enumerate(points)})
    folds = np.array([groups[key] for key, _ in points])

    sap_design = sap.verified_design(sap_assay)
    test_docs, targets = raw_geometry(sap_assay, raw)
    selected = {(row['id'], row['stop']) for row in targets}
    test_states = presealed_projection(sap_projection, sap_assay / 'stimuli.jsonl',
                                       sap_design['protocol_sha256'], heads, selected)
    test_x, test_points = feature_columns(
        ({**doc, 'stops': [stop for stop in doc['stops'] if (doc['id'], stop) in selected]}
         for doc in test_docs.values()), test_states)
    index = {point: row for row, point in enumerate(test_points)}
    forecasts = [{**row, 'forecasts': {}} for row in targets]
    fits, development = {}, {}
    for endpoint, stats in zip(OUTCOMES, data, strict=True):
        fits[endpoint], development[endpoint] = {}, {}
        for mode, count in MODEL_WIDTHS.items():
            features = dev_x[:, :count]
            test = test_x[:, :count]
            binary = endpoint == 'regressions_out'
            oof = []
            for fold in (1, 2, 3):
                train = folds != fold
                fit = fit_measurement(features[train], stats[train], binary)
                values, _ = outputs(features[~train], fit)
                losses = wloss(stats[~train], values, fit['variance'])
                oof.append({'batch': fold, 'outcomes': int(stats[~train, 0].sum()),
                            'nll': float(losses.sum() / stats[~train, 0].sum())})
            fit = fit_measurement(features, stats, binary)
            values, means = outputs(test, fit)
            if not np.isfinite(values).all() or not np.isfinite(means).all():
                raise ValueError('Nonfinite eye forecast')
            fits[endpoint][mode] = fit
            development[endpoint][mode] = oof
            for row in forecasts:
                position = index[row['id'], row['stop']]
                row['forecasts'].setdefault(endpoint, {})[mode] = {
                    'value': float(values[position]), 'mean': float(means[position]), 'variance': fit['variance'],
                }
    with TemporaryDirectory(dir=output.parent, prefix='.sealed-timkey-') as temporary:
        stage = Path(temporary) / 'forecast'
        stage.mkdir()
        write_rows(stage / 'predictions.jsonl', forecasts)
        write_json(stage / 'fit.json', {'models': fits, 'development': development,
                                       'measurement_totals': dict(zip(OUTCOMES, data[:, :, 0].sum(axis=1).astype(int).tolist()))})
        write_json(stage / 'seal.json', {
            'protocol_sha256': protocol_hash, 'forecast_code_sha256': digest(Path(__file__)),
            'joint_model_seal_sha256': JOINT_SEAL_SHA256,
            'joint_heads_sha256': digest(heads), 'sap_projection_sha256': digest(sap_projection / 'seal.json'),
            'development_projection_seals': seals, 'structural_source_sha256': {WORD_INDEX: WORD_INDEX_SHA256},
            'formal_outcome_columns_read': [], 'forecasts': len(forecasts),
            'files_sha256': {file.name: digest(file) for file in stage.iterdir()},
        })
        if output.exists():
            raise FileExistsError(output)
        stage.rename(output)


def formal_measurements(raw, targets):
    import numpy as np
    import rdata

    release = timkey_pin(raw)
    shape = len(OUTCOMES), len(targets)
    result = np.zeros(shape + (3,), dtype=np.float64)
    means = np.full(shape, np.nan, dtype=np.float64)
    target_index = {(row['item'], row['condition'], row['roi']): number for number, row in enumerate(targets)}
    data = rdata.read_rds(release)
    if data['subj'].nunique() != 368:
        raise ValueError('Author participant roster differs from Timkey frozen study')
    selected = data.loc[data['cond'].isin(sap.CONDITIONS)]
    for (item, condition), subset in selected.groupby(['item', 'cond'], observed=True):
        if subset['subj'].duplicated().any():
            raise ValueError('Repeated Timkey condition item for a reader')
        for roi in (0, 1):
            index = target_index[int(item), str(condition), roi]
            slot = targets[index]['slot']
            if set(subset['ROI'].dropna().astype(int)) != {slot - roi}:
                raise ValueError('Timkey recorded ROI disagrees with the author word index')
            gaze = subset[f'gzR{slot}'].to_numpy(dtype=np.float64, na_value=np.nan)
            regression = subset[f'regoutR{slot}'].to_numpy(dtype=np.float64, na_value=np.nan)
            path = subset[f'gpR{slot}'].to_numpy(dtype=np.float64, na_value=np.nan)
            viewed = (gaze > 0) & (gaze <= 2000) & np.isin(regression, [0, 1])
            backward = viewed & (regression == 1)
            vectors = (regression[viewed], gaze[viewed & (regression == 0)], gaze[backward],
                       path[backward & np.isfinite(path) & (path >= gaze)])
            for endpoint, observed in enumerate(vectors):
                if not len(observed):
                    continue
                means[endpoint, index] = observed.mean()
                values = observed if endpoint == 0 else np.log(observed)
                result[endpoint, index] = len(values), np.sum(values), np.dot(values, values)
    return result, means


def summarize_contrast(targets, number, endpoint, stats, means, roi, construction, rng):
    import numpy as np

    choices = {(row['item'], row['condition']): i for i, row in enumerate(targets) if row['roi'] == roi}
    constructions = sap.CONSTRUCTIONS if construction is None else (construction,)
    items = sorted({i for i, _ in choices})
    chosen = [choices[i, f'{kind}_{condition}'] for i in items for kind in constructions
              for condition in ('AMB', 'UAMB')]
    if len(items) != 24:
        raise ValueError('Timkey classic-garden-path cohort has changed')
    minimum = stats[chosen, 0].min()
    expected_pairs = len(chosen) // 2
    unsupported = int((stats[chosen, 0] <= 0).sum())
    interval_info = {'endpoint': endpoint, 'roi': roi, 'construction': construction or 'pooled',
                     'pairs': expected_pairs, 'positions_without_support': unsupported}
    if unsupported:
        return {**interval_info, 'status': 'unavailable'}
    counts = stats[chosen, 0].reshape(len(items), -1).sum(axis=1)
    observed = means[chosen].reshape(len(items), len(constructions), 2)
    human_delta = (observed[:, :, 0] - observed[:, :, 1]).mean(axis=1)
    predicted = {}
    for mode in MODEL_WIDTHS:
        rows = [targets[i]['forecasts'][endpoint][mode] for i in chosen]
        means_model = np.array([row['mean'] for row in rows]).reshape(observed.shape)
        deltas = (means_model[:, :, 0] - means_model[:, :, 1]).mean(axis=1)
        losses = wloss(stats[chosen], np.array([row['value'] for row in rows]), rows[0]['variance'])
        predicted[mode] = deltas, losses.reshape(len(items), -1).sum(axis=1)
    model_delta, model_loss = predicted['cogllm']
    baseline_loss = predicted['lexical_position'][1]
    draw = rng.integers(0, len(items), size=(2000, len(items)))
    scale = 100 if number == 0 else 1
    human_ci = np.quantile(human_delta[draw].mean(axis=1), [.025, .975])
    effect_gap_ci = np.quantile((model_delta - human_delta)[draw].mean(axis=1), [.025, .975])
    likelihood_gap = (model_loss - baseline_loss)[draw].sum(axis=1) / counts[draw].sum(axis=1)
    direction = 'indeterminate' if human_ci[0] <= 0 <= human_ci[1] else (
        'agrees' if np.sign(model_delta.mean()) == np.sign(human_delta.mean()) else 'disagrees')
    return {
        **interval_info, 'status': 'scored', 'direction': direction, 'observations': int(counts.sum()),
        'minimum_support_per_condition_position': int(minimum),
        'units': 'percentage_points' if scale == 100 else 'milliseconds',
        'human_effect': float(human_delta.mean() * scale), 'human_effect_ci95': (human_ci * scale).tolist(),
        'model_minus_human_ci95': (effect_gap_ci * scale).tolist(),
        'model_minus_lexical_position_nll_ci95': np.quantile(likelihood_gap, [.025, .975]).tolist(),
        'forecasts': {mode: {'effect': float(deltas.mean() * scale), 'nll': float(loss.sum() / counts.sum())}
                      for mode, (deltas, loss) in predicted.items()},
    }


def score(forecast_dir, raw, output):
    import numpy as np

    seal_path = forecast_dir / 'seal.json'
    seal = read_json(seal_path)
    sap.check_digest(PROTOCOL, seal['protocol_sha256'])
    sap.check_digest(Path(__file__), seal['forecast_code_sha256'])
    for name in ('predictions.jsonl', 'fit.json'):
        sap.check_digest(forecast_dir / name, seal['files_sha256'][name])
    if output.exists() or seal['formal_outcome_columns_read']:
        raise ValueError('Need sealed prospective eye predictions and a fresh score path')
    targets = list(stream_rows(forecast_dir / 'predictions.jsonl'))
    if len(targets) != 288 or len(targets) != seal['forecasts']:
        raise ValueError('Need all 288 Timkey classic targets')
    data, means = formal_measurements(raw, targets)
    rng = np.random.default_rng(1729)
    result = [
        summarize_contrast(targets, number, endpoint, data[number], means[number], roi, construction, rng)
        for number, endpoint in enumerate(OUTCOMES)
        for construction in (None, *sap.CONSTRUCTIONS)
        for roi in (PRIMARY[endpoint], 1 - PRIMARY[endpoint])
    ]
    write_json(output, {'forecast_seal_sha256': digest(seal_path), 'formal_release_sha256': SOURCE_PINS,
                        'primary': [row for row in result
                                    if row['construction'] == 'pooled' and row['roi'] == PRIMARY[row['endpoint']]],
                        'secondary': [row for row in result
                                      if row['construction'] != 'pooled' or row['roi'] != PRIMARY[row['endpoint']]]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    prep = subparsers.add_parser('prepare')
    for name in ('source', 'model', 'output'):
        prep.add_argument(f'--{name}', type=Path, required=True)
    predict = subparsers.add_parser('forecast')
    for name in ('assay', 'development-projection', 'sap-assay', 'sap-projection', 'heads', 'one-stop', 'raw', 'output'):
        predict.add_argument(f'--{name}', type=Path, required=True)
    evaluation = subparsers.add_parser('score')
    for name in ('forecast-dir', 'raw', 'output'):
        evaluation.add_argument(f'--{name}', type=Path, required=True)
    arguments = vars(parser.parse_args())
    command = arguments.pop('command')
    if command == 'prepare':
        prepare(**arguments)
    elif command == 'forecast':
        forecast(**arguments)
    else:
        score(**arguments)


if __name__ == '__main__':
    main()
