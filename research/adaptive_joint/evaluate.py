"""Score paired reading-responses after both adaptive-joint candidates are sealed."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata

import numpy as np

from research.backbone_transfer.extract import MODELS, digest, read_json, stream_rows, write_json, write_rows
from research.union_training import formal_anglin, formal_meco, formal_sap
from research.union_training.observation_effect_diagnostics import Development, statistic, verified_packet
from research.union_training.verify_observations import (
    COUNT_MODELS, continuous_nll, count_nll, load_forecast, verified_member,
)


CACHE = Path('.cache/cogllm')
UNION = CACHE / 'union-training'
FORMAL = UNION / 'formal-colab/cogllm-formal-20261121T203023Z-8488ca/extraction'
PAAPE = CACHE / 'semantic-tracking/context-6uv37-source'
TREATMENTS = ('adaptive', 'guarded')
INPUTS = CACHE / 'adaptive-joint/inputs'
FRICK_TARGET_TASKS = ('frick_target_first_report', 'frick_sentence_inconsistent')


@dataclass(frozen=True)
class Formal:
    name: str
    prepared: Path
    projection: Path
    seal: dict
    stimuli: list
    keys: tuple


def load_formal(directory):
    """Use only Qwen states whose cached grade-free extraction is independently verified."""
    full_seal = read_json(directory / 'seal.json')
    specification = read_json(MODELS)['models']['qwen']
    if full_seal['candidate'] != 'qwen' or full_seal['grade_free_extraction'] is not True:
        raise ValueError('Expected the released grade-free Qwen formal extraction')
    prepared_paths = {'sap': UNION / 'sap-spr-assay-v1', 'anglin': UNION / 'formal-anglin',
                      'meco': UNION / 'formal-meco'}
    records = {}
    for name, prepared in prepared_paths.items():
        projected = directory / name
        projected_seal = projected / 'seal.json'
        entry = full_seal['assessments'][name]
        if digest(projected_seal) != entry['seal_sha256']:
            raise ValueError(f'Changed cached projection receipt for {name}')
        seal = read_json(projected_seal)
        stimuli_file = prepared / 'stimuli.jsonl'
        stimuli = list(stream_rows(stimuli_file))
        expected = {
            'candidate': 'qwen', 'revision': specification['revision'], 'gpu': 'NVIDIA L4',
            'tokenizer_sha256': specification['sha256']['tokenizer.json'],
            'content_forward_per_document': 1, 'stimuli_sha256': digest(stimuli_file),
            'documents': entry['documents'], 'certified_stops': entry['positions'],
        }
        if any(seal.get(key) != value for key, value in expected.items()):
            raise ValueError(f'Incompatible cached formal data for {name}')
        for member, expected_hash in seal['file_sha256'].items():
            verified_member(projected, Path(member), expected_hash)
        geometry = [{'id': row['id'], 'stops': row['stops']} for row in stimuli]
        if geometry != list(stream_rows(projected / 'coordinates.jsonl')):
            raise ValueError(f'Formal text offsets changed for {name}')
        if len(geometry) != entry['documents']:
            raise ValueError(f'Formal document count changed for {name}')
        keys = tuple((row['id'], stop) for row in geometry for stop in row['stops'])
        if len(keys) != seal['certified_stops'] or len(set(keys)) != len(keys):
            raise ValueError(f'Formal observation positions are missing or duplicated for {name}')
        records[name] = Formal(name, prepared, projected, seal, stimuli, keys)
    formal_sap.verified_design(records['sap'].prepared)
    formal_anglin.verify_prepared(records['anglin'].prepared)
    meco = records['meco']
    formal_meco.checked_projection(meco.prepared, meco.projection, meco.stimuli)
    return records


def canonical(text):
    return unicodedata.normalize('NFKC', text).casefold()


def windows(text, name):
    text = canonical(text)
    if name == 'meco':
        chars = ''.join(char for char in text if char.isalnum())
        return {chars[index:index + 24] for index in range(len(chars) - 23)}
    words = re.findall(r'[^\W_]+', text)
    return set(zip(*(words[offset:] for offset in range(12)), strict=False))


def overlap_audit(formal, packet, token_stimuli):
    test_ids = {row['id'] for row in stream_rows(formal['sap'].prepared / 'sap-targets.jsonl')}
    exact, contexts = {}, {}
    for name, entry in formal.items():
        items = [row['text'] for row in entry.stimuli if name != 'sap' or row['id'] in test_ids]
        exact[name] = {''.join(char for char in canonical(text) if char.isalnum()) for text in items}
        contexts[name] = set().union(*(windows(text, name) for text in items))
    paths = {'observation_packet': packet / 'stimuli.jsonl', 'portable_token_packet': token_stimuli}
    audit = {}
    for label, path in paths.items():
        counts = {name: 0 for name in formal}
        matching = {name: [] for name in formal}
        for index, row in enumerate(stream_rows(path), 1):
            text = row['text']
            normalized = ''.join(char for char in canonical(text) if char.isalnum())
            for name in formal:
                hit = normalized in exact[name] or bool(windows(text, name) & contexts[name])
                if hit:
                    counts[name] += 1
                    matching[name].append(row['id'])
        audit[label] = {'sha256': digest(path), 'documents': index, 'overlap_counts': counts,
                        'matched_document_ids': matching}
        if any(counts.values()):
            raise ValueError(f'Formal text or neighboring literal window overlaps {label}: {counts}')
    return audit


def normalized_checkpoint(arm, binding):
    """Verify a pinned Qwen curve containing the folded shared adapter."""
    import torch

    curve = verified_member(arm, Path('curve.pt'), binding['curve.pt'])
    checkpoint = torch.load(curve, map_location='cpu', weights_only=True)
    model = read_json(MODELS)['models']['qwen']
    weight = checkpoint['reader']['final.weight']
    if (checkpoint['candidate'] != 'qwen' or checkpoint['revision'] != model['revision']
            or weight.ndim != 2 or weight.shape[1] != model['hidden_width']
            or weight.dtype != torch.float32 or not weight.isfinite().all()):
        raise ValueError(f'Candidate has an incompatible folded projection: {arm}')
    return weight


def normalized_states(formal, weight):
    import torch
    from torch.nn import functional as F

    states = np.load(formal.projection / 'states.npy', allow_pickle=False)
    if states.shape != (len(formal.keys), weight.shape[1]) or states.dtype != np.float32 or not np.isfinite(states).all():
        raise ValueError(f'Invalid causal Qwen states for {formal.name}')
    with torch.inference_mode():
        value = torch.from_numpy(states)
        adapted = F.linear(F.layer_norm(value, (value.shape[-1],)), weight).numpy()
    return dict(zip(formal.keys, adapted, strict=True))


def affine(values, archive, name):
    weight, bias = archive[f'{name}.weight'], archive[f'{name}.bias']
    if weight.shape != (len(bias), values.shape[-1]) or not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
        raise ValueError(f'Invalid observation weights for {name}')
    return values @ weight.T + bias


def state_at(stimuli, states, archive, state):
    keys = [(row['id'], stop) for row in stimuli for stop in row['stops']]
    values = np.asarray([states[key] for key in keys])
    adapted = np.tanh(affine(values, archive, 'shared.' + state))
    return dict(zip(keys, adapted, strict=True))


def forecast_sap(formal, states, archive, binding, output):
    targets = list(stream_rows(formal.prepared / 'sap-targets.jsonl'))
    calibration = list(stream_rows(formal.prepared / 'paape-targets.jsonl'))
    mapped = state_at(formal.stimuli, states, archive, 'processing')
    if mapped.keys() != {(row['id'], row['stop']) for row in targets + calibration}:
        raise ValueError('SAP and Paape targets must have the same certified mapped coordinates')
    observations = formal_sap.paape_observations(PAAPE / 'data_brms.Rda', calibration)
    train = np.asarray([formal_sap.features(row, mapped) for row, _ in observations], dtype=np.float64)
    observed = np.asarray([value for _, value in observations], dtype=np.float64)
    center = train.mean(axis=0)
    centered = train - center
    ridge = 0.01 * np.eye(train.shape[1])
    slopes = np.linalg.solve(centered.T @ centered / len(train) + ridge,
                             centered.T @ (observed - observed.mean()) / len(train))
    intercept = observed.mean() - center @ slopes
    predictions = [{'id': row['id'], 'stop': row['stop'],
                    'forecast': float(intercept + np.dot(formal_sap.features(row, mapped), slopes))}
                   for row in targets]
    output.mkdir(parents=True)
    write_rows(output / 'predictions.jsonl', predictions)
    write_json(output / 'seal.json', {
        **binding, 'protocol_sha256': digest(formal_sap.PROTOCOL), 'code_sha256': digest(Path(formal_sap.__file__)),
        'design_sha256': digest(formal.prepared / 'design.json'),
        'predictions_sha256': digest(output / 'predictions.jsonl'), 'prediction_count': len(predictions),
        'formal_participant_outcomes_used': False, 'calibration_source_sha256': formal_sap.PAAPE_RT_SHA256,
        'calibration_positions': len(observations), 'slopes': slopes.tolist(), 'intercept': float(intercept),
        'scale': 'development log-duration', 'ridge': 0.01,
    })


def forecast_anglin(formal, states, archive, binding, packet, output):
    from research.union_training.fit_observations import MIN_SCALE

    head = formal_anglin.HEAD
    contract = next(row for row in read_json(packet / 'index.json')['tasks'] if row['name'] == head)
    labels = np.asarray([row['outcome'] for row in stream_rows(packet / contract['file']) if row['role'] == 0])
    center, spread = labels.mean(), labels.std()
    if spread <= 0 or contract['states'] != ['belief'] or contract['likelihood'] != 'paired_gaussian':
        raise ValueError('Unrecognized Rodger source observation scale')
    mapped = state_at(formal.stimuli, states, archive, 'belief')
    name = f'measurements.{head}'
    post = float(archive[f'{name}.post_offset'])
    responses = []
    for row in formal.stimuli:
        readings = np.asarray([mapped[row['id'], stop] for stop in row['stops']])
        raw = affine(readings, archive, f'{name}.linear')
        # Original formal mapping applies the calibrated second-exposure offset to T2 and T3.
        means = (raw[:, 0] + np.asarray([0, post, post])) * spread + center
        scales = (np.logaddexp(0, raw[:, 1]) + MIN_SCALE) * spread
        probabilities = [formal_anglin.ordinal_probability(mu, sigma)
                         for mu, sigma in zip(means, scales, strict=True)]
        responses.append({'id': row['id'], 'rodger_scale_means': means.tolist(), 'rodger_scale_sd': scales.tolist(),
                          'probs': probabilities,
                          'mean_1_to_8': [float(np.dot(np.arange(1, 9), p)) for p in probabilities]})
    output.mkdir(parents=True)
    write_json(output / 'predictions.json', {**binding,
        'prepared_receipt_sha256': digest(formal.prepared / 'receipt.json'), 'responses': responses})
    write_json(output / 'seal.json', {
        **binding, 'predictions_sha256': digest(output / 'predictions.json'),
        'protocol_sha256': digest(formal_anglin.PROTOCOL), 'analysis_code_sha256': digest(Path(formal_anglin.__file__)),
    })


def forecast_meco(formal, states, archive, binding, packet, output):
    head = formal_meco.HEAD
    mapped = state_at(formal.stimuli, states, archive, 'interpretation')
    values = np.asarray([mapped[row['id'], row['stops'][0]] for row in formal.stimuli])
    logits = affine(values, archive, f'measurements.{head}.linear').astype(np.float64)
    if logits.shape[1] != 2:
        raise ValueError('MECO expects a binary correctness observation')
    contrast = logits[:, 1] - logits[:, 0]
    log_probs = -np.column_stack((np.logaddexp(0, contrast), np.logaddexp(0, -contrast)))
    forecast = [{'id': row['id'], 'log_probabilities': values} for row, values
                in zip(formal.stimuli, log_probs.tolist(), strict=True)]
    receipt = read_json(packet / 'receipt.json')
    baseline = formal_meco.baseline(packet, receipt, formal.stimuli)
    output.mkdir(parents=True)
    write_rows(output / 'forecasts.jsonl', forecast)
    write_json(output / 'forecast-seal.json', {
        **binding, 'family': formal_meco.FAMILY, 'readout': head, 'baseline': baseline,
        'protocol_sha256': digest(formal_meco.PROTOCOL), 'formal_holdout_sha256': digest(formal_meco.HOLDOUT),
        'analysis_code_sha256': digest(Path(formal_meco.__file__)),
        'data_sha256': formal_meco.source_pins()[formal_meco.OUTCOME],
        'forecasts_sha256': digest(output / 'forecasts.jsonl'),
    })


def build_forecasts(formal, packet, arm, member_hashes, output):
    weight = normalized_checkpoint(arm, member_hashes)
    heads = verified_member(arm, Path('heads.npz'), member_hashes['heads.npz'])
    predictors = {'sap': forecast_sap, 'anglin': partial(forecast_anglin, packet=packet),
                  'meco': partial(forecast_meco, packet=packet)}
    with np.load(heads, allow_pickle=False) as archive:
        for name, record in formal.items():
            binding = {'curve_sha256': member_hashes['curve.pt'], 'heads_sha256': member_hashes['heads.npz'],
                       'projection_seal_sha256': digest(record.projection / 'seal.json'),
                       'training_packet_receipt_sha256': digest(packet / 'receipt.json'),
                       'forecast_code_sha256': digest(Path(__file__))}
            states = normalized_states(record, weight)
            predictors[name](record, states, archive, binding, output=output / name)


def replay_losses(packet, index, predictions):
    """Independently recompute the held-out likelihoods under the published sample units."""
    losses = {}
    expected_arrays = set()
    with np.load(predictions, allow_pickle=False) as archive:
        for task in index.values():
            rows = list(stream_rows(packet / task['file']))
            for role, label in enumerate(('train', 'development')):
                indices = np.asarray([i for i, row in enumerate(rows) if row['role'] == role])
                if not len(indices):
                    continue
                prefix = f"{task['name']}.{label}"
                fetch = partial(load_forecast, archive, expected_arrays, prefix)
                selected = fetch('rows', indices.shape)
                if selected.dtype.kind not in 'iu' or not np.array_equal(selected, indices):
                    raise ValueError(f'Prediction rows do not match the registered {prefix} split')
                score = count_nll if task['likelihood'] in COUNT_MODELS else continuous_nll
                prediction_loss = score(task, rows, indices, fetch)
                if role == 1:
                    losses[task['name']] = {'source': task['source'], 'nll': prediction_loss, 'rows': len(indices)}
        if len(archive.files) != len(expected_arrays) or set(archive.files) != expected_arrays:
            raise ValueError('Unexpected per-task arrays in sealed predictions')
    return losses


def developmental_contrasts(base, packet, base_index, index, predictions):
    """Replay only historical contrasts whose sampled texts remain in development."""
    result = []
    with np.load(predictions, allow_pickle=False) as archive:
        diagnostics = Development(packet, index, archive)
        for task in base_index.values():
            if not task['dev_effects']:
                continue
            rows = list(stream_rows(base / task['file']))
            for effect in task['dev_effects']:
                coefficients = np.asarray(effect['coefficients'], dtype=np.float64)
                paired = [rows[position] for position in effect['indices']]
                report = {'task': task['name'], 'label': effect['label'], 'source': task['source'],
                          'scale': effect['scale'], 'matched_pairs': effect['matched_pairs']}
                if task['name'] in FRICK_TARGET_TASKS:
                    result.append({**report, 'status': 'reserved_for_portable_test',
                                   'reason': 'Frick target source families were removed from observation scoring'})
                    continue
                if (len(paired) != len(coefficients) or not np.isclose(coefficients.sum(), 0, atol=1e-10)
                        or any(row['split'] != 1 for row in paired)):
                    raise ValueError('Published development contrast crosses its original observation split')
                expected = coefficients @ [statistic(row['outcome'], task, effect) for row in paired]
                measurements = [diagnostics.measurement(task, (row['doc'], row['stop']), effect) for row in paired]
                if any(value is None for value in measurements):
                    result.append({**report, 'status': 'unsupported'})
                    continue
                observed, estimate = coefficients @ np.asarray(measurements)
                if not np.isclose(observed, expected, atol=1e-10) or not np.isclose(expected, effect['target'], atol=1e-10):
                    raise ValueError('Development effect could not be reconstructed from published contrasts')
                result.append({**report, 'status': 'replayed', 'human_difference': float(observed),
                               'model_difference': float(estimate), 'error': float(estimate - observed),
                               'standardized_absolute_error': float(abs(estimate - observed) / effect['scale'])})
    return result


def mean_source(values):
    return float(np.mean([np.mean(rows) for rows in values.values()]))


def source_bootstrap(values, seed=1729):
    # Resample sources as a unit to preserve dependence among endpoints from the same study.
    estimates = np.asarray([np.mean(rows) for _, rows in sorted(values.items())])
    if len(estimates) < 2:
        raise ValueError('A source summary needs more than one observed source')
    rng = np.random.default_rng(seed)
    replicate = rng.choice(estimates, size=(10000, len(estimates)), replace=True).mean(axis=1)
    return np.quantile(replicate, [.025, .975]).tolist()


def effect_key(row):
    return row['task'], row['label']


def compare_development(base, packet, arms, bindings, output):
    index, receipt = verified_packet(packet)
    base_index, base_receipt = verified_packet(base)
    if (receipt['base_receipt_sha256'] != digest(base / 'receipt.json')
            or receipt['formal_holdout_sha256'] != base_receipt['formal_holdout_sha256']):
        raise ValueError('Contrast specifications are from another registered study split')
    metrics, source_losses, effects = {}, {}, {}
    for name, files in arms.items():
        location = output / name
        location.mkdir(parents=True, exist_ok=True)
        predictions = verified_member(files, Path('predictions.npz'), bindings[name]['predictions.npz'])
        metrics[name] = replay_losses(packet, index, predictions)
        orig = location / 'development-effects.json'
        rows = developmental_contrasts(base, packet, base_index, index, predictions)
        effects[name] = {effect_key(row): row for row in rows if row['status'] == 'replayed'}
        report = {'registered_contrasts': len(rows), 'replayed_contrasts': len(effects[name]), 'effects': rows,
                  'predictions_sha256': bindings[name]['predictions.npz']}
        report['mean_standardized_absolute_error'] = float(np.mean(
            [row['standardized_absolute_error'] for row in effects[name].values()]))
        source_losses[name] = defaultdict(list)
        for task in metrics[name].values():
            source_losses[name][task['source']].append(task['nll'])
        write_json(orig, report)

    fixed = 'fixed'
    treatment, = arms.keys() - {fixed}
    if effects[fixed].keys() != effects[treatment].keys() or metrics[fixed].keys() != metrics[treatment].keys():
        raise ValueError('Comparison changes the evaluated observation set across arms')
    by_source_loss, by_source_effect = defaultdict(list), defaultdict(list)
    direction_counts = dict.fromkeys(arms, 0)
    effect_rows = []
    for task, fixed_loss in metrics[fixed].items():
        next_loss = metrics[treatment][task]
        if (fixed_loss['source'], fixed_loss['rows']) != (next_loss['source'], next_loss['rows']):
            raise ValueError(f'Changed development population for {task}')
        by_source_loss[fixed_loss['source']].append(next_loss['nll'] - fixed_loss['nll'])
    for key, before in effects[fixed].items():
        after = effects[treatment][key]
        human = before['human_difference']
        source = base_index[before['task']]['source']
        if human != after['human_difference'] or before['matched_pairs'] != after['matched_pairs']:
            raise ValueError('Human contrast or denominator differs across candidates')
        difference = after['standardized_absolute_error'] - before['standardized_absolute_error']
        by_source_effect[source].append(difference)
        directions = {fixed: human * before['model_difference'] > 0,
                      treatment: human * after['model_difference'] > 0}
        for arm in arms:
            direction_counts[arm] += directions[arm]
        effect_rows.append({
            'source': source, 'task': key[0], 'label': key[1], 'human': human, 'matched_pairs': before['matched_pairs'],
            fixed: before['model_difference'], treatment: after['model_difference'],
            'reference_scale': before['scale'], f'{treatment}_minus_fixed_scaled_absolute_error': difference,
            'direction_match': directions,
        })
    write_json(output / 'development-comparison.json', {
        'role': 'existing development diagnostics; may have guided model selection',
        'treatment': treatment,
        'development_source_balanced_nll': {arm: mean_source(source_losses[arm]) for arm in arms},
        f'{treatment}_minus_fixed_source_balanced_nll': mean_source(by_source_loss),
        'descriptive_source_bootstrap95_nll_difference': source_bootstrap(by_source_loss),
        'per_task_development': metrics,
        'direction_matches': direction_counts, 'nonzero_human_contrast_count': sum(x['human'] != 0 for x in effect_rows),
        'replayed_human_contrasts': len(effect_rows),
        f'{treatment}_minus_fixed_source_balanced_scaled_absolute_error': mean_source(by_source_effect),
        'descriptive_source_bootstrap95_scaled_absolute_error_difference': source_bootstrap(by_source_effect),
        'effect_rows': effect_rows,
    })


def sap_pair(assay, source, fixed, candidate, treatment='adaptive'):
    targets = list(stream_rows(assay / 'sap-targets.jsonl'))
    def matched_forecast(path):
        values = {(row['id'], row['stop']): row['forecast'] for row in stream_rows(path / 'predictions.jsonl')}
        return {(row['item'], row['condition'], row['roi']): values[row['id'], row['stop']] for row in targets}

    predictions = tuple(matched_forecast(path) for path in (fixed, candidate))
    _, release = formal_sap.sap_files(source)
    empirical, _ = formal_sap.sap_observations(release, predictions[0].keys())
    rng = np.random.default_rng(1729)
    comparisons = []
    for name in formal_sap.CONSTRUCTIONS:
        for roi in formal_sap.REGIONS:
            ambiguous, control = f'{name}_AMB', f'{name}_UAMB'
            items = sorted(item for item, label, key in empirical if label == ambiguous and key == roi
                           and (item, name, roi) != formal_sap.SAP_POSITION_EXCLUSION)
            differences = np.asarray([[vector[item, ambiguous, roi] - vector[item, control, roi] for item in items]
                                      for vector in (empirical, *predictions)])
            index = rng.integers(0, len(items), size=(2000, len(items)))
            differences_replicates = differences[:, index].mean(axis=2)
            errors = np.abs(differences_replicates[1:] - differences_replicates[0])
            h, f, a = differences.mean(axis=1)
            comparisons.append({'construction': name, 'roi': roi, 'items': len(items),
                                'human': float(h), 'fixed': float(f), treatment: float(a),
                                'no_effect_baseline_abs_error': float(abs(h)),
                                'fixed_abs_error': float(abs(f - h)), f'{treatment}_abs_error': float(abs(a - h)),
                                f'{treatment}_minus_fixed_abs_error': float(abs(a - h) - abs(f - h)),
                                'paired_item_bootstrap_95_interval': np.quantile(errors[1] - errors[0], [.025, .975]).tolist()})
    return comparisons


def anglin_pair(fixed, candidate, treatment='adaptive'):
    records = []
    for condition in formal_anglin.ARM_NAMES:
        for previous, current in zip(fixed['primary'][condition], candidate['primary'][condition], strict=True):
            human, f, a = previous['human_change'], previous['predicted_change'], current['predicted_change']
            if human != current['human_change'] or previous['n'] != current['n']:
                raise ValueError('The Anglin estimands differ across arms')
            low, high = previous['difference_ci95']
            # Subtracting two absolute errors is monotone in the human mean. Transforming the
            # shared equal-tail human bootstrap interval therefore gives the paired interval.
            interval = sorted(abs(a - value) - abs(f - value) for value in (f - high, f - low))
            records.append({'condition': condition, 'before': previous['before'], 'after': previous['after'],
                            'people': previous['n'], 'human': human, 'fixed': f, treatment: a,
                            'no_update_baseline_abs_error': abs(human),
                            f'{treatment}_minus_fixed_abs_error': abs(a - human) - abs(f - human),
                            'paired_participant_bootstrap_95_interval': interval})
    return records


def meco_pair(source, fixed, candidate, treatment='adaptive'):
    def values(path):
        return np.asarray([row['log_probabilities'] for row in stream_rows(path / 'forecasts.jsonl')])

    outcome = verified_member(source, Path(formal_meco.OUTCOME), formal_meco.source_pins()[formal_meco.OUTCOME])
    counts = formal_meco.outcome_cells(outcome)
    differences = (counts * (values(fixed) - values(candidate))).sum(axis=2)
    trials = counts.sum(axis=2)
    return {f'{treatment}_minus_fixed_nll': float(differences.sum() / trials.sum()),
            'paired_crossed_reader_passage_bootstrap_95_interval': formal_meco.crossed_interval(differences, trials),
            'chance_50pct_nll_nats': math.log(2), 'trials': int(trials.sum())}


def compare_formal(formal, output, treatment='adaptive'):
    sources = {'sap': UNION / 'heldout/sap', 'anglin': CACHE / 'anglin', 'meco': UNION / 'heldout/chinese-meco'}
    all_scores = {}
    for arm in ('fixed', treatment):
        root = output / arm
        formal_sap.score(formal['sap'].prepared, root / 'sap', sources['sap'], root / 'sap-score.json')
        formal_anglin.score(sources['anglin'], formal['anglin'].prepared, root / 'anglin', root / 'anglin-score.json')
        formal_meco.score(sources['meco'], root / 'meco', digest(root / 'meco/forecast-seal.json'), root / 'meco-score.json')
        all_scores[arm] = {key: read_json(root / f'{key}-score.json') for key in formal}
    fixed, candidate = (all_scores[arm] for arm in ('fixed', treatment))
    scope = 'held out from current fit; previously used in earlier CogLLM evaluations'
    if treatment == 'guarded':
        scope = ('retrospective retention follow-up. Formal results from the original fixed/adaptive comparison '
                 'were known before the guarded experiment. Outcomes remain excluded from checkpoint selection.')
    write_json(output / 'formal-comparison.json', {
        'scope': scope, 'treatment': treatment,
        'fixed': fixed, treatment: candidate,
        f'sap_{treatment}_minus_fixed': sap_pair(formal['sap'].prepared, sources['sap'],
                                                output / 'fixed/sap', output / treatment / 'sap', treatment),
        f'anglin_{treatment}_minus_fixed': anglin_pair(fixed['anglin'], candidate['anglin'], treatment),
        f'meco_{treatment}_minus_fixed': meco_pair(sources['meco'], output / 'fixed/meco',
                                                 output / treatment / 'meco', treatment),
    })


def summary(root, seeds, treatment='adaptive'):
    rows = []
    for seed in seeds:
        path = root / f'seed-{seed}'
        development = read_json(path / 'development-comparison.json')
        formal = read_json(path / 'formal-comparison.json')
        for arm in ('fixed', treatment):
            score = formal[arm]
            primary_sap = [row for row in score['sap']['results'] if row['roi'] == 0]
            rows.append({
                'seed': seed, 'arm': arm, 'development_nll': development['development_source_balanced_nll'][arm],
                'development_human_directions': development['direction_matches'][arm],
                'development_nonzero_human_effects': development['nonzero_human_contrast_count'],
                'sap_disambiguator_direction_agreement': sum(row['direction'] == 'agrees' for row in primary_sap),
                'sap_disambiguator_direction_determinate': sum(row['direction'] != 'indeterminate' for row in primary_sap),
                'anglin_directions': score['anglin']['directions_matched'],
                'anglin_within_tolerance': score['anglin']['within_tolerance'],
                'meco_nll': score['meco']['primary']['model'],
                'meco_source_baseline_nll': score['meco']['primary']['source_baseline'],
            })
    write_json(root / 'summary.json', {
        'paired_seeds': list(seeds), 'treatment': treatment, 'rows': rows,
        'scope': 'Repeated seeds use the same preexisting human outcomes.',
        'retrospective_followup_after_original_comparison': treatment == 'guarded',
    })


def verify_run(run, receipt_sha256, mode):
    seal_path = run / 'seal.json'
    seal = read_json(seal_path)
    if seal['input_receipt_sha256'] != receipt_sha256 or seal['formal_labels_in_worker'] is not False:
        raise ValueError('Comparison must use the original registered observations and keep formal outcomes local')
    inventory = seal['files_sha256']
    expected = ('curve.pt', 'heads.npz', 'predictions.npz')
    seeds = tuple(seal['seed_settings']['seeds'])
    if mode not in seal['seed_settings']['modes']:
        raise ValueError(f'Run does not declare the requested {mode} arm')
    if not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('Expected explicitly paired finite seed replications')
    arms = {}
    for seed in seeds:
        arm = run / f'{mode}-{seed}'
        binding = {relative: inventory[f'{arm.name}/{relative}'] for relative in expected}
        for relative, fingerprint in binding.items():
            verified_member(arm, Path(relative), fingerprint)
        arms[seed] = arm, binding
    return {
        'run_seal_sha256': digest(seal_path), 'protocol_sha256': seal['protocol_sha256'],
        'initial_checkpoint_sha256': seal['seed_sha256'], 'revision': seal['revision'],
        'input_receipt_sha256': receipt_sha256,
    }, arms


def verify_comparison(run, fixed_run, inputs, treatment, fixed_mode='fixed'):
    if treatment not in TREATMENTS or (fixed_run is None and treatment != 'adaptive'):
        raise ValueError('The guarded comparison requires an explicitly chosen sealed fixed run')
    fixed_run = run if fixed_run is None else fixed_run
    receipt_sha = digest(inputs / 'receipt.json')
    control_manifest, control = verify_run(fixed_run, receipt_sha, fixed_mode)
    treatment_manifest, treated = verify_run(run, receipt_sha, treatment)
    initial_binding = ('input_receipt_sha256', 'initial_checkpoint_sha256', 'revision')
    if control.keys() != treated.keys() or any(control_manifest[key] != treatment_manifest[key]
                                              for key in initial_binding):
        raise ValueError('Compare exactly paired seeds, input data, and initial model revision')
    if fixed_mode == 'guarded-fixed':
        control_settings = read_json(fixed_run / 'seal.json')['seed_settings']
        treatment_settings = read_json(run / 'seal.json')['seed_settings']
        control_settings.pop('modes')
        treatment_settings.pop('modes')
        if treatment != 'guarded' or control_settings != treatment_settings:
            raise ValueError('The retention comparison must change only the source weighting rule')
    return {'fixed': {**control_manifest, 'mode': fixed_mode}, treatment: {**treatment_manifest, 'mode': treatment}}, {
        seed: {'fixed': control[seed], treatment: treated[seed]} for seed in control
    }


def evaluate(run, inputs, source_observations, base, formal_cache, output, fixed_run=None, treatment='adaptive',
             fixed_mode='fixed'):
    if output.exists():
        raise FileExistsError(output)
    packet = inputs / 'observations'
    token_stimuli = inputs / 'token/stimuli.jsonl'
    receipt = read_json(inputs / 'receipt.json')
    if receipt['source_observation_receipt_sha256'] != digest(source_observations / 'receipt.json'):
        raise ValueError('The training view is unrelated to the complete grade-free observation texts')
    for relative in ('observations/receipt.json', 'token/stimuli.jsonl'):
        verified_member(inputs, Path(relative), receipt['files_sha256'][relative])
    source_runs, pairs = verify_comparison(run, fixed_run, inputs, treatment, fixed_mode)
    verified_packet(source_observations)
    verified_packet(packet)
    for name in (formal_anglin.HEAD, formal_meco.HEAD):
        member = f'{name}.jsonl'
        if digest(packet / member) != digest(source_observations / member):
            raise ValueError(f'Formal calibration is no longer aligned to the training-side observations: {name}')
    formal = load_formal(formal_cache)
    isolate = overlap_audit(formal, source_observations, token_stimuli)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.compare-') as temporary:
        stage = Path(temporary) / 'report'
        for seed, arms in pairs.items():
            for name, (directory, binding) in arms.items():
                build_forecasts(formal, source_observations, directory, binding, stage / f'seed-{seed}' / name)
        manifested = {path.relative_to(stage).as_posix(): digest(path)
                      for path in stage.rglob('*seal.json') if path.is_file()}
        if len(manifested) != len(pairs) * len(source_runs) * len(formal):
            raise ValueError('All formal predictions must be sealed before formal outcome access')
        write_json(stage / 'formal-prediction-lock.json', {
            'source_runs': source_runs, 'treatment': treatment,
            'all_formal_forecast_seals_sha256': manifested,
            'forecast_code_sha256': digest(Path(__file__)),
            'formal_participant_outcomes_used': False,
            'retrospective_followup_after_original_comparison': treatment == 'guarded',
            'formal_text_overlap_audit': isolate,
            'overlap_scope': 'Exact text, exact 12-word windows or 24-character MECO windows',
        })
        for seed, arms in pairs.items():
            paired = stage / f'seed-{seed}'
            directories = {name: directory for name, (directory, _) in arms.items()}
            bindings = {name: binding for name, (_, binding) in arms.items()}
            compare_development(base, packet, directories, bindings, paired)
            compare_formal(formal, paired, treatment)
        summary(stage, pairs, treatment)
        write_json(stage / 'seal.json', {
            'source_runs': source_runs, 'treatment': treatment,
            'packet_receipt_sha256': digest(packet / 'receipt.json'),
            'files_sha256': {path.relative_to(stage).as_posix(): digest(path)
                             for path in stage.rglob('*') if path.is_file()},
        })
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Sealed run exporting the treatment arm')
    parser.add_argument('--fixed-run', type=Path, help='Sealed source of the fixed arm for a follow-up')
    parser.add_argument('--fixed-mode', choices=('fixed', 'guarded-fixed'), default='fixed')
    parser.add_argument('--treatment', choices=TREATMENTS, default='adaptive')
    parser.add_argument('--inputs', type=Path, default=INPUTS)
    parser.add_argument('--source-observations', type=Path, default=UNION / 'observations-2')
    parser.add_argument('--base', type=Path, default=UNION / 'development-1')
    parser.add_argument('--formal-cache', type=Path, default=FORMAL)
    parser.add_argument('--output', type=Path, required=True)
    evaluate(**vars(parser.parse_args()))
