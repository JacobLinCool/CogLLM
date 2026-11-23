"""Evaluate frozen text readouts for within-study Rodger Study 1 belief updating."""

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

from research.semantic_tracking.rodger_belief_prepare import CONDITIONS, POLICIES, READERS, block_id
from research.semantic_tracking import rodger_belief_readout as readout
from research.semantic_tracking.gum_bridge_extract import CANDIDATE, TOKENIZER_SHA256


PROTOCOL = Path(__file__).with_name('rodger-belief-protocol.md')
CHECKPOINT = Path(__file__).resolve().parents[1] / 'results' / 'gradio-candidate-checkpoint'
MODES = ('text_only', 'known_prior')
METHODS = ('form', 'qwen', 'condition')
GRID = np.arange(101, dtype=np.float64)
WIDTH = 2048
PROJECTION_WIDTH = 128
FEATURE_FILES = ('coordinates.jsonl', 'terminal_states.npy', 'vectors.npy', 'causality.json')
DRAWS = 10_000
SEED = 735972
BATCH = 200
EQUIVALENCE_POINTS = 5


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def indexed(rows, key, name):
    lookup = {row[key]: row for row in rows}
    if len(lookup) != len(rows) or not rows:
        raise ValueError(f'Repeated or empty identifiers in {name}')
    return lookup


def load_assay(directory):
    audit_path = directory / 'source-audit.json'
    audit = read_json(audit_path)
    for name in ('stimuli.jsonl', 'stimulus_metadata.jsonl', 'outcomes.jsonl', 'folds.json'):
        if audit['output_sha256'][name] != digest(directory / name):
            raise ValueError(f'Rodger assay hash mismatch for {name}')
    stimuli = indexed(read_rows(directory / 'stimuli.jsonl'), 'id', 'stimuli')
    metadata = indexed(read_rows(directory / 'stimulus_metadata.jsonl'), 'id', 'stimulus metadata')
    trials = list(indexed(read_rows(directory / 'outcomes.jsonl'), 'trial_id', 'outcomes').values())
    folds = indexed(read_json(directory / 'folds.json')['folds'], 'id', 'folds')
    keys = {block_id(policy, condition) for policy in POLICIES for condition in (None, *CONDITIONS)}
    if not stimuli.keys() == metadata.keys() == keys or len(trials) != 6 * READERS:
        raise ValueError('Incomplete Rodger study inputs')

    groups = Counter()
    by_reader_policy, by_reader_condition = Counter(), Counter()
    for trial in trials:
        info = metadata[trial['id']]
        policy, condition, reader = trial['policy'], trial['condition'], trial['reader_id']
        ratings = trial['prior_effectiveness'], trial['posterior_effectiveness']
        if (policy != info['policy'] or condition not in CONDITIONS or condition != info['condition']
                or not all(type(x) is int and 0 <= x <= 100 for x in ratings)):
            raise ValueError('Unmatched evidence condition or noninteger 0–100 effectiveness rating')
        groups[trial['id']] += 1
        by_reader_policy[(reader, policy)] += 1
        by_reader_condition[(reader, condition)] += 1
    readers = sorted({reader for reader, _ in by_reader_policy})
    if (len(readers) != READERS or by_reader_policy != {(reader, policy): 1 for reader in readers for policy in POLICIES}
            or by_reader_condition != {(reader, condition): 1 for reader in readers for condition in CONDITIONS}):
        raise ValueError('Each of the 301 readers must contribute one rating pair per policy and per condition')
    if groups != audit['counts_by_stimulus']:
        raise ValueError('Observed evidence cell counts disagree with the prepared audit')

    for policy in POLICIES:
        fold = folds[f'holdout-{policy}']
        expected = {key for key, info in metadata.items() if info['policy'] == policy}
        if (set(fold['test_stimulus_ids']) != expected or set(fold['train_stimulus_ids']) != keys - expected
                or fold['test_policies'] != [policy] or set(fold['train_policies']) != set(POLICIES) - {policy}
                or fold['test_trials'] != READERS or fold['train_trials'] != 5 * READERS):
            raise ValueError('The recorded policy holdout fold changed')
    if len(folds) != len(POLICIES):
        raise ValueError('Expected exactly six leave-policy-out folds')
    return stimuli, metadata, trials, audit_path


def load_states(extraction, packet, stimuli, metadata, audit_path):
    seal_path = extraction / 'seal.json'
    seal = read_json(seal_path)
    packet_path = packet / 'preflight.json'
    extractor_path = packet / 'extractor-preflight.json'
    certificate, extractor = read_json(packet_path), read_json(extractor_path)
    stimulus_sha = digest(audit_path.with_name('stimuli.jsonl'))
    source_sha = digest(audit_path)
    stops = {key: row['stops'] for key, row in stimuli.items()}
    if (certificate['source_audit_sha256'] != source_sha or certificate['candidate_id'] != CANDIDATE
            or certificate['documents'] != stops or certificate['tokenizer_sha256'] != TOKENIZER_SHA256
            or not certificate['stimuli_sha256'] == stimulus_sha == extractor['stimuli_sha256']
            or extractor['tokenizer_sha256'] != TOKENIZER_SHA256
            or {row['id']: row['stops'] for row in extractor['documents']} != stops
            or extractor['certified_stops'] != 78):
        raise ValueError('The local candidate, assay, and extractor certificate disagree')
    checkpoint = read_json(CHECKPOINT / 'checkpoint.json')
    anchor = CHECKPOINT / 'anchor' / 'heads.npz'
    if checkpoint['checkpoint_sha256'] != CANDIDATE or digest(anchor) != checkpoint['files']['anchor/heads.npz']:
        raise ValueError('The registered app anchor weights do not match the frozen checkpoint')
    with np.load(anchor, allow_pickle=False) as weights:
        projection = weights['projection']
        projection_sha = sha256(projection.tobytes(order='C')).hexdigest()
    backbone = checkpoint['backbone']
    if (seal['stimuli_sha256'] != stimulus_sha or seal['assay_sha256'] != digest(packet_path)
            or seal['preflight_sha256'] != digest(extractor_path) or seal['protocol_sha256'] != digest(PROTOCOL)
            or any(seal[key] != backbone[key] for key in ('model', 'revision', 'transformers'))
            or seal['tokenizer_sha256'] != TOKENIZER_SHA256 or seal['candidate_id'] != CANDIDATE
            or seal['anchor_heads_sha256'] != checkpoint['files']['anchor/heads.npz']
            or seal['projection_sha256'] != projection_sha or seal['projection_width'] != PROJECTION_WIDTH
            or seal['compute_dtype'] != 'float32' or seal['terminal_state_dtype'] != 'float32'
            or seal['content_forward_per_document'] != 1):
        raise ValueError('The frozen extraction and Rodger protocol or assay seals do not match')
    for name in FEATURE_FILES:
        if digest(extraction / name) != seal['file_sha256'][name]:
            raise ValueError(f'Frozen extraction hash mismatch for {name}')
    causality = read_json(extraction / 'causality.json')['checks']
    if len(causality) != 12 or not all(row['passed'] and row['projected_tokens']['passed'] for row in causality):
        raise ValueError('The frozen extraction failed a causal prefix check')

    coordinates = indexed(read_rows(extraction / 'coordinates.jsonl'), 'id', 'extracted coordinates')
    h = np.load(extraction / 'terminal_states.npy', allow_pickle=False)
    z = np.load(extraction / 'vectors.npy', allow_pickle=False)
    if (coordinates.keys() != stimuli.keys() or h.shape != (42, WIDTH) or h.dtype != np.float32
            or z.shape != (78, PROJECTION_WIDTH) or z.dtype != np.float16
            or not np.isfinite(h).all() or not np.isfinite(z).all()):
        raise ValueError('Expected 42 frozen FP32 terminal states and 78 stored prefix projections')
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError('Malformed terminal-state norm')
    terminal = dict(zip(coordinates, (h / norms).astype(np.float64), strict=True))
    projected, offset = {}, 0
    for key, row in coordinates.items():
        stops = row['stops']
        if row.keys() != {'id', 'stops'} or stops != stimuli[key]['stops']:
            raise ValueError(f'An expected causal cut was excluded for {key}')
        projected[key] = z[offset:offset + len(stops)].astype(np.float64)
        offset += len(stops)
    errors = [np.linalg.norm(projected[key][0] - projected[meta['baseline_id']][0])
              for key, meta in metadata.items() if meta['condition'] is not None]
    if len(errors) != 36 or max(errors) > 0.002:
        raise ValueError('The paired before-evidence projection does not match its standalone prefix')
    return terminal, {
        'source_audit_sha256': source_sha, 'protocol_sha256': digest(PROTOCOL), 'feature_seal_sha256': digest(seal_path),
        'packet_preflight_sha256': digest(packet_path), 'extractor_preflight_sha256': digest(extractor_path),
        'candidate_id': CANDIDATE, 'projection_sha256': projection_sha,
        'paired_prefix_controls': len(errors), 'maximum_projected_prefix_l2': max(errors),
        'scorer_sha256': digest(Path(__file__)),
    }


def stimulus_features(stimuli, metadata, terminal):
    prior = {}
    after = {}
    for key, meta in metadata.items():
        if meta['condition'] is None:
            prior[meta['policy']] = float(np.log1p(len(stimuli[key]['text'].split()))), terminal[key]
            continue
        before_id = meta['baseline_id']
        text, stops = stimuli[key]['text'], stimuli[key]['stops']
        evidence = text[meta['pages'][1]['start']:]
        if (len(stops) != 2 or text[:stops[0]] != stimuli[before_id]['text']
                or text[stops[0]:meta['pages'][1]['start']] != '\n\n' or stops[1] != len(text) or not evidence):
            raise ValueError('Inconsistent baseline or evidence text across rating views')
        after[key] = (float(np.log1p(len(evidence.split()))), len(re.findall(r'[0-9]+', evidence)),
                      terminal[key] - terminal[before_id])
    return prior, after


def design(trials, features, mode, method):
    """Share exact views, with separate rows only for a genuinely observed prior."""
    prior, after = features
    stages = (0, 1) if mode == 'text_only' else (1,)
    assignments = np.empty((len(trials), len(stages)), dtype=np.int64)
    rows, indices = [], {}
    scalar_width = 4 + (mode == 'known_prior')
    blocks = ((scalar_width, scalar_width + WIDTH), (scalar_width + WIDTH, scalar_width + 2 * WIDTH)) if method == 'qwen' else ()
    for trial_index, trial in enumerate(trials):
        policy, condition, key = trial['policy'], trial['condition'], trial['id']
        prior_rating = trial['prior_effectiveness'] if mode == 'known_prior' else None
        for j, stage in enumerate(stages):
            identity = (key if stage else policy, stage, prior_rating)
            if identity not in indices:
                prior_words, prior_h = prior[policy]
                evidence_words, digits, delta = after[key] if stage else (0, 0, np.zeros(WIDTH))
                persistent = [prior_words, stage, evidence_words, digits]
                if mode == 'known_prior':
                    persistent.append(prior_rating)
                if method == 'qwen':
                    row = np.concatenate((persistent, prior_h, delta))
                elif method == 'condition':
                    row = np.array([*persistent, *(int(stage == 1 and condition == c) for c in CONDITIONS)])
                else:
                    row = np.asarray(persistent)
                indices[identity] = len(rows)
                rows.append(row)
            assignments[trial_index, j] = indices[identity]
    return np.stack(rows), assignments, blocks


def make_forecasts(trials, features):
    actual = np.array([(row['prior_effectiveness'], row['posterior_effectiveness']) for row in trials])
    trial_policies = np.array([row['policy'] for row in trials])
    forecasts, fits = {}, {}
    for mode in MODES:
        labels = actual if mode == 'text_only' else actual[:, 1:]
        predictions = np.full((len(trials), len(METHODS), labels.shape[1], len(GRID)), np.nan)
        fits[mode] = {}
        for i, method in enumerate(METHODS):
            matrix, indices, blocks = design(trials, features, mode, method)
            fits[mode][method] = {}
            for held_out in POLICIES:
                test = trial_policies == held_out
                trained_rows, assignment = np.unique(indices[~test].ravel(), return_inverse=True)
                if np.intersect1d(indices[test], trained_rows).size:
                    raise ValueError('A holdout policy reused a training material identity')
                counts = np.zeros((len(trained_rows), len(GRID)), dtype=np.int64)
                np.add.at(counts, (assignment, labels[~test].ravel()), 1)
                fitted = readout.fit(matrix[trained_rows], counts, blocks)
                test_rows, test_indices = np.unique(indices[test].ravel(), return_inverse=True)
                predicted = readout.probabilities(matrix[test_rows], fitted)[test_indices]
                predictions[test, i] = predicted.reshape((READERS, labels.shape[1], len(GRID)))
                fits[mode][method][held_out] = fitted
        if not (np.isfinite(predictions).all() and np.all(predictions > 0)
                and np.allclose(predictions.sum(axis=-1), 1, atol=1e-12, rtol=0)):
            raise ValueError('Invalid or incomplete grade forecasts')
        forecasts[mode] = predictions
    return forecasts, fits


def measure(predictions, prior, posterior, mode):
    nll = -np.log(np.take_along_axis(predictions[:, :, -1, :], posterior[:, None, None], axis=-1)[:, :, 0])
    means = predictions @ GRID
    post_mae = np.abs(means[:, :, -1] - posterior[:, None])
    if mode == 'text_only':
        pre_mae = np.abs(means[:, :, 0] - prior[:, None])
        pre_nll = -np.log(np.take_along_axis(predictions[:, :, 0, :], prior[:, None, None], axis=-1)[:, :, 0])
        losses = np.column_stack((nll, post_mae, pre_nll, pre_mae))
        starting = means[:, :, 0]
    else:
        losses = np.column_stack((nll, post_mae))
        starting = prior[:, None]
    updates = np.column_stack((posterior - prior, means[:, :, -1] - starting))
    return losses, updates


def aggregate(reader_weights, policy_weights, group_layouts):
    draws = len(reader_weights)
    losses = np.zeros((draws, group_layouts[0][2].shape[1]))
    loss_weight = np.zeros((draws, 1))
    updates = np.zeros((draws, len(CONDITIONS), group_layouts[0][3].shape[1]))
    policy_updates = []
    for i, (readers, conditions, loss, update) in enumerate(group_layouts):
        trial_weights = reader_weights[:, readers]
        material_weight = policy_weights[:, i:i + 1]
        losses += (trial_weights @ loss) * material_weight
        loss_weight += trial_weights.sum(axis=1, keepdims=True) * material_weight
        condition_means = []
        for include in conditions:
            group_weights = trial_weights[:, include]
            counts = group_weights.sum(axis=1, keepdims=True)
            if np.any((counts == 0) & (material_weight > 0)):
                raise ValueError('A resampled policy has an empty evidence cell')
            # Excluded policies may have zero counts and are given zero weight below.
            condition_means.append((group_weights @ update[include]) / np.maximum(counts, 1))
        policy_mean = np.stack(condition_means, axis=1)
        updates += policy_mean * material_weight[:, :, None]
        policy_updates.append(policy_mean)
    if np.any(loss_weight == 0):
        raise ValueError('A resample contains no response')
    return losses / loss_weight, updates / policy_weights.sum(axis=1)[:, None, None], policy_updates


def distribution(trials, losses, updates):
    reader_ids = sorted({row['reader_id'] for row in trials})
    reader_index = {reader: i for i, reader in enumerate(reader_ids)}
    reader_col = np.array([reader_index[row['reader_id']] for row in trials])
    policy_col = np.array([row['policy'] for row in trials])
    condition_col = np.array([row['condition'] for row in trials])
    group_layouts = []
    for policy in POLICIES:
        include = policy_col == policy
        conditions = [condition_col[include] == condition for condition in CONDITIONS]
        group_layouts.append((reader_col[include], conditions, losses[include], updates[include]))
    point_loss, point_update, policy_updates = aggregate(
        np.ones((1, READERS)), np.ones((1, len(POLICIES))), group_layouts,
    )
    rng = np.random.default_rng(SEED)
    sampled_losses, sampled_updates, fixed_losses, fixed_updates = [], [], [], []
    n_policy = len(POLICIES)
    for start in range(0, DRAWS, BATCH):
        batch = min(DRAWS - start, BATCH)
        readers = rng.multinomial(READERS, np.full(READERS, 1 / READERS), size=batch)
        policies = rng.multinomial(n_policy, np.full(n_policy, 1 / n_policy), size=batch)
        sample_loss, sample_update, _ = aggregate(readers, policies, group_layouts)
        fixed_loss, fixed_update, _ = aggregate(readers, np.ones_like(policies), group_layouts)
        sampled_losses.append(sample_loss)
        sampled_updates.append(sample_update)
        fixed_losses.append(fixed_loss)
        fixed_updates.append(fixed_update)
    return (point_loss[0], point_update[0], np.concatenate(policy_updates, axis=0), np.vstack(sampled_losses),
            np.vstack(sampled_updates), np.vstack(fixed_losses), np.vstack(fixed_updates))


def interval(point, samples, fixed):
    return {'estimate': float(point), 'ci95': np.quantile(samples, [0.025, 0.975]).tolist(),
            'fixed_policy_reader_only_ci95': np.quantile(fixed, [0.025, 0.975]).tolist()}


def summarize(trials, predictions, mode):
    prior = np.array([row['prior_effectiveness'] for row in trials])
    posterior = np.array([row['posterior_effectiveness'] for row in trials])
    losses, updates = measure(predictions, prior, posterior, mode)
    loss, update, policy_updates, loss_draw, update_draw, fixed_loss, fixed_update = distribution(trials, losses, updates)
    loss_names = ['posterior_nll_nats', 'posterior_mae_rating_points']
    if mode == 'text_only':
        loss_names += ['prior_nll_nats', 'prior_mae_rating_points']
    loss_intervals = [interval(value, loss_draw[:, i], fixed_loss[:, i]) for i, value in enumerate(loss)]
    metrics = {name: dict(zip(METHODS, loss_intervals[start:start + len(METHODS)], strict=True))
               for start, name in zip(range(0, len(loss_intervals), len(METHODS)), loss_names, strict=True)}
    nll_difference = interval(loss[1] - loss[0], loss_draw[:, 1] - loss_draw[:, 0], fixed_loss[:, 1] - fixed_loss[:, 0])
    effects = {}
    for index, condition in enumerate(CONDITIONS):
        if condition == 1:
            continue
        change = update[index] - update[0]
        change_draw = update_draw[:, index] - update_draw[:, 0]
        fixed_change = fixed_update[:, index] - fixed_update[:, 0]
        by_policy = policy_updates[:, index] - policy_updates[:, 0]
        human = by_policy[:, 0]
        train_only = (human.sum() - human) / (len(POLICIES) - 1)
        estimates = np.column_stack((by_policy[:, 1:], train_only))
        errors = estimates - human[:, None]
        comparison_names = (*METHODS, 'five_training_policy_effect_mean')
        effects[str(condition)] = {
            'name': CONDITIONS[condition],
            'difference_vs_control_rating_points': {
                label: interval(change[j], change_draw[:, j], fixed_change[:, j])
                for j, label in enumerate(('human', *METHODS))
            },
            'model_minus_human_rating_points': {
                method: interval(change[j] - change[0], change_draw[:, j] - change_draw[:, 0], fixed_change[:, j] - fixed_change[:, 0])
                for j, method in enumerate(METHODS, start=1)
            },
            'per_policy_effect_mae_rating_points': dict(zip(comparison_names, np.mean(np.abs(errors), axis=0).tolist(), strict=True)),
            'by_policy': {
                policy: {'human_effect_rating_points': float(human[i]),
                         'estimated_effect_rating_points': dict(zip(comparison_names, estimates[i].tolist(), strict=True)),
                         'effect_error_rating_points': dict(zip(comparison_names, errors[i].tolist(), strict=True))}
                for i, policy in enumerate(POLICIES)
            },
        }
    return {'metrics': metrics, 'posterior_nll_qwen_minus_form': nll_difference, 'conditions': effects}


def write_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + '\n')


def run(assay, extraction, packet, destination):
    if destination.exists():
        raise FileExistsError(destination)
    stimuli, metadata, trials, audit_path = load_assay(assay)
    terminal, provenance = load_states(extraction, packet, stimuli, metadata, audit_path)
    predictions, fits = make_forecasts(trials, stimulus_features(stimuli, metadata, terminal))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent, prefix='.rodger-score-') as work:
        staging = Path(work) / 'score'
        staging.mkdir()
        array_path = staging / 'predictions.npz'
        np.savez_compressed(array_path, trial_ids=[row['trial_id'] for row in trials], methods=METHODS, **predictions)
        write_json(staging / 'fits.json', fits)
        seal = {'predictions_sha256': digest(array_path), 'fits_sha256': digest(staging / 'fits.json'),
                'rating_levels': [0, 100], 'stage_axis': {'text_only': ['prior', 'posterior'], 'known_prior': ['posterior']},
                'provenance': {**provenance, 'readout_sha256': digest(Path(readout.__file__))}}
        write_json(staging / 'forecast-seal.json', seal)
        with np.load(array_path, allow_pickle=False) as archive:
            reports = {mode: summarize(trials, archive[mode], mode) for mode in MODES}
        primary = reports['text_only']
        contrast = primary['conditions']['4']
        difference = contrast['difference_vs_control_rating_points']
        low, high = contrast['model_minus_human_rating_points']['qwen']['ci95']
        gates = {
            'lower_posterior_nll': primary['posterior_nll_qwen_minus_form']['ci95'][1] < 0,
            'negative_human_and_model_effect': difference['human']['estimate'] < 0 and difference['qwen']['ci95'][1] < 0,
            'effect_error_within_five_points': -EQUIVALENCE_POINTS < low <= high < EQUIVALENCE_POINTS,
        }
        write_json(staging / 'results.json', {
            'scope': 'Study 1 internal leave-policy-out with shared readers, effectiveness ratings',
            'policies': len(POLICIES), 'readers': READERS, 'trials': len(trials),
            'bootstrap': {'draws': DRAWS, 'seed': SEED, 'resampling': 'crossed policy × reader, common paired multinomial weights, no refit'},
            'primary': {'mode': 'text_only', 'condition': 4, 'control': 1, 'gates': gates, 'passes': all(gates.values())},
            'modes': reports, 'forecast_seal_sha256': digest(staging / 'forecast-seal.json'),
        })
        staging.rename(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assay', type=Path, required=True)
    parser.add_argument('--extraction', type=Path, required=True)
    parser.add_argument('--packet', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.assay, args.extraction, args.packet, args.output)
