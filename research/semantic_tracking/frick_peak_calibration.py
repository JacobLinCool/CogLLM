"""Cross-fit single-prefix affine and peak-aware calibration of Frick hazards."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.semantic_tracking import frick_conditional_score as conditional
from research.semantic_tracking import frick_consistency_score as base
from research.semantic_tracking import frick_mass_calibration as mass
from research.semantic_tracking import frick_pair_score as paired
from research.semantic_tracking import frick_signals_score as signals


PROTOCOL = Path(__file__).with_name('frick_peak_calibration_protocol.md')
PROTOCOL_SHA256 = '11ca0e35fd20f807258934ec121a2b2d74f45ce3bc405e4627af731c4c46760e'
REFERENCE = paired.REFERENCE.parent / 'frick-signals-l4'
METHODS = ('peak', 'affine', 'attention_only', 'seven_feature_conditional', 'global', 'original_contextual')
RIDGE = 0.01
CENTER = np.array([0., 1., 0.])


def frozen_base(reference, provenance, signals_provenance, slots, stimuli):
    seal_path = reference / 'forecast-seal.json'
    seal = base.read_json(seal_path)
    score = base.read_json(reference / 'results.json')
    names = ('forecasts.jsonl', 'fits.json', 'axis-audit.json')
    if (score['forecast_seal_sha256'] != base.digest(seal_path)
            or seal['protocol_sha256'] != signals.PROTOCOL_SHA256
            or seal['signals_seal_sha256'] != signals_provenance['signals_seal_sha256']
            or any(seal['provenance'][field] != provenance[field]
                   for field in ('source_audit_sha256', 'state_extraction_sha256', 'candidate_id'))
            or any(base.digest(reference / name) != seal['file_sha256'][name] for name in names)):
        raise ValueError('The original attention baseline must match the exact frozen Frick source')
    rows = []
    expect = ((key, stop) for key in slots for stop in stimuli[key]['stops'][1:])
    for item, identity in zip(base.read_rows(reference / names[0]), expect, strict=True):
        if (item['id'], item['stop']) != identity:
            raise ValueError('The sealed attention forecast is misaligned')
        rows.append([item['first_logits'][method] for method in METHODS[2:]])
    fits = base.read_json(reference / names[1])['folds_by_arm']['attention_only']
    indices = {row['held_out_story']: row for row in fits}
    if len(indices) != len(fits) or not np.isfinite(rows).all():
        raise ValueError('Expected a single valid archived fit per Frick family')
    return np.asarray(rows).T, indices, base.digest(seal_path)


def split_four(training):
    return tuple(tuple(training[i::mass.INNER_FOLDS]) for i in range(mass.INNER_FOLDS))


def groups(stories, families):
    return sorted(key for story in stories for key in families[story].values())


def inner_attention(outer, training, validation, states, forms, families, metadata, slots, hits, totals):
    # Only the small base-fitting set can affect both tuning and coefficient estimation.
    allowed = groups(training, families)
    mask = np.ones(len(hits), dtype=bool)
    mask[mass.indices_for(allowed, slots, len(hits))] = False
    safe_hits, safe_totals = hits.copy(), totals.copy()
    safe_hits[mask] = safe_totals[mask] = -1
    safe_metadata = {key: metadata[key] for key in allowed}
    args = states, forms, families, safe_metadata, slots, safe_hits, safe_totals
    tuning = [conditional.make_fold(outer, tuple(key for key in training if key not in part), part,
                                    *args, inner_validation=True)[0] for part in split_four(training)]
    phase, contrast, _ = conditional.tune(tuning)
    fit, columns = conditional.make_fold(outer, training, validation, *args, inner_validation=False)
    baseline = conditional.fit_phase(fit, phase)
    if contrast:
        baseline = conditional.fit_conditional(fit, phase, baseline, contrast)
    return columns, base.predict_logits(fit.validate, baseline), phase, contrast


def peak_features(logits):
    current = np.asarray(logits, dtype=np.float64)
    if current.ndim != 1 or not len(current) or not np.isfinite(current).all():
        raise ValueError('Each visible story requires a finite single-path logit sequence')
    advance = np.zeros(len(current))
    advance[1:] = np.maximum(0, current[1:] - np.maximum.accumulate(current[:-1]))
    return np.column_stack((np.ones(len(current)), current, advance))


def training_weights(keys, indices, metadata, slots, hits, totals):
    weights = np.zeros(len(hits))
    positives = np.zeros(len(hits))
    count = len(keys)
    for key in keys:
        slot = slots[key]
        target = slot.start + metadata[key]['target_index'] - 1
        first, at_risk = totals[slot.start], totals[target]
        if min(first, at_risk) <= 0 or not slot.start <= target < slot.stop:
            raise ValueError('A calibration story lacks exact at-risk target support')
        weights[slot] = totals[slot] / (first * count)
        positives[slot] = hits[slot] / (first * count)
        weights[target] += 1 / count
        positives[target] += hits[target] / (at_risk * count)
    return weights[indices], positives[indices]


def objective(parameters, design, weights, positives):
    eta = design @ parameters
    shift = parameters - CENTER
    loss = weights @ np.logaddexp(0, eta) - positives @ eta + RIDGE / 2 * (shift @ shift)
    gradient = design.T @ (weights * expit(eta) - positives) + RIDGE * shift
    return float(loss), gradient


def fit_calibration(design, weights, positives, allow_peak):
    result = minimize(objective, CENTER, args=(design, weights, positives), jac=True, method='L-BFGS-B',
                      bounds=((None, None), (0, None), (0, None) if allow_peak else (0, 0)),
                      options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
    if not result.success or not np.isfinite(result.x).all():
        raise ArithmeticError(f'Frick causal calibration did not converge: {result.message}')
    return result.x, float(result.fun), int(result.nit)


def fit_folds(states, forms, families, metadata, slots, hits, totals, stored_fits, attention):
    predicted = np.full((2, len(hits)), np.nan)
    receipts = []
    order = tuple(sorted(families))
    if set(stored_fits) != set(order):
        raise ValueError('Attention fits do not cover the assigned Frick families')
    compared_prefixes = 0
    largest_replay_error = largest_prefix_error = 0.0
    for outer in order:
        development = tuple(story for story in order if story != outer)
        records = []
        oof = np.full(len(hits), np.nan)
        for validation in split_four(development):
            training = tuple(story for story in development if story not in validation)
            columns, values, phase, gamma = inner_attention(
                outer, training, validation, states, forms, families, metadata, slots, hits, totals,
            )
            if np.isfinite(oof[columns]).any():
                raise ValueError('A Frick calibration path appeared in multiple out-of-fold groups')
            oof[columns] = values
            records.append({'validation': validation, 'train': training, 'lambda': phase, 'gamma': gamma})
        keys = groups(development, families)
        columns = mass.indices_for(keys, slots, len(hits))
        if np.count_nonzero(np.isfinite(oof)) != len(columns):
            raise ValueError('Frick cross-fit calibration source paths are incomplete')
        design = np.vstack([peak_features(oof[slots[key]]) for key in keys])
        weights, positives = training_weights(keys, columns, metadata, slots, hits, totals)

        # Replay, rather than refit, the exact previously sealed outer base model.
        outer_keys = groups((outer,), families)
        safe_metadata = {key: metadata[key] for key in keys}
        outer_fit, test_columns = conditional.make_fold(outer, development, (outer,), states, forms, families,
                                                        safe_metadata, slots, hits, totals, inner_validation=False)
        stored = stored_fits[outer]
        if (stored['inner_validation_stories'] != [list(group) for group in split_four(development)]
                or stored['fitted']['train_at_risk_decisions'] != int(outer_fit.totals.sum())):
            raise ValueError('Saved outer attention fit used a different Frick fold')
        aligned = base.predict_logits(outer_fit.validate, stored['fitted'])
        error = float(np.max(np.abs(aligned - attention[test_columns])))
        if error > 1e-12:
            raise ValueError('The historical attention coefficients do not replay their causal forecasts')
        largest_replay_error = max(largest_replay_error, error)
        folded_fits = {}
        for i, method in enumerate(METHODS[:2]):
            parameters, loss, steps = fit_calibration(design, weights, positives, allow_peak=method == 'peak')
            for key in outer_keys:
                slot = slots[key]
                path = attention[slot]
                full = peak_features(path) @ parameters
                predicted[i, slot] = full
                for visible in range(1, len(full)):
                    short = peak_features(path[:visible]) @ parameters
                    largest_prefix_error = max(largest_prefix_error, float(np.max(np.abs(short - full[:visible]))))
                    compared_prefixes += 1
            folded_fits[method] = {'coefficients': parameters.tolist(), 'objective': loss, 'iterations': steps}
        receipts.append({'held_out_story': outer, 'calibration_groups': records, 'readouts': folded_fits,
                         'train_versions': len(keys), 'train_positions': len(columns)})
    if not np.isfinite(predicted).all() or largest_prefix_error > 1e-12:
        raise ValueError('A calibrated logit is missing or changes with unread suffix removal')
    return predicted, receipts, {
        'old_attention_replay_max_logit_change': largest_replay_error,
        'suffix_invariance_max_logit_change': largest_prefix_error,
        'suffix_invariance_path_prefix_checks': compared_prefixes,
        'excluded_calibration_and_outer_family_labels_masked_before_base_fit': True,
    }


def named_evaluation(logits, names, slots, families, metadata, trials, readers):
    values = conditional.evaluate(logits, slots, families, metadata, trials, readers)
    primary = names[0]
    values['methods'] = {name: values['methods'][old] for name, old in zip(names, conditional.METHODS, strict=True)}
    values['comparisons'] = {
        name: {key.replace('conditional_', f'{primary}_', 1): value for key, value in values['comparisons'][old].items()}
        for name, old in zip(names[1:], conditional.METHODS[1:], strict=True)
    }
    # The canonical helper's historical gate names depend on its fixed slot labels.
    values.pop('gate_results')
    values.pop('pass')
    return values


def evaluate(logits, slots, families, metadata, trials, readers, hits, totals):
    score = named_evaluation(logits[:4], METHODS[:4], slots, families, metadata, trials, readers)
    legacy_names = ('peak', 'affine', 'global', 'original_contextual')
    legacy = named_evaluation(logits[[0, 1, 4, 5]], legacy_names, slots, families, metadata, trials, readers)
    for name in legacy_names[2:]:
        score['methods'][name] = legacy['methods'][name]
        score['comparisons'][name] = legacy['comparisons'][name]
    affine = named_evaluation(logits[[1, 0, 2, 3]], ('affine', 'peak', *METHODS[2:4]),
                              slots, families, metadata, trials, readers)
    score['affine_comparisons'] = affine['comparisons']
    score['target_at_risk_probability'] = {}
    for condition in base.CONDITIONS:
        keys = [pair[condition] for pair in families.values()]
        indices = np.array([slots[key].start + metadata[key]['target_index'] - 1 for key in keys])
        human = hits[indices] / totals[indices]
        estimates = expit(logits[:, indices]).mean(axis=1)
        score['target_at_risk_probability'][condition] = {'human': float(human.mean()),
                                                         **dict(zip(METHODS, estimates.tolist(), strict=True))}

    def interval(method, endpoint):
        return score['comparisons'][method][f'peak_{endpoint}']['crossed_story_reader_ci95']

    def equivalent(summary):
        low, high = summary['crossed_story_reader_ci95']
        return -base.MARGIN < low and high < base.MARGIN

    peak = score['methods']['peak']
    score['primary_gates'] = {
        **{f'amplitude_error_reduced_vs_{name}': interval(name, 'target_absolute_error_reduction')[0] > 0
           for name in ('attention_only', 'affine')},
        **{f'nll_noninferior_to_{name}': interval(name, 'minus_comparator_nll')[1] < mass.MARGIN_NLL
           for name in ('attention_only', 'seven_feature_conditional')},
        **{f'pretarget_equivalent_{condition}': equivalent(
            peak['first_d_by_condition'][condition]['prediction_minus_human']['before_target'],
        ) for condition in base.CONDITIONS},
    }
    score['historical_gates'] = {
        'target_effect_positive': peak['target_effect']['crossed_story_reader_ci95'][0] > 0,
        'target_effect_equivalent': equivalent(peak['target_model_minus_human']),
        **{f'nll_noninferior_to_{name}': interval(name, 'minus_comparator_nll')[1] < mass.MARGIN_NLL
           for name in ('global', 'original_contextual')},
    }
    score['primary_pass'] = all(score['primary_gates'].values())
    return score


def run(assay, archived_extraction, archived_packet, signals_dir, signal_stimuli, signal_assay, signal_preflight,
        tokenizer, reference, output):
    if output.exists():
        raise FileExistsError(output)
    if base.digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('The registered Frick peak calibration protocol changed')
    audit, stimuli, metadata, families, texts = base.load_layout(assay)
    raw_states, provenance = base.load_states(assay, archived_extraction, archived_packet, base.PROTOCOL,
                                              stimuli, metadata, families)
    features, _, signal_provenance = signals.load_signal_features(
        signals_dir, signal_stimuli, signal_assay, signal_preflight, tokenizer,
        stimuli, metadata, raw_states, provenance,
    )
    keys = sorted(metadata)
    states = {key: paired.normalize_states(raw_states[key]) for key in keys}
    forms = {key: np.column_stack((paired.sentence_forms(texts[key]), features[key][:, signals.ARMS['attention_only']]))
             for key in keys}
    ends = np.r_[0, np.cumsum([len(forms[key]) for key in keys])]
    slots = {key: slice(start, stop) for key, start, stop in zip(keys, ends[:-1], ends[1:], strict=True)}
    frozen, stored_fits, reference_digest = frozen_base(reference, provenance, signal_provenance, slots, stimuli)
    trials, readers, hits, totals, inventory = base.load_outcomes(assay, audit, metadata, slots)
    new, fits, checks = fit_folds(states, forms, families, metadata, slots, hits[1], totals[1], stored_fits, frozen[0])
    logits = np.vstack((new, frozen))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.frick-peak-') as temporary:
        stage = Path(temporary) / 'scored'
        stage.mkdir()
        forecasts = stage / 'forecasts.jsonl'
        with forecasts.open('x', encoding='utf-8') as stream:
            for key, slot in slots.items():
                for stop, row in zip(stimuli[key]['stops'][1:], logits[:, slot].T, strict=True):
                    forecast = {'id': key, 'stop': stop, 'first_logits': dict(zip(METHODS, row.tolist(), strict=True))}
                    stream.write(json.dumps(forecast, sort_keys=True, allow_nan=False) + '\n')
        base.write_json(stage / 'fits.json', {'ridge': RIDGE, 'target_auxiliary_weight': 1.0, 'folds': fits})
        base.write_json(stage / 'verification.json', checks)
        seal_path = stage / 'forecast-seal.json'
        base.write_json(seal_path, {
            'provenance': provenance, 'protocol_sha256': PROTOCOL_SHA256, 'scorer_sha256': base.digest(Path(__file__)),
            'signals_source_sha256': signal_provenance['signals_seal_sha256'],
            'reference_forecast_seal_sha256': reference_digest,
            'file_sha256': {name: base.digest(stage / name) for name in ('fits.json', 'forecasts.jsonl', 'verification.json')},
        })
        replay = np.asarray([[row['first_logits'][method] for method in METHODS] for row in base.read_rows(forecasts)]).T
        if not np.array_equal(replay, logits):
            raise ValueError('Forecast seal serialization differs from the model probabilities')
        result = evaluate(replay, slots, families, metadata, trials, readers, hits[1], totals[1])
        base.write_json(stage / 'results.json', {
            **result, 'inventory': inventory, 'retrospective': True, 'forecast_seal_sha256': base.digest(seal_path),
        })
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('assay', 'archived-extraction', 'archived-packet', 'signals-dir', 'signal-stimuli', 'signal-assay',
                  'signal-preflight', 'tokenizer', 'output'):
        parser.add_argument(f'--{field}', required=True, type=Path)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    run(**vars(parser.parse_args()))
