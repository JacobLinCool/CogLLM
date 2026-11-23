"""Select effect supervision within AmbiStory train and seal forecasts for dev."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

if __name__ == '__main__':
    for variable in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = '1'

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax

from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3.run import save_json
from research.stage6 import ambistory_fit as baseline


DESIGN = Path(__file__).with_name('ambistory-effect-supervision.md')
BASELINE_PROTOCOL = DESIGN.parents[1] / 'results' / 'ambistory-fit-protocol-v2.json'
FOLDS = 5
FOLD_SALT = 'cogllm-ambistory-effect-v1'
BETAS = (0, 1, 4, 16, 64)
LEVELS = np.array(baseline.GRADES)
SPAN = int(LEVELS[-1] - LEVELS[0])
VOTE_NLL_MARGIN = 0.01
AMPLITUDE_MARGIN = 0.5
TIE_ATOL = 1e-12
RUNTIME = {'numpy': '2.4.6', 'scipy': '1.16.3'}
SOURCE_MEMBERS = frozenset({f'{split}/{name}' for split in baseline.SPLITS
                            for name in (baseline.STIMULI, baseline.DESIGN)} | {f'train/{baseline.OUTCOMES}'})
STAGED_MEMBERS = frozenset({'preparation.json', *(f'prepared/{name}' for name in SOURCE_MEMBERS)})


def protocol():
    reference = baseline.read_json(BASELINE_PROTOCOL)
    if reference != baseline.protocol():
        raise ValueError('The original v2 fit code or design differs from its sealed protocol')
    return {
        'status': 'follow_up_prespecified_before_training_selection_and_dev_scoring',
        'fit_code_sha256': digest(Path(__file__)), 'design_sha256': digest(DESIGN),
        'preparation_sha256': baseline.PREPARATION_SHA256, 'baseline_fit_protocol_sha256': digest(BASELINE_PROTOCOL),
        'fit_split': 'train', 'forecast_split': 'dev', 'fit_families': 378, 'forecast_families': 98,
        'forecast_items': 588, 'held_out_votes_expected': 2952, 'held_out_flagged_votes_expected': 44,
        'staged_members': sorted(STAGED_MEMBERS),
        'folds': {'group': 'casefolded homonym', 'count': FOLDS, 'hash': 'SHA-256', 'salt': FOLD_SALT,
                  'allocation': 'rank by digest of UTF-8 salt/homonym, tie by homonym, then rank modulo fold count'},
        'features': 'identical to v2, with form scaling fit within each original-training fit partition',
        'objective': {'ordinal': 'v2 summed per-item histogram cross-entropy and intercept-exempt L2',
                      'effect': 'beta * N / (2 * M) * sum(((predicted signed difference - observed signed difference) / 4)^2)',
                      'relative_pair_order': 'later - earlier by opaque ending ID for the same family and sense',
                      'grading_span': SPAN, 'beta_grid': list(BETAS), 'total_scaling': 'summed over N rated items'},
        'solver': {**{key: reference['objective'][key] for key in ('form_maxiter', 'context_maxiter', 'ftol', 'gtol')},
                   'method': 'L-BFGS-B', 'initialization': 'zero for beta=0, that fold beta=0 optimum for beta>0',
                   'numpy_scipy': RUNTIME, 'thread_count': 1},
        'selection': {'score': 'lowest pooled out-of-fold signed-update MAE, tie by smallest beta',
                      'eligibility': 'positive beta converges in all folds and strictly improves MAE versus beta=0',
                      'maximum_nll_increase_nats_per_vote': VOTE_NLL_MARGIN,
                      'empty_selection': 'seal_no_candidate_without_forecasts'},
        'dev_gates': {
            'effect_ending_minus_effect_prefix': 'paired 95% CI upper bounds below zero for NLL and update MAE',
            'effect_ending_minus_baseline_ending': 'paired 95% CI upper below zero for update MAE and below NLL margin',
            'nll_margin_nats_per_vote': VOTE_NLL_MARGIN,
            'effect_amplitude_minus_human': 'paired 90% CI entirely inside the open equivalence bounds',
            'equivalence_margin_grades': AMPLITUDE_MARGIN,
            'nll_subset': 'all variants, per vote', 'bootstrap': {'unit': 'casefolded homonym', 'draws': 10000, 'seed': 43},
        },
        'forecasts': {'files': ['baseline-predictions.jsonl', 'effect-predictions.jsonl'],
                      'fields': ['id', 'sense', *baseline.METHODS], 'grades': list(baseline.GRADES)},
    }


def check_preparation(path):
    if digest(path) != baseline.PREPARATION_SHA256:
        raise ValueError('Unexpected AmbiStory preparation edition')
    receipt = baseline.read_json(path)
    train, dev = (receipt['splits'][split] for split in ('train', 'dev'))
    if (train['story_families'] != 378 or dev['story_families'] != 98 or dev['sense_samples'] != 588
            or dev['votes'] != 2952 or dev['nonsensical_votes'] != 44):
        raise ValueError('Unexpected original train or dev inventory')
    return receipt


def stage_inputs(prepared, preparation, output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    receipt = check_preparation(preparation)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.ambistory-effect-input-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'input'
        stage.mkdir(mode=0o700)
        for name in sorted(SOURCE_MEMBERS):
            source = (prepared / name).read_bytes()
            if sha256(source).hexdigest() != receipt['files_sha256'][name]:
                raise ValueError(f'Source data differs from its preparation receipt: {name}')
            target = stage / 'prepared' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source)
            target.chmod(0o600)
        source = preparation.read_bytes()
        if sha256(source).hexdigest() != baseline.PREPARATION_SHA256:
            raise ValueError('Preparation changed during staging')
        (stage / 'preparation.json').write_bytes(source)
        stage.rename(output)


def check_staged(staged):
    paths = list(staged.rglob('*'))
    if (any(path.is_symlink() for path in paths)
            or {str(path.relative_to(staged)) for path in paths if path.is_file()} != STAGED_MEMBERS):
        raise ValueError('Expected the staged text, design and original train ratings allowlist')
    return check_preparation(staged / 'preparation.json')


def fold_map(groups):
    groups = sorted(set(groups), key=lambda name: (sha256(f'{FOLD_SALT}/{name}'.encode()).digest(), name))
    if len(groups) < FOLDS:
        raise ValueError('Too few original-training homonyms for the frozen fold count')
    return {name: rank % FOLDS for rank, name in enumerate(groups)}


def train_pairs(inventory, keys):
    _, families = inventory
    row_index = {key: index for index, key in enumerate(keys)}
    groups = {row['id']: row['homonym'].casefold() for variants in families.values() for row in variants}
    pairs = []
    for variants in families.values():
        earlier, later = sorted((row for row in variants if row['has_ending']), key=lambda row: row['id'])
        for sense in earlier['senses']:
            pairs.append([row_index[row['id'], sense['id']] for row in (earlier, later)])
    groups_by_row = np.array([groups[story] for story, _ in keys])
    pair_rows = np.array(pairs, dtype=np.intp)
    if (len(row_index) != len(keys) or len(keys) != 3 * len(pair_rows)
            or len(np.unique(pair_rows)) != pair_rows.size
            or np.any(groups_by_row[pair_rows[:, 0]] != groups_by_row[pair_rows[:, 1]])):
        raise ValueError('Every signed pair must join unique versions of the same sense and homonym')
    return groups_by_row, pair_rows


def differences(values, pairs):
    return values[pairs[:, 1]] - values[pairs[:, 0]]


def loss_and_gradient(flat, matrix, observed, pairs, human, beta):
    weights = flat.reshape(matrix.shape[1], len(LEVELS))
    log_p = log_softmax(matrix @ weights, axis=-1)
    p = np.exp(log_p)
    residual = p - observed
    ordinal = float(-np.sum(observed * log_p))
    penalty = float((baseline.causal_ordinal.PENALTY / 2) * np.sum(weights[1:] ** 2))
    means = p @ LEVELS
    error = differences(means, pairs) - human
    multiplier = beta * len(matrix) / (len(pairs) * SPAN ** 2)
    update = float(multiplier * (error @ error) / 2)
    per_row = np.zeros(len(matrix))
    # Each ending row occurs once, with opposite derivatives at the two endpoints.
    per_row[pairs[:, 1]] = multiplier * error
    per_row[pairs[:, 0]] = -multiplier * error
    residual += per_row[:, None] * p * (LEVELS - means[:, None])
    gradient = matrix.T @ residual
    gradient[1:] += baseline.causal_ordinal.PENALTY * weights[1:]
    return (ordinal + penalty + update, gradient.ravel()), (ordinal, penalty, update)


def fit_effect(matrix, observed, pairs, initial, beta):
    human = differences(observed @ LEVELS, pairs)

    def loss(flat):
        return loss_and_gradient(flat, matrix, observed, pairs, human, beta)[0]

    start = perf_counter()
    result = minimize(loss, initial.ravel(), method='L-BFGS-B', jac=True,
                      options={'maxiter': baseline.CONTEXT_MAX_ITERATIONS, 'ftol': 1e-12, 'gtol': 1e-7})
    finite = np.isfinite(result.fun) and np.isfinite(result.x).all() and np.isfinite(result.jac).all()
    report = {'success': bool(result.success and finite), 'iterations': int(result.nit),
              'evaluations': int(result.nfev), 'elapsed_seconds': perf_counter() - start, 'message': str(result.message)}
    if not report['success']:
        return None, report
    _, terms = loss_and_gradient(result.x, matrix, observed, pairs, human, beta)
    report.update(objective=float(result.fun), max_abs_gradient=float(np.max(np.abs(result.jac))),
                  loss_terms=dict(zip(('ordinal', 'l2', 'effect'), terms, strict=True)))
    return result.x.reshape(initial.shape), report


def fit_reference(matrix, observed, pairs):
    start = perf_counter()
    weights, report = baseline.fit_context(matrix, observed)
    elapsed = perf_counter() - start
    human = differences(observed @ LEVELS, pairs)
    _, terms = loss_and_gradient(weights.ravel(), matrix, observed, pairs, human, 0)
    report.update(success=True, elapsed_seconds=elapsed,
                  loss_terms=dict(zip(('ordinal', 'l2', 'effect'), terms, strict=True)))
    return weights, report


def vote_metrics(log_p, observed, counts, pairs):
    human = differences(observed @ LEVELS, pairs)
    model = differences(np.exp(log_p) @ LEVELS, pairs)
    direction = np.sign(np.where(np.abs(human) <= TIE_ATOL, 0, human))
    predicted_direction = np.sign(np.where(np.abs(model) <= TIE_ATOL, 0, model))
    return {'nll_nats_per_vote': float(-np.sum(counts[:, None] * observed * log_p) / counts.sum()),
            'signed_update_mae_grades': float(np.mean(np.abs(model - human))),
            'direction_agreement': float(np.mean((1 + direction * predicted_direction) / 2)),
            'human_mean_absolute_update_grades': float(np.mean(np.abs(human))),
            'model_mean_absolute_update_grades': float(np.mean(np.abs(model))),
            'mean_absolute_update_gap_grades': float(np.mean(np.abs(model) - np.abs(human)))}


def context_design(assembled, center, scale):
    _, forms, senses, _, endings, _ = assembled
    return baseline.with_context(baseline.base_design(forms, senses, center, scale), senses, endings)


def select_beta(train, groups, pairs, observed, counts):
    assignments = fold_map(groups)
    row_folds = np.array([assignments[key] for key in groups])
    oof = {beta: np.full(observed.shape, np.nan) for beta in BETAS}
    diagnostics = {'fold_homonyms': assignments, 'folds': []}
    for fold in range(FOLDS):
        fit = row_folds != fold
        center, scale = baseline._length_standardization(train[1][fit])
        design = context_design(train, center, scale)
        fit_rows = np.flatnonzero(fit)
        train_pairs = np.searchsorted(fit_rows, pairs[fit[pairs[:, 0]]])
        if len(fit_rows) != 3 * len(train_pairs):
            raise ValueError('Training fold split a family or variant pair')
        x, y = design[fit], observed[fit]
        initial, optimizer = fit_reference(x, y, train_pairs)
        fold_info = {'fold': fold, 'fit_items': len(x), 'fit_ending_pairs': len(train_pairs),
                     'held_out_items': int((~fit).sum()), 'optimizer': {0: optimizer}}
        diagnostics['folds'].append(fold_info)
        oof[0][~fit] = log_softmax(design[~fit] @ initial, axis=-1)
        for beta in BETAS[1:]:
            weights, report = fit_effect(x, y, train_pairs, initial, beta)
            fold_info['optimizer'][beta] = report
            if weights is not None:
                oof[beta][~fit] = log_softmax(design[~fit] @ weights, axis=-1)
    metrics = {beta: vote_metrics(log_p, observed, counts, pairs) for beta, log_p in oof.items()
               if np.isfinite(log_p).all()}
    diagnostics['oof'] = metrics
    reference = metrics[0]
    candidates = [beta for beta, result in metrics.items() if beta > 0
                  and result['nll_nats_per_vote'] - reference['nll_nats_per_vote'] <= VOTE_NLL_MARGIN
                  and result['signed_update_mae_grades'] < reference['signed_update_mae_grades']]
    diagnostics['eligible_betas'] = candidates
    if not candidates:
        return None, diagnostics
    beta = min(candidates, key=lambda value: (metrics[value]['signed_update_mae_grades'], value))
    diagnostics['selected_beta'] = beta
    return beta, diagnostics


def predict(assembled, form_weights, context_weights, center, scale):
    keys, forms, senses, common, ends, suffixed = assembled
    base = baseline.base_design(forms, senses, center, scale)
    form = np.repeat(base[::3] @ form_weights, 3, axis=0)
    prefix = np.repeat(baseline.with_context(base[::3], senses[::3], common[::3]) @ context_weights, 3, axis=0)
    ending = prefix.copy()
    ending[suffixed] = baseline.with_context(base[suffixed], senses[suffixed], ends[suffixed]) @ context_weights
    p = np.exp(log_softmax(np.stack((form, prefix, ending), axis=1), axis=-1))
    if not (np.isfinite(p).all() and (p > 0).all() and np.allclose(p.sum(axis=-1), 1, atol=1e-12, rtol=0)):
        raise ValueError('Expected finite, strictly positive five-grade forecasts')
    return dict(zip(keys, p, strict=True))


def write_forecasts(path, forecasts):
    write_jsonl(path, ({'id': story, 'sense': sense, **dict(zip(baseline.METHODS, values.tolist(), strict=True))}
                       for (story, sense), values in sorted(forecasts.items())))


def fit_heads(train, observed, pairs, counts, beta, diagnostics):
    center, scale = baseline._length_standardization(train[1])
    form = baseline.base_design(train[1], train[2], center, scale)
    form_weights = baseline.causal_ordinal.fit_softmax(form, observed)
    x = context_design(train, center, scale)
    base_weights, base_optimizer = fit_reference(x, observed, pairs)
    effect_weights, effect_optimizer = fit_effect(x, observed, pairs, base_weights, beta)
    if effect_weights is None:
        raise ArithmeticError(f'Full training refit failed at beta={beta}: {effect_optimizer}')
    diagnostics['full_training'] = {
        'baseline_optimizer': base_optimizer, 'effect_optimizer': effect_optimizer, 'items': len(x),
        'ending_pairs': len(pairs), 'votes': int(counts.sum()),
        'baseline_metrics': vote_metrics(log_softmax(x @ base_weights, axis=-1), observed, counts, pairs),
        'effect_metrics': vote_metrics(log_softmax(x @ effect_weights, axis=-1), observed, counts, pairs),
    }
    return {'form_weights': form_weights, 'baseline_context_weights': base_weights,
            'effect_context_weights': effect_weights, 'form_center': center, 'form_scale': scale}


def record_forecasts(stage, inventory, stories, meanings, heads):
    dev = baseline.assemble(inventory, stories, meanings)
    for name in ('baseline', 'effect'):
        forecasts = predict(dev, heads['form_weights'], heads[f'{name}_context_weights'],
                            heads['form_center'], heads['form_scale'])
        baseline.validate_invariance(inventory, forecasts)
        write_forecasts(stage / f'{name}-predictions.jsonl', forecasts)
    np.savez_compressed(stage / 'heads.npz', **heads)
    return len(dev[0])


def fit(staged, extraction, protocol_path, output):
    if output.exists():
        raise FileExistsError(output)
    if baseline.read_json(protocol_path) != protocol() or {name: version(name) for name in RUNTIME} != RUNTIME:
        raise ValueError('Effect-supervision preseal or numeric runtime differs from the frozen edition')
    receipt = check_staged(staged)
    prepared = staged / 'prepared'
    inventories = baseline.read_stories(prepared, receipt)
    stories, meanings, input_hashes, attested = baseline.read_features(
        extraction, staged / 'preparation.json', receipt, inventories)
    benchmark = baseline.compare_prefixes(inventories, stories)
    replay = baseline.compare_prefixes(inventories, stories, extraction_reference=True)
    if (attested['families'] != sum(row['families'] for row in replay.values())
            or attested['comparisons'] != sum(row['comparisons'] for row in replay.values())
            or any(not np.isclose(attested[field], max(row[field] for row in replay.values()), atol=1e-6, rtol=0)
                   for field in ('max_relative_rms', 'max_absolute'))):
        raise ValueError('Independent shared-prefix replay differs from its extraction seal')
    train = baseline.assemble(inventories['train'], stories, meanings)
    groups, pairs = train_pairs(inventories['train'], train[0])
    observed = baseline.read_train_outcomes(prepared, receipt, train[0])
    count_by_item = {(row['id'], row['sense_id']): len(row['ratings'])
                     for row in baseline.locked_rows(prepared, receipt, 'train', baseline.OUTCOMES)}
    counts = np.array([count_by_item[key] for key in train[0]])
    beta, diagnostics = select_beta(train, groups, pairs, observed, counts)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.ambistory-effect-fit-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'fit'
        stage.mkdir(mode=0o700)
        decision = {'status': 'no_candidate'}
        if beta is not None:
            heads = fit_heads(train, observed, pairs, counts, beta, diagnostics)
            size = record_forecasts(stage, inventories['dev'], stories, meanings, heads)
            decision = {'status': 'effect_and_baseline_forecasts_locked_before_dev_scoring',
                        'selected_beta': beta, 'forecast_samples': size}
        if (any(digest(extraction / name) != input_hashes[key] for key, name in baseline.FILES.items())
                or any(digest(prepared / name) != receipt['files_sha256'][name] for name in SOURCE_MEMBERS)
                or check_staged(staged) != receipt or baseline.read_json(protocol_path) != protocol()):
            raise ValueError('Source bytes changed during effect supervision')
        save_json(stage / 'train-diagnostics.json', diagnostics)
        save_json(stage / 'fit-seal.json', {
            **decision,
            'protocol_sha256': digest(protocol_path), 'source_preparation_sha256': digest(staged / 'preparation.json'),
            'input_sha256': {name: receipt['files_sha256'][name] for name in sorted(SOURCE_MEMBERS)},
            'extraction_sha256': input_hashes, 'runtime': RUNTIME, 'prefix_benchmark': benchmark,
            'fit_split': 'train', 'forecast_split': 'dev', 'fit_samples': len(train[0]),
            'methods': list(baseline.METHODS), 'grades': list(baseline.GRADES),
            'file_sha256': {path.name: digest(path) for path in sorted(stage.iterdir())},
        })
        stage.rename(output)


def preseal(output):
    with output.open('x', encoding='utf-8') as target:
        json.dump(protocol(), target, indent=2, allow_nan=False)
        target.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, paths in (('preseal', ('output',)), ('stage-inputs', ('prepared', 'preparation', 'output')),
                        ('fit', ('staged', 'extraction', 'protocol', 'output'))):
        command = commands.add_parser(name)
        for field in paths:
            command.add_argument(f'--{field}', type=Path, required=True)
    args = vars(parser.parse_args())
    subcommand = args.pop('command')
    if subcommand == 'fit':
        args['protocol_path'] = args.pop('protocol')
    {'preseal': preseal, 'stage-inputs': stage_inputs, 'fit': fit}[subcommand](**args)


if __name__ == '__main__':
    main()
