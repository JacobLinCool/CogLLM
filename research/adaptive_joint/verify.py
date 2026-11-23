"""Verify the sealed comparison and summarize matched training trajectories."""

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from statistics import mean, median

from research.backbone_transfer.extract import digest, read_json, write_json
from research.union_training.verify_observations import verified_member


def at_quality(history, limit):
    return next(({'epoch': row['epoch'], 'seconds': row['elapsed_seconds']}
                 for row in history if row['source_balanced_nll'] <= limit), None)


def profile(fit):
    history = fit['history']
    selected = history[fit['selected_epoch'] - 1]
    state = defaultdict(lambda: {'base_mass': 0.0, 'weighted_mass': 0.0, 'microbatches': 0})
    for row in fit['adaptive_policy']['objectives']:
        source = state[row['source']]
        source['base_mass'] += row['base_mass']
        source['weighted_mass'] += row['weighted_mass']
        source['microbatches'] += row['draws']
    total_base = sum(row['base_mass'] for row in state.values())
    total_weighted = sum(row['weighted_mass'] for row in state.values())
    if abs(total_base - total_weighted) > max(1e-8, 1e-10 * total_base):
        raise ValueError('The policy did not preserve total training weight')
    for row in state.values():
        row['base_weight_ratio'] = row['weighted_mass'] / row['base_mass']
    return {'epochs': len(history), 'stop': fit['stop_reason'], 'selected_epoch': fit['selected_epoch'],
            'best_dev_nll': selected['source_balanced_nll'], 'training_seconds_to_best': selected['elapsed_seconds'],
            'training_seconds_to_stop': history[-1]['elapsed_seconds'], 'total_fit_seconds': fit['seconds'],
            'updates': sum(row['updates'] for row in history),
            'ordinary_backward_calls': sum(row['ordinary_backward_calls'] for row in history),
            'gradient_clipped_fraction': sum(row['updates'] * row['clipped_fraction'] for row in history)
                                        / sum(row['updates'] for row in history),
            'median_epoch_p95_gradient_norm': median(row['p95_gradient_norm'] for row in history),
            'source_weight_range': [min(row['source_weight_range'][0] for row in history),
                                    max(row['source_weight_range'][1] for row in history)],
            'source_exposure': dict(sorted(state.items())), 'total_base_mass': total_base,
            'total_weighted_mass': total_weighted, 'params': fit['trainable_parameters'],
            'shared_export_max_abs': fit['forward_checkpoint_fold_max_abs'],
            'token_dev_at_selected': selected['token_dev'],
            **({'retention': fit['retention'], 'selected_epoch_adapter_projection_fraction':
                selected['adapter_projection_fraction']} if 'retention' in fit else {})}


def compare(fixed, adaptive, fixed_profile, adaptive_profile):
    last = min(len(fixed['history']), len(adaptive['history']))
    f, a = (value['history'][last - 1] for value in (fixed, adaptive))
    if f['source_microbatches'] != a['source_microbatches'] or f['updates'] != a['updates']:
        raise ValueError('Compared methods did not see the same exposures at the matched step')
    base = fixed['history'][fixed['selected_epoch'] - 1]['by_source_dev']
    changed = adaptive['history'][adaptive['selected_epoch'] - 1]['by_source_dev']
    if base.keys() != changed.keys():
        raise ValueError('Source inventories differ across the paired methods')
    threshold = fixed_profile['best_dev_nll'] + 0.001
    return {'matched_epochs': last, 'matched_updates': sum(row['updates'] for row in fixed['history'][:last]),
            'matched_dev_nll_delta': a['source_balanced_nll'] - f['source_balanced_nll'],
            'matched_time_delta_seconds': a['elapsed_seconds'] - f['elapsed_seconds'],
            'best_dev_nll_delta': adaptive_profile['best_dev_nll'] - fixed_profile['best_dev_nll'],
            'time_to_fixed_quality': {variant: at_quality(data['history'], threshold)
                                      for variant, data in (('fixed', fixed), ('adaptive', adaptive))},
            'fixed_quality_threshold': threshold,
            'per_source_selected_checkpoint_nll_delta': {name: changed[name] - base[name] for name in base},
            'source_nll_improved': sum(changed[name] < base[name] - 1e-8 for name in base),
            'source_nll_worse': sum(changed[name] > base[name] + 1e-8 for name in base)}


def collect(job, inputs, output):
    manifest = read_json(job / 'manifest.json')
    root = job / 'extraction'
    seal = read_json(root / 'seal.json')
    receipt = read_json(inputs / 'receipt.json')
    if (manifest['phase'] != 'finished' or not manifest['release_verified'] or not manifest['verified_outputs']
            or seal['protocol_sha256'] != manifest['protocol_sha256']
            or seal['input_receipt_sha256'] != manifest['dataset_receipt_sha256']
            or seal['input_receipt_sha256'] != digest(inputs / 'receipt.json')
            or seal['seed_sha256'] != receipt['seed_sha256'] or seal['formal_labels_in_worker']
            or len(seal['registered_observations']) != 62):
        raise ValueError('Collected comparison or release evidence does not match the sealed experiment')
    files = {name: verified_member(root, Path(name), expected) for name, expected in seal['files_sha256'].items()}
    output.mkdir(parents=True, exist_ok=True)
    for name, path in {**files, 'seal.json': root / 'seal.json', 'input-receipt.json': inputs / 'receipt.json'}.items():
        target = output / name
        target.parent.mkdir(exist_ok=True)
        copyfile(path, target)
    seeds = seal['seed_settings']['seeds']
    modes = seal['seed_settings']['modes']
    arms, fits = {}, {}
    for trial_seed in seeds:
        for name in modes:
            arm = f'{name}-{trial_seed}'
            fit = read_json(output / arm / 'fitting.json')
            if (fit['blocks'] != 65 or len(fit['source_names']) != 26
                    or fit['observation_training_rows'] != receipt['legacy_train_rows']):
                raise ValueError('An arm omitted one or more original observations or token sources')
            arms[arm] = profile(fit)
            fits[arm] = fit
    paired = ({str(seed): compare(fits[f'fixed-{seed}'], fits[f'adaptive-{seed}'],
                                  arms[f'fixed-{seed}'], arms[f'adaptive-{seed}']) for seed in seeds}
              if {'fixed', 'adaptive'} <= set(modes) else {})
    fields = ('run_id', 'candidate', 'model', 'revision', 'requested_gpu', 'cli_version', 'timeline', 'dataset_receipt_sha256',
              'formal_holdout_sha256', 'verified_outputs', 'release_verified', 'release_usage_recorded')
    safe_receipt = {key: manifest[key] for key in fields}
    times = manifest['timeline']
    safe_receipt['allocated_through_finished_seconds'] = (
        datetime.fromisoformat(times['finished']) - datetime.fromisoformat(times['allocated'])).total_seconds()
    write_json(output / 'run-verification.json', safe_receipt)
    write_json(output / 'training-comparison.json', {
        'run_seal_sha256': digest(output / 'seal.json'), 'arms': arms, 'paired': paired,
        'mean_best_dev_nll_delta': mean(item['best_dev_nll_delta'] for item in paired.values()) if paired else None,
        'mean_matched_seconds_delta': mean(item['matched_time_delta_seconds'] for item in paired.values()) if paired else None,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('job', 'inputs', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    collect(**vars(parser.parse_args()))
