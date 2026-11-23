"""Seal and launch the joint weighting comparison on an owned L4."""

import argparse
from pathlib import Path

from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest, read_json
from research.token_readout.colab import pack_job
from research.token_readout.transfer_colab import CODE as TOKEN_CODE


PROTOCOL = Path(__file__).with_name('protocol.md')
RETENTION = Path(__file__).with_name('retention.md')
RETENTION_CONTROL = Path(__file__).with_name('retention-control.md')
CODE = (*TOKEN_CODE, 'research/adaptive_joint/fit.py', 'research/adaptive_joint/policy.py',
        'research/union_training/fit_observations.py', 'research/union_training/observation_registry.py',
        'research/union_training/observation_losses.py')


def prepare(data, output, guarded=False, guarded_fixed=False):
    receipt = read_json(data / 'receipt.json')
    if (receipt['candidate'] != 'qwen' or receipt['formal_labels_in_worker'] is not False
            or receipt['positions'] != 164220 or receipt['observations'] != 62 or receipt['sources'] != 24):
        raise ValueError('Expected the compact full 24-source training packet with no formal observations')
    members = {name: data / name for name in receipt['files_sha256']}
    for name, path in members.items():
        if not path.is_file() or digest(path) != receipt['files_sha256'][name]:
            raise ValueError(f'Altered or missing joint training member: {name}')
    settings = {
        'seeds': [1729, 2718], 'modes': ['fixed', 'adaptive'], 'epochs': 28, 'min_epochs': 10,
        'min_dev_gain': 0.001, 'patience': 7, 'lr': 0.001, 'weight_decay': 0.015,
        'observation_batch': 256, 'token_batch': 3, 'microbatches_per_step': 4,
        'adaptive': {'warmup': 4, 'decay': 0.85, 'lower': 0.35, 'upper': 2.5},
    }
    if guarded or guarded_fixed:
        settings['modes'] = ['guarded-fixed' if guarded_fixed else 'guarded']
        settings['retention'] = {'frick': 0.05, 'evidence': 0.12, 'radius': 0.10}
    protocol = RETENTION_CONTROL if guarded_fixed else RETENTION if guarded else PROTOCOL
    purpose = 'fixed-retention-qwen' if guarded_fixed else 'adaptive-retention-qwen' if guarded else 'adaptive-joint-qwen'
    pack_job(output, protocol, purpose, 'adaptive-joint', CODE,
             {**members, 'receipt.json': data / 'receipt.json'}, {'settings.json': settings},
             3000, {'dataset_receipt_sha256': digest(data / 'receipt.json'),
                    'formal_holdout_sha256': receipt['formal_holdout_sha256']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    setup = actions.add_parser('prepare')
    setup.add_argument('--data', required=True, type=Path)
    setup.add_argument('--output', required=True, type=Path)
    training = actions.add_parser('run')
    training.add_argument('job', type=Path)
    for action in (setup, training):
        mode = action.add_mutually_exclusive_group()
        mode.add_argument('--guarded', action='store_true')
        mode.add_argument('--guarded-fixed', action='store_true')
    options = vars(parser.parse_args())
    if options.pop('action') == 'prepare':
        prepare(**options)
    else:
        guarded, guarded_fixed = options.pop('guarded'), options.pop('guarded_fixed')
        protocol = RETENTION_CONTROL if guarded_fixed else RETENTION if guarded else PROTOCOL
        run(**options, protocol=protocol)
