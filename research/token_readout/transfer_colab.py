"""Prepare sealed portable readout jobs with local-only final observations."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from research.backbone_transfer.colab import CANDIDATES, MODEL_SOURCE, run
from research.backbone_transfer.extract import digest, read_json, stream_rows, write_rows
from research.token_readout.colab import COMMON_CODE, UI_CODE, pack_job


PROTOCOL = Path(__file__).with_name('transfer-protocol.md')
LINKED_PROTOCOL = Path(__file__).with_name('transfer-iirc-protocol.md')
PROTOCOLS = {'base': PROTOCOL, 'linked': LINKED_PROTOCOL}
CODE = (*COMMON_CODE, 'research/token_readout/backbone.py', 'research/token_readout/readout.py',
        'research/token_readout/transfer_fit.py')
PORTABLE_UI_CODE = (*UI_CODE, *(path for path in CODE if path not in COMMON_CODE),
                    'research/token_readout/portable_runner.py', 'research/token_readout/portable_live.py')


def prepare(dataset, output, candidate, experiment='base', arm_seconds=420, epochs=38):
    receipt = read_json(dataset / 'receipt.json')
    for name, expected in receipt['file_sha256'].items():
        if digest(dataset / name) != expected:
            raise ValueError(f'Transfer data changed after certification: {name}')
    spec = read_json(MODEL_SOURCE / 'models.json')['models'][candidate]
    if receipt['tokenizer_sha256'][candidate] != spec['sha256']['tokenizer.json']:
        raise ValueError('Tokenizer geometry belongs to another backbone snapshot')
    paths = {'stimuli.jsonl': dataset / 'stimuli.jsonl', 'learning.json': dataset / f'{candidate}-learn.json',
             'layout.json': dataset / f'{candidate}.json'}
    layout = read_json(paths['layout.json'])['rows']
    learning = read_json(paths['learning.json'])
    ids = {row['id'] for sets in learning.values() for rows in sets.values() for row in rows}
    if (set(learning) != {'train', 'dev'} or len(layout) != receipt['documents']
            or ids != {identity for identity, row in layout.items() if row['split'] in learning}
            or any(layout[row['id']]['split'] != split for split, sets in learning.items()
                   for rows in sets.values() for row in rows)):
        raise ValueError('Remote learning payload differs from the sealed train/dev split')
    if bool(learning['train'].get('linked')) != (experiment == 'linked'):
        raise ValueError('The sealed study must match the presence of IIRC training observations')
    for row in layout.values():
        labels = {'event_counts', 'censor_count', 'window_tokens', 'sham_tokens', 'cue_span'}
        if row.keys() & labels:
            raise ValueError('Final observation leaked into tokenizer layout')
    settings = {
        'seed': 1729, 'epochs': epochs, 'lr': 0.0007, 'weight_decay': 0.035,
        'minimum_epochs': 10, 'patience': 7, 'min_dev_gain': 0.0005, 'arm_seconds': arm_seconds,
        'technical_controls': [
            'An instrument loses pressure during each reading. The next section describes how calibration isolates the loss.',
            '這段開始介紹概念和實驗設定。後面再解釋如何辨識差異並回到先前的描述。',
        ],
    }
    purpose = f'portable-{candidate}' if experiment == 'base' else f'portable-linked-{candidate}'
    pack_job(output, PROTOCOLS[experiment], purpose, 'portable-token-readout', CODE, paths,
             {'settings.json': settings}, 2 * arm_seconds + 900,
             {'dataset_receipt_sha256': digest(dataset / 'receipt.json')}, candidate=candidate)


def prepare_ui(dataset, output, candidate, checkpoint, experiment='base'):
    seal_path = checkpoint.parent / 'seal.json'
    seal = read_json(seal_path)
    protocol = PROTOCOLS[experiment]
    if (seal['protocol_sha256'] != digest(protocol) or seal['candidate'] != candidate
            or seal['files_sha256'].get(checkpoint.name) != digest(checkpoint)):
        raise ValueError('HTTP verification requires a sealed checkpoint from this comparison')
    receipt = read_json(dataset / 'receipt.json')
    for name in ('stimuli.jsonl', 'design.json'):
        if digest(dataset / name) != receipt['file_sha256'][name]:
            raise ValueError('Preview source differs from the certified transfer dataset')
    group = read_json(dataset / 'design.json')['controls'][0]
    texts = {row['id']: row['text'] for row in stream_rows(dataset / 'stimuli.jsonl')}
    extended = texts[group['variants']['early']['id']]
    stops = receipt['structural_alignment']['control_shared_stops'][group['group_id']]
    prefix = extended[:stops['common_prefix_end']]
    if len(prefix) >= len(extended):
        raise ValueError('The preview requires a strict certified prefix')
    with TemporaryDirectory() as temporary:
        stimuli = Path(temporary) / 'stimuli.jsonl'
        write_rows(stimuli, [
            {'id': 'prefix', 'text': prefix},
            {'id': 'extended', 'text': extended},
            {'id': 'unicode', 'text': '本段說明校正流程。紀錄包含 e\u0301 與符號 🧪。\r\n\r\n後續說明校正時機。'},
        ])
        pack_job(output, protocol, f'portable-ui-{candidate}', 'portable-token-ui', PORTABLE_UI_CODE,
                 {'stimuli.jsonl': stimuli, 'checkpoint.pt': checkpoint}, {}, 900,
                 {'checkpoint_sha256': digest(checkpoint), 'fit_seal_sha256': digest(seal_path)}, candidate=candidate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    setup = actions.add_parser('prepare')
    for name in ('dataset', 'output'):
        setup.add_argument('--' + name, type=Path, required=True)
    setup.add_argument('--candidate', choices=CANDIDATES, required=True)
    setup.add_argument('--arm-seconds', type=int, default=420)
    setup.add_argument('--epochs', type=int, default=38)
    ui = actions.add_parser('prepare-ui')
    for name in ('dataset', 'output', 'checkpoint'):
        ui.add_argument('--' + name, type=Path, required=True)
    ui.add_argument('--candidate', choices=CANDIDATES, required=True)
    execution = actions.add_parser('run')
    execution.add_argument('job', type=Path)
    for command in (setup, ui, execution):
        command.add_argument('--experiment', choices=tuple(PROTOCOLS), default='base')
    options = vars(parser.parse_args())
    action = options.pop('action')
    if action == 'prepare':
        prepare(**options)
    elif action == 'prepare-ui':
        prepare_ui(**options)
    else:
        protocol = PROTOCOLS[options.pop('experiment')]
        run(**options, protocol=protocol)
