"""Seal a train/score split and schedule a bounded, owned Qwen Colab experiment."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile
from tempfile import TemporaryDirectory
from uuid import uuid4

from research.backbone_transfer.colab import MODEL_SOURCE, run
from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.semantic_tracking.iirc_linked_colab import CLI_VERSION, Colab, PROJECT, verify_code_inventory


COMMON_CODE = (
    'research/backbone_transfer/extract.py',
    'research/stage3/probe.py',
    'research/stage3/run.py',
    'research/stage5/complex_tokenization.py',
    'research/token_readout/extraction.py',
    'research/token_readout/model.py',
)
CODE = (*COMMON_CODE, 'research/token_readout/fit.py')
UI_CODE = (*COMMON_CODE, 'research/token_readout/app.py', 'research/token_readout/verify_app.py',
           'research/token_readout/runner.py', 'research/token_readout/live.py')


def select_keys(row, keys):
    return {name: row[name] for name in keys}


def split_inputs(design):
    training = {name: [row for row in design[name] if row['split'] == 'train'] for name in ('frick', 'evidence')}
    validation = {
        'frick': [select_keys(row, ('id', 'sentence_spans', 'event_ends'))
                  for row in design['frick'] if row['split'] == 'test'],
        'evidence': [{**select_keys(row, ('id', 'article_start')),
                      'queries': [select_keys(query, ('id', 'span')) for query in row['queries']]}
                     for row in design['evidence'] if row['split'] == 'test'],
        'controls': [{'id': variant['id'], 'anchor_span': row['anchor_span']}
                     for row in design['controls'] for variant in row['variants'].values()],
    }
    if any(not rows for rows in (*training.values(), *validation.values())):
        raise ValueError('The experiment requires eligible data for all declared endpoints')
    if not all(row['split'] == 'test' for row in design['controls']):
        raise ValueError('Designed controls must be reserved for evaluation')
    return training, validation


def pack_job(output, protocol, purpose, kind, code_paths, members, json_files, budget, provenance, *, candidate='qwen',model_spec=None):
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    job = output / f'cogllm-{purpose}-{stamp}-{uuid4().hex[:6]}'
    model_path = MODEL_SOURCE / 'models.json'
    model = read_json(model_path)['models'][candidate] if model_spec is None else model_spec
    with TemporaryDirectory(prefix='.readout-', dir=output) as temporary:
        stage = Path(temporary) / job.name
        stage.mkdir()
        code = {name: PROJECT / name for name in code_paths}
        verify_code_inventory(code)
        files = {**members, **code, 'research/backbone_transfer/models.json': model_path, 'protocol.md': protocol}
        for name, content in json_files.items():
            path = stage / name
            write_json(path, content)
            files[name] = path
        hashes = {name: digest(path) for name, path in sorted(files.items())}
        payload = stage / 'payload.tar.gz'
        with tarfile.open(payload, 'x:gz', compresslevel=2) as package:
            for name, path in sorted(files.items()):
                package.add(path, arcname=name, recursive=False)
        shutil.copyfile(MODEL_SOURCE / 'worker.py', stage / 'worker.py')
        data = {
            'run_id': job.name, 'candidate': candidate, 'model': model['repo'], 'revision': model['revision'],
            'cli_version': CLI_VERSION, 'requested_gpu': 'L4', 'time_budget_s': budget,
            'payload_members_sha256': hashes, 'payload_sha256': digest(payload), 'timeline': {},
            'worker_sha256': digest(stage / 'worker.py'), 'protocol_sha256': digest(protocol),
            'extraction_kind': kind, 'heldout_outcomes_in_worker': False, **provenance,
        }
        write_json(stage / 'plan.json', data)
        Colab(stage, data).update('prepared')
        stage.rename(job)
    print(job.resolve(), flush=True)


def prepare(dataset, output, protocol, smoke=False, arm_seconds=1100, epochs=48):
    paths = {name: dataset / name for name in ('stimuli.jsonl', 'design.json')}
    receipt = read_json(dataset / 'receipt.json')
    if {name: digest(path) for name, path in paths.items()} != receipt['file_sha256']:
        raise ValueError('Prepared stimuli or observation schema changed')
    model_path = MODEL_SOURCE / 'models.json'
    model = read_json(model_path)['models']['qwen']
    if model['revision'] != receipt['model_revision'] or model['sha256']['tokenizer.json'] != receipt['tokenizer_sha256']:
        raise ValueError('Dataset token coordinates are bound to another tokenizer')
    training, layout = split_inputs(read_json(paths['design.json']))
    documents = list(stream_rows(paths['stimuli.jsonl']))
    ids = [d['id'] for d in documents]
    referenced = [r['id'] for collection in (training, layout) for rows in collection.values() for r in rows]
    if (len(set(ids)) != len(ids) or sorted(ids) != sorted(referenced)
            or any(set(row) != {'id', 'text'} or not row['text'] for row in documents)):
        raise ValueError('Input IDs do not match disjoint training and labels-free scoring layouts')
    settings = {
        'seed': 1729, 'epochs': epochs, 'head_lr': 0.0007, 'adapter_lr': 0.00025, 'weight_decay': 0.02,
        'minimum_epoch_gain': 0.003, 'frozen_seconds': arm_seconds, 'lora_seconds': arm_seconds, 'smoke': smoke,
        'technical_controls': [
            'A sample loses water during preparation. The next section explains the correction for the measured loss.',
            '新的概念要保留先前內容的前綴狀態。接下來這段文字在因果遮罩下只影響後續輸出。',
        ],
    }
    pack_job(output, protocol, 'token-smoke' if smoke else 'token-fit', 'token-readout', CODE,
             {'stimuli.jsonl': paths['stimuli.jsonl']},
             {'training.json': training, 'layout.json': layout, 'settings.json': settings},
             900 if smoke else 2 * arm_seconds + 1200, {'dataset_receipt_sha256': digest(dataset / 'receipt.json')})


def prepare_ui(dataset, output, protocol, checkpoint):
    root = checkpoint.parent
    seal = read_json(root / 'seal.json')
    if (seal['protocol_sha256'] != digest(protocol)
            or seal['files_sha256'].get(checkpoint.name) != digest(checkpoint)):
        raise ValueError('The HTTP run requires a sealed evaluated candidate')
    receipt = read_json(dataset / 'receipt.json')
    for member, checksum in receipt['file_sha256'].items():
        if digest(dataset / member) != checksum:
            raise ValueError('Design or stimulus bytes changed before app verification')
    group = read_json(dataset / 'design.json')['controls'][0]
    observed = {row['id']: row['text'] for row in stream_rows(dataset / 'stimuli.jsonl')}
    extended = observed[group['variants']['early']['id']]
    prefix = extended[:group['common_prefix_end']]
    if prefix == extended or not observed[group['variants']['late']['id']].startswith(prefix):
        raise ValueError('HTTP verification requires the sealed extended-prefix example')
    with TemporaryDirectory() as temporary:
        from research.backbone_transfer.extract import write_rows

        paths = Path(temporary) / 'stimuli.jsonl'
        write_rows(paths, [
            {'id': 'prefix', 'text': prefix}, {'id': 'extended', 'text': extended},
            {'id': 'unicode', 'text': '前一段介紹校準值。觀察記錄含 e\u0301 與符號 🧪。\r\n\r\n下一段解釋何時採用校準值。'},
        ])
        pack_job(output, protocol, 'token-ui', 'token-readout-ui', UI_CODE,
                 {'stimuli.jsonl': paths, 'checkpoint.pt': checkpoint}, {}, 900,
                 {'checkpoint_sha256': digest(checkpoint), 'fit_seal_sha256': digest(root / 'seal.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    setup = actions.add_parser('prepare')
    for name in ('dataset', 'output', 'protocol'):
        setup.add_argument('--' + name, type=Path, required=True)
    setup.add_argument('--smoke', action='store_true')
    setup.add_argument('--arm-seconds', type=int, default=1100)
    setup.add_argument('--epochs', type=int, default=48)
    ui = actions.add_parser('prepare-ui')
    for name in ('dataset', 'output', 'protocol', 'checkpoint'):
        ui.add_argument('--' + name, type=Path, required=True)
    execution = actions.add_parser('run')
    execution.add_argument('job', type=Path)
    execution.add_argument('--protocol', type=Path, required=True)
    arguments = vars(parser.parse_args())
    action = arguments.pop('action')
    {'prepare': prepare, 'prepare-ui': prepare_ui, 'run': run}[action](**arguments)
