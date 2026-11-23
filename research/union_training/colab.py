"""Pack the frozen union training protocol for the owned L4 controller."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile
from uuid import uuid4

from research.backbone_transfer.colab import MODEL_SOURCE, run
from research.backbone_transfer.extract import digest, read_json, write_json
from research.semantic_tracking.iirc_linked_colab import CLI_VERSION, Colab

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'protocol.md'
WORK_SECONDS = 42 * 60


def prepare(packet, output, holdout):
    receipt = read_json(packet / 'receipt.json')
    if digest(holdout) != receipt['formal_holdout_sha256']:
        raise ValueError('Formal verification manifest changed after development packing')
    names = set(receipt['packet_sha256'])
    index = read_json(packet / 'index.json')
    if names != {'stimuli.jsonl', 'index.json'} | {row['file'] for row in index['tasks']}:
        raise ValueError('Unexpected files in the development packet')
    if any(Path(name).name != name or digest(packet / name) != receipt['packet_sha256'][name] for name in names):
        raise ValueError('Development packet has unexpected or altered files')
    registered = read_json(MODEL_SOURCE / 'models.json')['models']['qwen']
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    job = output / f'cogllm-union-{timestamp}-{uuid4().hex[:8]}'
    job.mkdir(parents=True, mode=0o700)
    sources = {
        'research/backbone_transfer/models.json': MODEL_SOURCE / 'models.json',
        'research/backbone_transfer/extract.py': MODEL_SOURCE / 'extract.py',
        'research/joint/train.py': HERE.parent / 'joint/train.py',
        **{f'research/union_training/{name}': HERE / name for name in ('fit.py', 'run_l4.py')},
        **{name: packet / name for name in names | {'receipt.json'}},
    }
    payload = job / 'payload.tar.gz'
    with tarfile.open(payload, 'x:gz', compresslevel=2) as archive:
        for name, path in sorted(sources.items()):
            archive.add(path, arcname=name, recursive=False)
    shutil.copyfile(MODEL_SOURCE / 'worker.py', job / 'worker.py')
    data = {
        'run_id': job.name, 'candidate': 'qwen', 'model': registered['repo'], 'revision': registered['revision'],
        'cli_version': CLI_VERSION, 'requested_gpu': 'L4', 'time_budget_s': WORK_SECONDS,
        'payload_members_sha256': {name: digest(path) for name, path in sources.items()},
        'payload_sha256': digest(payload), 'worker_sha256': digest(job / 'worker.py'),
        'protocol_sha256': digest(PROTOCOL), 'extraction_kind': 'union-training', 'timeline': {},
        'formal_holdout_sha256': digest(holdout), 'input_documents': receipt['development_documents'],
        'input_tokens': receipt['input_tokens'],
    }
    write_json(job / 'plan.json', data)
    Colab(job, data).update('prepared')
    print(job.resolve(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    setup = actions.add_parser('prepare')
    for name in ('packet', 'output', 'holdout'):
        setup.add_argument(f'--{name}', required=True, type=Path)
    actions.add_parser('run').add_argument('job', type=Path)
    args = vars(parser.parse_args())
    if args.pop('action') == 'prepare':
        prepare(**args)
    else:
        run(args['job'], protocol=PROTOCOL)
