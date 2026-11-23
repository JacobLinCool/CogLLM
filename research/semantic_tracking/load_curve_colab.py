"""Pack and run the human-label-free load-curve extraction on one owned L4."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile
from uuid import uuid4

from research.backbone_transfer.colab import MODEL_SOURCE, run
from research.backbone_transfer.extract import digest, read_json, stream_rows, write_rows
from research.semantic_tracking.iirc_linked_colab import CLI_VERSION, Colab, PROJECT
from research.semantic_tracking.iirc_linked_colab_worker import write_json
from research.semantic_tracking.load_curve_extract import documents_from


PROTOCOL = Path(__file__).with_name('semantic-load-curve-pilot.md')
WORK_SECONDS = 24 * 60


def prepare(inputs, tokenizer_path, output, protocol=PROTOCOL):
    from tokenizers import Tokenizer

    model = read_json(MODEL_SOURCE / 'models.json')['models']['qwen']
    if digest(tokenizer_path) != model['sha256']['tokenizer.json']:
        raise ValueError('Preflight needs the pinned Qwen tokenizer')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    job = output / f'cogllm-loadcurve-{timestamp}-{uuid4().hex[:8]}'
    job.mkdir(parents=True, mode=0o700)
    rows = []
    for path in inputs:
        for row in stream_rows(path):
            if row.keys() not in ({'id', 'text'}, {'id', 'text', 'stops'}):
                raise ValueError(f'Model input contains non-stimulus fields: {path.name}')
            rows.append({'id': row['id'], 'text': row['text']})
    write_rows(job / 'stimuli.jsonl', sorted(rows, key=lambda row: row['id']))
    documents, token_count = documents_from(job / 'stimuli.jsonl', Tokenizer.from_file(str(tokenizer_path)))
    sources = {
        'research/backbone_transfer/models.json': MODEL_SOURCE / 'models.json',
        'research/backbone_transfer/extract.py': MODEL_SOURCE / 'extract.py',
        'research/semantic_tracking/load_curve_extract.py': PROJECT / 'research/semantic_tracking/load_curve_extract.py',
        'stimuli.jsonl': job / 'stimuli.jsonl',
    }
    payload = job / 'payload.tar.gz'
    with tarfile.open(payload, 'x:gz', compresslevel=2) as archive:
        for name, path in sorted(sources.items()):
            archive.add(path, arcname=name, recursive=False)
    shutil.copyfile(MODEL_SOURCE / 'worker.py', job / 'worker.py')
    data = {
        'run_id': job.name, 'candidate': 'qwen', 'model': model['repo'], 'revision': model['revision'],
        'cli_version': CLI_VERSION, 'requested_gpu': 'L4', 'time_budget_s': WORK_SECONDS,
        'payload_members_sha256': {name: digest(path) for name, path in sources.items()},
        'payload_sha256': digest(payload), 'worker_sha256': digest(job / 'worker.py'),
        'protocol_sha256': digest(protocol), 'extraction_kind': 'load-curve', 'timeline': {},
        'input_documents': len(documents), 'input_tokens': token_count,
    }
    write_json(job / 'plan.json', data)
    Colab(job, data).update('prepared')
    print(job.resolve(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    prep = actions.add_parser('prepare')
    prep.add_argument('--input', type=Path, action='append', required=True, dest='inputs')
    prep.add_argument('--tokenizer', type=Path, required=True, dest='tokenizer_path')
    prep.add_argument('--output', type=Path, required=True)
    execute = actions.add_parser('run')
    execute.add_argument('job', type=Path)
    for command in (prep, execute):
        command.add_argument('--protocol', type=Path, default=PROTOCOL)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.inputs, args.tokenizer_path, args.output, args.protocol)
    else:
        run(args.job, protocol=args.protocol)


if __name__ == '__main__':
    main()
