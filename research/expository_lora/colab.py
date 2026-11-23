"""Seal the expository-reading comparison and run it on one owned Colab L4."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile
from uuid import uuid4

from research.backbone_transfer.colab import MODEL_SOURCE, run
from research.backbone_transfer.extract import digest, load_documents, read_json, stream_rows, write_json, write_rows
from research.semantic_tracking.iirc_linked_colab import CLI_VERSION, Colab

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / 'protocol.md'
WORK_SECONDS = 36 * 60


def prepare(assay, tokenizer_path, output):
    from tokenizers import Tokenizer

    registry = MODEL_SOURCE / 'models.json'
    model = read_json(registry)['models']['qwen']
    receipt = assay / 'source-audit.json'
    audit = read_json(receipt)
    if digest(tokenizer_path) != model['sha256']['tokenizer.json'] or audit['tokenizer_sha256'] != digest(tokenizer_path):
        raise ValueError('The prepared assay must use the pinned model tokenizer')
    for name, sha in audit['files_sha256'].items():
        if digest(assay / name) != sha:
            raise ValueError(f'Prepared assay member changed: {name}')

    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    job = output / f'cogllm-expository-{timestamp}-{uuid4().hex[:8]}'
    job.mkdir(parents=True, mode=0o700)
    corpus = {}
    rows = []
    for dataset in ('plaba', 'kerwer'):
        subset = list(stream_rows(assay / dataset / 'stimuli.jsonl'))
        corpus[dataset] = [row['id'] for row in subset]
        rows.extend(subset)
    write_rows(job / 'stimuli.jsonl', rows)
    documents = load_documents(job / 'stimuli.jsonl', Tokenizer.from_file(str(tokenizer_path)))
    if any(doc.missing for doc in documents.values()):
        raise ValueError('Every planned readout stop must be prefix-stable')

    plaba_spans = {(row['id'], rating['stop']): rating['start']
                   for row in stream_rows(assay / 'plaba/ratings.jsonl') for rating in row['ratings']}
    study = read_json(HERE.parent / 'results/kerwer-transfer-protocol.json')
    kerwer_spans = {(key, stop): start for key in corpus['kerwer']
                    for start, stop, _ in study['alignment']['served'][key]}
    write_rows(job / 'spans.jsonl', [
        {'id': key, 'starts': [dict_of_spans[(key, stop)] for stop in documents[key].certified]}
        for dataset, dict_of_spans in (('plaba', plaba_spans), ('kerwer', kerwer_spans)) for key in corpus[dataset]
    ])
    write_json(job / 'assay.json', {
        'assay_sha256': digest(receipt),
        'corpus': corpus,
        'held_ids': {index: value['held_ids'] for index, value in audit['folds'].items()},
        'stimulus_file_sha256': {key: audit['files_sha256'][key + '/stimuli.jsonl'] for key in corpus},
    })
    sources = {
        'research/backbone_transfer/models.json': registry,
        'research/backbone_transfer/extract.py': MODEL_SOURCE / 'extract.py',
        'research/expository_lora/fit.py': HERE / 'fit.py',
        'research/expository_lora/polish.py': HERE / 'polish.py',
        'research/expository_lora/run_l4.py': HERE / 'run_l4.py',
        **{name: job / name for name in ('stimuli.jsonl', 'spans.jsonl', 'assay.json')},
    }
    for name in ('ratings', 'pairs'):
        sources[f'labels/full/{name}.jsonl'] = assay / 'plaba' / f'{name}.jsonl'
        for fold in range(5):
            sources[f'labels/{fold}/{name}.jsonl'] = assay / f'plaba/folds/{fold}/{name}.jsonl'
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
        'protocol_sha256': digest(PROTOCOL), 'extraction_kind': 'expository-lora', 'timeline': {},
        'assay_sha256': digest(receipt), 'input_documents': len(documents),
        'input_tokens': sum(len(doc.ids) for doc in documents.values()),
    }
    write_json(job / 'plan.json', data)
    Colab(job, data).update('prepared')
    print(job.resolve(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    setup = actions.add_parser('prepare')
    for name in ('assay', 'tokenizer_path', 'output'):
        setup.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    actions.add_parser('run').add_argument('job', type=Path)
    args = vars(parser.parse_args())
    if args.pop('action') == 'prepare':
        prepare(**args)
    else:
        run(args['job'], protocol=PROTOCOL)


if __name__ == '__main__':
    main()
