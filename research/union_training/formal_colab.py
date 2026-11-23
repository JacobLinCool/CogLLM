"""Seal only grade-free assessment stimuli for the owned Colab extractor."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile
import tarfile
from tempfile import TemporaryDirectory
from uuid import uuid4

from research.backbone_transfer.colab import MODEL_SOURCE, run
from research.backbone_transfer.extract import MAX_INPUT_TOKENS, digest, read_json, stream_rows, write_json
from research.semantic_tracking.iirc_linked_colab import CLI_VERSION, Colab, PROJECT, verify_code_inventory


MODULES = (
    'research/backbone_transfer/extract.py',
    'research/stage3/probe.py',
    'research/stage3/run.py',
    'research/stage5/complex_tokenization.py',
    'research/union_training/formal_extract.py',
)


def prepare(inputs, rules, output, protocol):
    code = {name: PROJECT / name for name in MODULES}
    verify_code_inventory(code)
    model_path = MODEL_SOURCE / 'models.json'
    model = read_json(model_path)['models']['qwen']
    rule_paths = dict(value.split('=', 1) for value in rules)
    if len(rule_paths) != len(rules):
        raise ValueError('Expected one preregistration per study')
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    job = output / f'cogllm-formal-{stamp}-{uuid4().hex[:6]}'
    with TemporaryDirectory(prefix='.formal-', dir=output) as temporary:
        staging = Path(temporary) / job.name
        staging.mkdir()
        members = {**code, 'research/backbone_transfer/models.json': model_path}
        inventory = {}
        for value in inputs:
            alias, text_path = value.split('=', 1)
            if not alias.isascii() or not alias.replace('-', '').isalnum() or alias in inventory:
                raise ValueError(f'Unexpected study alias {alias!r}')
            path = Path(text_path)
            ids = set()
            for row in stream_rows(path):
                if (set(row) != {'id', 'text', 'stops'} or row['id'] in ids
                        or not row['text'] or not row['stops']
                        or len(row['stops']) > MAX_INPUT_TOKENS
                        or row['stops'] != sorted(set(row['stops']))
                        or any(type(stop) is not int or not 0 < stop <= len(row['text']) for stop in row['stops'])):
                    raise ValueError(f'{alias} includes a response field, duplicate, or invalid geometry')
                ids.add(row['id'])
            if not ids:
                raise ValueError(f'Empty assessment {alias}')
            inventory[alias] = {'stimuli_sha256': digest(path),
                                'protocol_sha256': digest(Path(rule_paths[alias]))}
            members[f'formal/{alias}/stimuli.jsonl'] = path
        if set(inventory) != set(rule_paths):
            raise ValueError('Formal stimuli and preregistrations differ')
        manifest = staging / 'inventory.json'
        write_json(manifest, inventory)
        members['formal/inventory.json'] = manifest
        marker = staging / 'stimuli.jsonl'
        marker.touch()
        members['stimuli.jsonl'] = marker
        hashes = {name: digest(path) for name, path in sorted(members.items())}
        payload = staging / 'payload.tar.gz'
        with tarfile.open(payload, 'x:gz', compresslevel=2) as archive:
            for name, path in sorted(members.items()):
                archive.add(path, arcname=name, recursive=False)
        copyfile(MODEL_SOURCE / 'worker.py', staging / 'worker.py')
        plan = {
            'run_id': job.name, 'candidate': 'qwen', 'model': model['repo'], 'revision': model['revision'],
            'cli_version': CLI_VERSION, 'requested_gpu': 'L4', 'time_budget_s': 15 * 60,
            'payload_members_sha256': hashes, 'payload_sha256': digest(payload),
            'worker_sha256': digest(staging / 'worker.py'), 'protocol_sha256': digest(protocol),
            'extraction_kind': 'observation-formal-stops', 'timeline': {}, 'grade_free_inputs': inventory,
        }
        write_json(staging / 'plan.json', plan)
        Colab(staging, plan).update('prepared')
        staging.rename(job)
    print(job.resolve(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    setup = actions.add_parser('prepare')
    setup.add_argument('--input', action='append', dest='inputs', required=True)
    setup.add_argument('--rule', action='append', dest='rules', required=True)
    setup.add_argument('--output', type=Path, required=True)
    setup.add_argument('--protocol', type=Path, required=True)
    execution = actions.add_parser('run')
    execution.add_argument('job', type=Path)
    execution.add_argument('--protocol', type=Path, required=True)
    arguments = vars(parser.parse_args())
    action = arguments.pop('action')
    (prepare if action == 'prepare' else run)(**arguments)
