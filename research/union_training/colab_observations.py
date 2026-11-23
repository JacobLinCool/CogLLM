"""Seal one schema-2 observation run for the owned Colab L4 controller."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from shutil import copyfile
import tarfile
from tempfile import TemporaryDirectory
from uuid import uuid4

from research.backbone_transfer.colab import MODEL_SOURCE, run
from research.backbone_transfer.extract import digest, read_json, write_json
from research.semantic_tracking.iirc_linked_colab import CLI_VERSION, Colab, PROJECT, verify_code_inventory
from research.union_training.run_observations_l4 import PACKET_SCHEMA, verify_packet


PROTOCOL = Path(__file__).with_name('observation-protocol.md')
WORK_SECONDS = 42 * 60
SMOKE_SECONDS = 12 * 60
WORKER_MODULES = (
    'research/backbone_transfer/extract.py',
    'research/stage3/probe.py',
    'research/stage3/run.py',
    'research/stage5/complex_tokenization.py',
    'research/union_training/fit_observations.py',
    'research/union_training/observation_losses.py',
    'research/union_training/observation_registry.py',
    'research/union_training/run_observations_l4.py',
)


def prepare(packet, output, holdout):
    receipt = verify_packet(packet)
    purpose = receipt.get('purpose', 'full_training')
    if purpose not in {'full_training', 'hardware_smoke'}:
        raise ValueError('Unrecognized purpose for the observation packet')
    holdout_sha256 = digest(holdout)
    if holdout_sha256 != receipt['formal_holdout_sha256']:
        raise ValueError('Formal verification manifest changed after observation packing')
    model_path = MODEL_SOURCE / 'models.json'
    model = read_json(model_path)['models']['qwen']
    if model['sha256']['tokenizer.json'] != receipt['tokenizer_sha256']:
        raise ValueError('Observation packet does not use the registered frozen Qwen tokenizer')

    code = {name: PROJECT / name for name in WORKER_MODULES}
    verify_code_inventory(code)
    members = {
        **code,
        'research/backbone_transfer/models.json': model_path,
        **{name: packet / name for name in set(receipt['packet_sha256']) | {'receipt.json'}},
    }
    inputs = {name: digest(path) for name, path in members.items()}
    if any(inputs[name] != sha256 for name, sha256 in receipt['packet_sha256'].items()):
        raise ValueError('The observation packet changed while sealing the Colab bundle')
    protocol_sha256 = digest(PROTOCOL)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output.mkdir(parents=True, exist_ok=True)
    job = output / f'cogllm-observations-{timestamp}-{uuid4().hex[:8]}'
    with TemporaryDirectory(dir=output, prefix='.staging-observation-') as temporary:
        staging = Path(temporary) / job.name
        staging.mkdir(mode=0o700)
        payload = staging / 'payload.tar.gz'
        with tarfile.open(payload, 'x:gz', compresslevel=2) as archive:
            for name, path in sorted(members.items()):
                archive.add(path, arcname=name, recursive=False)
        copyfile(MODEL_SOURCE / 'worker.py', staging / 'worker.py')
        plan = {
            'run_id': job.name, 'candidate': 'qwen', 'model': model['repo'], 'revision': model['revision'],
            'cli_version': CLI_VERSION, 'requested_gpu': 'L4', 'purpose': purpose,
            'time_budget_s': SMOKE_SECONDS if purpose == 'hardware_smoke' else WORK_SECONDS,
            'payload_members_sha256': inputs, 'payload_sha256': digest(payload),
            'worker_sha256': digest(staging / 'worker.py'), 'protocol_sha256': protocol_sha256,
            'extraction_kind': 'union-observations', 'timeline': {},
            'formal_holdout_sha256': holdout_sha256, 'packet_schema_version': PACKET_SCHEMA,
            'packet_receipt_sha256': inputs['receipt.json'], 'base_receipt_sha256': receipt['base_receipt_sha256'],
            'input_documents': receipt['documents'], 'input_tokens': receipt['input_tokens'],
            'input_positions': receipt['positions'],
        }
        write_json(staging / 'plan.json', plan)
        Colab(staging, plan).update('prepared')
        staging.rename(job)
    print(job.resolve(), flush=True)
    return job


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    setup = actions.add_parser('prepare')
    for name in ('packet', 'output', 'holdout'):
        setup.add_argument(f'--{name}', type=Path, required=True)
    actions.add_parser('run').add_argument('job', type=Path)
    args = vars(parser.parse_args())
    if args.pop('action') == 'prepare':
        prepare(**args)
    else:
        run(args['job'], protocol=PROTOCOL)
