"""Seal the common inputs and collect one owned Colab L4 job per backbone."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import signal
import tarfile
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from uuid import uuid4

from research.backbone_transfer.worker import CHUNK_BYTES, digest
from research.semantic_tracking.iirc_linked_colab import CLI_VERSION, Colab, PROJECT, run_lock
from research.semantic_tracking.iirc_linked_colab_worker import inspect_archive, read_json, write_json


MODEL_SOURCE = Path(__file__).parent
PROTOCOL = MODEL_SOURCE / 'protocol.md'
CANDIDATES = ('qwen', 'lfm3', 'gemma4')
DATASETS = ('plaba', 'complex', 'frick')
WORK_SECONDS = 32 * 60
MAX_OUTPUT_BYTES = 350 * 1024 * 1024


def combine(prepared, output):
    output.mkdir(parents=True, exist_ok=True)
    stimuli = output / 'stimuli.jsonl'
    targets = output / 'targets.jsonl'
    unseen, studies = set(), {}
    with stimuli.open('xb') as combined:
        for dataset in DATASETS:
            path = prepared / ('frick-both' if dataset == 'frick' else dataset) / 'stimuli.jsonl'
            files = path.read_bytes()
            rows = [json.loads(line) for line in files.splitlines()]
            if not rows or not files.endswith(b'\n'):
                raise ValueError(f'No complete grade-free {dataset} stimulus rows')
            for row in rows:
                if row.keys() != {'id', 'text', 'stops'} or row['id'] in unseen:
                    raise ValueError('Non-grade-free or duplicate stimulus ID')
                unseen.add(row['id'])
            studies[dataset] = {'documents': len(rows), 'stops': sum(len(row['stops']) for row in rows),
                                'sha256': digest(path)}
            combined.write(files)
    target_path = prepared / 'complex' / 'targets.jsonl'
    rows = [json.loads(row) for row in target_path.read_bytes().splitlines()]
    identities = set()
    for row in rows:
        if (row.keys() != {'id', 'start', 'end'} or row['id'] not in unseen
                or not 0 <= row['start'] < row['end'] or (row['id'], row['start'], row['end']) in identities):
            raise ValueError('Expected deduplicated grade-free target intervals')
        identities.add((row['id'], row['start'], row['end']))
    shutil.copyfile(target_path, targets)
    write_json(output / 'inventory.json', {'studies': studies, 'stimuli_sha256': digest(stimuli),
                                           'targets_sha256': digest(targets), 'targets': len(rows),
                                           'protocol_sha256': digest(PROTOCOL)})


def prepare(prepared, output, packing, candidate=None):
    combine(prepared, packing)
    output.mkdir(parents=True, exist_ok=True)
    sources = {f'research/backbone_transfer/{name}': MODEL_SOURCE / name for name in ('extract.py', 'models.json')}
    sources.update({name: PROJECT / name for name in ('research/stage5/complex_tokenization.py',
                                                      'research/stage3/probe.py', 'research/stage3/run.py')})
    sources.update({name: packing / name for name in ('stimuli.jsonl', 'targets.jsonl')})
    inputs = {name: digest(path) for name, path in sources.items()}
    models = read_json(MODEL_SOURCE / 'models.json')['models']
    if set(models) != set(CANDIDATES):
        raise ValueError('Expected the three registered model aliases')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    suffix = uuid4().hex[:8]
    with TemporaryDirectory(dir=output, prefix='.staging-') as temporary:
        payload = Path(temporary) / 'payload.tar.gz'
        with tarfile.open(payload, 'x:gz', compresslevel=2) as archive:
            for name, path in sorted(sources.items()):
                archive.add(path, arcname=name, recursive=False)
        for alias in CANDIDATES if candidate is None else (candidate,):
            job = output / f'cogllm-backbone-{alias}-{timestamp}-{suffix}'
            job.mkdir(mode=0o700)
            shutil.copyfile(payload, job / 'payload.tar.gz')
            shutil.copyfile(MODEL_SOURCE / 'worker.py', job / 'worker.py')
            model = models[alias]
            data = {'run_id': job.name, 'candidate': alias, 'model': model['repo'], 'revision': model['revision'],
                    'cli_version': CLI_VERSION, 'requested_gpu': 'L4', 'time_budget_s': WORK_SECONDS,
                    'payload_members_sha256': inputs, 'payload_sha256': digest(payload),
                    'worker_sha256': digest(job / 'worker.py'), 'protocol_sha256': digest(PROTOCOL),
                    'extraction_kind': 'stops', 'timeline': {}}
            write_json(job / 'plan.json', data)
            Colab(job, data).update('prepared')
            print(alias, str(job.resolve()), flush=True)


def collect(machine, status, deadline):
    job, plan = machine.job, machine.data
    if status['phase'] != 'finished' or 'result' not in status:
        machine.download('/worker.log', 'remote-worker.log', timeout=30)
        raise RuntimeError(f'Owned Colab worker failed: {status.get("error", "unknown")}. See remote-worker.log')
    result = status['result']
    parts, expected = result['parts'], result['members_sha256']
    if (not 0 < result['archive_bytes'] <= MAX_OUTPUT_BYTES or not parts or len(parts) > 32
            or sum(entry['bytes'] for entry in parts) != result['archive_bytes']):
        raise ValueError('Invalid extraction archive size or piece count')
    for name in expected:
        path = PurePosixPath(name)
        if len(path.parts) < 2 or path.parts[0] != 'extraction' or '..' in path.parts:
            raise ValueError('Unexpected path in researcher result archive')
    with (job / 'results.tar.gz').open('xb') as combined:
        for i, entry in enumerate(parts):
            name = f'part-{i:03}.bin'
            if (entry['name'] != name or not 0 < entry['bytes'] <= CHUNK_BYTES
                    or i < len(parts) - 1 and entry['bytes'] != CHUNK_BYTES):
                raise ValueError('An output piece does not correspond to its declared order and size')
            for _ in range(3):
                if monotonic() + 45 >= deadline:
                    raise TimeoutError(f'Download deadline before {name}')
                if machine.download('/' + name, name, timeout=65):
                    break
            else:
                raise TimeoutError(f'Colab did not return {name}')
            part = job / name
            if part.stat().st_size != entry['bytes'] or digest(part) != entry['sha256']:
                raise ValueError(f'Transferred output piece failed integrity verification: {name}')
            with part.open('rb') as stream:
                shutil.copyfileobj(stream, combined)
            part.unlink()
    artifact = job / 'results.tar.gz'
    if digest(artifact) != result['archive_sha256']:
        raise ValueError('Reconstructed extraction archive failed integrity verification')
    if inspect_archive(artifact, expected.keys(), destination=job) != expected:
        raise ValueError('At least one extracted file differs from the worker verification')
    seal = read_json(job / 'extraction' / 'seal.json')
    if seal['protocol_sha256'] != plan['protocol_sha256']:
        raise ValueError('The actual extraction was run with a different protocol')
    if plan['extraction_kind'] == 'union-observations':
        cached = read_json(job / 'extraction' / 'extraction.json')
        sealed = {f'extraction/{name}': checksum for name, checksum in seal['files_sha256'].items()}
        manifest = {name: checksum for name, checksum in expected.items() if name != 'extraction/seal.json'}
        if (sealed != manifest
                or any(seal[key] != plan[key] for key in ('packet_schema_version', 'packet_receipt_sha256',
                                                         'base_receipt_sha256', 'formal_holdout_sha256'))
                or seal['formal_data_in_worker'] is not False or seal['one_forward_per_stimulus'] is not True
                or seal['encoder_revision'] != plan['revision'] or seal['encoder_gpu'] != 'NVIDIA L4'
                or cached['documents'] != plan['input_documents'] or cached['certified_stops'] != plan['input_positions']
                or cached['stimuli_sha256'] != plan['payload_members_sha256']['stimuli.jsonl']):
            raise ValueError('The observation result is not bound to the prepared corpus and frozen extraction')
    plan['artifact'] = result
    plan['verified_outputs'] = True


def run(job, protocol=PROTOCOL):
    plan = read_json(job / 'manifest.json')
    machine = Colab(job, plan)
    if (plan['phase'] != 'prepared' or plan['run_id'] != job.name
            or plan['payload_sha256'] != digest(job / 'payload.tar.gz')
            or plan['worker_sha256'] != digest(job / 'worker.py')
            or plan['protocol_sha256'] != digest(protocol)):
        raise ValueError('Expected an unused job with exact code, contents, and experiment protocol')
    if inspect_archive(job / 'payload.tar.gz', plan['payload_members_sha256'].keys()) != plan['payload_members_sha256']:
        raise ValueError('Unrecognized inputs in the prepared grade-free payload')
    previous_handlers = {}
    attempted = False

    def interrupt(signum, _):
        raise KeyboardInterrupt(f'Received signal {signum}')

    with run_lock(job, 'x'):
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, interrupt)
        deadline = monotonic() + plan['time_budget_s']
        try:
            before = machine.sessions()
            plan['endpoints_before_allocation'] = sorted(before)
            if machine.endpoint(before) is not None:
                raise RuntimeError('The new session name is already assigned')
            machine.usage('before-allocation', required=True)
            machine.update('allocating')
            attempted = True
            machine.cli('new', '-s', job.name, '--gpu', 'L4', timeout=80)
            endpoint = machine.endpoint(machine.sessions())
            if endpoint is None or endpoint in before:
                raise RuntimeError('Could not confirm ownership of the assigned L4')
            plan['endpoint'] = endpoint
            machine.update('allocated')
            machine.usage('after-allocation')
            for name, suffix in (('payload.tar.gz', '.tar.gz'), ('worker.py', '.worker.py'), ('plan.json', '.plan.json')):
                machine.cli('upload', '-s', job.name, job / name, machine.remote + suffix, timeout=70)
            machine.launch(deadline - 150)
            machine.update('worker-running')
            usage_at = monotonic() + 600
            while monotonic() + 60 < deadline:
                if machine.download('/status.json', 'remote-status.json', timeout=20):
                    status = read_json(job / 'remote-status.json')
                    if any(status[name] != plan[name] for name in ('run_id', 'candidate', 'payload_sha256', 'protocol_sha256')):
                        raise ValueError('Remote state belongs to a different run')
                    if status['finished']:
                        collect(machine, status, deadline)
                        break
                    phase = f'remote-{status["phase"]}'
                    if plan['phase'] != phase:
                        machine.update(phase)
                else:
                    machine.check_connection()
                if monotonic() >= usage_at:
                    machine.usage('periodic')
                    usage_at = monotonic() + 600
                sleep(12)
            if not plan.get('verified_outputs'):
                machine.download('/worker.log', 'remote-worker.log', timeout=15)
                raise TimeoutError('The owned L4 reached its bounded extraction or transfer deadline')
        except (Exception, KeyboardInterrupt) as error:
            plan['failure'] = str(error)
        finally:
            for signum in previous_handlers:
                signal.signal(signum, signal.SIG_IGN)
            if attempted:
                machine.update('releasing')
                machine.usage('before-stop', timeout=12)
                plan['release_verified'] = machine.release()
                plan['release_usage_recorded'] = machine.usage('after-stop')
            success = plan.get('verified_outputs') and plan.get('release_verified') and 'failure' not in plan
            machine.update('finished' if success else 'incomplete')
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        if not success:
            raise RuntimeError(plan.get('failure', 'Colab release could not be verified'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    setup = actions.add_parser('prepare')
    for name in ('prepared', 'output', 'packing'):
        setup.add_argument('--' + name, type=Path, required=True)
    setup.add_argument('--candidate', choices=CANDIDATES)
    actions.add_parser('run').add_argument('job', type=Path)
    arguments = vars(parser.parse_args())
    command = arguments.pop('action')
    (prepare if command == 'prepare' else run)(**arguments)
