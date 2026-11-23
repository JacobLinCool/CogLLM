"""Stage, verify, and release a fit-only matched IIRC job with no Qwen execution."""

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tarfile
from time import monotonic
from uuid import uuid4
from zipfile import BadZipFile, ZipFile

from research.semantic_tracking import iirc_linked_colab as owned
from research.semantic_tracking import iirc_linked_colab_worker as linked
from research.semantic_tracking import iirc_matched_colab_worker as worker


METHODS = {'geometry', 'path_text', 'anchor_conditioned'}
CASES = {'train': 3240, 'dev': 114}


def source_checkpoint(job):
    source = worker.read_json(job / 'manifest.json')
    directory = job / 'checkpoints'
    receipt_file = directory / 'receipt.json'
    receipt = worker.read_json(receipt_file)
    if (source['run_id'] != job.name or source['phase'] not in ('finished', 'failed')
            or source.get('release_verified') is not True or receipt['run_id'] != job.name
            or len(receipt['shards']) != receipt['shards_total']):
        raise ValueError('Matched replay requires a complete checkpoint from a released official L4 run')
    linked.check_proofs(directory, receipt, source['data_members_sha256']['assay.json'],
                        source['payload_members_sha256'], source['data_members_sha256'])
    return source, worker.digest(receipt_file)


def copy_parts(features, job):
    feature_hash = hashlib.sha256()
    parts = []
    with features.open('rb') as stream:
        while block := stream.read(worker.PART_BYTES):
            (job / worker.part_name(len(parts))).write_bytes(block)
            feature_hash.update(block)
            parts.append({'bytes': len(block), 'sha256': hashlib.sha256(block).hexdigest()})
    return {'sha256': feature_hash.hexdigest(), 'bytes': sum(part['bytes'] for part in parts), 'parts': parts}


def prepare(prepared, replay, original_job, output):
    original, checkpoint_sha = source_checkpoint(original_job)
    packet = prepared / 'fit-inputs.tar.gz'
    input_hashes = worker.inspect(packet, worker.INPUTS)
    assay_path = prepared / 'assay.json'
    assay = worker.read_json(assay_path)
    extraction_path = replay / 'extraction.json'
    extraction = worker.read_json(extraction_path)
    assay_hash = input_hashes['assay.json']
    if (assay['cases'] != CASES or (assay['model'], assay['revision']) != (linked.MODEL, linked.REVISION)
            or assay_hash != worker.digest(assay_path)
            or {name: value for name, value in input_hashes.items() if name != 'assay.json'} != assay['inputs_sha256']
            or assay['original_stimuli_sha256'] != {split: original['data_members_sha256'][f'{split}/stimuli.jsonl']
                                                    for split in CASES}
            or assay['protocol_sha256'] != worker.digest(owned.PROJECT / 'research/iirc-identity-holdout.md')
            or assay['code_sha256'] != worker.digest(owned.PROJECT / 'research/semantic_tracking/iirc_linked_identity.py')
            or extraction['source_checkpoint_sha256'] != checkpoint_sha
            or (extraction['model'], extraction['revision']) != (assay['model'], assay['revision'])
            or extraction['assay_sha256'] != assay_hash
            or extraction['inputs_sha256'] != {'assay': assay_hash, **{f'{split}/stimuli.jsonl':
                                               input_hashes[f'{split}/stimuli.jsonl'] for split in CASES}}):
        raise ValueError('Matched inputs and features do not prove the same freeze and original L4 bound checkpoint')
    original_payload = original_job / 'payload.tar.gz'
    if worker.digest(original_payload) != original['payload_sha256']:
        raise ValueError('The official source payload no longer matches its archive receipt')
    with tarfile.open(original_payload, 'r:gz') as archive:
        code = {name: archive.extractfile(name).read() for name in worker.CODE}
    source_hashes = {name: hashlib.sha256(body).hexdigest() for name, body in code.items()}
    if any(value != original['payload_members_sha256'][name] for name, value in source_hashes.items()):
        raise ValueError('Matched fit source differs from its official archived bytes')
    for name, body in code.items():
        compile(body, name, 'exec')

    run_id = f'cogllm-iirc-matched-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}'
    job = output / run_id
    job.mkdir(mode=0o700, parents=True)
    features = copy_parts(replay / 'features.npz', job)
    if features['sha256'] != extraction['features_sha256']:
        raise ValueError('The matched replay feature archive differs from its extraction receipt')
    plan = {'source_run_id': original['run_id'], 'source_checkpoint_sha256': checkpoint_sha,
            'input_archive_sha256': worker.digest(packet), 'inputs_sha256': input_hashes, 'features': features,
            'extraction_sha256': worker.digest(extraction_path), 'source_sha256': source_hashes}
    worker.write_json(job / 'bundle.json', plan)
    shutil.copyfile(worker.__file__, job / 'worker.py')
    payload = job / 'payload.tar.gz'
    with tarfile.open(payload, 'w:gz', compresslevel=1) as archive:
        for name, body in sorted(code.items()):
            member = tarfile.TarInfo(name)
            member.size = len(body)
            archive.addfile(member, BytesIO(body))
        for name, path in {'bundle.json': job / 'bundle.json', 'fit-inputs.tar.gz': packet,
                           'extraction.json': extraction_path}.items():
            archive.add(path, arcname=name, recursive=False)
    data = {'run_id': run_id, 'schema_version': 1, 'phase': 'prepared', 'timeline': {'prepared': owned.utc()},
            'runtime_budget_s': worker.MAX_SECONDS, 'requested_gpu': 'L4', 'fit_backend': 'cpu', 'plan': plan,
            'payload_sha256': worker.digest(payload), 'worker_sha256': worker.digest(job / 'worker.py')}
    worker.write_json(job / 'manifest.json', data)
    verify(job, data)
    print(job.resolve(), flush=True)


def verify(job, data):
    plan = worker.read_json(job / 'bundle.json')
    payload = job / 'payload.tar.gz'
    hashes = worker.inspect(payload, worker.PAYLOAD)
    if (not worker.RUN_ID.fullmatch(job.name) or data['run_id'] != job.name or data['schema_version'] != 1
            or data['runtime_budget_s'] != worker.MAX_SECONDS or data['fit_backend'] != 'cpu'
            or plan != data['plan'] or hashes['bundle.json'] != worker.digest(job / 'bundle.json')
            or hashes['fit-inputs.tar.gz'] != plan['input_archive_sha256'] or hashes['extraction.json'] != plan['extraction_sha256']
            or {name: hashes[name] for name in worker.CODE} != plan['source_sha256']
            or worker.digest(payload) != data['payload_sha256'] or worker.digest(job / 'worker.py') != data['worker_sha256']):
        raise ValueError('Prepared matched fit files differ from sealed metadata')
    features = plan['features']
    files = features['parts']
    feature_hash = hashlib.sha256()
    if (not files or sum(item['bytes'] for item in files) != features['bytes']
            or any(item['bytes'] != worker.PART_BYTES for item in files[:-1])):
        raise ValueError('Matched feature transfer has an invalid part inventory')
    for index, item in enumerate(files):
        path = job / worker.part_name(index)
        if not 0 < path.stat().st_size == item['bytes'] <= worker.PART_BYTES:
            raise ValueError('Matched feature transfer part size differs from the prepared seal')
        body = path.read_bytes()
        if hashlib.sha256(body).hexdigest() != item['sha256']:
            raise ValueError('Matched feature transfer part digest differs from the prepared seal')
        feature_hash.update(body)
    if feature_hash.hexdigest() != features['sha256']:
        raise ValueError('The matched feature parts do not reconstruct the sealed replay archive')


def verify_results(job, data, status):
    plan = data['plan']
    result = job / 'results.tar.gz'
    if (not status['completed'] or any(status[key] != data[key] for key in ('run_id', 'payload_sha256', 'worker_sha256'))
            or not 0 < result.stat().st_size <= worker.MAX_RETURN or worker.digest(result) != status['archive_sha256']
            or worker.inspect(result, worker.OUTPUTS) != status['output_sha256']):
        raise ValueError('The returned matched fit archive does not match the remote receipt')
    output = job / 'outputs'
    output.mkdir()
    with tarfile.open(result, 'r:gz') as archive:
        archive.extractall(output, filter='data')
    runtime = worker.read_json(output / 'runtime.json')
    fit = worker.read_json(output / 'fit/fit-manifest.json')
    hashes = plan['inputs_sha256']
    if (runtime['device'] != 'NVIDIA L4' or runtime['fit_backend'] != 'cpu'
            or runtime['source_sha256'] != plan['source_sha256']
            or runtime['source_checkpoint_sha256'] != plan['source_checkpoint_sha256']
            or runtime['packages'] != dict(item.split('==') for item in worker.REQUIREMENTS)
            or (fit['model'], fit['revision']) != (linked.MODEL, linked.REVISION)
            or fit['assay_sha256'] != hashes['assay.json'] or fit['extraction_sha256'] != plan['extraction_sha256']
            or fit['features_sha256'] != plan['features']['sha256']
            or fit['inputs_sha256'] != {key: value for key, value in hashes.items() if key != 'assay.json'}
            or fit['fit_paths'] != CASES['train'] or fit['heldout_forecast_paths'] != CASES['dev']
            or fit['fit_dtype'] != 'float64' or fit['optimizer']['name'] != 'L-BFGS-B'
            or set(fit['fits']) != METHODS or fit['heads_sha256'] != status['output_sha256']['fit/heads.npz']
            or fit['predictions_sha256'] != status['output_sha256']['fit/predictions.jsonl']):
        raise ValueError('The matched fit does not bind the frozen inputs or fitted remote runtime')
    with ZipFile(output / 'fit/heads.npz') as heads:
        entries = {f'{name}.npy' for name in METHODS | {'projection', 'geometry_center', 'geometry_scale'}}
        if len(heads.namelist()) != len(entries) or set(heads.namelist()) != entries or heads.testzip() is not None:
            raise ValueError('The matched head array inventory differs from the fitted methods')
    with tarfile.open(job / 'payload.tar.gz', 'r:gz') as payload, payload.extractfile('fit-inputs.tar.gz') as part:
        with tarfile.open(fileobj=part, mode='r|gz') as inputs:
            for member in inputs:
                if member.name == 'dev/stimuli.jsonl':
                    with inputs.extractfile(member) as rows:
                        expected = [(row['id'], len(row['stops']) - 1) for row in map(json.loads, rows)]
                    break
            else:
                raise ValueError('The matched held-out stop inventory is absent')
    with (output / 'fit/predictions.jsonl').open(encoding='utf-8') as predictions:
        for (identifier, count), row in zip(expected, map(json.loads, predictions), strict=True):
            if (row['case_id'] != identifier or row['methods'].keys() != METHODS
                    or any(len(values) != count or any(not math.isfinite(value) or not 0 < value < 1 for value in values)
                           for values in row['methods'].values())):
                raise ValueError('The matched hazards do not match the exact held-out stop inventory')


def release(machine):
    machine.update('releasing')
    machine.usage('before-stop', timeout=12)
    data = machine.data
    data['endpoint_absent_after_stop'] = machine.release()
    data['release_usage_recorded'] = machine.usage('after-stop')
    data['release_verified'] = data['endpoint_absent_after_stop'] and data['release_usage_recorded']


def run_owned(machine):
    job, data = machine.job, machine.data
    attempted = False

    def interrupt(signum, _):
        signal.signal(signum, signal.SIG_IGN)
        raise KeyboardInterrupt(f'Matched fit interrupted by signal {signum}')

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupt)
    try:
        if not re.search(rf'^Version: {re.escape(owned.CLI_VERSION)}$', machine.cli('version', timeout=20).stdout, re.M):
            raise ValueError(f'Expected Colab CLI {owned.CLI_VERSION}')
        before = machine.sessions()
        if machine.endpoint(before) is not None:
            raise ValueError('The matched fit runtime name is already owned')
        data['endpoints_before_allocation'] = sorted(before)
        machine.usage('before-allocation', required=True)
        machine.update('allocating')
        deadline = monotonic() + worker.MAX_SECONDS - owned.RELEASE_RESERVE_S
        attempted = True
        machine.cli('new', '-s', job.name, '--gpu', 'L4', timeout=75)
        endpoint = machine.endpoint(machine.sessions())
        if not endpoint or endpoint in before:
            raise RuntimeError('Unable to identify the owned matched L4 runtime')
        data['endpoint'] = endpoint
        machine.update('allocated')
        machine.usage('after-allocation')
        transfers = [('payload.tar.gz', '.tar.gz'), ('worker.py', '.worker.py')]
        transfers.extend((name, f'.{name}') for name in map(worker.part_name, range(len(data['plan']['features']['parts']))))
        for local, suffix in transfers:
            remaining = deadline - monotonic() - 90
            if remaining <= 0:
                raise TimeoutError('Matched fit transfer reached the reserved cleanup window')
            machine.cli('upload', '-s', job.name, job / local, f'{machine.remote}{suffix}', timeout=min(60, remaining))
        machine.launch(deadline)
        machine.update('worker-running')
        status = machine.wait(deadline)
        if not machine.download('/results.tar.gz', 'results.tar.gz', timeout=max(1, min(25, deadline - monotonic()))):
            raise RuntimeError('Could not collect the sealed matched fit')
        verify_results(job, data, status)
        data['artifacts_verified'] = True
    except (KeyError, ValueError, TypeError, OSError, RuntimeError, subprocess.SubprocessError,
            tarfile.TarError, BadZipFile, KeyboardInterrupt) as error:
        data['failure'] = str(error) or type(error).__name__
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, signal.SIG_IGN)
        if attempted:
            release(machine)
        ok = 'failure' not in data and data.get('artifacts_verified') and data.get('release_verified')
        machine.update('finished' if ok else 'failed')
    return int(not ok)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pack = commands.add_parser('prepare')
    for name in ('prepared', 'replay', 'original-job', 'output'):
        pack.add_argument(f'--{name}', required=True, type=Path)
    for name in ('verify', 'run', 'release'):
        commands.add_parser(name).add_argument('job', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.prepared, args.replay, args.original_job, args.output)
    else:
        job = args.job.resolve()
        data = worker.read_json(job / 'manifest.json')
        if args.action == 'release':
            with owned.run_lock(job, 'r+'):
                data = worker.read_json(job / 'manifest.json')
                if not worker.RUN_ID.fullmatch(job.name) or data['run_id'] != job.name or 'allocating' not in data['timeline']:
                    raise ValueError('No matched L4 allocation to recover')
                machine = owned.Colab(job, data)
                release(machine)
                machine.update('release-verified' if data['release_verified'] else 'release-unverified')
                return int(not data['release_verified'])
        verify(job, data)
        if args.action == 'run':
            with owned.run_lock(job, 'x'):
                return run_owned(owned.Colab(job, data))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
