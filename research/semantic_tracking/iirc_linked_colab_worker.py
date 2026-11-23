"""Run the sealed linked-evidence assay inside its owned Colab L4 runtime."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from threading import Event, Thread
from time import monotonic
import traceback
from zipfile import ZIP_STORED, ZipFile


DATA = frozenset({'assay.json', 'train/stimuli.jsonl', 'train/queries.jsonl', 'train/intervals.jsonl',
                  'dev/stimuli.jsonl', 'dev/queries.jsonl'})
CODE = frozenset(f'research/{name}.py' for name in (
    'core_features/extract', 'core_features/fit', 'semantic_tracking/iirc_linked_extract',
    'semantic_tracking/iirc_linked_fit', 'semantic_tracking/iirc_linked_pack', 'semantic_tracking/iirc_prepare',
    'semantic_tracking/joint_model', 'semantic_tracking/qasper_fit', 'stage2/records', 'stage3/probe', 'stage3/run',
))
PAYLOAD = CODE | {'inputs.tar.gz'}
OUTPUTS = frozenset({'features/plan.json', 'features/causality.json', 'features/progress.json',
                     'features/extraction.json', 'fit/heads.npz', 'fit/predictions.jsonl',
                     'fit/fit-manifest.json', 'worker.log'})
TELEMETRY = re.compile(r'features/telemetry-shard-[0-9]{4}\.jsonl')
METHODS = ('geometry', 'path_text', 'anchor_conditioned')
MODEL = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
TRANSFORMERS = '5.17.0'
RUN_ID = re.compile(r'cogllm-iirc-linked-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}')
RESULTS = 'results.tar.gz'
MAX_RETURN_BYTES = 16 * 1024 * 1024
MAX_RUNTIME_S = 75 * 60
CHECKPOINT_META = frozenset({'receipt.json', 'plan.json', 'causality.json'})
SHARD_ARRAYS = frozenset(f'{name}.npy' for name in ('signature', 'ids', 'stops', 'states', 'forward_s', 'peak_reserved_bytes'))
SHA256 = re.compile(r'[0-9a-f]{64}')


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def inspect_archive(path, expected, destination=None):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        names = {member.name for member in members}
        if (names != expected or len(names) != len(members) or not all(member.isfile() for member in members)):
            raise ValueError(f'Archive members differ from the allowlist: {path.name}')
        hashes = {}
        for member in members:
            with archive.extractfile(member) as handle:
                hashes[member.name] = hashlib.file_digest(handle, 'sha256').hexdigest()
        if destination is not None:
            archive.extractall(destination, filter='data')
    return hashes


def shard_name(number):
    return f'shard-{number:04d}.npz'


def check_checkpoint(receipt):
    if (set(receipt) != {'run_id', 'assay_sha256', 'plan_sha256', 'causality_sha256', 'shards_total', 'shards'}
            or not RUN_ID.fullmatch(receipt['run_id']) or not all(SHA256.fullmatch(receipt[name]) for name in
                ('assay_sha256', 'plan_sha256', 'causality_sha256'))
            or type(receipt['shards_total']) is not int or not 1 <= len(receipt['shards']) <= receipt['shards_total'] <= 9999):
        raise ValueError('Invalid linked checkpoint receipt')
    for index, item in enumerate(receipt['shards']):
        if (set(item) != {'number', 'bytes', 'sha256'} or type(item['number']) is not int or item['number'] != index
                or type(item['bytes']) is not int or not 0 < item['bytes'] <= MAX_RETURN_BYTES
                or not SHA256.fullmatch(item['sha256'])):
            raise ValueError('Checkpoint receipt must describe an exact contiguous shard prefix')


def check_proofs(directory, receipt, assay_sha, source_hashes, data_hashes):
    check_checkpoint(receipt)
    plan_path, causal_path = directory / 'plan.json', directory / 'causality.json'
    plan, causal = read_json(plan_path), read_json(causal_path)
    if (receipt['assay_sha256'] != assay_sha or digest(plan_path) != receipt['plan_sha256']
            or digest(causal_path) != receipt['causality_sha256'] or causal['plan_sha256'] != receipt['plan_sha256']
            or plan['assay_sha256'] != assay_sha or plan['shards'] != receipt['shards_total']
            or plan['model'] != MODEL or plan['revision'] != REVISION or plan['transformers'] != TRANSFORMERS
            or plan['gpu'] != 'NVIDIA L4' or plan['dtype'] != 'float32'
            or plan['code_sha256'] != {name.rsplit('/', 1)[-1]: source_hashes[name] for name in (
                'research/semantic_tracking/iirc_linked_extract.py', 'research/core_features/extract.py')}
            or plan['inputs_sha256'] != {'assay': assay_sha, **{name: data_hashes[name] for name in (
                'train/stimuli.jsonl', 'dev/stimuli.jsonl')}}
            or len(causal['checks']) != 12 or not all(row['passed'] and row['projected_tokens']['passed']
                                                    for row in causal['checks'])):
        raise ValueError('Checkpoint plan, causal controls or extraction input/source digests differ')


def check_shard(path, receipt):
    if path.stat().st_size != receipt['bytes'] or digest(path) != receipt['sha256']:
        raise ValueError(f'Checkpoint byte count or digest differs: {path.name}')
    with ZipFile(path) as arrays:
        entries = arrays.infolist()
        if (len(entries) != len(SHARD_ARRAYS) or {item.filename for item in entries} != SHARD_ARRAYS
                or any(item.compress_type != ZIP_STORED for item in entries) or arrays.testzip() is not None):
            raise ValueError(f'Checkpoint arrays differ from the extractor format: {path.name}')


def set_proofs(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('plan.json', 'causality.json'):
        shutil.copyfile(source / name, destination / name)


def restore(root, resume_sha256, status):
    archive = root.with_suffix('.resume.tar.gz')
    if digest(archive) != resume_sha256:
        raise ValueError('Resume metadata archive differs from the prepared digest')
    proofs = root / 'checkpoint-proofs'
    inspect_archive(archive, CHECKPOINT_META, proofs)
    receipt = read_json(proofs / 'receipt.json')
    check_proofs(proofs, receipt, status['data_members_sha256']['assay.json'],
                 status['payload_members_sha256'], status['data_members_sha256'])
    features = root / 'features'
    set_proofs(proofs, features)
    (features / 'shards').mkdir()
    for item in receipt['shards']:
        name = shard_name(item['number'])
        source = root.with_suffix(f'.resume-{name}')
        check_shard(source, item)
        source.replace(features / 'shards' / name)
    status['resumed_from'] = {'run_id': receipt['run_id'], 'plan_sha256': receipt['plan_sha256'],
                              'shards': len(receipt['shards']), 'metadata_sha256': resume_sha256}
    receipt['run_id'] = root.name
    write_json(root / 'checkpoint-index.json', receipt)


def publish_checkpoints(root, status):
    features = root / 'features'
    progress_path = features / 'progress.json'
    if not progress_path.is_file():
        return
    progress = read_json(progress_path)
    if not progress['shards_completed']:
        return
    index_path = root / 'checkpoint-index.json'
    proofs = root / 'checkpoint-proofs'
    if index_path.is_file():
        receipt = read_json(index_path)
    else:
        set_proofs(features, proofs)
        receipt = {'run_id': root.name, 'assay_sha256': status['data_members_sha256']['assay.json'],
                   'plan_sha256': digest(proofs / 'plan.json'), 'causality_sha256': digest(proofs / 'causality.json'),
                   'shards_total': progress['shards'], 'shards': []}
    if progress['plan_sha256'] != receipt['plan_sha256'] or progress['shards'] != receipt['shards_total']:
        raise ValueError('Worker progress changed its extraction plan')
    if progress['shards_completed'] <= len(receipt['shards']):
        return
    for number in range(len(receipt['shards']), progress['shards_completed']):
        path = features / 'shards' / shard_name(number)
        receipt['shards'].append({'number': number, 'bytes': path.stat().st_size, 'sha256': digest(path)})
    check_proofs(proofs, receipt, status['data_members_sha256']['assay.json'],
                 status['payload_members_sha256'], status['data_members_sha256'])
    write_json(index_path, receipt)


def watch_checkpoints(root, status, stop, errors):
    try:
        while not stop.wait(2):
            publish_checkpoints(root, status)
    except Exception as error:
        errors.append(error)
        write_json(root / 'checkpoint-error.json', {'run_id': root.name, 'message': str(error)})


def preflight():
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected one CUDA device')
    gpu = torch.cuda.get_device_properties(0)
    if gpu.name != 'NVIDIA L4' or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError(f'Expected a Colab L4 with native BF16, got {gpu.name}')
    return {'name': gpu.name, 'bytes': gpu.total_memory, 'torch': torch.__version__, 'cuda': torch.version.cuda}


def run(root, payload_sha256, worker_sha256, seconds, resume_sha256=None):
    os.umask(0o077)
    if not RUN_ID.fullmatch(root.name) or root.parent != Path('/content') or not 0 < seconds <= MAX_RUNTIME_S:
        raise ValueError('Invalid linked-evidence worker configuration')
    started = monotonic()
    root.mkdir()
    status_path = root / 'status.json'
    worker_log = root / 'worker.log'
    status = {'run_id': root.name, 'phase': 'preflight', 'finished': False, 'completed': False,
              'phase_seconds': {}}
    write_json(status_path, status)
    try:
        payload = root.with_suffix('.tar.gz')
        if digest(payload) != payload_sha256 or digest(Path(__file__)) != worker_sha256:
            raise ValueError('Remote payload or worker differs from the prepared files')
        status.update(payload_sha256=payload_sha256, worker_sha256=worker_sha256, device=preflight())
        status['payload_members_sha256'] = inspect_archive(payload, PAYLOAD, root)
        status['data_members_sha256'] = inspect_archive(root / 'inputs.tar.gz', DATA, root / 'inputs')
        assay = read_json(root / 'inputs/assay.json')
        if (assay['model'] != MODEL or assay['revision'] != REVISION or
                {name: status['data_members_sha256'][name] for name in DATA - {'assay.json'}} != assay['inputs_sha256']):
            raise ValueError('Remote data members disagree with the sealed assay')
        if resume_sha256 is not None:
            restore(root, resume_sha256, status)

        environment = {**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                       'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1'}
        stages = (
            ('install', 120, ('-m', 'pip', 'install', '--quiet', f'transformers=={TRANSFORMERS}')),
            ('extract', seconds, ('-m', 'research.semantic_tracking.iirc_linked_extract',
                                  '--inputs', 'inputs', '--output', 'features')),
            ('fit', 180, ('-m', 'research.semantic_tracking.iirc_linked_fit', '--inputs', 'inputs',
                          '--features', 'features/features.npz', '--extraction', 'features/extraction.json',
                          '--output', 'fit')),
        )
        with worker_log.open('w', encoding='utf-8') as log:
            for phase, maximum, command in stages:
                status['phase'] = phase
                write_json(status_path, status)
                stage_start = monotonic()
                timeout = min(maximum, started + seconds - stage_start)
                if timeout <= 0:
                    raise TimeoutError('The linked-evidence worker reached its deadline')
                stop = Event()
                errors = []
                observer = None
                if phase == 'extract':
                    observer = Thread(target=watch_checkpoints, args=(root, status, stop, errors))
                    observer.start()
                try:
                    subprocess.run([sys.executable, *command], cwd=root, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, timeout=timeout, check=True)
                finally:
                    if observer is not None:
                        stop.set()
                        observer.join()
                        if errors:
                            raise RuntimeError('Checkpoint publication failed') from errors[0]
                        publish_checkpoints(root, status)
                    status['phase_seconds'][phase] = monotonic() - stage_start

        outputs = OUTPUTS | {str(path.relative_to(root)) for path in (root / 'features').glob('telemetry-shard-*.jsonl')}
        status['output_sha256'] = {name: digest(root / name) for name in sorted(outputs)}
        archive_path = root / RESULTS
        partial = archive_path.with_suffix('.pending')
        with tarfile.open(partial, 'w:gz', compresslevel=1) as archive:
            for name in sorted(outputs):
                archive.add(root / name, arcname=name, recursive=False)
        archive_path = partial.replace(archive_path)
        if archive_path.stat().st_size > MAX_RETURN_BYTES:
            raise ValueError('Result archive exceeds the return size limit')
        status.update(completed=True, phase='complete', archive_sha256=digest(archive_path))
    except BaseException as error:
        status['error_type'] = type(error).__name__
        with worker_log.open('a+b') as log:
            size = log.seek(0, os.SEEK_END)
            log.seek(max(0, size - 4096))
            status['stage_log_tail'] = log.read().decode('utf-8', errors='replace')
        with worker_log.open('a', encoding='utf-8') as log:
            traceback.print_exc(file=log)
        raise
    finally:
        status.update(finished=True, elapsed_s=monotonic() - started)
        write_json(status_path, status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--payload-sha256', required=True)
    parser.add_argument('--worker-sha256', required=True)
    parser.add_argument('--seconds', type=int, required=True)
    parser.add_argument('--resume-sha256')
    run(**vars(parser.parse_args()))
