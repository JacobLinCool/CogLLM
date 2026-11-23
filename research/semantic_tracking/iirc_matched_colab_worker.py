"""Run the frozen, CPU numerical matched-IIRC fit on one owned Colab L4."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from time import monotonic


RUN_ID = re.compile(r'cogllm-iirc-matched-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}')
CODE = frozenset(f'research/{name}.py' for name in (
    'semantic_tracking/iirc_linked_fit', 'semantic_tracking/iirc_linked_pack', 'semantic_tracking/iirc_prepare',
    'stage2/records', 'stage3/probe', 'stage3/run',
))
INPUTS = frozenset({'assay.json', 'train/intervals.jsonl', 'train/stimuli.jsonl', 'train/queries.jsonl',
                    'dev/stimuli.jsonl', 'dev/queries.jsonl'})
PAYLOAD = CODE | {'bundle.json', 'fit-inputs.tar.gz', 'extraction.json'}
OUTPUTS = {'runtime.json', 'fit/heads.npz', 'fit/predictions.jsonl', 'fit/fit-manifest.json', 'worker.log'}
REQUIREMENTS = ('numpy==2.4.6', 'scipy==1.16.3')
MAX_SECONDS = 15 * 60
MAX_RETURN = 16 * 1024 * 1024
PART_BYTES = 8 * 1024 * 1024


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, data):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def inspect(path, expected, destination=None):
    with tarfile.open(path, 'r:gz') as archive:
        files = archive.getmembers()
        if ({file.name for file in files} != set(expected) or len(files) != len(expected)
                or any(not file.isfile() for file in files)):
            raise ValueError(f'Unexpected archive members: {path.name}')
        hashes = {}
        for file in files:
            with archive.extractfile(file) as stream:
                hashes[file.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        if destination is not None:
            archive.extractall(destination, filter='data')
    return hashes


def part_name(index):
    return f'feature-{index:04d}.part'


def assemble_features(root, spec):
    target = root / 'features.npz'
    with target.open('wb') as output:
        for index, item in enumerate(spec['parts']):
            path = root.with_suffix(f'.{part_name(index)}')
            if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise ValueError(f'Matched feature transfer part {index} failed verification')
            with path.open('rb') as part:
                shutil.copyfileobj(part, output)
    if target.stat().st_size != spec['bytes'] or digest(target) != spec['sha256']:
        raise ValueError('The reassembled matched feature archive differs from its sealed bytes')


def device():
    import torch

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4':
        raise RuntimeError('Expected one owned NVIDIA L4 allocation')
    return {'device': torch.cuda.get_device_name(0), 'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'fit_backend': 'cpu'}


def invoke(root, args, env, log, deadline, limit):
    remaining = min(limit, deadline - monotonic())
    if remaining <= 0:
        raise TimeoutError('Matched fit exceeded its runtime deadline')
    subprocess.run([sys.executable, *args], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                   timeout=remaining, check=True)


def run(root, payload_sha256, worker_sha256, seconds):
    os.umask(0o077)
    if not RUN_ID.fullmatch(root.name) or root.parent != Path('/content') or not 0 < seconds <= MAX_SECONDS:
        raise ValueError('Invalid owned matched fit configuration')
    root.mkdir()
    deadline = monotonic() + seconds
    status = {'run_id': root.name, 'payload_sha256': payload_sha256, 'worker_sha256': worker_sha256,
              'phase': 'setup', 'finished': False, 'completed': False}
    status_path = root / 'status.json'
    write_json(status_path, status)
    try:
        payload = root.with_suffix('.tar.gz')
        if digest(payload) != payload_sha256 or digest(Path(__file__)) != worker_sha256:
            raise ValueError('Matched payload or worker digest differs from the local seal')
        hashes = inspect(payload, PAYLOAD, root)
        plan = read_json(root / 'bundle.json')
        if ({name: hashes[name] for name in CODE} != plan['source_sha256']
                or hashes['fit-inputs.tar.gz'] != plan['input_archive_sha256']
                or hashes['extraction.json'] != plan['extraction_sha256']
                or inspect(root / 'fit-inputs.tar.gz', INPUTS, root / 'inputs') != plan['inputs_sha256']):
            raise ValueError('Matched fit materials differ from the sealed plan')
        assemble_features(root, plan['features'])
        hardware = device()
        env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
               'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
        with (root / 'worker.log').open('w', encoding='utf-8') as log:
            invoke(root, ('-m', 'pip', 'install', '--quiet', *REQUIREMENTS), env, log, deadline, 110)
            packages = {name: version(name) for name in ('numpy', 'scipy')}
            write_json(root / 'runtime.json', {**hardware, 'packages': packages, 'source_sha256': plan['source_sha256'],
                                               'source_checkpoint_sha256': plan['source_checkpoint_sha256']})
            status['phase'] = 'matched-fit'
            write_json(status_path, status)
            invoke(root, ('-m', 'research.semantic_tracking.iirc_linked_fit', '--inputs', 'inputs',
                          '--features', 'features.npz', '--extraction', 'extraction.json', '--output', 'fit'),
                   env, log, deadline, 240)
        with tarfile.open(root / 'results.tar.gz', 'w:gz', compresslevel=1) as archive:
            for name in sorted(OUTPUTS):
                archive.add(root / name, arcname=name, recursive=False)
        results = root / 'results.tar.gz'
        if results.stat().st_size > MAX_RETURN:
            raise ValueError('Matched fit results exceed the bounded transfer size')
        status.update(phase='sealed', completed=True, archive_sha256=digest(results),
                      output_sha256={name: digest(root / name) for name in OUTPUTS})
    except Exception as error:
        status['error_type'] = type(error).__name__
        status['stage_log_tail'] = str(error)
        raise
    finally:
        status['finished'] = True
        write_json(status_path, status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--payload-sha256', required=True)
    parser.add_argument('--worker-sha256', required=True)
    parser.add_argument('--seconds', required=True, type=int)
    run(**vars(parser.parse_args()))
