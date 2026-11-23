"""Fit the sealed StudentAnnotations and AmbiStory readouts on one owned L4."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from time import monotonic


RUN_ID = re.compile(r'cogllm-ordinal-fits-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}')
MAX_SECONDS = 18 * 60
MAX_RETURN = 16 * 1024 * 1024
REQUIREMENTS = ('numpy==2.4.6', 'scipy==1.16.3')
STAGES = ('student-annotations', 'ambistory')
EFFECT_STAGE = 'ambistory-effect'
KERWER_STAGE = 'kerwer-background'
KERWER_CODE = frozenset({'research/stage6/kerwer_reader_background_fit.py', 'research/stage6/kerwer_proportional_odds.py'})
KERWER_PACKET = f'{KERWER_STAGE}/fit-packet.json'
WEIGHTS = {'student-annotations': 'fold-heads.npz', 'ambistory': 'heads.npz'}
CODE = frozenset(f'research/{name}' for name in (
    'stage6/student_annotations_fit.py', 'stage6/ambistory_fit.py', 'stage6/ambistory_extract.py',
    'stage6/student-annotations-background.md', 'stage6/ambistory-context-update.md',
    'stage6/ambistory-numerical-amendment.md',
    'stage3/causal_ordinal.py', 'stage3/causal_export.py', 'stage3/probe.py', 'stage3/run.py',
    'core_features/readback_score.py', 'core_features/extract.py', 'core_features/fit.py',
    'semantic_tracking/iirc_prepare.py', 'semantic_tracking/joint_model.py', 'semantic_tracking/qasper_fit.py',
    'stage2/records.py',
))
STUDENT_PREPARED = frozenset(f'student-annotations/prepared/{name}' for name in (
    'preparation-receipt.json', 'model_input/stimuli.jsonl', 'model_input/certified-stops.jsonl',
    'model_input/extraction-plan.json', 'local/trial-design.jsonl', 'local/readers.jsonl', 'local/sentences.jsonl',
    'local/outcomes.jsonl',
))
AMBISTORY_PREPARED = frozenset(f'ambistory/prepared/{split}/{name}' for split in ('train', 'dev', 'test')
                               for name in ('model_input/stimuli.jsonl', 'design/stops.jsonl')) | {
                                   'ambistory/prepared/train/scoring/ordinal-ratings.jsonl', 'ambistory/preparation.json',
                               }
EFFECT_PREPARED = frozenset(name.replace('ambistory/', f'{EFFECT_STAGE}/staged/', 1) for name in AMBISTORY_PREPARED)
EFFECT_CODE = frozenset({'research/stage6/ambistory_effect_fit.py', 'research/stage6/ambistory-effect-supervision.md',
                        'research/results/ambistory-fit-protocol-v2.json'})
PREPARED = {'student-annotations': STUDENT_PREPARED, 'ambistory': AMBISTORY_PREPARED, EFFECT_STAGE: EFFECT_PREPARED}


def scope_members(stages):
    if tuple(stages) == (KERWER_STAGE,):
        return KERWER_CODE | {'bundle.json', KERWER_PACKET}
    if tuple(stages) not in (STAGES, ('ambistory',), ('student-annotations',), (EFFECT_STAGE,)):
        raise ValueError('Unexpected ordinal stage scope')
    code = CODE | EFFECT_CODE if EFFECT_STAGE in stages else CODE
    return code | {'bundle.json', *(f'{stage}/fit-protocol.json' for stage in stages)} | frozenset.union(
        *(PREPARED[stage] for stage in stages))


def digest(path):
    with path.open('rb') as contents:
        return hashlib.file_digest(contents, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def unpack(path, expected, destination=None):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        if ({member.name for member in members} != set(expected) or len(members) != len(expected)
                or not all(member.isfile() for member in members)):
            raise ValueError(f'Unexpected archive members: {path.name}')
        hashes = {}
        for member in members:
            with archive.extractfile(member) as content:
                hashes[member.name] = hashlib.file_digest(content, 'sha256').hexdigest()
        if destination is not None:
            archive.extractall(destination, filter='data')
        return hashes


def extraction_stage(stage):
    return 'ambistory' if stage == EFFECT_STAGE else stage


def fit_files(stage, outcome=None):
    if stage == KERWER_STAGE:
        return {'fit-seal.json', 'predictions.jsonl', 'heads.json'}
    if stage == EFFECT_STAGE:
        files = {'fit-seal.json', 'train-diagnostics.json'}
        if outcome != 'no_candidate':
            files.update({'baseline-predictions.jsonl', 'effect-predictions.jsonl', 'heads.npz'})
        return files
    return {'fit-seal.json', 'predictions.jsonl', WEIGHTS[stage]}


def output_files(stage, outcome=None):
    return {'runtime.json', 'run.log', *fit_files(stage, outcome)}


def device():
    import torch

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4':
        raise RuntimeError('Expected a single NVIDIA L4 allocation')
    return {'device': 'NVIDIA L4', 'cuda': torch.version.cuda, 'torch': torch.__version__, 'fit_backend': 'cpu',
            'vram_bytes': torch.cuda.get_device_properties(0).total_memory}


def invoke(root, command, env, log, deadline, limit):
    seconds = min(limit, deadline - monotonic())
    if seconds <= 0:
        raise TimeoutError('The ordinal fit reached its bounded deadline')
    subprocess.run([sys.executable, *command], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                   timeout=seconds, check=True)


def fit_command(stage, study):
    if stage == KERWER_STAGE:
        packet = study / 'fit-packet.json'
        return ('-m', 'research.stage6.kerwer_reader_background_fit', 'fit', '--packet', str(packet),
                '--checksum', digest(packet), '--output', str(study / 'fit'))
    command = ('-m', f"research.stage6.{stage.replace('-', '_')}_fit", 'fit',
               '--extraction', str(study / 'extraction'), '--output', str(study / 'fit'))
    if stage == EFFECT_STAGE:
        return command + ('--staged', str(study / 'staged'), '--protocol', str(study / 'fit-protocol.json'))
    command += ('--prepared', str(study / 'prepared'), '--fit-protocol', str(study / 'fit-protocol.json'))
    if stage == 'ambistory':
        command += ('--preparation', str(study / 'preparation.json'))
    return command


def fit_stage(root, stage, status, plan, env, deadline):
    study = root / stage
    log_path = study / 'stage.log'
    receipt = {'state': 'running'}
    status['phase'] = f'{stage}-fit'
    status['stages'][stage] = receipt
    write(root / 'status.json', status)
    start = monotonic()
    with log_path.open('w', encoding='utf-8') as log:
        try:
            if stage in plan['extractions']:
                expected = plan['extractions'][stage]
                archive = root.with_suffix(f'.{stage}.tar.gz')
                if digest(archive) != expected['archive_sha256'] or unpack(archive, expected['members'], study) != expected['members']:
                    raise ValueError(f'{stage} extraction archive differs from its verified V11 receipt')
            fit_log = study / 'fit.log'
            with fit_log.open('w', encoding='utf-8') as fit_output:
                limit = {'student-annotations': 480, 'ambistory': 135, EFFECT_STAGE: 480, KERWER_STAGE: 300}[stage]
                invoke(root, fit_command(stage, study), env, fit_output, deadline, limit)
            if stage == EFFECT_STAGE:
                outcome = read(study / 'fit/fit-seal.json')['status']
                if outcome not in {'no_candidate', 'effect_and_baseline_forecasts_locked_before_dev_scoring'}:
                    raise ValueError('Unexpected AmbiStory effect fit outcome')
                receipt['fit_outcome'] = outcome
            receipt['state'] = 'complete'
        except Exception as error:
            receipt.update(state='failed', error_type=type(error).__name__)
            print(error, file=log)
    logs = study / 'fit.log'
    members = {'runtime.json': root / 'runtime.json', 'run.log': logs if logs.is_file() else log_path}
    if receipt['state'] == 'complete':
        members.update({name: study / 'fit' / name for name in fit_files(stage, receipt.get('fit_outcome'))})
    path = root / 'return' / f'{stage}.tar.gz'
    with tarfile.open(path, 'w:gz', compresslevel=1) as archive:
        for name, item in members.items():
            archive.add(item, arcname=name, recursive=False)
    if path.stat().st_size > MAX_RETURN:
        raise ValueError(f'{stage} fit output exceeds the bounded return transfer')
    receipt.update(sha256=digest(path), bytes=path.stat().st_size, members={name: digest(item) for name, item in members.items()},
                   elapsed_s=monotonic() - start)
    write(root / 'status.json', status)


def run(root, payload_sha256, worker_sha256, seconds):
    os.umask(0o077)
    if not RUN_ID.fullmatch(root.name) or root.parent != Path('/content') or not 0 < seconds <= MAX_SECONDS:
        raise ValueError('Invalid owned ordinal fit configuration')
    root.mkdir()
    (root / 'return').mkdir()
    deadline = monotonic() + seconds
    status = {'run_id': root.name, 'payload_sha256': payload_sha256, 'worker_sha256': worker_sha256,
              'phase': 'setup', 'finished': False, 'stages': {}}
    write(root / 'status.json', status)
    try:
        payload = root.with_suffix('.tar.gz')
        if digest(payload) != payload_sha256 or digest(Path(__file__)) != worker_sha256:
            raise ValueError('Ordinal fit payload or worker differs from its local seal')
        with tarfile.open(payload, 'r:gz') as bundle, bundle.extractfile('bundle.json') as stream:
            plan = json.load(stream)
        stages = plan['stages']
        hashes = unpack(payload, scope_members(stages), root)
        if {name: value for name, value in hashes.items() if name != 'bundle.json'} != plan['materials_sha256']:
            raise ValueError('Ordinal fit materials differ from the sealed fit plan')
        status['stages'] = {stage: {'state': 'pending'} for stage in stages}
        write(root / 'status.json', status)
        hardware = device()
        env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1',
               'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
        with (root / 'setup.log').open('w', encoding='utf-8') as log:
            invoke(root, ('-m', 'pip', 'install', '--quiet', *REQUIREMENTS), env, log, deadline, 120)
        write(root / 'runtime.json', {**hardware, 'packages': {name: version(name) for name in ('numpy', 'scipy')},
                                     'source_job_manifest_sha256': plan['source_job_manifest_sha256']})
        for stage in stages:
            fit_stage(root, stage, status, plan, env, deadline)
        status['phase'] = 'sealed'
    except Exception as error:
        status.update(phase='worker-failed', error_type=type(error).__name__)
        raise
    finally:
        status['finished'] = True
        write(root / 'status.json', status)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--payload-sha256', required=True)
    parser.add_argument('--worker-sha256', required=True)
    parser.add_argument('--seconds', type=int, required=True)
    run(**vars(parser.parse_args()))
