"""Prepare, monitor, verify and release one owned Colab L4 linked-evidence assay."""

import argparse
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from importlib.util import resolve_name
from io import BytesIO
import json
import math
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
from time import monotonic, sleep
from uuid import uuid4
from zipfile import BadZipFile

from research.semantic_tracking import iirc_linked_colab_worker as worker


PROJECT = Path(__file__).resolve().parents[2]
CLI = Path.home() / '.local/bin/colab'
USAGE = Path.home() / '.codex/skills/colab-gpu/scripts/record_usage.py'
SESSION_ROW = re.compile(r'^\[([^\[\]]+)\]\s+(\S+)\s+\| Hardware: [^|]+\| Shape: [^|]+\| Variant: \w+\s*$', re.M)
NO_SESSIONS = '[colab] No active sessions found on server.'
CLI_VERSION = '0.7.4'
RELEASE_RESERVE_S = 175
POLL_S = 20
USAGE_INTERVAL_S = 600
PRIOR_L4_TOKENS_PER_SECOND = 2419.39567213605
STATUS_FAILURES_BEFORE_ENDPOINT_CHECK = 3


def utc():
    return datetime.now(timezone.utc).isoformat()


def manifest_path(job):
    return job / 'manifest.json'


@contextmanager
def run_lock(job, mode):
    with (job / 'run.lock').open(mode, encoding='ascii') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('A local driver already owns this runtime') from error
        yield


def inventory(inputs):
    stops = {}
    for split in ('train', 'dev'):
        with inputs.extractfile(f'{split}/stimuli.jsonl') as handle:
            stops[split] = {row['id']: len(row['stops']) for row in map(json.loads, handle)}
    return stops


def verify_code_inventory(files):
    for path in files.values():
        source = path.relative_to(PROJECT)
        package = '.'.join(source.parent.parts)
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(source))):
            if isinstance(node, ast.ImportFrom):
                base = resolve_name('.' * node.level + (node.module or ''), package)
                imports = (base, *(f'{base}.{name.name}' for name in node.names))
            elif isinstance(node, ast.Import):
                imports = tuple(name.name for name in node.names)
            else:
                continue
            for name in imports:
                if name.startswith('research.'):
                    parts = name.replace('.', '/')
                    for dependency in (f'{parts}.py', f'{parts}/__init__.py'):
                        if (PROJECT / dependency).is_file() and dependency not in files:
                            raise ValueError(f'Worker import {dependency} from {source} is absent from the payload')


def resume_source(job, data_hashes, input_hash):
    previous = worker.read_json(manifest_path(job))
    inherited = verify_payload(job, previous)
    if (previous['data_members_sha256'] != data_hashes
            or previous['payload_members_sha256']['inputs.tar.gz'] != input_hash):
        raise ValueError('Requested linked inputs differ from the interrupted job archive')
    checkpoints = job / 'checkpoints'
    receipt = worker.read_json(checkpoints / 'receipt.json')
    checkpoint_proofs(checkpoints, receipt, previous)
    if receipt['run_id'] != job.name and receipt != inherited:
        raise ValueError('Durable checkpoint receipt is not bound to the interrupted job')
    return previous, receipt


def prepare(inputs, output, resume_from=None):
    input_path = inputs / 'linked-evidence-model-inputs.tar.gz'
    receipt = worker.read_json(inputs / 'pack-verification.json')
    assay_path = inputs / 'assay.json'
    assay = worker.read_json(assay_path)
    data_hashes = worker.inspect_archive(input_path, worker.DATA)
    input_hash = worker.digest(input_path)
    if (data_hashes != receipt['members_sha256'] or input_hash != receipt['archive_sha256']
            or worker.digest(assay_path) != data_hashes['assay.json']
            or {name: data_hashes[name] for name in worker.DATA - {'assay.json'}} != assay['inputs_sha256']
            or assay['model'] != worker.MODEL or assay['revision'] != worker.REVISION):
        raise ValueError('Linked assay seal or model snapshot does not match the packaged inputs')

    prior = checkpoint = None
    if resume_from is not None:
        prior, checkpoint = resume_source(resume_from, data_hashes, input_hash)

    run_id = f'cogllm-iirc-linked-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}'
    job = output / run_id
    job.mkdir(mode=0o700, parents=True)
    payload = job / 'payload.tar.gz'
    if prior is None:
        files = {name: PROJECT / name for name in worker.CODE}
        verify_code_inventory(files)
        for path in files.values():
            compile(path.read_bytes(), str(path), 'exec')
        files['inputs.tar.gz'] = input_path
        with tarfile.open(payload, 'w:gz', compresslevel=1) as package:
            for name, path in sorted(files.items()):
                package.add(path, arcname=name, recursive=False)
        payload_hash = worker.digest(payload)
        payload_members = {name: worker.digest(path) for name, path in files.items()}
    else:
        # Checkpoints bind the archived researcher code and inputs, not the current worktree.
        shutil.copyfile(resume_from / 'payload.tar.gz', payload)
        payload_hash = prior['payload_sha256']
        payload_members = prior['payload_members_sha256']
    shutil.copyfile(worker.__file__, job / 'worker.py')
    shutil.copyfile(assay_path, job / 'assay.json')
    with tarfile.open(input_path, 'r:gz') as archive:
        stops = inventory(archive)
    data = {
        'run_id': run_id, 'schema_version': 2, 'phase': 'prepared', 'timeline': {'prepared': utc()}, 'cli_version': CLI_VERSION,
        'requested_gpu': 'L4', 'runtime_budget_s': worker.MAX_RUNTIME_S,
        'payload_sha256': payload_hash, 'worker_sha256': worker.digest(job / 'worker.py'),
        'payload_members_sha256': payload_members,
        'data_members_sha256': data_hashes, 'cases': assay['cases'], 'tokens': assay['tokens'],
        'certified_stops': sum(sum(counts.values()) for counts in stops.values()),
        'previous_l4_forward_tokens_per_second': PRIOR_L4_TOKENS_PER_SECOND,
        'linear_forward_seconds': sum(assay['tokens'].values()) / PRIOR_L4_TOKENS_PER_SECOND,
    }
    if prior is not None:
        data['resume_payload_source_run_id'] = prior['run_id']
        prepare_resume(job, data, resume_from / 'checkpoints', checkpoint)
    worker.write_json(manifest_path(job), data)
    verify_payload(job, data)
    print(str(job.resolve()), flush=True)


def verify_payload(job, data):
    payload = job / 'payload.tar.gz'
    assay = job / 'assay.json'
    metadata = worker.read_json(assay)
    if (not worker.RUN_ID.fullmatch(job.name) or data['run_id'] != job.name or data.get('schema_version') != 2
            or worker.digest(assay) != data['data_members_sha256']['assay.json']
            or worker.digest(payload) != data['payload_sha256']
            or worker.digest(job / 'worker.py') != data['worker_sha256']
            or data['cases'] != metadata['cases'] or data['tokens'] != metadata['tokens']
            or worker.inspect_archive(payload, worker.PAYLOAD) != data['payload_members_sha256']):
        raise ValueError('Prepared job files differ from their sealed manifest')
    if 'resume' in data:
        return verify_resume(job, data)


def checkpoint_proofs(directory, receipt, data):
    worker.check_proofs(directory, receipt, data['data_members_sha256']['assay.json'],
                        data['payload_members_sha256'], data['data_members_sha256'])


def prepare_resume(job, data, source, receipt):
    checkpoints = job / 'checkpoints'
    worker.set_proofs(source, checkpoints)
    checkpoint_proofs(checkpoints, receipt, data)
    for item in receipt['shards']:
        name = worker.shard_name(item['number'])
        shutil.copyfile(source / name, checkpoints / name)
        worker.check_shard(checkpoints / name, item)
    worker.write_json(checkpoints / 'receipt.json', receipt)
    meta = job / 'resume.tar.gz'
    with tarfile.open(meta, 'w:gz') as archive:
        for name in sorted(worker.CHECKPOINT_META):
            archive.add(checkpoints / name, arcname=name, recursive=False)
    data['resume'] = {'run_id': receipt['run_id'], 'plan_sha256': receipt['plan_sha256'],
                      'shards': len(receipt['shards']), 'metadata_sha256': worker.digest(meta)}
    data['durable_checkpoints'] = {'shards': len(receipt['shards']), 'total': receipt['shards_total'],
                                    'plan_sha256': receipt['plan_sha256']}


def verify_resume(job, data):
    meta = job / 'resume.tar.gz'
    resume = data['resume']
    if worker.digest(meta) != resume['metadata_sha256']:
        raise ValueError('Prepared resume metadata differs from the new job seal')
    hashes = worker.inspect_archive(meta, worker.CHECKPOINT_META)
    with tarfile.open(meta, 'r:gz') as archive, archive.extractfile('receipt.json') as stream:
        receipt = json.load(stream)
    checkpoints = job / 'checkpoints'
    checkpoint_proofs(checkpoints, receipt, data)
    if (hashes['plan.json'] != receipt['plan_sha256'] or hashes['causality.json'] != receipt['causality_sha256']
            or receipt['run_id'] != resume['run_id'] or receipt['plan_sha256'] != resume['plan_sha256']
            or len(receipt['shards']) != resume['shards']):
        raise ValueError('Resume receipt differs from the checked checkpoint proofs')
    for item in receipt['shards']:
        worker.check_shard(checkpoints / worker.shard_name(item['number']), item)
    return receipt


def write_launcher(job, data, seconds):
    root = f'/content/{job.name}'
    invocation = [f'{root}.worker.py', '--root', root, '--seconds', str(seconds),
                  '--payload-sha256', data['payload_sha256'], '--worker-sha256', data['worker_sha256']]
    if 'resume' in data:
        invocation.extend(('--resume-sha256', data['resume']['metadata_sha256']))
    path = job / 'launch.py'
    path.write_text(f'''from pathlib import Path
import os
import subprocess
import sys

pid_file = Path({root + '.pid'!r})
try:
    claim = pid_file.open('x', encoding='ascii')
except FileExistsError:
    print('Linked-evidence worker already claimed')
else:
    try:
        with claim, Path({root + '.launch.log'!r}).open('w', encoding='utf-8') as log:
            process = subprocess.Popen([sys.executable, '-u', *{invocation!r}], cwd='/content',
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, close_fds=True)
            claim.write(str(process.pid))
            claim.flush()
            os.fsync(claim.fileno())
    except BaseException:
        pid_file.unlink(missing_ok=True)
        raise
    print('Linked-evidence worker launched')
''', encoding='utf-8')
    compile(path.read_bytes(), str(path), 'exec')
    return path


def parse_sessions(output):
    rows = SESSION_ROW.findall(output)
    if not rows and NO_SESSIONS not in output:
        raise ValueError('Could not parse the server session list')
    if len({endpoint for _, endpoint in rows}) != len(rows):
        raise ValueError('Server session list includes duplicate endpoints')
    return {endpoint: name for name, endpoint in rows}


class Colab:
    def __init__(self, job, data):
        self.job = job
        self.data = data
        self.remote = f'/content/{job.name}'

    def update(self, phase):
        self.data['phase'] = phase
        self.data['timeline'][phase] = utc()
        worker.write_json(manifest_path(self.job), self.data)
        print(phase, flush=True)

    def cli(self, *args, timeout=25, check=True):
        log_path = self.job / f'{args[0]}.log'
        result = subprocess.run([str(CLI), '--auth', 'adc', *map(str, args)], capture_output=True,
                                text=True, timeout=timeout)
        with log_path.open('a', encoding='utf-8') as log:
            log.write(result.stdout)
            log.write(result.stderr)
        if check and result.returncode:
            raise RuntimeError(f'Colab {args[0]} returned {result.returncode}, see {log_path.name}')
        return result

    def sessions(self):
        return parse_sessions(self.cli('sessions', timeout=20).stdout)

    def endpoint(self, sessions):
        matches = [endpoint for endpoint, name in sessions.items() if name == self.job.name]
        if len(matches) > 1:
            raise ValueError('Multiple server sessions match the owned run name')
        return matches[0] if matches else None

    def usage(self, phase, required=False, timeout=32):
        try:
            with (self.job / 'usage.log').open('a', encoding='utf-8') as log:
                subprocess.run([sys.executable, str(USAGE), '--project', str(PROJECT), '--run-id', self.job.name,
                                '--session', self.job.name, '--phase', phase, '--output', str(self.job / 'usage.jsonl'),
                                '--cli', str(CLI)], stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
            return True
        except (OSError, subprocess.SubprocessError) as error:
            self.data.setdefault('usage_errors', {})[phase] = type(error).__name__
            if required:
                raise RuntimeError(f'Account usage unavailable at {phase}, see usage.log') from error
            return False

    def download(self, remote_name, local_name, timeout=13):
        destination = self.job / local_name
        partial = destination.with_suffix('.downloading')
        partial.unlink(missing_ok=True)
        try:
            if self.cli('download', '-s', self.job.name, f'{self.remote}{remote_name}', partial,
                        timeout=timeout, check=False).returncode:
                return False
            partial.replace(destination)
            return True
        except subprocess.TimeoutExpired:
            return False
        finally:
            partial.unlink(missing_ok=True)

    def launch(self, deadline):
        seconds = int(deadline - monotonic()) - 30
        launcher = write_launcher(self.job, self.data, seconds)
        for _ in range(2):
            try:
                self.cli('exec', '-s', self.job.name, '-f', launcher, '--timeout', '45', timeout=57, check=False)
            except subprocess.TimeoutExpired:
                with (self.job / 'exec.log').open('a', encoding='utf-8') as log:
                    log.write('Colab exec timed out during worker launch; checking for its detached process.\n')
            if (self.download('.pid', 'remote-worker.pid') and
                    re.fullmatch(r'[1-9][0-9]*', (self.job / 'remote-worker.pid').read_text(encoding='ascii'))):
                return
        self.download('.launch.log', 'remote-launch.log')
        raise RuntimeError('The owned runtime did not confirm the detached worker')

    def upload_resume(self, receipt, deadline):
        files = [('resume.tar.gz', '.resume.tar.gz')]
        for item in receipt['shards']:
            name = worker.shard_name(item['number'])
            files.append((f'checkpoints/{name}', f'.resume-{name}'))
        for filename, suffix in files:
            remaining = deadline - monotonic() - 90
            if remaining <= 0:
                raise TimeoutError('Resume upload exhausted the owned runtime transfer budget')
            self.cli('upload', '-s', self.job.name, self.job / filename, f'{self.remote}{suffix}', timeout=min(30, remaining))

    def pull_checkpoints(self, deadline):
        if monotonic() + 38 >= deadline or not self.download('/checkpoint-index.json', 'remote-checkpoint-index.json'):
            return None
        remote = worker.read_json(self.job / 'remote-checkpoint-index.json')
        worker.check_checkpoint(remote)
        if remote['run_id'] != self.job.name or remote['assay_sha256'] != self.data['data_members_sha256']['assay.json']:
            raise ValueError('Checkpoint index comes from another run or assay')

        directory = self.job / 'checkpoints'
        directory.mkdir(exist_ok=True)
        marker = directory / 'receipt.json'
        committed = []
        if marker.is_file():
            local = worker.read_json(marker)
            checkpoint_proofs(directory, local, self.data)
            committed = local['shards']
            if (any(remote[name] != local[name] for name in ('plan_sha256', 'causality_sha256', 'shards_total'))
                    or remote['shards'][:len(committed)] != committed):
                raise ValueError('Remote checkpoint index disagrees with the durable local prefix')
        else:
            for name in ('plan.json', 'causality.json'):
                if not self.download(f'/checkpoint-proofs/{name}', f'checkpoints/{name}'):
                    return None
            checkpoint_proofs(directory, remote, self.data)

        for item in remote['shards'][len(committed):]:
            remaining = deadline - monotonic() - 25
            if remaining <= 0:
                break
            filename = worker.shard_name(item['number'])
            local = directory / filename
            if not self.download(f'/features/shards/{filename}', f'checkpoints/{filename}', timeout=min(20, remaining)):
                break
            try:
                worker.check_shard(local, item)
            except (OSError, ValueError, BadZipFile):
                local.unlink(missing_ok=True)
                raise
            committed.append(item)
            worker.write_json(marker, {**remote, 'shards': committed})
            self.data['durable_checkpoints'] = {'shards': len(committed), 'total': remote['shards_total'],
                                                'plan_sha256': remote['plan_sha256']}
            self.update(f'checkpoint-local-{len(committed)}')
        return len(committed)

    def check_connection(self):
        try:
            sessions = self.sessions()
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            return
        if self.data['endpoint'] not in sessions and self.endpoint(sessions) is None:
            self.data['session_lost'] = True
            self.update('session-lost')
            count = self.data.get('durable_checkpoints', {}).get('shards', 0)
            raise RuntimeError(f'The owned L4 endpoint disappeared. Preserved {count} verified local checkpoints.')

    def wait(self, deadline):
        previous = None
        previous_shard = 0
        next_usage = monotonic() + USAGE_INTERVAL_S
        initial_deadline = monotonic() + 50
        missing_status = 0
        checkpoint_lag = 0
        while monotonic() + 25 < deadline:
            if self.download('/status.json', 'remote-status.json'):
                missing_status = 0
                status = worker.read_json(self.job / 'remote-status.json')
                if (status['run_id'] != self.job.name or any(key in status and status[key] != self.data[key]
                        for key in ('payload_sha256', 'worker_sha256'))):
                    raise ValueError('Remote status belongs to a different run')
                if status['phase'] != previous:
                    previous = status['phase']
                    self.update(f'remote-{previous}')
                if previous in {'extract', 'fit', 'complete'}:
                    saved = self.pull_checkpoints(deadline)
                    checkpoint_lag = (checkpoint_lag + 1 if previous_shard > (saved or 0) else 0)
                    if checkpoint_lag >= 2 and self.download('/checkpoint-error.json', 'checkpoint-error.json', timeout=10):
                        error = worker.read_json(self.job / 'checkpoint-error.json')
                        if error['run_id'] == self.job.name:
                            raise RuntimeError(f'Could not publish a durable linked checkpoint: {error["message"]}')
                if status['finished']:
                    if status['completed']:
                        return status
                    self.download('/worker.log', 'remote-worker.log', timeout=10)
                    diagnostic = status.get('stage_log_tail', '').strip().splitlines()
                    detail = f' ({diagnostic[-1]})' if diagnostic else ''
                    raise RuntimeError(f'The remote {previous} stage failed with {status["error_type"]}{detail}')
                if previous == 'extract' and self.download('/features/progress.json', 'remote-progress.json'):
                    progress = worker.read_json(self.job / 'remote-progress.json')
                    if progress['shards_completed'] > previous_shard:
                        previous_shard = progress['shards_completed']
                        self.data['latest_extraction'] = progress
                        self.update(f'extraction-shard-{previous_shard}')
            else:
                missing_status += 1
                if missing_status >= STATUS_FAILURES_BEFORE_ENDPOINT_CHECK:
                    self.check_connection()
                    missing_status = 0
                if previous is None and monotonic() >= initial_deadline:
                    self.download('.launch.log', 'remote-launch.log', timeout=10)
                    raise RuntimeError('The detached worker did not publish an initial status')
            if monotonic() >= next_usage and monotonic() + 60 < deadline:
                self.usage('periodic')
                next_usage = monotonic() + USAGE_INTERVAL_S
            sleep(min(POLL_S, max(0, deadline - monotonic() - 25)))
        raise TimeoutError('The owned runtime reached its extraction and fit deadline')

    def release(self):
        endpoint = self.data.get('endpoint')
        for _ in range(2):
            try:
                sessions = self.sessions()
                named = self.endpoint(sessions)
                if endpoint is None and named is None:
                    extras = sessions.keys() - self.data['endpoints_before_allocation']
                    if extras:
                        raise RuntimeError('Allocation failed before its endpoint could be identified')
                    return True
                endpoint = endpoint or named
                self.data['endpoint'] = endpoint
                worker.write_json(manifest_path(self.job), self.data)
                if named == endpoint:
                    self.cli('stop', '-s', self.job.name, timeout=25, check=False)
                    sessions = self.sessions()
                if endpoint not in sessions and self.endpoint(sessions) is None:
                    return True
            except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as error:
                self.data['release_error'] = str(error)
        return False


def verify_results(job, data, status):
    archive = job / worker.RESULTS
    names = set(status['output_sha256'])
    if (status['run_id'] != data['run_id'] or not status['completed'] or status['device']['name'] != 'NVIDIA L4'
            or status.get('resumed_from') != data.get('resume')
            or any(status[name] != data[name] for name in ('payload_sha256', 'worker_sha256',
                   'payload_members_sha256', 'data_members_sha256')) or worker.digest(archive) != status['archive_sha256']
            or not worker.OUTPUTS <= names or any(not worker.TELEMETRY.fullmatch(name) for name in names - worker.OUTPUTS)
            or not archive.stat().st_size <= worker.MAX_RETURN_BYTES):
        raise ValueError('Result provenance failed verification')
    if worker.inspect_archive(archive, names) != status['output_sha256']:
        raise ValueError('Sealed result member digests differ')
    output = job / 'outputs'
    output.mkdir()
    with tarfile.open(archive, 'r:gz') as package:
        package.extractall(output, filter='data')
    extraction = worker.read_json(output / 'features/extraction.json')
    checkpoint = job / 'checkpoints/receipt.json'
    if checkpoint.is_file():
        receipt = worker.read_json(checkpoint)
        if (receipt['plan_sha256'] != extraction['plan_sha256'] or any(
                extraction['per_shard'][item['number']]['sha256'] != item['sha256'] for item in receipt['shards'])):
            raise ValueError('Final extraction differs from the durable checkpoint prefix')
    causality = worker.read_json(output / 'features/causality.json')
    progress = worker.read_json(output / 'features/progress.json')
    seal = worker.read_json(output / 'fit/fit-manifest.json')
    targets = data['cases']
    input_hashes = data['data_members_sha256']
    token_total = sum(data['tokens'].values())
    extraction_inputs = {'assay': input_hashes['assay.json'], 'train/stimuli.jsonl': input_hashes['train/stimuli.jsonl'],
                         'dev/stimuli.jsonl': input_hashes['dev/stimuli.jsonl']}
    if (extraction['model'] != worker.MODEL or extraction['revision'] != worker.REVISION
            or extraction['gpu'] != 'NVIDIA L4' or extraction['transformers'] != worker.TRANSFORMERS
            or extraction['assay_sha256'] != input_hashes['assay.json'] or extraction['tokens'] != token_total
            or extraction['inputs_sha256'] != extraction_inputs
            or extraction['input_paths'] != sum(targets.values()) or extraction['feature_width'] != 128
            or extraction['certified_character_stops'] != data['certified_stops'] or extraction['dtype'] != 'float32'
            or extraction['max_batch_tokens'] != 6144 or extraction['max_batch_examples'] != 24
            or extraction['causality_sha256'] != status['output_sha256']['features/causality.json']
            or extraction['plan_sha256'] != status['output_sha256']['features/plan.json']
            or progress['plan_sha256'] != extraction['plan_sha256'] or not progress['complete']
            or progress['tokens_completed'] != token_total or progress['shards_completed'] != extraction['shards']
            or causality['plan_sha256'] != extraction['plan_sha256'] or len(causality['checks']) != 12
            or not all(row['passed'] and row['projected_tokens']['passed'] for row in causality['checks'])
            or seal['features_sha256'] != extraction['features_sha256']
            or seal['extraction_sha256'] != status['output_sha256']['features/extraction.json']
            or seal['assay_sha256'] != input_hashes['assay.json'] or seal['model'] != worker.MODEL
            or seal['inputs_sha256'] != {name: value for name, value in input_hashes.items() if name != 'assay.json'}
            or seal['revision'] != worker.REVISION or seal['fit_paths'] != targets['train']
            or seal['heldout_forecast_paths'] != targets['dev'] or set(seal['fits']) != set(worker.METHODS)
            or seal['heads_sha256'] != status['output_sha256']['fit/heads.npz']
            or seal['predictions_sha256'] != status['output_sha256']['fit/predictions.jsonl']):
        raise ValueError('Result receipts disagree with the sealed linked assay')

    with tarfile.open(job / 'payload.tar.gz', 'r:gz') as payload:
        with payload.extractfile('inputs.tar.gz') as input_file:
            with tarfile.open(fileobj=BytesIO(input_file.read()), mode='r:gz') as inputs:
                expected = {case_id: n - 1 for case_id, n in inventory(inputs)['dev'].items()}
    with (output / 'fit/predictions.jsonl').open(encoding='utf-8') as forecasts:
        for row in map(json.loads, forecasts):
            count = expected.pop(row['case_id'])
            if (set(row['methods']) != set(worker.METHODS) or any(
                    len(hazards) != count or not all(math.isfinite(h) and 0 < h < 1 for h in hazards)
                    for hazards in row['methods'].values())):
                raise ValueError('Forecast probabilities or coordinates differ from the planned held-out rows')
    if expected:
        raise ValueError('Forecasts are missing planned held-out cases')


def run(job):
    data = worker.read_json(manifest_path(job))
    verify_payload(job, data)
    with run_lock(job, 'x'):
        return run_owned(job, data)


def run_owned(job, data):
    machine = Colab(job, data)
    allocation_attempted = False

    def on_signal(signum, _):
        signal.signal(signum, signal.SIG_IGN)
        raise KeyboardInterrupt(f'Interrupted by signal {signum}')

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, on_signal)

    try:
        if re.search(rf'^Version: {re.escape(CLI_VERSION)}$', machine.cli('version', timeout=20).stdout, re.M) is None:
            raise RuntimeError(f'Expected shared Colab CLI {CLI_VERSION}')
        before = machine.sessions()
        if machine.endpoint(before) is not None:
            raise RuntimeError('The prepared session name is already registered on the server')
        data['endpoints_before_allocation'] = sorted(before)
        machine.usage('before-allocation', required=True)
        machine.update('allocating')
        deadline = monotonic() + worker.MAX_RUNTIME_S - RELEASE_RESERVE_S
        allocation_attempted = True
        machine.cli('new', '-s', job.name, '--gpu', 'L4', timeout=80)
        endpoint = machine.endpoint(machine.sessions())
        if endpoint is None or endpoint in before:
            raise RuntimeError('No newly created endpoint matched the owned L4 run name')
        data['endpoint'] = endpoint
        machine.update('allocated')
        machine.usage('after-allocation')
        machine.update('uploading')
        for source, suffix in (('payload.tar.gz', '.tar.gz'), ('worker.py', '.worker.py')):
            machine.cli('upload', '-s', job.name, job / source, f'{machine.remote}{suffix}', timeout=70)
        if 'resume' in data:
            machine.update('uploading-resume')
            machine.upload_resume(verify_resume(job, data), deadline)
        machine.launch(deadline)
        machine.update('worker-running')
        status = machine.wait(deadline)
        remaining = deadline - monotonic() + 24
        if remaining <= 0 or not machine.download(f'/{worker.RESULTS}', worker.RESULTS, timeout=min(remaining, 30)):
            raise RuntimeError('Failed to retrieve the sealed results from the owned runtime')
        data['result_retrieved'] = True
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        data['failure'] = str(error) or type(error).__name__
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, signal.SIG_IGN)
        if allocation_attempted:
            machine.update('releasing')
            machine.usage('before-stop', timeout=12)
            data['endpoint_absent_after_stop'] = machine.release()
            data['release_usage_recorded'] = machine.usage('after-stop')
            data['release_verified'] = data['endpoint_absent_after_stop'] and data['release_usage_recorded']
        if data.get('result_retrieved'):
            try:
                verify_results(job, data, status)
                data['artifacts_verified'] = True
            except (OSError, ValueError, KeyError, tarfile.TarError) as error:
                data['failure'] = str(error) or type(error).__name__
        ok = 'failure' not in data and data.get('release_verified') and data.get('artifacts_verified')
        machine.update('finished' if ok else 'failed')
        if not ok:
            print(data.get('failure', 'Release could not be verified, see manifest.json'), file=sys.stderr)
    return 0 if ok else 1


def release_interrupted(job):
    with run_lock(job, 'r+'):
        data = worker.read_json(manifest_path(job))
        if (not worker.RUN_ID.fullmatch(job.name) or data['run_id'] != job.name or
                'allocating' not in data['timeline']):
            raise ValueError('No attempted allocation to recover for this run')
        machine = Colab(job, data)
        machine.update('recovering-release')
        machine.usage('before-stop', timeout=12)
        data['endpoint_absent_after_stop'] = machine.release()
        data['release_usage_recorded'] = machine.usage('after-stop')
        data['release_verified'] = data['endpoint_absent_after_stop'] and data['release_usage_recorded']
        machine.update('release-verified' if data['release_verified'] else 'release-unverified')
        return 0 if data['release_verified'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    prep = actions.add_parser('prepare')
    prep.add_argument('--inputs', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--resume-from', type=Path)
    for name in ('verify', 'run', 'status', 'release'):
        actions.add_parser(name).add_argument('job', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.inputs, args.output, args.resume_from)
    elif args.action == 'run':
        return run(args.job.resolve())
    elif args.action == 'release':
        return release_interrupted(args.job.resolve())
    else:
        data = worker.read_json(manifest_path(args.job))
        if args.action == 'verify':
            verify_payload(args.job, data)
            print('Prepared linked-evidence payload verified')
        else:
            print(json.dumps(data, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
