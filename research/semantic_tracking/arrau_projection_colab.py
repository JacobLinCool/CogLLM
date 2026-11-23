"""Run the verified frozen-app projector on one sealed, grade-free packet."""

import argparse
from datetime import datetime, timezone
from gzip import GzipFile
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
import signal
import tarfile
from time import monotonic, sleep
from uuid import uuid4

from research.semantic_tracking import beukers_extract as extractor
from research.semantic_tracking.iirc_linked_colab import Colab, PROJECT, run_lock
from research.semantic_tracking.iirc_linked_colab_worker import digest, inspect_archive, read_json


PROTOCOL = Path(__file__).with_name('arrau-bridge-protocol.md')
REFERENCE = PROJECT / 'research/results/external-projection-l4'
ARCHIVE = 'projection-inputs.tar.gz'
STIMULI = 'model_input/stimuli.jsonl'
EXTRACTOR_SOURCE = 'research/semantic_tracking/beukers_extract.py'
WORKER_SOURCE = 'research/gradio_app/cuda_verification_batch_worker.py'
ANCHOR_HEADS = 'research/results/gradio-candidate-checkpoint/anchor/heads.npz'
SECONDS = 12 * 60
STAGES = {
    'setup': {'run.log', 'runtime.json'},
    'projection': {'causality.json', 'coordinates.jsonl', 'run.log', 'seal.json',
                   'telemetry.jsonl', 'terminal_states.npy', 'vectors.npy'},
}


def package(output, members):
    hashes = {}
    with output.open('xb') as destination, GzipFile(fileobj=destination, mode='wb', mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as archive:
            for name, contents in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size = len(contents)
                archive.addfile(info, BytesIO(contents))
                hashes[name] = sha256(contents).hexdigest()
    return hashes


def prepare(packet, previous, root, protocol=PROTOCOL, raw_stops=False):
    expected = read_json(REFERENCE / 'run-verification.json')
    worker = previous / 'worker.py'
    previous_archive = previous / ARCHIVE
    if (digest(worker) != expected['worker_sha256']
            or digest(previous_archive) != expected['payload_members_sha256'][ARCHIVE]
            or digest(previous / 'bundle.json') != expected['payload_members_sha256']['bundle.json']):
        raise ValueError('Expected the last verified Gradio projection worker and its source archive')
    bundle = read_json(previous / 'bundle.json')
    tracked = read_json(REFERENCE / 'extraction.json')
    receipt = read_json(packet / 'preflight.json')
    certified_path = packet / 'extractor-preflight.json'
    certified = read_json(certified_path)
    stimuli = packet / 'stimuli.jsonl'
    rows = extractor.read_stimuli(stimuli)
    extractor.validate_preflight(certified, rows, digest(stimuli))
    if (receipt['stimuli_sha256'] != digest(stimuli) or receipt['candidate_id'] != tracked['candidate_id']
            or len(receipt['documents']) != certified['encoded_documents']):
        raise ValueError('Source receipt disagrees with the frozen candidate or inference packet')
    original = bundle['archives'][ARCHIVE]['members']
    with tarfile.open(previous_archive, 'r:gz') as source:
        if set(source.getnames()) != original.keys() or len(source.getmembers()) != len(original):
            raise ValueError('Unexpected reference archive members')
        members = {member.name: source.extractfile(member).read() for member in source.getmembers()
                   if member.isfile() and member.name not in (STIMULI, 'preflight.json')}
    if any(not name.startswith('research/') for name in members) or len(members) != len(original) - 2:
        raise ValueError('Reference archive contains unexpected files beside the old stimuli')
    if any(sha256(contents).hexdigest() != original[name] for name, contents in members.items()):
        raise ValueError('A frozen source or candidate artifact differs from its reference checksum')
    replaced = {STIMULI, 'preflight.json'}
    if raw_stops:
        from research.gradio_app.cuda_verification_batch_worker import PROJECTION_MAX_BYTES

        if any(row['stops'][-1] != len(row['text']) for row in rows):
            raise ValueError('The projection export requires an exact full-text stop for every input')
        stops, documents = certified['certified_stops'], certified['encoded_documents']
        hidden_width = tracked['terminal_states_shape'][1]
        array_bytes = 4 * hidden_width * (stops + documents) + 2 * extractor.PROJECTION_WIDTH * stops
        if array_bytes >= PROJECTION_MAX_BYTES:
            raise ValueError('Requested raw and projected arrays exceed the projection return budget')
        if EXTRACTOR_SOURCE not in members:
            raise ValueError('Reference archive does not contain the extractor approved for replacement')
        members[EXTRACTOR_SOURCE] = (PROJECT / EXTRACTOR_SOURCE).read_bytes()
        replaced.add(EXTRACTOR_SOURCE)
        worker = PROJECT / WORKER_SOURCE
    members.update({STIMULI: stimuli.read_bytes(), 'preflight.json': certified_path.read_bytes()})

    job = root / f'cogllm-cuda-checks-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}'
    job.mkdir(parents=True)
    shutil.copyfile(worker, job / 'worker.py')
    staged = package(job / ARCHIVE, members)
    if any(staged[name] != sha for name, sha in original.items() if name not in replaced):
        raise ValueError('A frozen model or source artifact changed')
    bundle['archives'][ARCHIVE].update(sha256=digest(job / ARCHIVE), members=staged)
    bundle['projection'].update(
        **{name: certified[name] for name in ('certified_stops', 'encoded_documents', 'input_tokens', 'stimuli_sha256')},
        preflight_sha256=digest(certified_path), assay_sha256=digest(packet / 'preflight.json'),
        protocol_sha256=digest(protocol),
    )
    if raw_stops:
        bundle['projection'].update(raw_stops=True, source_code_sha256=staged[EXTRACTOR_SOURCE])
    bundle_bytes = (json.dumps(bundle, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    (job / 'bundle.json').write_bytes(bundle_bytes)
    hashes = package(job / 'payload.tar.gz', {'bundle.json': bundle_bytes, ARCHIVE: (job / ARCHIVE).read_bytes()})
    data = {'run_id': job.name, 'operation': protocol.stem, 'cli_version': '0.7.4',
            'requested_hardware': 'NVIDIA L4', 'time_budget_s': SECONDS, 'timeline': {},
            'payload_members_sha256': hashes, 'payload_sha256': digest(job / 'payload.tar.gz'),
            'worker_sha256': digest(job / 'worker.py'), 'projection': bundle['projection'],
            'reference_run_id': expected['run_id'], 'expected_projection_sha256': tracked['projection_sha256']}
    machine = Colab(job, data)
    machine.update('prepared')
    print(job, flush=True)
    return job


def validate_raw_states(evidence, extraction, job):
    import numpy as np

    with tarfile.open(job / ARCHIVE, 'r:gz') as archive:
        certificate = archive.extractfile('preflight.json').read()
    if sha256(certificate).hexdigest() != extraction['preflight_sha256']:
        raise ValueError('Raw state validation requires the sealed tokenizer certificate')
    certified = json.loads(certificate)
    with (evidence / 'coordinates.jsonl').open(encoding='utf-8') as stream:
        coordinates = [json.loads(line) for line in stream]
    expected = [{name: doc[name] for name in ('id', 'stops')} for doc in certified['documents']]
    if coordinates != expected:
        raise ValueError('Raw state coordinates differ from the ordered certified stops')
    ends = np.cumsum([len(row['stops']) for row in coordinates]) - 1
    count = int(ends[-1]) + 1
    width = extractor.PROJECTION_WIDTH
    hidden_width = read_json(REFERENCE / 'extraction.json')['terminal_states_shape'][1]
    if (extraction['raw_stops'] is not True or count != extraction['certified_stops']
            or len(ends) != extraction['encoded_documents']
            or extraction['states_shape'] != [count, hidden_width] or extraction['states_dtype'] != 'float32'
            or extraction['terminal_states_shape'] != [len(ends), hidden_width]
            or extraction['terminal_state_dtype'] != 'float32'
            or extraction['vectors_shape'] != [count, width] or extraction['storage_dtype'] != 'float16'
            or extraction['compute_dtype'] != 'float32' or extraction['projection_width'] != width):
        raise ValueError('Unexpected raw, terminal, or projected state declaration')
    checks = read_json(evidence / 'causality.json')['checks']
    if len(checks) != 12 or not all(row['passed'] is True and row['projected_tokens']['passed'] is True for row in checks):
        raise ValueError('Raw and projected states require all 12 independent causal checks')

    def matrix_file(name, shape, dtype):
        path = evidence / name
        matrix = np.load(path, mmap_mode='r', allow_pickle=False)
        if (not isinstance(matrix, np.memmap) or matrix.shape != shape or matrix.dtype != dtype
                or not matrix.flags.c_contiguous or matrix.offset + matrix.nbytes != path.stat().st_size):
            raise ValueError(f'Invalid sealed NumPy matrix: {name}')
        return matrix

    states = matrix_file('states.npy', (count, hidden_width), np.float32)
    terminal = matrix_file('terminal_states.npy', (len(ends), hidden_width), np.float32)
    vectors = matrix_file('vectors.npy', (count, width), np.float16)
    anchor_heads = PROJECT / ANCHOR_HEADS
    if digest(anchor_heads) != extraction['anchor_heads_sha256']:
        raise ValueError('The raw-state replay requires the registered local anchor checkpoint')
    with np.load(anchor_heads, allow_pickle=False) as heads:
        projection = heads['projection']
    if (projection.shape != (hidden_width, width) or projection.dtype != np.float32
            or sha256(projection.tobytes(order='C')).hexdigest() != extraction['projection_sha256']):
        raise ValueError('The raw-state replay projection differs from the sealed projection')
    maximum_error = 0.0
    for start in range(0, count, 256):
        block = slice(start, min(start + 256, count))
        raw = states[block]
        terminal_rows = (ends >= block.start) & (ends < block.stop)
        if (not np.isfinite(raw).all() or not np.any(raw, axis=1).all()
                or not np.array_equal(raw[ends[terminal_rows] - start], terminal[terminal_rows])):
            raise ValueError('Invalid raw stop states or disordered terminal rows')
        projected = raw @ projection
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        if not (np.isfinite(norms) & (norms > 0)).all():
            raise ValueError('Raw stop states failed frozen projection replay')
        error = float(np.max(np.linalg.norm(projected / norms - vectors[block], axis=1)))
        if not np.isfinite(error) or error > extractor.FP16_TOLERANCE + 1e-5:
            raise ValueError('Raw and projected stop states disagree under independent FP32 projection replay')
        maximum_error = max(maximum_error, error)
    return {'rows': count, 'width': hidden_width, 'dtype': 'float32', 'max_projection_replay_l2': maximum_error}


def download_archive(machine, filename, deadline):
    for _ in range(2):
        seconds = int(deadline - monotonic())
        if seconds <= 0:
            break
        if machine.download(f'/return/{filename}', filename, timeout=min(55, seconds)):
            return
    raise RuntimeError(f'Could not download the sealed archive {filename} within the transfer budget')


def collect(machine, status, deadline):
    data, job = machine.data, machine.job
    if (any(status[key] != data[key] for key in ('run_id', 'payload_sha256', 'worker_sha256'))
            or status['stages'].keys() != STAGES.keys()):
        raise ValueError('Remote status belongs to a different job')
    raw_stops = data['projection'].get('raw_stops', False)
    required_projection = STAGES['projection'] | {'states.npy'} if raw_stops else STAGES['projection']
    for stage, required in {**STAGES, 'projection': required_projection}.items():
        entry = status['stages'][stage]
        if entry['state'] != 'complete':
            if entry['state'] == 'failed':
                machine.download(f'/return/{stage}.tar.gz', f'{stage}-failed.tar.gz', timeout=20)
            raise RuntimeError(f'The owned L4 stage did not complete: {stage}/{entry["state"]}')
        filename = f'{stage}.tar.gz'
        download_archive(machine, filename, deadline)
        archive = job / filename
        if (archive.stat().st_size != entry['archive_bytes'] or digest(archive) != entry['archive_sha256']
                or entry['members'].keys() != required):
            raise ValueError(f'Unexpected {stage} archive')
        if inspect_archive(archive, required, job / 'outputs' / stage) != entry['members']:
            raise ValueError(f'Member checksum mismatch in {stage} archive')
        machine.update(f'retrieved-{stage}')
    extraction = read_json(job / 'outputs/projection/seal.json')
    runtime = read_json(job / 'outputs/setup/runtime.json')
    paired = ('assay_sha256', 'candidate_id', 'certified_stops', 'encoded_documents', 'input_tokens',
              'preflight_sha256', 'protocol_sha256', 'source_code_sha256', 'stimuli_sha256')
    evidence = job / 'outputs/projection'
    if (any(extraction[name] != data['projection'][name] for name in paired)
            or extraction['projection_sha256'] != data['expected_projection_sha256']
            or extraction['gpu'] != runtime['device'] or runtime['device'] != 'NVIDIA L4'
            or extraction['content_forward_per_document'] != 1 or extraction['independent_causal_checks'] != 12
            or extraction['file_sha256'].keys() != required_projection - {'run.log', 'seal.json'}
            or any(digest(evidence / name) != sha for name, sha in extraction['file_sha256'].items())):
        raise ValueError('Output artifacts disagree with the frozen candidate and requested work')
    if raw_stops:
        data['raw_states_verification'] = validate_raw_states(evidence, extraction, job)
    data['stages'] = status['stages']
    data['verified_outputs'] = True


def run(job):
    data = read_json(job / 'manifest.json')
    if (data['phase'] != 'prepared' or job.name != data['run_id']
            or digest(job / 'payload.tar.gz') != data['payload_sha256']
            or digest(job / 'worker.py') != data['worker_sha256']):
        raise ValueError('Expected intact, unused owned runtime preparation')
    with run_lock(job, 'x'):
        machine = Colab(job, data)
        attempted = False

        def interrupted(signum, _):
            raise KeyboardInterrupt(f'Received signal {signum}')

        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, interrupted)
        try:
            before = machine.sessions()
            data['endpoints_before_allocation'] = sorted(before)
            if machine.endpoint(before) is not None:
                raise RuntimeError('The prepared session name is already present')
            machine.usage('before-allocation', required=True)
            deadline = monotonic() + SECONDS - 120
            machine.update('allocating')
            attempted = True
            machine.cli('new', '-s', job.name, '--gpu', 'L4', timeout=70)
            endpoint = machine.endpoint(machine.sessions())
            if endpoint is None or endpoint in before:
                raise RuntimeError('Could not identify a newly allocated owned endpoint')
            data['endpoint'] = endpoint
            machine.update('allocated')
            machine.usage('after-allocation')
            for name, suffix in (('payload.tar.gz', '.tar.gz'), ('worker.py', '.worker.py')):
                machine.cli('upload', '-s', job.name, job / name, machine.remote + suffix, timeout=65)
            machine.launch(deadline)
            machine.update('worker-running')
            while monotonic() + 65 < deadline:
                if machine.download('/status.json', 'remote-status.json'):
                    status = read_json(job / 'remote-status.json')
                    if status['run_id'] != job.name:
                        raise ValueError('Received status for another runtime')
                    if status['finished']:
                        collect(machine, status, deadline)
                        break
                    phase = f'remote-{status["phase"]}'
                    if data['phase'] != phase:
                        machine.update(phase)
                else:
                    machine.check_connection()
                sleep(9)
            if not data.get('verified_outputs'):
                raise TimeoutError('The owned L4 exhausted its extraction or transfer time')
        except (Exception, KeyboardInterrupt) as error:
            data['failure'] = str(error)
        finally:
            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.signal(signum, signal.SIG_IGN)
            if attempted:
                machine.update('releasing')
                machine.usage('before-stop', timeout=12)
                data['endpoint_absent_after_stop'] = machine.release()
                data['release_usage_recorded'] = machine.usage('after-stop')
                data['release_verified'] = data['endpoint_absent_after_stop'] and data['release_usage_recorded']
            success = data.get('verified_outputs') and data.get('release_verified') and 'failure' not in data
            machine.update('finished' if success else 'incomplete')
        if not success:
            raise RuntimeError(data.get('failure', 'Runtime release was not verified'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest='command', required=True)
    staging = command.add_parser('prepare')
    for option in ('packet', 'previous', 'root'):
        staging.add_argument(f'--{option}', type=Path, required=True)
    staging.add_argument('--protocol', type=Path, default=PROTOCOL)
    staging.add_argument('--raw-stops', action='store_true', help='include the FP32 final-layer state at every certified stop')
    command.add_parser('run').add_argument('job', type=Path)
    arguments = vars(parser.parse_args())
    action = arguments.pop('command')
    (prepare if action == 'prepare' else run)(**arguments)
