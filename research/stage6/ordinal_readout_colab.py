"""Stage and verify two independent ordinal fits from released V11 extractions."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tarfile
from time import monotonic, sleep
from uuid import uuid4

from research.gradio_app import cuda_verification_batch as extraction_batch
from research.gradio_app import cuda_verification_batch_worker as extraction_worker
from research.semantic_tracking import iirc_linked_colab as owned
from research.semantic_tracking.iirc_matched_colab import release
from research.stage6 import ordinal_readout_colab_worker as worker


ROOT = owned.PROJECT
METHODS = {'student-annotations': {'length_group', 'qwen_additive', 'qwen_interaction', 'blank_background'},
           'ambistory': {'form_control', 'common_prefix', 'ending'}}
EFFECT = worker.EFFECT_STAGE
KERWER = worker.KERWER_STAGE
EFFECT_PROVENANCE = {'feature_manifest': 'extraction.json', 'story_features': 'story-states.npz',
                     'sense_features': 'sense-states.npz', 'causality': 'causality.json', 'extraction_seal': 'seal.json'}


def certified_extractions(source, stages):
    path = owned.manifest_path(source)
    data = worker.read(path)
    extraction_batch.verify(source, data)
    if (data['schema_version'] != 11 or data['phase'] != 'finished' or data['release_verified'] is not True
            or not data['endpoint_absent_after_stop'] or not data['release_usage_recorded']):
        raise ValueError('Ordinal fit requires the completed and released V11 source')
    receipts = {}
    for stage in stages:
        name = worker.extraction_stage(stage)
        receipt = data['stages'][name]
        archive = source / f'{name}.tar.gz'
        if (receipt['state'] != 'complete' or receipt['verified'] is not True
                or receipt['members'].keys() != extraction_worker.OUTPUTS[name]
                or receipt['archive_bytes'] != archive.stat().st_size or receipt['archive_sha256'] != worker.digest(archive)
                or worker.unpack(archive, receipt['members']) != receipt['members']):
            raise ValueError(f'{stage} does not have a sealed verified extraction')
        receipts[stage] = {key: receipt[key] for key in ('archive_bytes', 'archive_sha256', 'members')}
    return data['run_id'], worker.digest(path), receipts


def material_paths(student, ambistory, stages, ambistory_protocol, effect_staged, effect_protocol):
    files = {name: ROOT / name for name in worker.CODE}
    results = ROOT / 'research/results'
    if 'student-annotations' in stages:
        files.update({name: student / name.removeprefix('student-annotations/prepared/') for name in worker.STUDENT_PREPARED})
        files['student-annotations/fit-protocol.json'] = results / 'student-annotations-fit-protocol.json'
    if 'ambistory' in stages:
        files.update({name: ambistory / name.removeprefix('ambistory/prepared/') for name in worker.AMBISTORY_PREPARED
                      if name.startswith('ambistory/prepared/')})
        files['ambistory/preparation.json'] = results / 'ambistory-preparation.json'
        files['ambistory/fit-protocol.json'] = ambistory_protocol
    if EFFECT in stages:
        files.update({name: ROOT / name for name in worker.EFFECT_CODE})
        files.update({name: effect_staged / name.removeprefix(f'{EFFECT}/staged/') for name in worker.EFFECT_PREPARED})
        files[f'{EFFECT}/fit-protocol.json'] = effect_protocol
    return files


def validate_frozen_inputs(files, hashes, stages):
    if EFFECT in stages:
        protocol = worker.read(files[f'{EFFECT}/fit-protocol.json'])
        prep = f'{EFFECT}/staged/preparation.json'
        prepared = worker.read(files[prep])
        if (protocol['preparation_sha256'] != hashes[prep]
                or protocol['fit_code_sha256'] != hashes['research/stage6/ambistory_effect_fit.py']
                or protocol['design_sha256'] != hashes['research/stage6/ambistory-effect-supervision.md']
                or protocol['baseline_fit_protocol_sha256'] != hashes['research/results/ambistory-fit-protocol-v2.json']
                or set(protocol['staged_members']) != {name.removeprefix(f'{EFFECT}/staged/') for name in worker.EFFECT_PREPARED}
                or any(hashes[name] != prepared['files_sha256'][name.removeprefix(f'{EFFECT}/staged/prepared/')]
                       for name in worker.EFFECT_PREPARED - {prep})):
            raise ValueError('AmbiStory effect staged files or protocol differ from their committed hashes')
    for stage, prep_field, prep_path, source_name, design, hash_key in (
        ('student-annotations', 'preparation_sha256', 'student-annotations/prepared/preparation-receipt.json',
         'student_annotations_fit.py', 'student-annotations-background.md', 'sha256'),
        ('ambistory', 'input_preparation_sha256', 'ambistory/preparation.json',
         'ambistory_fit.py', 'ambistory-context-update.md', 'files_sha256'),
    ):
        if stage not in stages:
            continue
        prepared = worker.read(files[prep_path])
        if any(hashes[name] != prepared[hash_key][name.removeprefix(f'{stage}/prepared/')]
               for name in worker.PREPARED[stage] - {prep_path}):
            raise ValueError(f'{stage} prepared inputs differ from the frozen release')
        protocol = worker.read(files[f'{stage}/fit-protocol.json'])
        if (protocol[prep_field] != hashes[prep_path]
                or protocol['fit_code_sha256'] != hashes[f'research/stage6/{source_name}']
                or protocol['design_protocol_sha256'] != hashes[f'research/stage6/{design}']
                or protocol['softmax_implementation_sha256'] != hashes['research/stage3/causal_ordinal.py']
                or protocol['length_scaler_sha256'] != hashes['research/stage3/probe.py']
                or (stage == 'ambistory' and protocol['numerical_amendment_sha256'] !=
                    hashes['research/stage6/ambistory-numerical-amendment.md'])):
            raise ValueError(f'{stage} numerical source differs from its fit preregistration')


def prepare(source, student, ambistory, output, stages, ambistory_protocol, effect_staged, effect_protocol):
    run_id, source_sha, extractions = certified_extractions(source, stages)
    files = material_paths(student, ambistory, stages, ambistory_protocol, effect_staged, effect_protocol)
    hashes = {name: worker.digest(path) for name, path in files.items()}
    extract_sources = worker.read(source / 'bundle.json')['archives'][extraction_worker.AMBISTORY_SOURCE_ARCHIVE]['members']
    if any(hashes[name] != expected for name, expected in extract_sources.items()):
        raise ValueError('AmbiStory extraction-source bytes differ from its verified V11 packet')
    validate_frozen_inputs(files, hashes, stages)
    owned.verify_code_inventory({name: path for name, path in files.items() if name.endswith('.py')})
    plan = {'source_run_id': run_id, 'source_job_manifest_sha256': source_sha, 'stages': list(stages), 'extractions': extractions,
            'materials_sha256': hashes}
    archives = {stage: source / f'{worker.extraction_stage(stage)}.tar.gz' for stage in stages}
    stage_job(output, plan, files, archives)


def prepare_kerwer(staged, prepared_receipt, source, output):
    packet = staged / 'fit-packet.json'
    data = worker.read(packet)
    staging = worker.read(staged / 'staging-receipt.json')
    prep = worker.read(prepared_receipt)
    receipt_path = source / 'run-verification.json'
    receipt = worker.read(receipt_path)
    protocol = ROOT / 'research/stage6/kerwer-reader-background-protocol.md'
    if (worker.digest(packet) != staging['packet_sha256']
            or worker.digest(staged / 'test-map.jsonl') != staging['test_map_sha256']
            or worker.digest(prepared_receipt) != staging['prepared_receipt_sha256']
            or any(data[key] != staging[key] for key in ('candidate_id', 'protocol_sha256', 'trial_count'))
            or data['candidate_id'] != prep['source']['candidate_id'] or data['trial_count'] != prep['counts']['trials']
            or data['protocol_sha256'] != prep['protocol_sha256'] or data['protocol_sha256'] != worker.digest(protocol)
            or receipt['phase'] != 'finished' or receipt['release_verified'] is not True
            or receipt['schema_version'] != 8 or receipt['checkpoint_sha256'] != data['candidate_id']
            or worker.digest(receipt_path) != prep['source']['v8_release_receipt_sha256']):
        raise ValueError('Kerwer sufficient statistics, prior candidate or protocol differ from their sealed copies')
    files = {name: ROOT / name for name in worker.KERWER_CODE}
    files[worker.KERWER_PACKET] = packet
    plan = {'source_run_id': receipt['run_id'], 'source_job_manifest_sha256': worker.digest(receipt_path),
            'stages': [KERWER], 'extractions': {}, 'materials_sha256': {name: worker.digest(path) for name, path in files.items()}}
    stage_job(output, plan, files, {})


def stage_job(output, plan, files, archives):
    name = f'cogllm-ordinal-fits-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}'
    job = output / name
    job.mkdir(mode=0o700, parents=True)
    (job / 'extractions').mkdir()
    for name, archive in archives.items():
        shutil.copyfile(archive, job / 'extractions' / f'{name}.tar.gz')
    worker.write(job / 'bundle.json', plan)
    shutil.copyfile(worker.__file__, job / 'worker.py')
    payload = job / 'payload.tar.gz'
    with tarfile.open(payload, 'w:gz', compresslevel=1) as archive:
        for member, path in sorted({**files, 'bundle.json': job / 'bundle.json'}.items()):
            archive.add(path, arcname=member, recursive=False)
    data = {'run_id': name, 'schema_version': 1, 'phase': 'prepared', 'timeline': {'prepared': owned.utc()},
            'runtime_budget_s': worker.MAX_SECONDS, 'requested_gpu': 'L4', 'fit_backend': 'cpu', 'plan': plan,
            'payload_sha256': worker.digest(payload), 'worker_sha256': worker.digest(job / 'worker.py')}
    worker.write(owned.manifest_path(job), data)
    verify(job, data)
    print(job.resolve(), flush=True)


def verify(job, data):
    plan = worker.read(job / 'bundle.json')
    payload = job / 'payload.tar.gz'
    hashes = worker.unpack(payload, worker.scope_members(plan['stages']))
    if (not worker.RUN_ID.fullmatch(job.name) or data['run_id'] != job.name or data['schema_version'] != 1
            or data['runtime_budget_s'] != worker.MAX_SECONDS or data['fit_backend'] != 'cpu' or data['plan'] != plan
            or set(plan['extractions']) != (set(plan['stages']) - {KERWER})
            or hashes.pop('bundle.json') != worker.digest(job / 'bundle.json') or hashes != plan['materials_sha256']
            or worker.digest(payload) != data['payload_sha256'] or worker.digest(job / 'worker.py') != data['worker_sha256']):
        raise ValueError('Prepared ordinal fit differs from its payload or worker seal')
    for stage, receipt in plan['extractions'].items():
        archive = job / 'extractions' / f'{stage}.tar.gz'
        if (receipt['members'].keys() != extraction_worker.OUTPUTS[worker.extraction_stage(stage)]
                or archive.stat().st_size != receipt['archive_bytes'] or worker.digest(archive) != receipt['archive_sha256']):
            raise ValueError(f'Prepared {stage} extraction differs from its original verified bytes')


def expected_keys(job, stage):
    relative = {'student-annotations': 'student-annotations/prepared/local/trial-design.jsonl',
                'ambistory': 'ambistory/prepared/test/design/stops.jsonl',
                EFFECT: f'{EFFECT}/staged/prepared/dev/design/stops.jsonl'}[stage]
    with tarfile.open(job / 'payload.tar.gz', 'r:gz') as package, package.extractfile(relative) as stream:
        rows = list(map(json.loads, stream))
    if stage == 'student-annotations':
        return {row['trial'] for row in rows}, ('trial',)
    return {(row['id'], sense['id']) for row in rows for sense in row['senses']}, ('id', 'sense')


def verify_predictions(path, expected, fields, methods, grades=5):
    remaining = expected.copy()
    with path.open(encoding='utf-8') as stream:
        for row in map(json.loads, stream):
            key = row[fields[0]] if len(fields) == 1 else tuple(row[field] for field in fields)
            if (key not in remaining or row.keys() != methods | set(fields)
                    or any(len(row[method]) != grades or not all(math.isfinite(p) and 0 < p <= 1 for p in row[method])
                           or not math.isclose(sum(row[method]), 1, rel_tol=0, abs_tol=1e-12) for method in methods)):
                raise ValueError(f'Unexpected forecast ID or ordinal probability in {path.name}')
            remaining.remove(key)
    if remaining:
        raise ValueError(f'Preregistered forecast IDs missing in {path.name}')


def verify_effect_result(job, plan, output, fit, receipt):
    extraction = plan['extractions'][EFFECT]['members']
    root = f'{EFFECT}/staged/'
    source_hashes = plan['materials_sha256']
    expected_inputs = {name.removeprefix(root + 'prepared/'): source_hashes[name]
                       for name in worker.EFFECT_PREPARED - {root + 'preparation.json'}}
    outcome = receipt['fit_outcome']
    if (outcome not in {'no_candidate', 'effect_and_baseline_forecasts_locked_before_dev_scoring'}
            or fit['status'] != outcome or fit['protocol_sha256'] != source_hashes[f'{EFFECT}/fit-protocol.json']
            or fit['source_preparation_sha256'] != source_hashes[root + 'preparation.json']
            or fit['input_sha256'] != expected_inputs or fit['fit_split'] != 'train' or fit['forecast_split'] != 'dev'
            or fit['extraction_sha256'] != {key: extraction[f'extraction/{name}'] for key, name in EFFECT_PROVENANCE.items()}
            or fit['file_sha256'] != {name: receipt['members'][name]
                                      for name in worker.fit_files(EFFECT, outcome) - {'fit-seal.json'}}
            or fit['fit_samples'] != 2268 or set(fit['methods']) != METHODS['ambistory']
            or fit['grades'] != [1, 2, 3, 4, 5]):
        raise ValueError('AmbiStory effect receipt disagrees with training features, source or protocol')
    if outcome == 'no_candidate':
        if 'selected_beta' in fit or 'forecast_samples' in fit:
            raise ValueError('An unsuccessful training selection cannot include a dev forecast')
        return
    expected, fields = expected_keys(job, EFFECT)
    if fit['forecast_samples'] != len(expected) or fit['selected_beta'] not in {1, 4, 16, 64}:
        raise ValueError('AmbiStory effect forecast count or training-selected beta differs from the protocol')
    for name in ('baseline-predictions.jsonl', 'effect-predictions.jsonl'):
        verify_predictions(output / name, expected, fields, METHODS['ambistory'])


def verify_kerwer_result(job, plan, output, fit, receipt):
    hashes = plan['materials_sha256']
    with tarfile.open(job / 'payload.tar.gz', 'r:gz') as archive, archive.extractfile(worker.KERWER_PACKET) as stream:
        packet = json.load(stream)
    if (any(fit[key] != packet[key] for key in ('candidate_id', 'protocol_sha256', 'trial_count'))
            or fit['packet_sha256'] != hashes[worker.KERWER_PACKET]
            or fit['fit_code_sha256'] != hashes['research/stage6/kerwer_reader_background_fit.py']
            or fit['kernel_code_sha256'] != hashes['research/stage6/kerwer_proportional_odds.py']
            or fit['heads_sha256'] != receipt['members']['heads.json']
            or fit['predictions_sha256'] != receipt['members']['predictions.jsonl']):
        raise ValueError('Kerwer predictions do not match their sufficient-statistic, source or weight seals')
    expected = {f'{index:04d}' for index in range(packet['trial_count'])}
    verify_predictions(output / 'predictions.jsonl', expected, ('trial',),
                       {'L', 'LP', 'T', 'TP', 'TPI', 'TP_blank', 'TPI_blank'}, grades=8)


def verify_return(job, data, stage, receipt):
    archive = job / f'{stage}-fit.tar.gz'
    output = job / 'outputs' / stage
    complete = receipt['state'] == 'complete'
    names = worker.output_files(stage, receipt.get('fit_outcome')) if complete else {'runtime.json', 'run.log'}
    if (not 0 < archive.stat().st_size == receipt['bytes'] <= worker.MAX_RETURN
            or worker.digest(archive) != receipt['sha256'] or receipt['members'].keys() != names
            or worker.unpack(archive, names) != receipt['members']):
        raise ValueError(f'Invalid returned {stage} fit archive')
    worker.unpack(archive, names, output)
    if not complete:
        return False
    plan = data['plan']
    runtime = worker.read(output / 'runtime.json')
    fit = worker.read(output / 'fit-seal.json')
    if (runtime['device'] != 'NVIDIA L4' or runtime['fit_backend'] != 'cpu'
            or runtime['source_job_manifest_sha256'] != plan['source_job_manifest_sha256']
            or runtime['packages'] != dict(value.split('==') for value in worker.REQUIREMENTS)):
        raise ValueError(f'{stage} runtime differs from the prespecified L4 environment')
    if stage == EFFECT:
        if fit['runtime'] != runtime['packages']:
            raise ValueError('AmbiStory effect fit runtime differs from its allocation receipt')
        verify_effect_result(job, plan, output, fit, receipt)
        return True
    if stage == KERWER:
        if fit['runtime'] != runtime['packages']:
            raise ValueError('Kerwer fit runtime differs from its allocation receipt')
        verify_kerwer_result(job, plan, output, fit, receipt)
        return True
    extraction = plan['extractions'][stage]['members']
    if stage == 'student-annotations':
        artifacts = {'extraction': 'extraction.json', 'features': 'features.npz', 'causality': 'causality.json'}
        weights_field, counts_field, code = 'fold_weights_sha256', 'forecast_trials', 'student_annotations_fit.py'
    else:
        artifacts = EFFECT_PROVENANCE
        weights_field, counts_field, code = 'weights_sha256', 'forecast_samples', 'ambistory_fit.py'
    expected, fields = expected_keys(job, stage)
    methods = METHODS[stage]
    if (fit['numeric_runtime'] != runtime['packages'] or fit['status'] != 'fit_and_forecasts_locked_before_scoring'
            or fit['fit_code_sha256'] != plan['materials_sha256'][f'research/stage6/{code}']
            or fit['protocol_sha256'] != plan['materials_sha256'][f'{stage}/fit-protocol.json']
            or any(fit['input_sha256'][key] != extraction[f'extraction/{name}'] for key, name in artifacts.items())
            or fit[weights_field] != receipt['members'][worker.WEIGHTS[stage]]
            or fit['predictions_sha256'] != receipt['members']['predictions.jsonl'] or fit[counts_field] != len(expected)
            or set(fit['methods']) != methods or fit['grades'] != [1, 2, 3, 4, 5]):
        raise ValueError(f'{stage} fit receipt differs from its preregistration or frozen feature source')
    verify_predictions(output / 'predictions.jsonl', expected, fields, methods)
    return True


def collect(machine, deadline):
    data, job = machine.data, machine.job
    expected = set(data['plan']['stages'])
    results = data.setdefault('stages', {})
    missing = 0
    while monotonic() + 25 < deadline:
        if machine.download('/status.json', 'remote-status.json'):
            missing = 0
            status = worker.read(job / 'remote-status.json')
            reported = set(status['stages'])
            if (any(status[key] != data[key] for key in ('run_id', 'payload_sha256', 'worker_sha256'))
                    or (reported != expected and (reported or status['phase'] not in {'setup', 'worker-failed'}))):
                raise ValueError('Ordinal fit returned status from a different payload')
            for stage, receipt in status['stages'].items():
                if stage in results or receipt['state'] not in ('complete', 'failed'):
                    continue
                name = f'{stage}-fit.tar.gz'
                if not machine.download(f'/return/{stage}.tar.gz', name, timeout=max(1, min(40, deadline - monotonic() - 15))):
                    continue
                try:
                    results[stage] = {**receipt, 'verified': verify_return(job, data, stage, receipt)}
                except (ValueError, TypeError, KeyError, OSError, tarfile.TarError) as error:
                    results[stage] = {**receipt, 'verified': False, 'error': str(error)}
                machine.update(f'retrieved-{stage}')
            if status['finished'] and (results.keys() == expected or status['phase'] == 'worker-failed'):
                if status['phase'] == 'worker-failed':
                    machine.download('.launch.log', 'remote-launch.log', timeout=10)
                return
        else:
            missing += 1
            if missing == 3:
                machine.check_connection()
                missing = 0
        sleep(5)
    raise TimeoutError('Ordinal fit reached shutdown deadline; previously verified stages were retained')


def run_owned(machine):
    data, job = machine.data, machine.job
    attempted = False

    def interrupt(signum, _):
        signal.signal(signum, signal.SIG_IGN)
        raise KeyboardInterrupt(f'Ordinal fit interrupted by signal {signum}')

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupt)
    try:
        if not re.search(rf'^Version: {re.escape(owned.CLI_VERSION)}$', machine.cli('version', timeout=20).stdout, re.M):
            raise ValueError(f'Expected Colab CLI {owned.CLI_VERSION}')
        before = machine.sessions()
        if machine.endpoint(before) is not None:
            raise ValueError('The ordinal fit runtime name is already owned')
        data['endpoints_before_allocation'] = sorted(before)
        machine.usage('before-allocation', required=True)
        machine.update('allocating')
        deadline = monotonic() + worker.MAX_SECONDS - owned.RELEASE_RESERVE_S
        attempted = True
        machine.cli('new', '-s', job.name, '--gpu', 'L4', timeout=75)
        endpoint = machine.endpoint(machine.sessions())
        if not endpoint or endpoint in before:
            raise RuntimeError('Unable to identify the owned ordinal-fit L4')
        data['endpoint'] = endpoint
        machine.update('allocated')
        machine.usage('after-allocation')
        transfers = [('payload.tar.gz', '.tar.gz'), ('worker.py', '.worker.py')]
        transfers.extend((f'extractions/{stage}.tar.gz', f'.{stage}.tar.gz') for stage in data['plan']['extractions'])
        for local, suffix in transfers:
            remaining = deadline - monotonic() - 45
            if remaining <= 0:
                raise TimeoutError('Ordinal fit transfer reached the shutdown window')
            machine.cli('upload', '-s', job.name, job / local, f'{machine.remote}{suffix}', timeout=min(65, remaining))
        machine.launch(deadline)
        machine.update('worker-running')
        collect(machine, deadline)
    except (KeyError, ValueError, TypeError, OSError, RuntimeError, subprocess.SubprocessError,
            tarfile.TarError, KeyboardInterrupt) as error:
        data['failure'] = str(error) or type(error).__name__
    finally:
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, signal.SIG_IGN)
        if attempted:
            release(machine)
        ok = ('failure' not in data and data.get('release_verified') and
              all(data.get('stages', {}).get(stage, {}).get('verified') for stage in data['plan']['stages']))
        machine.update('finished' if ok else 'failed')
    return int(not ok)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    prep = actions.add_parser('prepare')
    prep.add_argument('--source-job', type=Path, required=True)
    prep.add_argument('--student', type=Path, default=ROOT / '.cache/cogllm/studentannotations/guide-v1')
    prep.add_argument('--ambistory', type=Path, default=ROOT / '.cache/cogllm/ambistory/causal-v3')
    prep.add_argument('--only', choices=(*worker.STAGES, EFFECT))
    prep.add_argument('--ambistory-fit-protocol', type=Path, default=ROOT / 'research/results/ambistory-fit-protocol-v2.json')
    prep.add_argument('--effect-staged', type=Path, default=ROOT / '.cache/cogllm/ambistory/effect-inputs')
    prep.add_argument('--effect-fit-protocol', type=Path, default=ROOT / 'research/results/ambistory-effect-fit-protocol.json')
    prep.add_argument('--output', type=Path, required=True)
    kerwer = actions.add_parser('prepare-kerwer')
    kerwer.add_argument('--staged', type=Path, required=True)
    kerwer.add_argument('--prepared-receipt', type=Path, required=True)
    kerwer.add_argument('--source', type=Path, default=ROOT / 'research/results/gradio-candidate-l4')
    kerwer.add_argument('--output', type=Path, required=True)
    for action in ('verify', 'run', 'release'):
        actions.add_parser(action).add_argument('job', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        stages = (args.only,) if args.only else worker.STAGES
        prepare(args.source_job, args.student, args.ambistory, args.output, stages, args.ambistory_fit_protocol,
                args.effect_staged, args.effect_fit_protocol)
        return 0
    if args.action == 'prepare-kerwer':
        prepare_kerwer(args.staged, args.prepared_receipt, args.source, args.output)
        return 0
    job = args.job.resolve()
    data = worker.read(owned.manifest_path(job))
    if args.action == 'release':
        with owned.run_lock(job, 'r+'):
            data = worker.read(owned.manifest_path(job))
            if data['run_id'] != job.name or not worker.RUN_ID.fullmatch(job.name) or 'allocating' not in data['timeline']:
                raise ValueError('No owned ordinal-fit allocation to recover')
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
