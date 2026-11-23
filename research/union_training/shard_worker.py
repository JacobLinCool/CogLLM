"""Run durable frozen-state shards and the joint fitter on one owned L4 session."""

import argparse
from contextlib import contextmanager
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
from shutil import copyfile, copyfileobj
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from time import perf_counter

from research.backbone_transfer.extract import (
    MODELS, checked_digest, digest, extract_loaded, load_frozen_backbone, read_json, write_json,
)
from research.union_training.observation_shards import assemble_shards, verify_plan, verify_shard
from research.union_training.run_observations_l4 import FIT_FILES, RESULT_FILES


PART_BYTES = 12 * 1024 * 1024
ARCHIVE_LIMIT = 180 * 1024 * 1024
MAX_BATCH_DOCUMENTS = 48
MAX_BATCH_TOKENS = 6144
PROJECTION_FILES = frozenset({'states.npy', 'coordinates.jsonl', 'preflight.json', 'causality.json', 'seal.json'})


def atomic_json(path, value):
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


@contextmanager
def work_status(root, operation):
    """Serialize changes inside this runtime and publish terminal status even on errors."""
    with (root / '.shard-worker.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = {'operation': operation, 'finished': False}
        destination = root / f'{operation}-status.json'

        def update(phase, **attributes):
            status.update(phase=phase, **attributes)
            atomic_json(destination, status)

        update('starting')
        try:
            yield update
        except BaseException as error:
            update('failed', finished=True, error=type(error).__name__, message=str(error))
            raise
        else:
            update('finished', finished=True)


def run_inputs(root):
    packet, plan_dir = root / 'packet', root / 'plan'
    plan = verify_plan(packet, plan_dir)
    receipt = read_json(packet / 'receipt.json')
    spec = read_json(MODELS)['models']['qwen']
    identity = {key: plan[key] for key in ('parent_receipt_sha256', 'protocol_sha256')}
    return packet, plan_dir, receipt, spec, identity, [item['id'] for item in plan['shards']]


def shard_id(value):
    return (isinstance(value, str) and value.startswith('shard-') and len(value) == 10
            and value[6:].isascii() and value[6:].isdigit())


def check_requested(requested, registered):
    if not requested or any(not shard_id(identifier) for identifier in requested):
        raise ValueError('Requested shards must be a nonempty ordered subset of the sealed plan')
    selected = set(requested)
    if requested != [identifier for identifier in registered if identifier in selected]:
        raise ValueError('Requested shards must be a nonempty ordered subset of the sealed plan')


def bootstrap(root, update):
    _, plan_dir, _, spec, identity, _ = run_inputs(root)
    saved = root / 'bootstrap.json'
    plan_hash = digest(plan_dir / 'plan.json')
    if saved.exists():
        previous = read_json(saved)
        if (previous['plan_sha256'] != plan_hash or previous['revision'] != spec['revision']
                or any(previous[key] != value for key, value in identity.items())
                or not Path(previous['snapshot']).is_dir()):
            raise ValueError('Runtime bootstrap receipt is inconsistent with this plan')
        update('ready', revision=previous['revision'], gpu=previous['gpu'])
        return

    update('dependencies')
    manifest = read_json(MODELS)
    version = spec.get('transformers', manifest['transformers'])
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', f'transformers=={version}',
                    'accelerate==1.13.0'], check=True, timeout=120)

    import torch
    from huggingface_hub import snapshot_download

    if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4':
        raise RuntimeError('The frozen observation run requires one NVIDIA L4')
    update('snapshot_download', gpu=torch.cuda.get_device_name(0), revision=spec['revision'])
    snapshot = Path(snapshot_download(repo_id=spec['repo'], revision=spec['revision'], token=False,
                                     allow_patterns=spec['files'], max_workers=6))
    if snapshot.name != spec['revision'] or any(not (snapshot / member).is_file() for member in spec['files']):
        raise ValueError('The pinned Qwen snapshot is incomplete')
    write_json(saved, {
        **identity, 'plan_sha256': plan_hash, 'revision': spec['revision'], 'snapshot': str(snapshot),
        'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda, 'torch': torch.__version__,
        'transformers': version, 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
    })


def raw_members(directory, expected):
    members = {path.name: path for path in directory.iterdir()}
    if set(members) != expected or any(not path.is_file() or path.is_symlink() for path in members.values()):
        raise ValueError(f'Unrecognized artifact inventory in {directory.name}')
    return members


def transport(root, identifier, directory, expected, identity):
    """Atomically expose bounded, checksummed transport pieces after a raw artifact is sealed."""
    members = raw_members(directory, expected)
    hashes = {name: digest(path) for name, path in sorted(members.items())}
    out = root / 'transport' / identifier
    if out.exists():
        prior = read_json(out / 'status.json')
        if (prior['id'] != identifier or prior['members_sha256'] != hashes
                or any(prior[key] != value for key, value in identity.items())
                or any(digest(out / entry['name']) != entry['sha256'] for entry in prior['parts'])):
            raise ValueError('Existing transport differs from its raw artifact')
        return prior

    out.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=f'.{identifier}-', dir=out.parent) as temporary:
        stage = Path(temporary) / 'transport'
        stage.mkdir()
        archive = Path(temporary) / 'artifact.tar.gz'
        with tarfile.open(archive, 'x:gz', compresslevel=1) as bundle:
            for name, path in sorted(members.items()):
                bundle.add(path, arcname=name, recursive=False)
        archive_bytes = archive.stat().st_size
        if not 0 < archive_bytes <= ARCHIVE_LIMIT:
            raise ValueError(f'Transport artifact exceeds the sealed {ARCHIVE_LIMIT}-byte envelope')
        parts = []
        archive_hash = sha256()
        with archive.open('rb') as stream:
            for index, data in enumerate(iter(lambda: stream.read(PART_BYTES), b'')):
                name = f'part-{index:03}.bin'
                (stage / name).write_bytes(data)
                archive_hash.update(data)
                parts.append({'name': name, 'bytes': len(data), 'sha256': sha256(data).hexdigest()})
        status = {
            **identity, 'id': identifier, 'phase': 'finished', 'finished': True,
            'parts': parts, 'archive_bytes': archive_bytes, 'archive_sha256': archive_hash.hexdigest(),
            'members_sha256': hashes,
        }
        write_json(stage / 'status.json', status)
        stage.rename(out)
    return status


def extract_all(root, only, update):
    _, plan, _, spec, identity, ids = run_inputs(root)
    wanted = read_json(only)
    if not isinstance(wanted, list):
        raise ValueError('Extractor needs a JSON list of shard IDs')
    check_requested(wanted, ids)
    boot = read_json(root / 'bootstrap.json')
    if any(boot[key] != value for key, value in identity.items()) or boot['plan_sha256'] != digest(plan / 'plan.json'):
        raise ValueError('The downloaded model runtime does not belong to this exposure plan')

    extracted = root / 'extracted'
    extracted.mkdir(exist_ok=True)
    backbone = None
    completed = []
    update('preparing_shards', completed=completed, requested=wanted, current=None, request_sha256=digest(only))
    for identifier in wanted:
        stimulus_dir, projection = plan / identifier, extracted / identifier
        receipt = read_json(stimulus_dir / 'receipt.json')
        stimuli = stimulus_dir / 'stimuli.jsonl'
        if (receipt['id'] != identifier or receipt['revision'] != spec['revision']
                or receipt['stimuli_sha256'] != digest(stimuli)
                or any(receipt[key] != value for key, value in identity.items())):
            raise ValueError(f'Shard input differs from its receipt: {identifier}')
        if not projection.exists():
            if backbone is None:
                update('loading_model', current=identifier)
                backbone = load_frozen_backbone('qwen', Path(boot['snapshot']))
                update('extracting', model_load_seconds=backbone.load_seconds)
            update('extracting', current=identifier)
            extract_loaded(backbone, stimuli, projection, identity['protocol_sha256'],
                           max_batch_documents=MAX_BATCH_DOCUMENTS, max_batch_tokens=MAX_BATCH_TOKENS)
        verify_shard(stimulus_dir, projection)
        seal = read_json(projection / 'seal.json')
        if (seal['max_batch_documents'], seal['max_batch_tokens']) != (MAX_BATCH_DOCUMENTS, MAX_BATCH_TOKENS):
            raise ValueError('The shard was extracted with a different batching contract')
        update('archiving', current=identifier)
        transport(root, identifier, projection, PROJECTION_FILES, identity)
        completed.append(identifier)
        update('shard_ready', completed=completed, current=None)


def join_parts(directory, identifier, expected_sha256, archive, identity):
    status = read_json(directory / 'status.json')
    if (status['id'] != identifier or status['archive_sha256'] != expected_sha256
            or any(status[key] != value for key, value in identity.items())
            or status['phase'] != 'finished' or status['finished'] is not True
            or set(status['members_sha256']) != PROJECTION_FILES):
        raise ValueError(f'Transport receipt differs from the recovery request for {identifier}')
    parts = status['parts']
    if not parts or not 0 < status['archive_bytes'] <= ARCHIVE_LIMIT:
        raise ValueError('Invalid restore transport size')
    inventory = {item.name for item in directory.iterdir()}
    if inventory != {'status.json'} | {f'part-{i:03}.bin' for i in range(len(parts))}:
        raise ValueError('Invalid restore transport part paths')
    total = 0
    hash_archive = sha256()
    with archive.open('xb') as output:
        for index, item in enumerate(parts):
            expected_name = f'part-{index:03}.bin'
            path = directory / expected_name
            if (item['name'] != expected_name or not path.is_file() or path.is_symlink()
                    or not 0 < item['bytes'] <= PART_BYTES or path.stat().st_size != item['bytes']
                    or (index + 1 < len(parts) and item['bytes'] != PART_BYTES)):
                raise ValueError('Invalid restore transport part length or order')
            data = path.read_bytes()
            if sha256(data).hexdigest() != item['sha256']:
                raise ValueError(f'Restore transport part checksum mismatch: {expected_name}')
            total += len(data)
            if total > status['archive_bytes']:
                raise ValueError('Restore transport exceeds its declared size')
            hash_archive.update(data)
            output.write(data)
    if total != status['archive_bytes'] or hash_archive.hexdigest() != expected_sha256:
        raise ValueError('The reassembled transport differs from the verified local artifact')
    return status['members_sha256']


def restore_archive(root, identifier, archive, expected_sha256, member_hashes=None):
    if archive.stat().st_size > ARCHIVE_LIMIT or digest(archive) != expected_sha256:
        raise ValueError('Restore archive differs from the verified local artifact')
    stimulus_dir = root / 'plan' / identifier
    out_root = root / 'extracted'
    out_root.mkdir(exist_ok=True)
    destination = out_root / identifier
    with TemporaryDirectory(dir=out_root, prefix=f'.restore-{identifier}-') as temporary:
        stage = Path(temporary) / 'projection'
        stage.mkdir()
        with tarfile.open(archive, 'r:gz') as compressed:
            files = compressed.getmembers()
            if (len(files) != len(PROJECTION_FILES) or {file.name for file in files} != PROJECTION_FILES
                    or any(not file.isfile() for file in files)
                    or sum(file.size for file in files) > ARCHIVE_LIMIT):
                raise ValueError('Archive contains invalid extraction artifacts')
            for item in files:
                path = stage / item.name
                with compressed.extractfile(item) as source, path.open('xb') as output:
                    copyfileobj(source, output)
                if member_hashes is not None and digest(path) != member_hashes[item.name]:
                    raise ValueError(f'Restored artifact differs from transport receipt: {item.name}')
        verify_shard(stimulus_dir, stage)
        seal = read_json(stage / 'seal.json')
        if (seal['max_batch_documents'], seal['max_batch_tokens']) != (MAX_BATCH_DOCUMENTS, MAX_BATCH_TOKENS):
            raise ValueError('Restored artifact uses the wrong extraction batching settings')
        if destination.exists():
            if digest(destination / 'seal.json') != digest(stage / 'seal.json'):
                raise ValueError(f'Refusing to replace a conflicting recovered shard {identifier}')
            verify_shard(stimulus_dir, destination)
        else:
            stage.rename(destination)


def restore_many(root, only, update):
    _, _, _, _, identity, ids = run_inputs(root)
    requested = read_json(only)
    if not isinstance(requested, list) or any(
            not isinstance(entry, dict) or set(entry) != {'id', 'sha256'} for entry in requested):
        raise ValueError('Restore request must be an ordered list of shard IDs and archive SHA-256')
    wanted = [entry['id'] for entry in requested]
    check_requested(wanted, ids)
    completed = []
    update('preparing_restore', completed=completed, requested=wanted, current=None, request_sha256=digest(only))
    for entry in requested:
        identifier, expected = entry['id'], checked_digest(entry['sha256'])
        update('restoring', current=identifier, completed=completed)
        with TemporaryDirectory(dir=root, prefix=f'.incoming-{identifier}-') as temporary:
            archive = Path(temporary) / 'artifact.tar.gz'
            members = join_parts(root / 'incoming' / identifier, identifier, expected, archive, identity)
            restore_archive(root, identifier, archive, expected, members)
        completed.append(identifier)
    update('ready', completed=completed, current=None)


def fit(root, update):
    packet, plan_dir, receipt, spec, identity, _ = run_inputs(root)
    output = root / 'fitted'
    if not output.exists():
        start = perf_counter()
        with TemporaryDirectory(dir=root, prefix='.fit-observations-') as temporary:
            work = Path(temporary)
            projection, trained = work / 'projection', work / 'trained'
            update('assembling')
            assemble_shards(packet, plan_dir, root / 'extracted', projection)
            aggregate = projection / 'seal.json'
            cached = read_json(aggregate)
            if (cached['documents'] != receipt['documents'] or cached['certified_stops'] != receipt['positions']
                    or cached['tokenizer_sha256'] != receipt['tokenizer_sha256']
                    or cached['stimuli_sha256'] != receipt['packet_sha256']['stimuli.jsonl']
                    or cached['protocol_sha256'] != identity['protocol_sha256']
                    or cached['revision'] != spec['revision'] or cached['gpu'] != 'NVIDIA L4'
                    or cached['content_forward_per_document'] != 1):
                raise ValueError('The combined extraction differs from the full observation contract')
            from research.union_training.fit_observations import run

            update('training', documents=cached['documents'], positions=cached['certified_stops'])
            continuation = root / 'continuation'

            def progress(epoch, score, best_epoch, best_score):
                if epoch % 3 == 0:
                    update('training', epoch=epoch, development_nll=score,
                           best_epoch=best_epoch, best_development_nll=best_score)

            run(packet, projection, trained, device_name='cuda',
                continuation=continuation if continuation.is_dir() else None, progress=progress)
            raw_members(trained, FIT_FILES | {'seal.json'})
            copyfile(projection / 'causality.json', trained / 'causality.json')
            copyfile(aggregate, trained / 'extraction.json')
            seal = read_json(trained / 'seal.json')
            if seal['packet_receipt_sha256'] != identity['parent_receipt_sha256']:
                raise ValueError('Training returned a different observation receipt')
            atomic_json(trained / 'seal.json', {
                **seal, **identity, 'packet_schema_version': 2,
                'base_receipt_sha256': receipt['base_receipt_sha256'],
                'formal_holdout_sha256': receipt['formal_holdout_sha256'], 'formal_data_in_worker': False,
                'encoder_revision': cached['revision'], 'encoder_gpu': cached['gpu'],
                'one_forward_per_stimulus': True, 'shard_plan_sha256': digest(plan_dir / 'plan.json'),
                'assembly_and_fit_seconds': perf_counter() - start,
                'files_sha256': {name: digest(trained / name) for name in sorted(RESULT_FILES)},
            })
            trained.rename(output)
    update('archiving')
    transport(root, 'fit', output, RESULT_FILES | {'seal.json'}, identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='command', required=True)
    for name in ('bootstrap', 'extract-all', 'restore-many', 'fit'):
        action = actions.add_parser(name)
        action.add_argument('--root', required=True, type=Path)
        if name in {'extract-all', 'restore-many'}:
            action.add_argument('--only', required=True, type=Path)
    args = parser.parse_args()
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '2'
    operations = {
        'bootstrap': ('bootstrap', bootstrap), 'extract-all': ('extract', extract_all),
        'restore-many': ('restore', restore_many), 'fit': ('fit', fit),
    }
    operation, run = operations[args.command]
    with work_status(args.root, operation) as update:
        if args.command in {'extract-all', 'restore-many'}:
            run(args.root, args.only, update)
        else:
            run(args.root, update)


if __name__ == '__main__':
    main()
