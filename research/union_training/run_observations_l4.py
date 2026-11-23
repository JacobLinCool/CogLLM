"""Run schema-2 observation training after one frozen Qwen extraction."""

import argparse
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from time import perf_counter

from research.backbone_transfer.extract import checked_digest, digest, extract, read_json, write_json


PACKET_SCHEMA = 2
FIT_FILES = frozenset({'fitting.json', 'heads.npz', 'predictions.npz'})
RESULT_FILES = FIT_FILES | {'causality.json', 'extraction.json'}


def verify_packet(packet):
    """Verify the exact observation packet members before bundling or extraction."""
    receipt = read_json(packet / 'receipt.json')
    index = read_json(packet / 'index.json')
    if index['schema_version'] != PACKET_SCHEMA or not index['tasks']:
        raise ValueError('Observation training requires a schema-2 packet with registered tasks')
    tasks = {row['name']: row['file'] for row in index['tasks']}
    if (len(tasks) != len(index['tasks']) or set(tasks) != set(receipt['endpoints'])
            or any(filename != f'{name}.jsonl' for name, filename in tasks.items())):
        raise ValueError('Observation tasks disagree with their receipt')
    expected = receipt['packet_sha256']
    if set(expected) != {'index.json', 'stimuli.jsonl'} | set(tasks.values()):
        raise ValueError('The observation receipt has missing or unexpected members')
    for name, sha256 in expected.items():
        if Path(name).name != name or digest(packet / name) != sha256:
            raise ValueError(f'Observation packet verification failed for {name}')
    for name in ('documents', 'positions', 'input_tokens'):
        if type(receipt[name]) is not int or receipt[name] < 1:
            raise ValueError(f'Observation receipt requires a positive integer for {name}')
    return receipt


def experiment(candidate, model, stimuli, output, protocol_sha256):
    if candidate != 'qwen' or stimuli.name != 'stimuli.jsonl' or output.exists():
        raise ValueError('Expected a fresh output and the registered frozen Qwen observation packet')
    packet = stimuli.parent
    receipt = verify_packet(packet)
    output.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    with TemporaryDirectory(dir=output.parent, prefix='.union-observation-fit-') as temporary:
        work = Path(temporary)
        projection, trained = work / 'projection', work / 'trained'
        extract(candidate, model, stimuli, projection, protocol_sha256,
                max_batch_documents=24, max_batch_tokens=3072)
        extraction_path = projection / 'seal.json'
        cached = read_json(extraction_path)
        if (cached['documents'] != receipt['documents'] or cached['certified_stops'] != receipt['positions']
                or cached['tokenizer_sha256'] != receipt['tokenizer_sha256']
                or cached['stimuli_sha256'] != receipt['packet_sha256']['stimuli.jsonl']
                or cached['protocol_sha256'] != protocol_sha256 or cached['gpu'] != 'NVIDIA L4'
                or cached['content_forward_per_document'] != 1):
            raise ValueError('The frozen extraction differs from the prepared observation geometry')

        from research.union_training.fit_observations import run

        print(f"Fitting {len(receipt['endpoints'])} observation blocks from {cached['certified_stops']} positions",
              flush=True)
        run(packet, projection, trained, device_name='cuda')
        artifacts = list(trained.iterdir())
        if ({path.name for path in artifacts} != FIT_FILES | {'seal.json'}
                or any(not path.is_file() or path.is_symlink() for path in artifacts)):
            raise ValueError('Observation training returned an unexpected artifact inventory')
        copyfile(projection / 'causality.json', trained / 'causality.json')
        copyfile(extraction_path, trained / 'extraction.json')
        seal_path = trained / 'seal.json'
        fit_receipt = read_json(seal_path)
        seal_path.unlink()
        write_json(seal_path, {
            **fit_receipt,
            'packet_schema_version': PACKET_SCHEMA,
            'packet_receipt_sha256': digest(packet / 'receipt.json'),
            'base_receipt_sha256': receipt['base_receipt_sha256'],
            'protocol_sha256': protocol_sha256,
            'formal_holdout_sha256': receipt['formal_holdout_sha256'],
            'formal_data_in_worker': False,
            'encoder_revision': cached['revision'],
            'encoder_gpu': cached['gpu'],
            'one_forward_per_stimulus': True,
            'experiment_seconds': perf_counter() - start,
            'files_sha256': {name: digest(trained / name) for name in sorted(RESULT_FILES)},
        })
        trained.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=('qwen',), required=True)
    for name in ('model', 'stimuli', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--protocol-sha256', type=checked_digest, required=True)
    experiment(**vars(parser.parse_args()))
