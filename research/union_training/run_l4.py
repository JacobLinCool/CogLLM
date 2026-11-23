"""Extract sealed development stimuli once and fit the common endpoint models on L4."""

import argparse
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from research.backbone_transfer.extract import extract, digest, read_json, stream_rows, write_json
from research.union_training.fit import BATCH_SIZE, EPOCHS, METHODS, SEED, run


def pack_features(packet, extraction, output):
    output.mkdir()
    state_path = extraction / 'states.npy'
    states = np.load(state_path, mmap_mode='r', allow_pickle=False)
    indices = {}
    for row in stream_rows(extraction / 'coordinates.jsonl'):
        for stop in row['stops']:
            key = row['id'], stop
            if key in indices:
                raise ValueError('Duplicate frozen state coordinate')
            indices[key] = len(indices)
    if states.shape != (len(indices), 2048) or states.dtype != np.float32:
        raise ValueError('Incomplete or incompatible frozen Qwen states')
    index = read_json(packet / 'index.json')
    tasks = []
    for task in index['tasks']:
        name = task['name']
        records = list(stream_rows(packet / task['file']))
        filename = f'{name}.npz'
        chosen = [indices[row['doc'], row['stop']] for row in records]
        np.savez(output / filename,
                 features=states[chosen],
                 outcomes=np.asarray([row['outcome'] for row in records], dtype=np.float64),
                 split=np.asarray([row['split'] for row in records], dtype=np.int8),
                 id=np.asarray([row['id'] for row in records]),
                 group=np.asarray([row['family'] for row in records]))
        tasks.append({k: task[k] for k in ('name', 'kind', 'labels', 'effects')} | {'file': filename})
    write_json(output / 'index.json', {'tasks': tasks})
    return index


def development_effects(design, prediction_path, output):
    result = {}
    with np.load(prediction_path, allow_pickle=False) as forecasts:
        for task in design['tasks']:
            name = task['name']
            if not task['dev_effects']:
                continue
            endpoints = []
            for effect in task['dev_effects']:
                difference = {}
                for method in METHODS:
                    values = forecasts[f'{method}.{name}.predictions'][effect['indices']]
                    if task['kind'] == 'gaussian':
                        scalar = values[:, 0]
                    elif effect['metric'] == 'category':
                        scalar = values[:, effect['category']]
                    else:
                        scalar = values @ np.arange(values.shape[1])
                    estimate = float(np.dot(effect['coefficients'], scalar))
                    difference[method] = {'prediction': estimate, 'error': estimate - effect['target']}
                endpoints.append({k: effect[k] for k in ('label', 'target', 'matched_pairs', 'scale')}
                                 | {'methods': difference})
            result[name] = endpoints
    write_json(output, result)


def experiment(candidate, model, stimuli, output, protocol_sha256):
    packet = stimuli.parent
    receipt_path = packet / 'receipt.json'
    receipt = read_json(receipt_path)
    if any(digest(packet / name) != value for name, value in receipt['packet_sha256'].items()):
        raise ValueError('Development inputs differ from the prepared corpus receipt')
    if candidate != 'qwen' or output.exists():
        raise ValueError('Expected a fresh directory and the registered frozen Qwen')
    start = perf_counter()
    with TemporaryDirectory(dir=output.parent, prefix='.union-fit-') as directory:
        work = Path(directory)
        projection, arrays, trained = (work / name for name in ('projection', 'arrays', 'trained'))
        extract(candidate, model, stimuli, projection, protocol_sha256,
                max_batch_documents=24, max_batch_tokens=3072)
        extraction_seal = read_json(projection / 'seal.json')
        if (extraction_seal['documents'] != receipt['development_documents']
                or extraction_seal['tokenizer_sha256'] != receipt['tokenizer_sha256']):
            raise ValueError('Runtime causal geometry differs from the development packet')
        design = pack_features(packet, projection, arrays)
        print(f"Packed {len(design['tasks'])} aligned human endpoints on L4", flush=True)
        run(arrays, trained, 'cuda', EPOCHS, BATCH_SIZE, SEED)
        development_effects(design, trained / 'predictions.npz', trained / 'development-effects.json')
        copyfile(projection / 'causality.json', trained / 'causality.json')
        copyfile(projection / 'seal.json', trained / 'extraction.json')
        fitting = read_json(trained / 'seal.json')
        (trained / 'seal.json').unlink()
        write_json(trained / 'seal.json', {
            **fitting, 'protocol_sha256': protocol_sha256, 'packet_receipt_sha256': digest(receipt_path),
            'formal_holdout_sha256': receipt['formal_holdout_sha256'], 'formal_data_in_worker': False,
            'encoder_revision': extraction_seal['revision'], 'encoder_gpu': extraction_seal['gpu'],
            'one_forward_per_stimulus': extraction_seal['content_forward_per_document'] == 1,
            'experiment_seconds': perf_counter() - start,
            'files_sha256': {path.name: digest(path) for path in sorted(trained.iterdir())},
        })
        trained.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    for name in ('model', 'stimuli', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    experiment(**vars(parser.parse_args()))
