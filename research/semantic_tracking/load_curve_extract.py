"""Collect every native Qwen token from a single causal backbone call per text."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from research.backbone_transfer.extract import (
    MAX_INPUT_TOKENS, Document, batches, causal_controls, checked_digest, digest, forward,
    load_language_model, stream_rows, verified_model, write_json, write_rows,
)


PROJECTION_SEED = 38542
PROJECTION_WIDTH = 160
MAX_CORPUS_TOKENS = 135_000


def documents_from(path, tokenizer):
    documents = {}
    token_count = 0
    for row in stream_rows(path):
        if row.keys() != {'id', 'text'}:
            raise ValueError('Dense extraction accepts only opaque IDs and unannotated text')
        key, text = row['id'], row['text']
        if not isinstance(key, str) or not key or key in documents or not isinstance(text, str) or not text.strip():
            raise ValueError('Invalid or repeated text identity')
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if (not 3 <= len(encoded.ids) <= MAX_INPUT_TOKENS or len(encoded.ids) != len(encoded.offsets)
                or any(not 0 <= a < b <= len(text) for a, b in encoded.offsets)
                or any(a[1] > b[1] for a, b in zip(encoded.offsets, encoded.offsets[1:]))):
            raise ValueError(f'Unexpected native Qwen token geometry for {key}')
        documents[key] = Document(key, text, encoded.ids, encoded.offsets, {}, {})
        token_count += len(encoded.ids)
    if not documents or token_count > MAX_CORPUS_TOKENS:
        raise ValueError('Dense corpus is empty or exceeds the registered token budget')
    return documents, token_count


def extract(model, stimuli, output, protocol_sha256):
    import numpy as np
    import torch
    import transformers
    from tokenizers import Tokenizer

    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['USE_HUB_KERNELS'] = 'NO'
    if output.exists():
        raise FileExistsError(output)
    _, model_spec = verified_model(model, 'qwen', tokenizer_only=True)
    tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
    documents, count = documents_from(stimuli, tokenizer)
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    encoder, spec = load_language_model(model, 'qwen', torch, transformers)
    load_seconds = perf_counter() - start
    controls = causal_controls(encoder, documents, tokenizer, spec['pad_id'], torch)
    projection = np.random.default_rng(PROJECTION_SEED).standard_normal(
        (spec['hidden_width'], PROJECTION_WIDTH), dtype=np.float32)
    projection = torch.from_numpy(projection).to('cuda')
    index = 0
    row_slices = {}
    for key, row in documents.items():
        row_slices[key] = slice(index, index + len(row.ids))
        index += len(row.ids)
    if index != count or spec != model_spec:
        raise RuntimeError('Corpus or model inventory changed during extraction')

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.load-curve-') as folder:
        stage = Path(folder) / 'extraction'
        stage.mkdir()
        write_json(stage / 'causality.json', controls)
        write_rows(stage / 'coordinates.jsonl', (
            {'id': key, 'token_ids': row.ids, 'offsets': row.offsets} for key, row in documents.items()
        ))
        preflight = {'documents': len(documents), 'tokens': count,
                     'maximum_document_tokens': max(len(row.ids) for row in documents.values())}
        write_json(stage / 'preflight.json', preflight)
        shape = (count, PROJECTION_WIDTH)
        states = np.lib.format.open_memmap(stage / 'states.npy', mode='w+', dtype=np.float32, shape=shape)
        start = perf_counter()
        calls = 0
        for batch in batches(documents):
            states_cuda = forward(encoder, [row.ids for row in batch], spec['pad_id'], torch)
            with torch.inference_mode():
                features = (torch.nn.functional.normalize(states_cuda, dim=-1) @ projection).cpu().numpy()
            if not np.isfinite(features).all():
                raise RuntimeError('Nonfinite native-token projection')
            for i, row in enumerate(batch):
                states[row_slices[row.identifier]] = features[i, :len(row.ids)]
            del states_cuda
            calls += 1
        states.flush()
        del states
        write_json(stage / 'seal.json', {
            **preflight, 'candidate': 'qwen', 'model': spec['repo'], 'revision': spec['revision'],
            'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda, 'torch': str(torch.__version__),
            'transformers': transformers.__version__, 'compute_dtype': 'float32', 'states_dtype': 'float32',
            'shape': shape, 'projection_seed': PROJECTION_SEED,
            'protocol_sha256': protocol_sha256, 'stimuli_sha256': digest(stimuli),
            'tokenizer_sha256': spec['sha256']['tokenizer.json'],
            'source_sha256': digest(Path(__file__)), 'content_forward_per_document': 1,
            'content_forward_batches': calls, 'independent_causal_checks': len(controls['checks']),
            'model_load_and_verification_s': load_seconds, 'forward_projection_export_s': perf_counter() - start,
            'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
            'file_sha256': {name: digest(stage / name) for name in
                            ('causality.json', 'coordinates.jsonl', 'preflight.json', 'states.npy')},
        })
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=('qwen',), required=True)
    for name in ('model', 'stimuli', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--protocol-sha256', type=checked_digest, required=True)
    args = parser.parse_args()
    extract(args.model, args.stimuli, args.output, args.protocol_sha256)


if __name__ == '__main__':
    main()
