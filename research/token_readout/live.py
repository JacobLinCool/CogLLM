"""Verify the selected Qwen candidate through actual Gradio HTTP on Colab L4."""

import argparse
from pathlib import Path
from time import monotonic

import torch

from research.backbone_transfer.extract import digest, stream_rows, write_json
from research.token_readout.app import CHANNELS
from research.token_readout.runner import load, verify_mock_runtime
from research.token_readout.verify_app import verify_http


PREFIX_LIMIT = 1e-3
STIMULUS_ORDER = ('prefix', 'extended', 'unicode')


def read_inputs(path):
    rows = list(stream_rows(path))
    if (len(rows) != len(STIMULUS_ORDER) or any(set(row) != {'id', 'text'} for row in rows)
            or {row['id'] for row in rows} != set(STIMULUS_ORDER)
            or any(type(row['text']) is not str or not row['text'].strip() for row in rows)):
        raise ValueError('Expected separate grade-free prefix, extended, and Unicode stimulus records')
    identities = {row['id']: row['text'] for row in rows}
    texts = [identities[name] for name in STIMULUS_ORDER]
    if (len(set(texts)) != len(texts) or not texts[1].startswith(texts[0])
            or not any(ord(ch) > 127 for ch in texts[2])):
        raise ValueError('HTTP examples need a strict text extension and a separate Unicode article')
    return texts


def prefix_verification(tokenizer, prefix, extended):
    def encode(payload):
        return tokenizer.encode(payload['text'], add_special_tokens=False)

    opening, continued = encode(prefix), encode(extended)
    n = len(opening.ids)
    if (n < 4 or opening.ids != continued.ids[:n] or opening.offsets != continued.offsets[:n]
            or n != len(prefix['tokens']) or len(continued.ids) != len(extended['tokens'])):
        raise ValueError('Extension changed the tokenization of the original prefix or the HTTP trace')
    differences = {channel: 0.0 for channel in CHANNELS}
    pointer_differences = 0
    for (start, end), left, right in zip(opening.offsets, prefix['tokens'], extended['tokens'][:n], strict=True):
        if (left['start'], left['end']) != (start, end) or (right['start'], right['end']) != (start, end):
            raise AssertionError('Prefix token coordinates changed between real HTTP requests')
        for channel in differences:
            differences[channel] = max(differences[channel], abs(left[channel] - right[channel]))
        if (left['pointer_start'], left['pointer_end']) != (right['pointer_start'], right['pointer_end']):
            pointer_differences += 1
    if pointer_differences or any(value > PREFIX_LIMIT for value in differences.values()):
        raise AssertionError(f'Actual Gradio traces depended on unread suffix: {differences}, {pointer_differences} pointers')
    return {'checked_tokens': n, 'maximum_absolute_difference': differences,
            'absolute_tolerance': PREFIX_LIMIT, 'changed_prefix_pointers': pointer_differences}


def run(candidate, model, stimuli, output, protocol_sha256, checkpoint):
    if candidate != 'qwen' or not torch.cuda.is_available() or torch.cuda.get_device_name() != 'NVIDIA L4':
        raise RuntimeError('The candidate HTTP verification requires Qwen on NVIDIA L4')
    if len(protocol_sha256) != 64 or any(char not in '0123456789abcdef' for char in protocol_sha256):
        raise ValueError('Expected the sealed protocol SHA-256')
    runner = load(model=model, checkpoint=checkpoint)
    verify_candidate(runner, stimuli, output, protocol_sha256)


def verify_candidate(runner, stimuli, output, protocol_sha256):
    texts = read_inputs(stimuli)
    mock = verify_mock_runtime()
    traces = []

    def observe(text):
        trace = runner(text)
        traces.append(trace)
        return trace

    output.mkdir(exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    started = monotonic()
    report = verify_http(observe, texts, forward_count=runner.forward_count,
                         save_preview=output / 'preview.html', preview_request=2)
    if runner.forward_count() != len(texts) or len(traces) != len(texts):
        raise AssertionError('Gradio verification did not use one backbone call per article')
    causal = prefix_verification(runner.runtime.tokenizer, traces[0], traces[1])
    write_json(output / 'causality.json', causal)
    write_json(output / 'traces.json', dict(zip(STIMULUS_ORDER, traces, strict=True)))
    verification = {**report, 'runtime': 'NVIDIA L4', 'model_revision': runner.runtime.revision,
                    'cpu_mock': mock, 'elapsed_http_seconds': monotonic() - started,
                    'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                    'actual_network': 'Gradio HTTP on loopback', 'total_backbone_forwards': runner.forward_count()}
    write_json(output / 'verification.json', verification)
    write_json(output / 'seal.json', {'protocol_sha256': protocol_sha256, 'checkpoint_sha256': runner.checkpoint,
                                      'stimuli_sha256': digest(stimuli),
                                      'artifacts_sha256': {path.name: digest(path) for path in sorted(output.iterdir())}})
    print(verification, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    for name in ('model', 'stimuli', 'output', 'checkpoint'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
