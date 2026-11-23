"""Read certified AmbiStory stops and standalone senses with one frozen Qwen backbone."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from threading import Event, Thread
from time import perf_counter


SPLITS = ('train', 'dev', 'test')
KINDS = ('definition', 'example')
STORY_FILES = tuple(f'{split}/stimuli.jsonl' for split in SPLITS)
INPUT_MEMBERS = frozenset({*STORY_FILES, 'senses.jsonl', 'shared-prefixes.jsonl'})
PROJECTION_WIDTH = 128
PROJECTION_SHA256 = '8b0fcbc13c1b64ca0784741816e04fdf758d83e5fda95ee30e1d753c518dea91'
INTERIOR_CONTROLS = 'independently certified interior boundaries of the input text'
PREFIX_TOLERANCE = 0.002


def digest(path):
    with path.open('rb') as contents:
        return hashlib.file_digest(contents, 'sha256').hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8') as contents:
        return list(map(json.loads, contents))


def read_inputs(directory):
    stories = {}
    split_counts = {}
    for name in STORY_FILES:
        rows = read_rows(directory / name)
        for row in rows:
            identifier, text, stops = (row[key] for key in ('id', 'text', 'stops'))
            if (row.keys() != {'id', 'text', 'stops'} or re.fullmatch(r'[0-9a-f]{64}', identifier) is None
                    or identifier in stories or not isinstance(text, str) or not text.strip()
                    or not isinstance(stops, list) or len(stops) not in (3, 4) or stops != sorted(set(stops))
                    or any(type(stop) is not int or not 0 < stop <= len(text) for stop in stops)
                    or stops[-1] != len(text)):
                raise ValueError('Invalid grade-free AmbiStory story row')
            stories[identifier] = row
        split_counts[name.split('/')[0]] = len(rows)
    senses = {}
    definitions = {}
    for row in read_rows(directory / 'senses.jsonl'):
        identifier, kind, text = (row[key] for key in ('id', 'kind', 'text'))
        key = (identifier, kind)
        if (row.keys() != {'id', 'kind', 'text'} or re.fullmatch(r'[0-9a-f]{64}', identifier) is None
                or identifier in stories or kind not in KINDS or key in senses
                or not isinstance(text, str) or not text.strip()):
            raise ValueError('Invalid standalone AmbiStory sense text')
        senses[key] = text
        definitions.setdefault(identifier, set()).add(kind)
    if not senses or any(set(KINDS) != kinds for kinds in definitions.values()):
        raise ValueError('Each AmbiStory sense needs one definition and one example')

    prefixes = read_rows(directory / 'shared-prefixes.jsonl')
    seen = set()
    for row in prefixes:
        cases, stops = row['ids'], row['stops']
        if (row.keys() != {'ids', 'stops'} or len(cases) != 3 or len(set(cases)) != 3
                or any(case not in stories or case in seen for case in cases)
                or len(stops) != 3 or stops != sorted(set(stops)) or any(type(stop) is not int for stop in stops)
                or any(not set(stops) <= set(stories[case]['stops']) for case in cases)
                or len({stories[case]['text'][:stops[-1]] for case in cases}) != 1):
            raise ValueError('Invalid AmbiStory shared-prefix control')
        seen.update(cases)
    if not seen or seen != stories.keys():
        raise ValueError('Shared-prefix controls must cover every AmbiStory variant')
    return stories, senses, prefixes, split_counts


def prefix_report(keys, vectors, prefixes):
    import numpy as np

    rows = dict(zip(keys, vectors, strict=True))
    pairs = [(rows[row['ids'][0], stop], rows[variant, stop]) for row in prefixes
             for stop in row['stops'] for variant in row['ids'][1:]]
    expected, actual = np.asarray(pairs, dtype=np.float64).transpose(1, 0, 2)
    delta = actual - expected
    relative = np.linalg.norm(delta, axis=1) / np.linalg.norm(expected, axis=1)
    max_relative = float(relative.max())
    if not 0 <= max_relative <= PREFIX_TOLERANCE:
        raise ValueError('AmbiStory shared-prefix states changed across ending variants')
    return {'families': len(prefixes), 'comparisons': len(pairs), 'max_relative_rms': max_relative,
            'max_absolute': float(np.max(np.abs(delta)))}


def extract_rows(backbone, documents, tokenizer, matrix, torch):
    import numpy as np

    from research.core_features.extract import batches_for, project_tokens
    from research.stage3.run import encode

    vectors = np.empty((sum(len(stops) for _, _, stops in documents), PROJECTION_WIDTH), dtype=np.float32)
    ids, offsets = [], []
    batches = 0
    for samples in batches_for(documents, 8, None):
        hidden = encode(backbone, [tokens for _, tokens, _ in samples], tokenizer.pad_token_id, torch)
        positions = [(i, position) for i, (_, _, stops) in enumerate(samples) for _, position in stops]
        chosen = hidden[[i for i, _ in positions], [position for _, position in positions]]
        projected = project_tokens(chosen, matrix, torch).cpu().numpy()
        vectors[len(ids):len(ids) + len(projected)] = projected
        ids.extend(identifier for identifier, _, stops in samples for _ in stops)
        offsets.extend(end for _, _, stops in samples for end, _ in stops)
        if batches == 0 and torch.cuda.mem_get_info()[0] < 2 * 1024**3:
            raise RuntimeError('AmbiStory pilot left less than 2 GiB L4 headroom')
        batches += 1
        del hidden, chosen, projected
    return ids, offsets, vectors, batches


def run(inputs, model, output):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    from research.core_features.extract import causal_checks, plan, token_control_documents
    from research.semantic_tracking.qasper_fit import projection
    from research.stage3.run import MODEL, REVISION, TRANSFORMERS_VERSION, sample_gpu, save_json

    started = perf_counter()
    if (model.name != REVISION or transformers.__version__ != TRANSFORMERS_VERSION
            or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4'
            or not torch.cuda.is_bf16_supported(including_emulation=False)):
        raise RuntimeError('Expected a pinned Qwen snapshot and one NVIDIA L4')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    stories, senses, prefixes, splits = read_inputs(inputs)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise ValueError('Expected the pinned fast tokenizer with a padding token')
    offsets = {}
    story_rows = list(stories.values())
    story_docs, story_rejected = plan(tokenizer, story_rows, token_offsets=offsets, batch_prefixes=True)
    sense_rows = [{'id': f'{identifier}/{kind}', 'text': text, 'stops': [len(text.rstrip())]}
                  for (identifier, kind), text in senses.items()]
    sense_docs, sense_rejected = plan(tokenizer, sense_rows, batch_prefixes=True)
    if (story_rejected or sense_rejected or len(story_docs) != len(stories) or len(sense_docs) != len(senses)):
        raise ValueError('Every supplied AmbiStory stop and standalone sense end must pass runtime certification')
    projection_matrix = projection()
    projection_sha = hashlib.sha256(projection_matrix.tobytes(order='C')).hexdigest()
    if projection_matrix.shape != (2048, PROJECTION_WIDTH) or projection_sha != PROJECTION_SHA256:
        raise ValueError('AmbiStory sign projection differs from the frozen QASPER matrix')
    diagnostics = token_control_documents(tokenizer, story_rows, story_docs, offsets)

    qwen, loading = Qwen3_5ForCausalLM.from_pretrained(
        model, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa',
        use_safetensors=True, output_loading_info=True,
    )
    if any(loading.values()) or next(qwen.model.parameters()).dtype != torch.bfloat16:
        raise ValueError(f'Pinned Qwen checkpoint did not load exactly: {loading}')
    backbone = qwen.model.eval().requires_grad_(False).to(device='cuda', dtype=torch.float32)
    del qwen
    matrix = torch.as_tensor(projection_matrix, device='cuda')
    checks = causal_checks(backbone, diagnostics, tokenizer.pad_token_id, torch, token_projection=matrix)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'causality.json', {'checks': checks, 'control_stops': INTERIOR_CONTROLS})
    if not all(row['passed'] and row['projected_tokens']['passed'] for row in checks):
        raise RuntimeError('AmbiStory causal controls failed for raw or projected states')

    torch.cuda.reset_peak_memory_stats()
    halt = Event()
    sampler = Thread(target=sample_gpu, args=(output / 'telemetry.jsonl', halt))
    sampler.start()
    try:
        story_ids, stops, story_states, story_batches = extract_rows(backbone, story_docs, tokenizer, matrix, torch)
        sense_keys, _, sense_states, sense_batches = extract_rows(backbone, sense_docs, tokenizer, matrix, torch)
    finally:
        halt.set()
        sampler.join()
    common_prefix = prefix_report(zip(story_ids, stops, strict=True), story_states, prefixes)
    sense_ids, kinds = zip(*(value.split('/') for value in sense_keys), strict=True)
    np.savez(output / 'story-states.npz', ids=np.asarray(story_ids), stops=np.asarray(stops), states=story_states)
    np.savez(output / 'sense-states.npz', ids=np.asarray(sense_ids), kinds=np.asarray(kinds), states=sense_states)
    counts = Counter(kinds)
    save_json(output / 'extraction.json', {
        'model': MODEL, 'revision': REVISION, 'transformers': transformers.__version__, 'gpu': 'NVIDIA L4',
        'dtype': 'float32', 'projection_width': PROJECTION_WIDTH, 'projection_sha256': projection_sha,
        'stories': len(stories), 'story_splits': splits, 'story_stops': len(story_ids),
        'story_tokens': sum(len(tokens) for _, tokens, _ in story_docs), 'story_batches': story_batches,
        'sense_definitions': counts['definition'], 'sense_examples': counts['example'], 'sense_batches': sense_batches,
        'sense_tokens': sum(len(tokens) for _, tokens, _ in sense_docs), 'common_prefix': common_prefix,
        'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(), 'elapsed_s': perf_counter() - started,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'model', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
