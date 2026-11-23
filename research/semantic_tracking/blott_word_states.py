"""Reconstruct grade-free causal word updates from a sealed Blott L4 export."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research.semantic_tracking import frick_consistency_score as records
from research.semantic_tracking.gum_bridge_extract import CANDIDATE, TOKENIZER_SHA256


SOURCE_RECEIPT = Path(__file__).resolve().parents[1] / 'results' / 'blott-word-source-audit.json'


@dataclass(frozen=True)
class WordUpdates:
    text: str
    spans: tuple[tuple[int, int], ...]
    token_indices: tuple[int, ...]
    ordinals: tuple[int, ...]
    features: dict


def word_updates(text, words, token_ends, states_by_end):
    spans, indices, ordinals, surface, deltas = [], [], [], [], []
    for i, (previous, word) in enumerate(zip(words, words[1:]), 2):
        prior_start, prior_end = previous
        start, end = word
        if prior_end not in states_by_end or end not in states_by_end:
            continue
        spans.append((start, end))
        indices.append(token_ends[end])
        ordinals.append(i)
        surface.append((end, end - start, prior_end - prior_start))
        deltas.append(states_by_end[end] - states_by_end[prior_end])
    if not spans:
        raise ValueError('A Blott stimulus has no exact adjacent word states to score')
    order = np.log(ordinals)
    state_difference = np.stack(deltas)
    change_norm = np.einsum('ij,ij->i', state_difference, state_difference, optimize=False) / 2
    return WordUpdates(text, tuple(spans), tuple(indices), tuple(ordinals), {
        'position': np.column_stack((order, np.square(order))),
        'surface': np.log(surface), 'change_norm': change_norm[:, None], 'delta': state_difference,
    })


def load_updates(work, extraction, tokenizer_path, protocol_sha):
    from tokenizers import Tokenizer

    source = records.read_json(SOURCE_RECEIPT)
    if any(records.digest(work / name) != sha for name, sha in source['packet_sha256'].items()):
        raise ValueError('The frozen Blott word packet differs from the grade-free source audit')
    if records.digest(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError('Only the frozen Qwen tokenizer certifies original token indices')
    tokenize = Tokenizer.from_file(str(tokenizer_path))
    packet = work / 'packet'
    seal_path = extraction / 'seal.json'
    seal = records.read_json(seal_path)
    expected = {'candidate_id': CANDIDATE, 'protocol_sha256': protocol_sha, 'gpu': 'NVIDIA L4',
                'assay_sha256': records.digest(packet / 'preflight.json'),
                'preflight_sha256': records.digest(packet / 'extractor-preflight.json'),
                'stimuli_sha256': records.digest(packet / 'stimuli.jsonl'), 'tokenizer_sha256': TOKENIZER_SHA256,
                'states_dtype': 'float32', 'raw_stops': True, 'content_forward_per_document': 1,
                'independent_causal_checks': 12}
    files = ('states.npy', 'coordinates.jsonl', 'causality.json')
    if (any(seal.get(k) != v for k, v in expected.items())
            or any(records.digest(extraction / name) != seal['file_sha256'][name] for name in files)):
        raise ValueError('The raw word-state export does not match the sealed benchmark')
    checks = records.read_json(extraction / 'causality.json')['checks']
    if len(checks) != 12 or not all(c['passed'] and c['projected_tokens']['passed'] for c in checks):
        raise ValueError('The exported word states missed a required independent causal check')
    stimuli = records.indexed(records.read_rows(packet / 'stimuli.jsonl'), 'Blott word stimuli')
    layout = records.indexed(records.read_rows(work / 'prepared' / 'layout.jsonl'), 'Blott word layout')
    coordinates = records.indexed(records.read_rows(extraction / 'coordinates.jsonl'), 'Blott word coordinates')
    if stimuli.keys() != layout.keys() or stimuli.keys() != coordinates.keys() or len(stimuli) != 276:
        raise ValueError('Blott extraction must cover the registered 276 sentences exactly')
    shape = (4533, 2048)
    states = np.load(extraction / 'states.npy', mmap_mode='r', allow_pickle=False)
    if states.shape != shape or seal['states_shape'] != list(shape) or states.dtype != np.float32:
        raise ValueError('The FP32 state array failed shape or dtype validation')

    result, offset = {}, 0
    for key, coordinate in coordinates.items():
        stops, item, text = coordinate['stops'], layout[key], stimuli[key]['text']
        if stops != stimuli[key]['stops'] or stops[-1] != len(text) or item['sentence_stop'] != len(text):
            raise ValueError('A word-state export failed source-coordinate alignment')
        chunk = states[offset:offset + len(stops)].astype(np.float64)
        offset += len(stops)
        norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        if not np.isfinite(norms).all() or np.any(norms <= 0):
            raise ValueError('Every candidate state must have a finite nonzero norm')
        unit_states = dict(zip(stops, chunk / norms, strict=True))
        words = tuple(tuple(row['span']) for row in item['words'])
        encoded = tokenize.encode(text, add_special_tokens=False)
        token_ends = {end: i for i, (start, end) in enumerate(encoded.offsets) if start < end}
        if not set(stops) <= token_ends.keys():
            raise ValueError('A certified stop lacks its required final Qwen token')
        candidates = word_updates(text, words, token_ends, unit_states)
        if candidates.spans != tuple(tuple(row['span']) for row in item['words'] if row['update_eligible']):
            raise ValueError('Words must be scored using their immediately preceding certified lexical word')
        for n in range(candidates.ordinals[0], len(words)):
            end = words[n - 1][1]
            short = word_updates(text[:end], words[:n], token_ends, {k: v for k, v in unit_states.items() if k <= end})
            for name, array in short.features.items():
                if not np.array_equal(array, candidates.features[name][:len(array)]):
                    raise ValueError('A candidate feature depends on text after its visible word')
        result[key] = candidates
    if offset != len(states) or sum(len(item.ordinals) for item in result.values()) != 3981:
        raise ValueError('Blott did not reproduce every source stop and eligible adjacent word update')
    return result, {'extraction_sha256': records.digest(seal_path), 'source_sha256': records.digest(SOURCE_RECEIPT),
                    'input_documents': len(result), 'certified_stops': offset, 'eligible_updates': 3981}
