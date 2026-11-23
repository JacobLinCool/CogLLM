"""Extract the served Qwen backbone's causal paragraph stops for OneStop."""

import argparse
from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import perf_counter


BREAK = '\n\n'
CHECKS = 6
PREFIX_TOLERANCE = 0.0002
WIDTH = 2048
TOKENIZER_SHA256 = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'


@dataclass(frozen=True)
class Article:
    identifier: str
    text: str
    stops: tuple[int, ...]

    @property
    def serialized(self):
        return self.text + BREAK


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read_stimuli(path, expected_sha):
    if digest(path) != expected_sha:
        raise ValueError('The OneStop stimuli do not match their sealed source bytes')
    documents = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if set(row) != {'id', 'text', 'stops'}:
                raise ValueError('OneStop model inputs may contain only article ID, body and source stops')
            case, text, stops = row['id'], row['text'], row['stops']
            if (not isinstance(text, str) or not text.strip() or sha256(text.encode()).hexdigest() != case
                    or not isinstance(stops, list) or not stops
                    or any(type(stop) is not int or not 0 < stop <= len(text) for stop in stops)
                    or stops != sorted(set(stops))
                    or any(not (text + BREAK).startswith(BREAK, stop) for stop in stops)):
                raise ValueError('Invalid OneStop article or source stop')
            documents.append(Article(case, text, tuple(stops)))
    if not documents or len({doc.identifier for doc in documents}) != len(documents):
        raise ValueError('Expected nonempty OneStop inputs with unique source fingerprints')
    return documents


def certified_stops(document, tokenizer):
    text = document.serialized
    whole = tokenizer.encode(text, add_special_tokens=False)
    positions = []
    for stop in document.stops:
        boundary = stop + len(BREAK)
        prefix = tokenizer.encode(text[:boundary], add_special_tokens=False).ids
        count = len(prefix)
        if (not count or whole.ids[:count] != prefix or whole.offsets[count - 1][1] != boundary
                or any(right > boundary for _, right in whole.offsets[:count])
                or count < len(whole.ids) and whole.offsets[count][0] < boundary):
            raise ValueError(f'Uncertified OneStop causal prefix {document.identifier} at {stop}')
        positions.append(count - 1)
    return tuple(whole.ids), tuple(positions)


def index_digest(ids, stops, tokens):
    index = list(zip(ids, stops, tokens, strict=True))
    return sha256(json.dumps(index, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def preflight(path, expected_sha, tokenizer_path):
    from tokenizers import Tokenizer

    if digest(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError('Preflight expected the candidate tokenizer configuration')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    documents = read_stimuli(path, expected_sha)
    ids, stops, indices = [], [], []
    tokens_total = 0
    for doc in documents:
        token_ids, positions = certified_stops(doc, tokenizer)
        ids.extend([doc.identifier] * len(positions))
        stops.extend(doc.stops)
        indices.extend(positions)
        tokens_total += len(token_ids)
    return {'stimuli_sha256': expected_sha, 'documents': len(documents), 'stops': len(stops),
            'input_tokens': tokens_total, 'positions_sha256': index_digest(ids, stops, indices),
            'tokenizer_sha256': TOKENIZER_SHA256}


def causal_controls(backbone, document, stop, token, reference):
    import numpy as np

    prefix = document.serialized[:stop + len(BREAK)]
    alternate = prefix + 'A later passage describes a different event in the same place.' + BREAK
    reference = reference.astype(np.float64)
    report = {}
    for name, text in (('standalone_prefix', prefix), ('replaced_suffix', alternate)):
        encoded = backbone.encode(text)
        if (encoded.tokens[:token + 1] != tuple(backbone.tokenizer.encode(prefix, add_special_tokens=False).ids)
                or encoded.offsets[token][1] != len(prefix)):
            raise ValueError('OneStop causal control changed the certified prefix tokenization')
        difference = encoded.states[token].astype(np.float64) - reference
        relative = float(np.linalg.norm(difference) / np.linalg.norm(reference))
        if not 0 <= relative <= PREFIX_TOLERANCE:
            raise ValueError(f'OneStop future context changed the paragraph state: {relative}')
        report[name] = relative
    return report


def extract(backbone, documents, output):
    import numpy as np

    total_stops = sum(len(doc.stops) for doc in documents)
    states = np.empty((total_stops, WIDTH), dtype=np.float32)
    ids, stops, indices = [], [], []
    model_calls = 0
    tokens_total = 0

    def count_forward(*_):
        nonlocal model_calls
        model_calls += 1

    candidates = [doc.identifier for doc in documents if doc.stops[0] < len(doc.text)]
    if len(candidates) < CHECKS:
        raise ValueError('Need six articles with a stop before unobserved source text')
    test_ids = {candidates[index * (len(candidates) - 1) // (CHECKS - 1)] for index in range(CHECKS)}
    checks = {}
    hook = backbone.encoder.register_forward_hook(count_forward)
    try:
        for number, doc in enumerate(documents, 1):
            tokens, positions = certified_stops(doc, backbone.tokenizer)
            previous = model_calls
            encoded = backbone.encode(doc.serialized)
            if encoded.tokens != tokens or model_calls != previous + 1:
                raise ValueError('The official encoder did not process the source in exactly one forward')
            states[len(ids):len(ids) + len(positions)] = encoded.states[list(positions)]
            ids.extend([doc.identifier] * len(doc.stops))
            stops.extend(doc.stops)
            indices.extend(positions)
            tokens_total += len(tokens)
            if doc.identifier in test_ids:
                checks[doc.identifier] = causal_controls(
                    backbone, doc, doc.stops[0], positions[0], encoded.states[positions[0]])
            if number == 1 or number % 40 == 0:
                print(json.dumps({'completed_documents': number, 'tokens_processed': tokens_total}), flush=True)
    finally:
        hook.remove()
    if model_calls != len(documents) + 2 * CHECKS or set(checks) != test_ids:
        raise ValueError('The number of observation and control forwards differs from the fixed plan')
    np.savez_compressed(output / 'features.npz', ids=np.asarray(ids), stops=np.asarray(stops, dtype=np.int32),
                        tokens=np.asarray(indices, dtype=np.int32), states=states)
    return {'documents': len(documents), 'stops': total_stops, 'input_tokens': tokens_total, 'width': WIDTH,
            'positions_sha256': index_digest(ids, stops, indices),
            'model_forwards': len(documents), 'causality_control_forwards': 2 * CHECKS, 'causality_controls': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('model', 'checkpoint', 'stimuli', 'output'):
        parser.add_argument(f'--{option}', type=Path, required=True)
    parser.add_argument('--stimuli-sha256', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    documents = read_stimuli(args.stimuli, args.stimuli_sha256)

    import torch

    from research.gradio_app.checkpoint import load_checkpoint
    from research.gradio_app.cuda import CudaCausalQwen
    from research.stage3.run import sample_gpu

    checkpoint = load_checkpoint(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    backbone = CudaCausalQwen(args.model)
    initialized = perf_counter()
    with TemporaryDirectory(prefix='.onestop-output-', dir=args.output.parent) as staging:
        state = Path(staging) / 'extract'
        state.mkdir()
        telemetry = state / 'telemetry.jsonl'
        done = Event()
        sampling = Thread(target=sample_gpu, args=(telemetry, done))
        torch.cuda.reset_peak_memory_stats()
        sampling.start()
        try:
            summary = extract(backbone, documents, state)
        finally:
            done.set()
            sampling.join()
        sealed = {
            **summary, 'candidate_id': checkpoint.identifier, 'model_revision': args.model.name,
            'device': 'NVIDIA L4 FP32', 'stimuli_sha256': args.stimuli_sha256,
            'source_code_sha256': digest(Path(__file__)), 'features_sha256': digest(state / 'features.npz'),
            'telemetry_sha256': digest(telemetry), 'initialization_seconds': initialized - start,
            'extraction_seconds': perf_counter() - initialized,
            'cuda_peak_memory_allocated_bytes': torch.cuda.max_memory_allocated(),
            'causality_tolerance': PREFIX_TOLERANCE,
        }
        (state / 'seal.json').write_text(json.dumps(sealed, indent=2, sort_keys=True, allow_nan=False) + '\n',
                                        encoding='utf-8')
        state.rename(args.output)


if __name__ == '__main__':
    main()
