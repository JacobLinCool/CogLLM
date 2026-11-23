"""Prepare outcome-free abstracts and extract local and persistent scores in one pass."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.cache/cogllm'
RESULTS = ROOT / 'research/results'
CHECKPOINT = 'c1704d4436d767fa43bdb59e9216698514e1f6bd1d45cad4b60d0f018d84abb5'
WORD = re.compile(r"[^\W_]+(?:[’'\-][^\W_]+)*")
CHANNELS = ('unresolved', 'opened', 'resolved', 'transition')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare():
    output = CACHE / 'reading-validation'
    output.mkdir(exist_ok=True)
    packet = CACHE / 'expository-lora/prepared'
    sources = {}
    assays = json.loads((packet / 'source-audit.json').read_text())
    with (output / 'local-stimuli.jsonl').open('w') as stream:
        for study in ('plaba', 'kerwer'):
            relative = f'{study}/stimuli.jsonl'
            sources[relative] = digest(packet / relative)
            if sources[relative] != assays['files_sha256'][relative]:
                raise ValueError('Stimulus content differs from the sealed source')
            for row in rows(packet / relative):
                stimulus = {'id': f"{study}/{row['id']}", 'text': row['text']}
                stream.write(json.dumps(stimulus, ensure_ascii=False) + '\n')
    head = RESULTS / 'gradio-candidate-checkpoint/lexical/heads.npz'
    manifest = json.loads((head.parent / 'fitting.json').read_text())
    if digest(head) != manifest['artifacts']['heads.npz']['sha256']:
        raise ValueError('CompLex lexical-head checksum mismatch')
    with np.load(head, allow_pickle=False) as fits:
        method = fits['methods'].tolist().index('target_end')
        fraction = fits['fractions'].tolist().index(100)
        weight, bias = fits['weights_target_end'][fraction], fits['intercepts'][method, fraction]
        if not (np.all(weight == weight[0]) and np.all(bias == bias[0])):
            raise ValueError('Full-training lexical heads disagree')
        np.savez(output / 'local-lexical.npz', w=weight[0].astype('float64'), b=bias[0])
    seal = dict(stimuli_sha256=digest(output / 'local-stimuli.jsonl'),
                lexical_sha256=digest(output / 'local-lexical.npz'),
                original_lexical_sha256=digest(head), checkpoint_sha256=CHECKPOINT,
                protocol_sha256=digest(Path(__file__).with_name('local-protocol.md')), sources=sources)
    print(json.dumps(seal, indent=2))
    return seal


def run(runtime, reader, checkpoint, input_path, output_dir):
    import torch
    from research.stage5.complex_tokenization import token_span
    from research.token_readout.runner import trace_rows

    inputs, destination = Path(input_path), Path(output_dir)
    destination.mkdir(exist_ok=True, parents=True)
    if checkpoint['candidate'] != 'qwen' or runtime.revision != checkpoint['revision']:
        raise ValueError('Expected the pinned Qwen text backbone')
    native = reader.state_dict()
    if set(checkpoint['reader']) != set(native) or any(
        not torch.equal(native[k].cpu(), v.cpu()) for k, v in checkpoint['reader'].items()
    ):
        raise ValueError('Reader differs from its loaded checkpoint')
    lexical_file = inputs.with_name('local-lexical.npz')
    with np.load(lexical_file, allow_pickle=False) as lexical:
        w = torch.as_tensor(lexical['w'], device=runtime.device, dtype=torch.float64)
        b = float(lexical['b'])
    if w.shape != (2048,) or not torch.isfinite(w).all():
        raise ValueError('Malformed lexical readout')
    # A data-independent projection supplies a clean comparison to the trained readout projection.
    generator = np.random.default_rng(78491)
    reduced, _ = np.linalg.qr(generator.standard_normal((2048, 32)))
    projection = torch.from_numpy(reduced.astype(np.float32)).to(runtime.device)
    scale = checkpoint['scale']
    summary, files, seen = [], [], set()
    def count_call(_module, _args):
        files.append(1)
    with runtime.encoder.register_forward_pre_hook(count_call), torch.no_grad():
        for row in rows(inputs):
            key, text = row['id'], row['text']
            if key in seen or set(row) != {'id', 'text'}:
                raise ValueError('Expected distinct grade-free documents')
            seen.add(key)
            before = len(files)
            ft = runtime.encode(text)
            if len(files) != before + 1:
                raise ValueError('Expected one backbone pass per document')
            final, middle = reader.project(ft)
            scores = reader.score_projected(final, middle, ft.relations)
            scores = replace(scores, opening=scores.opening * scale['opening_gain'],
                             resolution=scores.resolution * scale['resolution_gain'] + scale['resolution_bias'])
            trace = trace_rows(text, ft.offsets, scores, checkpoint['priors'], reader.anchor_tokens)
            word_index = []
            for word in WORD.finditer(text):
                _, interval = token_span(text, ft.offsets, *word.span())
                if interval is not None:
                    end = interval[1]
                    prefix = runtime.tokenizer.encode(text[:word.end()], add_special_tokens=False).ids
                    if tuple(prefix) == ft.token_ids[:end + 1]:
                        word_index.append((*word.span(), end))
                        continue
                word_index.append((*word.span(), -1))
            hidden = torch.nn.functional.normalize(ft.hidden.double(), dim=1)
            with (destination / f'{before:03d}.npz').open('wb') as stream:
                np.savez_compressed(stream,
                    offsets=np.asarray(ft.offsets, dtype=np.int32),
                    words=np.asarray(word_index, dtype=np.int32),
                    trace=np.array([[t[c] for c in CHANNELS] for t in trace], dtype=np.float32),
                    lexical=(hidden @ w + b).clamp(0, 1).float().cpu().numpy(),
                    projection=(hidden.float() @ projection).cpu().numpy(),
                    shared=final.cpu().numpy(),
                )
            file = destination / f'{before:03d}.npz'
            summary.append(dict(id=key, file=file.name, sha256=digest(file),
                                text_sha256=hashlib.sha256(text.encode()).hexdigest(), tokens=len(ft.token_ids)))
    seal = dict(revision=runtime.revision, passes=len(files), documents=summary, channels=CHANNELS,
                stimuli_sha256=digest(inputs), lexical_sha256=digest(lexical_file),
                projection_seed=78491, code_sha256=digest(Path(__file__)))
    (destination / 'seal.json').write_text(json.dumps(seal, indent=2) + '\n')


if __name__ == '__main__':
    prepare()
