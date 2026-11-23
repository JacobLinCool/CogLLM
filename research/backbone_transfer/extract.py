"""Extract comparable causal text states from pinned small language models on one L4."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from time import perf_counter


MODELS = Path(__file__).with_name('models.json')
MAX_INPUT_TOKENS = 2048
MAX_BATCH_TOKENS = 1024
MAX_BATCH_DOCUMENTS = 8
MAX_PREFIX_BATCH = 128
MAX_CONTROLS = 6
RELATIVE_RMS_LIMIT = 0.001
REPLACEMENT_SUFFIXES = (' A separate note follows.', '\n\nThis is a different passage.')


@dataclass(frozen=True)
class Document:
    identifier: str
    text: str
    ids: list[int]
    offsets: list[tuple[int, int]]
    certified: dict[int, int]
    missing: dict[int, str]


@dataclass(frozen=True)
class FrozenBackbone:
    candidate: str
    model: Path
    encoder: object
    spec: dict
    load_seconds: float


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, content):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(content, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def stream_rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def write_rows(path, content):
    with path.open('x', encoding='utf-8') as stream:
        for row in content:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def checked_digest(value):
    if re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise argparse.ArgumentTypeError('Expected a lower-case SHA-256 digest')
    return value


def verified_model(model, alias, *, tokenizer_only=False):
    manifest = read_json(MODELS)
    spec = manifest['models'][alias]
    hashes = spec['sha256']
    if manifest['schema'] != 1 or set(spec['files']) != set(hashes):
        raise ValueError('The model file and SHA-256 inventories must match')
    files = ('tokenizer.json',) if tokenizer_only else spec['files']
    if not tokenizer_only and model.name != spec['revision']:
        raise ValueError('Expected the named local immutable model snapshot')
    for name in files:
        if digest(model / name) != hashes[name]:
            raise ValueError(f'{alias} snapshot file digest differs: {name}')
    if not tokenizer_only:
        local_code = {path.name for path in model.glob('*.py')}
        if local_code != {name for name in files if name.endswith('.py')}:
            raise ValueError('Snapshot contains Python files outside the audited code allowlist')
    return spec.get('transformers', manifest['transformers']), spec


def load_documents(path, tokenizer):
    documents = {}
    for row in stream_rows(path):
        if not isinstance(row, dict) or set(row) != {'id', 'text', 'stops'}:
            raise ValueError('Inference input accepts only id, exact text, and Unicode end stops')
        key, text, stops = (row[name] for name in ('id', 'text', 'stops'))
        if (type(key) is not str or not key or key in documents or type(text) is not str
                or not text.strip() or type(stops) is not list or not stops
                or any(type(stop) is not int or not 0 < stop <= len(text) for stop in stops)
                or stops != sorted(set(stops))):
            raise ValueError('A source document identity, text or stop is invalid')
        encoded = tokenizer.encode(text, add_special_tokens=False)
        documents[key] = certify(key, text, stops, encoded, tokenizer)
    if not documents or not any(doc.certified for doc in documents.values()):
        raise ValueError('No exact text states could be certified')
    return documents


def certify(key, text, stops, encoded, tokenizer):
    ids, offsets = encoded.ids, encoded.offsets
    if (not ids or len(ids) != len(offsets)
            or any(not 0 <= left <= right <= len(text) for left, right in offsets)):
        raise ValueError(f'Invalid Unicode tokenization for {key}')
    if len(ids) > MAX_INPUT_TOKENS:
        raise ValueError(f'{key} exceeds the registered extraction length of {MAX_INPUT_TOKENS} tokens')

    ends = {right: i for i, (left, right) in enumerate(offsets) if left < right}
    candidates = [(stop, ends[stop]) for stop in stops if stop in ends]
    missing = {stop: 'no_token_end' for stop in stops if stop not in ends}
    certified = {}
    for offset in range(0, len(candidates), MAX_PREFIX_BATCH):
        batch = candidates[offset:offset + MAX_PREFIX_BATCH]
        independent = tokenizer.encode_batch([text[:stop] for stop, _ in batch], add_special_tokens=False)
        for (stop, index), prefix in zip(batch, independent, strict=True):
            if prefix.ids != ids[:index + 1] or any(right > stop for _, right in offsets[:index + 1]):
                missing[stop] = 'unstable_prefix'
            else:
                certified[stop] = index
    return Document(key, text, ids, offsets, certified, dict(sorted(missing.items())))


def target_geometry(path, documents):
    if path is None:
        return []
    from research.stage5.complex_tokenization import token_span

    records, seen = [], set()
    for row in stream_rows(path):
        if not isinstance(row, dict) or set(row) != {'id', 'start', 'end'} or row['id'] not in documents:
            raise ValueError('Target coordinates accept only an existing model-input id and start/end')
        key, start, stop = row['id'], row['start'], row['end']
        doc = documents[key]
        if (type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(doc.text)
                or (stop not in doc.certified and stop not in doc.missing) or (key, start, stop) in seen):
            raise ValueError(f'Unexpected or duplicate grade-free target in {key}')
        seen.add((key, start, stop))
        status, pair = token_span(doc.text, doc.offsets, start, stop)
        if stop not in doc.certified:
            status = doc.missing[stop]
        elif pair is not None and pair[1] != doc.certified[stop]:
            status = 'target_last_token_differs_from_prefix_endpoint'
        eligible = status == 'exact_nonwhite_span'
        records.append({
            **row, 'status': status, 'eligible': eligible,
            'start_token': pair[0] if eligible else None,
            'end_token': pair[1] if eligible else None,
        })
    if not records:
        raise ValueError('An accompanying target file must contain at least one grade-free span')
    return records


def geometry(alias, tokenizer_root, stimuli, targets, protocol_sha256):
    from tokenizers import Tokenizer, __version__

    _, spec = verified_model(tokenizer_root, alias, tokenizer_only=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_root / 'tokenizer.json'))
    documents = load_documents(stimuli, tokenizer)
    target_rows = target_geometry(targets, documents)
    errors = Counter(reason for row in documents.values() for reason in row.missing.values())
    result = {
        'candidate': alias,
        'model': spec['repo'],
        'revision': spec['revision'],
        'protocol_sha256': protocol_sha256,
        'stimuli_sha256': digest(stimuli),
        'targets_sha256': digest(targets) if targets else None,
        'tokenizer_sha256': spec['sha256']['tokenizer.json'],
        'tokenizers': __version__,
        'special_tokens': False,
        'documents': len(documents),
        'input_tokens': sum(len(row.ids) for row in documents.values()),
        'max_input_tokens': max(len(row.ids) for row in documents.values()),
        'requested_stops': sum(len(row.certified) + len(row.missing) for row in documents.values()),
        'certified_stops': sum(len(row.certified) for row in documents.values()),
        'unavailable_stops_by_reason': dict(errors),
        'details': [{
            'id': row.identifier,
            'stops': list(row.certified),
            'token_indices': list(row.certified.values()),
            'input_tokens': len(row.ids),
            'tokenization_sha256': sha256(json.dumps([row.ids, row.offsets], separators=(',', ':')).encode()).hexdigest(),
            'unavailable': [{'stop': stop, 'reason': reason} for stop, reason in row.missing.items()],
        } for row in documents.values()],
        'target_geometry': target_rows,
        'target_geometry_by_status': dict(Counter(row['status'] for row in target_rows)),
    }
    return tokenizer, documents, result


def lfm_config(model):
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    raw = read_json(model / 'config.json')
    attention_types = ('full_attention', 'sliding_attention')
    if (raw['model_type'] != 'lfm3' or set(raw['rope_parameters']) != set(attention_types)
            or set(raw['layer_types']) != {'conv', *attention_types}):
        raise ValueError('Expected the exact LFM3 convolution and attention configuration')
    cls = get_class_from_dynamic_module('configuration_lfm3.Lfm3Config', model, local_files_only=True)
    if cls._attention_layer_types != attention_types:
        raise ValueError('The LFM3 config class disagrees with its pinned attention layer types')
    legacy_validator = cls.validate_layer_type
    validators = [check for check in cls.__class_validators__ if check is not legacy_validator]
    if len(validators) != len(cls.__class_validators__) - 1 or cls.validate_architecture not in validators:
        raise ValueError('Unexpected LFM3 configuration validators')
    standardize = cls.standardize_rope_params

    def preserve_attention_rope(config):
        # Transformers adds flat defaults intended for single-RoPE models. LFM3 already specifies both attention maps.
        if (config.rope_parameters.pop('rope_theta') != config.default_theta
                or config.rope_parameters != raw['rope_parameters']):
            raise ValueError('Unexpected change to the original LFM3 attention rotary parameters')
        standardize(config)

    cls._rope_type_labels = attention_types
    cls.standardize_rope_params = preserve_attention_rope
    # LFM3's author validator already checks the layer sequence and keeps the actual "conv" type.
    cls.__class_validators__ = validators
    config = cls.from_pretrained(model, local_files_only=True)
    if config.layer_types != raw['layer_types'] or config.rope_parameters != raw['rope_parameters']:
        raise ValueError('Unexpected LFM3 metadata after Hugging Face config normalization')
    config._attn_implementation = 'sdpa'
    return config


def load_language_model(model, alias, torch, transformers, *, serving_device=None):
    version, spec = verified_model(model, alias)
    if transformers.__version__ != version:
        raise RuntimeError(f'Expected Transformers {version} for the pinned model')
    if serving_device is None:
        if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4':
            raise RuntimeError('The research extraction protocol requires one NVIDIA L4')
        device = 'cuda'
    else:
        if alias != 'qwen' or serving_device not in ('cpu', 'cuda'):
            raise ValueError('The public writer requires Qwen on CPU or CUDA')
        device = serving_device
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32 = False
    options = {
        'local_files_only': True,
        'attn_implementation': 'sdpa',
        'dtype': torch.float32,
        'device_map': {'': device},
        'use_safetensors': True,
        'use_kernels': False,
        'output_loading_info': True,
    }
    ignored = ()
    if alias == 'gemma4':
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

        outer = transformers.AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        if outer.model_type != 'gemma4' or outer.text_config.model_type != 'gemma4_text':
            raise ValueError('Expected the Gemma 4 text subconfiguration')
        outer.text_config._attn_implementation = 'sdpa'
        text_model, loading = Gemma4TextModel.from_pretrained(
            model, config=outer.text_config, key_mapping={r'^model\.language_model\.': ''}, **options,
        )
        ignored = ('model.audio_tower.', 'model.embed_audio.', 'model.embed_vision.', 'model.vision_tower.')
    else:
        if alias == 'lfm3':
            # This local snapshot contains only the three reviewed, hash-pinned author source files.
            config = lfm_config(model)
            whole, loading = transformers.AutoModelForCausalLM.from_pretrained(
                model, config=config, trust_remote_code=True, **options,
            )
            if whole.config.layer_types != config.layer_types or whole.config.rope_parameters != config.rope_parameters:
                raise RuntimeError('Loading changed an author-defined LFM3 layer or attention rotary setting')
        else:
            whole, loading = transformers.Qwen3_5ForCausalLM.from_pretrained(model, **options)
        if type(whole).__name__ != spec['architecture']:
            raise RuntimeError('Resolved causal model differs from the official manifest')
        text_model = whole.model
        del whole
    unexpected = [key for key in loading.get('unexpected_keys', ()) if not key.startswith(ignored)]
    if unexpected or any(value for field, value in loading.items() if field != 'unexpected_keys'):
        raise RuntimeError(f'The requested text backbone did not load exactly: {loading}')
    if (text_model.config.hidden_size != spec['hidden_width'] or len(text_model.layers) != spec['layers']
            or any(value.dtype != torch.float32 or value.device.type != device for value in text_model.parameters())):
        raise RuntimeError(f'Loaded text layers must match the frozen architecture and FP32 {device} contract')
    return text_model.eval().requires_grad_(False), spec


def forward(encoder, sequences, pad_id, torch):
    length = max(map(len, sequences))
    ids = [row + [pad_id] * (length - len(row)) for row in sequences]
    masks = [[1] * len(row) + [0] * (length - len(row)) for row in sequences]
    with torch.inference_mode():
        answer = encoder(input_ids=torch.tensor(ids, device='cuda', dtype=torch.long),
                         attention_mask=torch.tensor(masks, device='cuda', dtype=torch.long),
                         use_cache=False, output_hidden_states=False, output_attentions=False)
    states = answer.last_hidden_state
    if states.dtype != torch.float32 or states.shape[:2] != (len(ids), length):
        raise RuntimeError('Expected FP32 final normalized hidden states for every input token')
    return states


def causal_controls(encoder, documents, tokenizer, pad_id, torch):
    checks = []
    # Spread fixed, label-free controls across the input order and document lengths.
    ordered = sorted(documents.values(), key=lambda doc: (len(doc.ids), doc.identifier))
    positions = sorted({round((len(ordered) - 1) * i / (MAX_CONTROLS - 1)) for i in range(MAX_CONTROLS)})
    for i in positions:
        doc = ordered[i]
        candidates = sorted(
            {(right, index) for index, (left, right) in enumerate(doc.offsets)
             if left < right and 2 <= index < len(doc.ids) - 1},
            key=lambda pair: abs(pair[1] - len(doc.ids) // 2),
        )
        for end, index in candidates:
            truncated = tokenizer.encode(doc.text[:end], add_special_tokens=False).ids
            if truncated != doc.ids[:index + 1] or any(right > end for _, right in doc.offsets[:index + 1]):
                continue
            alternatives = tokenizer.encode_batch([doc.text[:end] + suffix for suffix in REPLACEMENT_SUFFIXES],
                                                  add_special_tokens=False)
            replacement = next((item.ids for item in alternatives
                                if item.ids[:len(truncated)] == truncated and len(item.ids) > len(truncated)
                                and item.ids != doc.ids
                                and all(right <= end for _, right in item.offsets[:len(truncated)])), None)
            if replacement is not None:
                break
        else:
            continue
        whole = forward(encoder, [doc.ids], pad_id, torch)[0, :len(truncated)]
        for variant, tokens in (('truncation', truncated), ('replacement', replacement)):
            prefix = forward(encoder, [tokens], pad_id, torch)[0, :len(truncated)]
            error = (whole - prefix).square().mean(dim=-1).sqrt()
            scale = prefix.square().mean(dim=-1).sqrt().clamp_min(1e-8)
            relative = float((error / scale).max().item())
            absolute = float((whole - prefix).abs().max().item())
            passed = bool(torch.isfinite(whole).all().item() and torch.isfinite(prefix).all().item()
                          and relative <= RELATIVE_RMS_LIMIT)
            checks.append({'id': doc.identifier, 'variant': variant, 'character_end': end,
                           'prefix_tokens': len(truncated), 'max_relative_rms': relative,
                           'max_absolute': absolute, 'passed': passed})
    if len(checks) < 2 * min(3, len(documents)) or not all(row['passed'] for row in checks):
        raise RuntimeError(f'Independent exact-prefix forward test failed: {checks}')
    return {'checks': checks, 'relative_tolerance': RELATIVE_RMS_LIMIT,
            'comparison': 'original source states vs independently retokenized truncation and suffix replacement'}


def batches(documents, max_documents=MAX_BATCH_DOCUMENTS, max_tokens=MAX_BATCH_TOKENS):
    classed = defaultdict(list)
    for row in documents.values():
        length = len(row.ids)
        classed[(length - 1) // 16].append(row)
    for bucket in reversed(sorted(classed)):
        group = classed[bucket]
        largest = max(len(row.ids) for row in group)
        size = max(1, min(max_documents, max_tokens // largest))
        for start in range(0, len(group), size):
            yield group[start:start + size]


def load_frozen_backbone(candidate, model):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['USE_HUB_KERNELS'] = 'NO'
    import torch
    import transformers

    before_model = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    encoder, spec = load_language_model(model, candidate, torch, transformers)
    return FrozenBackbone(candidate, model, encoder, spec, perf_counter() - before_model)


def extract(candidate, model, stimuli, output, protocol_sha256, targets=None, preflight=None,
            *, max_batch_documents=MAX_BATCH_DOCUMENTS, max_batch_tokens=MAX_BATCH_TOKENS):
    if output.exists():
        raise FileExistsError(output)
    tokenizer, documents, summary = geometry(candidate, model, stimuli, targets, protocol_sha256)
    if preflight is not None and read_json(preflight) != summary:
        raise ValueError('Frozen tokenizer preflight differs from the runtime Unicode geometry')
    backbone = load_frozen_backbone(candidate, model)
    write_projection(backbone, tokenizer, documents, summary, output, backbone.load_seconds,
                     max_batch_documents, max_batch_tokens)


def extract_loaded(backbone, stimuli, output, protocol_sha256,
                   *, max_batch_documents=MAX_BATCH_DOCUMENTS, max_batch_tokens=MAX_BATCH_TOKENS):
    """Extract another disjoint document set without reloading its frozen weights."""
    import torch

    if output.exists():
        raise FileExistsError(output)
    tokenizer, documents, summary = geometry(backbone.candidate, backbone.model, stimuli, None, protocol_sha256)
    torch.cuda.reset_peak_memory_stats()
    write_projection(backbone, tokenizer, documents, summary, output, 0.0,
                     max_batch_documents, max_batch_tokens)


def write_projection(backbone, tokenizer, documents, summary, output, load_seconds,
                     max_batch_documents, max_batch_tokens):
    import numpy as np
    import torch
    import transformers

    encoder, spec = backbone.encoder, backbone.spec
    controls = causal_controls(encoder, documents, tokenizer, spec['pad_id'], torch)
    ranges = {}
    start = 0
    coordinates = []
    for row in documents.values():
        end = start + len(row.certified)
        ranges[row.identifier] = slice(start, end)
        coordinates.append({'id': row.identifier, 'stops': list(row.certified)})
        start = end

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.backbone-transfer-') as temporary:
        stage = Path(temporary) / 'extraction'
        stage.mkdir()
        coordinates_path = stage / 'coordinates.jsonl'
        preflight_path = stage / 'preflight.json'
        causality_path = stage / 'causality.json'
        states_path = stage / 'states.npy'
        write_rows(coordinates_path, coordinates)
        write_json(preflight_path, summary)
        write_json(causality_path, controls)
        shape = (start, spec['hidden_width'])
        states = np.lib.format.open_memmap(states_path, mode='w+', dtype=np.float32, shape=shape)
        start_content = perf_counter()
        calls = 0
        for batch in batches(documents, max_batch_documents, max_batch_tokens):
            hidden = forward(encoder, [row.ids for row in batch], spec['pad_id'], torch)
            if hidden.shape[-1] != spec['hidden_width']:
                raise RuntimeError('A final layer returned an unexpected representation width')
            selected = [(i, token) for i, row in enumerate(batch) for token in row.certified.values()]
            if selected:
                indices, positions = zip(*selected, strict=True)
                picked = hidden[list(indices), list(positions)].cpu().numpy()
                if not np.isfinite(picked).all():
                    raise RuntimeError('Encountered a nonfinite extracted hidden state')
                row_start = 0
                for row in batch:
                    next_start = row_start + len(row.certified)
                    states[ranges[row.identifier]] = picked[row_start:next_start]
                    row_start = next_start
            del hidden
            calls += 1
        states.flush()
        del states
        write_json(stage / 'seal.json', {
            'candidate': backbone.candidate,
            'model': spec['repo'],
            'revision': spec['revision'],
            'architecture': spec['architecture'],
            'hidden_width': spec['hidden_width'],
            'layers': spec['layers'],
            'model_files_sha256': spec['sha256'],
            'config_adapter': spec.get('config_adapter'),
            'model_manifest_sha256': digest(MODELS),
            'source_code_sha256': digest(Path(__file__)),
            'protocol_sha256': summary['protocol_sha256'],
            'stimuli_sha256': summary['stimuli_sha256'],
            'targets_sha256': summary['targets_sha256'],
            'tokenizer_sha256': summary['tokenizer_sha256'],
            'states_dtype': 'float32',
            'compute_dtype': 'float32',
            'states_shape': list(shape),
            'gpu': torch.cuda.get_device_name(0),
            'torch': torch.__version__,
            'transformers': transformers.__version__,
            'cuda': torch.version.cuda,
            'content_forward_per_document': 1,
            'content_forward_batches': calls,
            'max_batch_documents': max_batch_documents,
            'max_batch_tokens': max_batch_tokens,
            'independent_causal_checks': len(controls['checks']),
            'documents': summary['documents'],
            'certified_stops': summary['certified_stops'],
            'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved(),
            'model_load_and_verification_s': load_seconds,
            'forward_and_readout_s': perf_counter() - start_content,
            'file_sha256': {item.name: digest(item) for item in (coordinates_path, preflight_path, causality_path, states_path)},
        })
        stage.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', choices=tuple(read_json(MODELS)['models']), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--stimuli', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol-sha256', type=checked_digest, required=True)
    parser.add_argument('--targets', type=Path)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--geometry-only', action='store_true')
    args = parser.parse_args()
    if args.geometry_only:
        _, _, receipt = geometry(args.candidate, args.model, args.stimuli, args.targets, args.protocol_sha256)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, receipt)
    else:
        extract(args.candidate, args.model, args.stimuli, args.output, args.protocol_sha256, args.targets, args.preflight)


if __name__ == '__main__':
    main()
