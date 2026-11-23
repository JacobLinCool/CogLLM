"""Extract features for the locked Stage 3 comparison on a Colab L4."""

import argparse
import json
import os
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import numpy as np

from .feature_signals import FEATURE_NAMES, signals_for_document
from .run import (
    MODEL,
    REVISION,
    TRANSFORMERS_VERSION,
    difference,
    optional_version,
    sample_gpu,
    save_json,
    sentence_position,
)


ATTENTION_LAYERS = (4, 8, 12, 16, 20, 24)
INTERMEDIATE_BLOCK = 11  # Zero-based index in selective hidden-state capture.


def tokenize_documents(tokenizer, data_dir):
    documents = []
    with (data_dir / "stimuli.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            document = json.loads(line)
            text = document["text"]
            encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
            if not ids or len(ids) != len(offsets) or not document["segments"]:
                raise ValueError(f"Invalid alignment for {document['id']}")
            sentences = []
            previous_end = -1
            for segment in document["segments"]:
                start, boundary = segment["start"], segment["end"]
                end = sentence_position(text, segment, offsets)
                indices = [
                    i for i, (left, right) in enumerate(offsets)
                    if text[max(left, start):min(right, boundary)].strip()
                ]
                if (
                    not indices or indices[0] <= previous_end or indices[-1] != end
                    or any(text[left:start].strip() for i in indices if (left := offsets[i][0]) < start)
                    or any(offsets[i][1] > boundary for i in indices)
                ):
                    raise ValueError(f"Token crosses an annotated sentence boundary: {segment['id']}")
                sentences.append((segment["id"], indices))
                previous_end = end
            documents.append((document["id"], ids, sentences))
    return sorted(documents, key=lambda row: (len(row[1]), row[0]))


def encode(model, sequences, pad_id, torch):
    width = max(map(len, sequences))
    ids = [seq + [pad_id] * (width - len(seq)) for seq in sequences]
    mask = [[1] * len(seq) + [0] * (width - len(seq)) for seq in sequences]
    with torch.inference_mode():
        result = model.model(
            input_ids=torch.tensor(ids, device="cuda", dtype=torch.long),
            attention_mask=torch.tensor(mask, device="cuda", dtype=torch.long),
            output_hidden_states=[INTERMEDIATE_BLOCK],
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    if (
        len(result.hidden_states) != 24 or result.hidden_states[INTERMEDIATE_BLOCK] is None
        or len(result.attentions) != len(ATTENTION_LAYERS)
        or any(layer is None for layer in result.attentions)
    ):
        raise RuntimeError("Expected one block-12 hidden state and six eager attention tensors")
    return result.last_hidden_state, result.hidden_states[INTERMEDIATE_BLOCK], result.attentions


def document_row(captured, row, length):
    final, middle, attention = captured
    return final[row, :length], middle[row, :length], tuple(a[row, :, :length, :length] for a in attention)


def causal_checks(model, documents, pad_id, torch):
    original = documents[len(documents) // 2][1]
    alternative = documents[-1][1]
    full = encode(model, [original], pad_id, torch)
    checks = []
    for prefix_length in (32, 65, 97):
        prefix = original[:prefix_length]
        suffix_replaced = prefix + alternative[:len(original) - prefix_length]
        changed = encode(model, [suffix_replaced], pad_id, torch)
        if difference(full[0][0, prefix_length:], changed[0][0, prefix_length:])["max_token_relative_rms"] <= 0.02:
            raise ValueError("The positive-control suffix must alter the suffix hidden states")
        reference = document_row(full, 0, prefix_length)
        # Every attention query in this prefix retains its original <= query key region.
        sentences = [("test", list(range(prefix_length // 2, prefix_length)))]
        reference_signals = signals_for_document(torch, model, prefix, *reference, sentences)
        for name, encoded, tolerance in (
            ("different_suffix_same_length", changed, 0.002),
            ("truncate", encode(model, [prefix], pad_id, torch), 0.02),
            ("right_pad_in_batch", encode(model, [prefix, original], pad_id, torch), 0.02),
        ):
            row = document_row(encoded, 0, prefix_length)
            recorded = signals_for_document(torch, model, prefix, *row, sentences)
            final_error = difference(reference[0], row[0])["max_token_relative_rms"]
            intermediate_error = difference(reference[1], row[1])["max_token_relative_rms"]
            attention_error = max(float((a - b).abs().max().item()) for a, b in zip(reference[2], row[2], strict=True))
            signal_errors = {key: float(np.max(np.abs(reference_signals[key] - recorded[key]))) for key in FEATURE_NAMES}
            checks.append({
                "comparison": name,
                "prefix_tokens": prefix_length,
                "final_hidden_max_relative_rms": final_error,
                "block12_hidden_max_relative_rms": intermediate_error,
                "hidden_tolerance": tolerance,
                "attention_max_absolute": attention_error,
                "attention_tolerance": 0.0002,
                "sentence_signal_max_absolute": signal_errors,
                "signal_tolerance": 0.001,
                "passed": (
                    max(final_error, intermediate_error) <= tolerance
                    and attention_error <= 0.0002 and max(signal_errors.values()) <= 0.001
                ),
            })
    return checks


def extract(model, documents, pad_id, output, torch):
    ids, blocks, batches = [], {"sentence_hidden": [], **{name: [] for name in FEATURE_NAMES}}, []
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(documents), 4):
        batch = documents[start:start + 4]
        started = perf_counter()
        captured = encode(model, [doc[1] for doc in batch], pad_id, torch)
        for row, (_, tokens, sentences) in enumerate(batch):
            final, middle, attention = document_row(captured, row, len(tokens))
            features = signals_for_document(torch, model, tokens, final, middle, attention, sentences)
            ids.extend(sentence_id for sentence_id, _ in sentences)
            blocks["sentence_hidden"].append(final[[positions[-1] for _, positions in sentences]].cpu().numpy())
            for name, array in features.items():
                blocks[name].append(array)
        del final, middle, attention, captured
        torch.cuda.synchronize()
        batches.append({
            "documents": len(batch),
            "tokens": sum(len(doc[1]) for doc in batch),
            "elapsed_s": perf_counter() - started,
        })
        if start % 16 == 0 or start + len(batch) == len(documents):
            partial = output / "features.partial.npz"
            np.savez_compressed(partial, ids=np.array(ids), **{name: np.concatenate(v) for name, v in blocks.items()})
            partial.replace(output / "features.npz")
            print(f"Encoded {start + len(batch)}/{len(documents)} articles", flush=True)
    elapsed = sum(batch["elapsed_s"] for batch in batches)
    tokens = sum(batch["tokens"] for batch in batches)
    save_json(output / "extraction.json", {
        "articles": len(documents),
        "sentences": len(ids),
        "backbone_forward_calls": len(batches),
        "tokens": tokens,
        "elapsed_s_excluding_checkpoint_writes": elapsed,
        "tokens_per_second": tokens / elapsed,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "batches": batches,
    })


def run(data_dir, output):
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Expected Transformers {TRANSFORMERS_VERSION}, got {transformers.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("One CUDA device is required")
    device = torch.cuda.get_device_properties(0)
    if not device.name.endswith(" L4"):
        raise RuntimeError(f"Expected NVIDIA L4, got {device.name}")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "environment.json", {
        "gpu": device.name,
        "vram_bytes": device.total_memory,
        "compute_capability": [device.major, device.minor],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "model": MODEL,
        "revision": REVISION,
        "checkpoint_load_dtype": "bfloat16",
        "compute_dtype": "float32",
        "attention_backend": "eager",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "full_attention_layers_one_based": ATTENTION_LAYERS,
        "intermediate_layer_one_based": INTERMEDIATE_BLOCK + 1,
        "sentence_signal_columns": FEATURE_NAMES,
        "causal_conv1d": optional_version("causal-conv1d"),
        "flash_linear_attention": optional_version("flash-linear-attention"),
    })
    started = perf_counter()
    stop = Event()
    telemetry = Thread(target=sample_gpu, args=(output / "telemetry.jsonl", stop))
    telemetry.start()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
        if not tokenizer.is_fast or tokenizer.pad_token_id is None:
            raise RuntimeError("A fast tokenizer with a defined padding token is required")
        documents = tokenize_documents(tokenizer, data_dir)
        model, loading = Qwen3_5ForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="eager",
            use_safetensors=True, output_loading_info=True,
        )
        if any(loading.values()):
            raise RuntimeError(f"Pinned Qwen weights did not load exactly: {loading}")
        layers = tuple(
            i for i, kind in enumerate(model.model.config.layer_types, start=1) if kind == "full_attention"
        )
        if layers != ATTENTION_LAYERS:
            raise RuntimeError(f"Unexpected full-attention layers: {layers}")
        model.eval().requires_grad_(False)
        model.model.to(device="cuda", dtype=torch.float32)
        model.tie_weights()
        if (
            model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
            or model.lm_head.weight.dtype != torch.float32
        ):
            raise RuntimeError("Expected the vocabulary projection to share the token embedding weights")
        ready = perf_counter()
        checks = causal_checks(model, documents, tokenizer.pad_token_id, torch)
        save_json(output / "causality.json", {"checks": checks})
        if not all(check["passed"] for check in checks):
            raise RuntimeError("An eager prefix check failed")
        print("Eager hidden, attention, and projected-feature causality checks passed", flush=True)
        extract(model, documents, tokenizer.pad_token_id, output, torch)
        save_json(output / "timings.json", {
            "model_and_tokenizer_setup_s": ready - started,
            "total_script_s": perf_counter() - started,
        })
    finally:
        stop.set()
        telemetry.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Directory containing stimuli.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    run(args.data, args.output)
