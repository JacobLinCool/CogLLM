"""Extract one-pass Qwen features on a Colab L4 and fit the PLABA probe."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from threading import Event, Thread
from time import perf_counter

from .probe import evaluate


MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
TRANSFORMERS_VERSION = "5.17.0"


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def optional_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def sentence_position(text, segment, offsets):
    start, end = segment["start"], segment["end"]
    tail = start + len(text[start:end].rstrip()) - 1
    eligible = [i for i, (left, right) in enumerate(offsets) if left <= tail < right <= end]
    if tail < start or not eligible:
        raise ValueError(f"No causal token ending the annotated sentence: {segment['id']}")
    return eligible[-1]


def tokenize_stimuli(tokenizer, data_dir):
    documents = []
    with (data_dir / "stimuli.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            stimulus = json.loads(line)
            text = stimulus["text"]
            tokens = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            ids = tokens["input_ids"]
            offsets = tokens["offset_mapping"]
            if not ids or len(ids) != len(offsets) or not stimulus["segments"]:
                raise ValueError(f"Invalid tokenization: {stimulus['id']}")
            sentences = [
                (segment["id"], sentence_position(text, segment, offsets))
                for segment in stimulus["segments"]
            ]
            documents.append((stimulus["id"], ids, sentences))
    return sorted(documents, key=lambda document: (len(document[1]), document[0]))


def encode(language, sequences, pad_id, torch):
    width = max(map(len, sequences))
    ids = [seq + [pad_id] * (width - len(seq)) for seq in sequences]
    mask = [[1] * len(seq) + [0] * (width - len(seq)) for seq in sequences]
    with torch.inference_mode():
        outputs = language(
            input_ids=torch.tensor(ids, device="cuda", dtype=torch.long),
            attention_mask=torch.tensor(mask, device="cuda", dtype=torch.long),
            use_cache=False,
        )
    return outputs.last_hidden_state


def difference(left, right):
    left, right = left.float(), right.float()
    error = (left - right).square().mean(dim=-1).sqrt()
    denominator = left.square().mean(dim=-1).sqrt().clamp_min(1e-8)
    return {
        "max_token_relative_rms": float((error / denominator).max().item()),
        "max_absolute": float((left - right).abs().max().item()),
    }


def run_causal_checks(language, documents, pad_id, torch):
    tokens = documents[len(documents) // 2][1]
    other_tokens = documents[-1][1]
    full = encode(language, [tokens], pad_id, torch)[0]
    checks = []
    for prefix_length in (32, 65, 97):
        if prefix_length + 8 >= len(tokens):
            raise ValueError("Pilot article is too short to test the preregistered positions")
        prefix = tokens[:prefix_length]
        other = prefix + other_tokens[:len(tokens) - prefix_length]
        if other == tokens:
            raise ValueError("Replacement suffix must change the input")
        replaced = encode(language, [other], pad_id, torch)[0]
        truncated = encode(language, [prefix], pad_id, torch)[0]
        padded = encode(language, [prefix, tokens], pad_id, torch)[0, :prefix_length]
        for name, comparison, tolerance in (
            ("different_suffix_same_length", replaced[:prefix_length], 0.002),
            ("truncate", truncated, 0.02),
            ("right_pad_in_batch", padded, 0.02),
        ):
            metrics = difference(full[:prefix_length], comparison)
            checks.append({
                "comparison": name,
                "prefix_tokens": prefix_length,
                "original_tokens": len(tokens),
                "max_token_relative_rms_tolerance": tolerance,
                **metrics,
                "passed": metrics["max_token_relative_rms"] <= tolerance,
            })
        changed = difference(full[prefix_length:], replaced[prefix_length:])
        checks.append({
            "comparison": "positive_control_different_suffix",
            "prefix_tokens": prefix_length,
            **changed,
            "passed": changed["max_token_relative_rms"] > 0.02,
        })
    return checks


def collect_features(language, documents, pad_id, output, torch, np):
    feature_ids, features, timings = [], [], []
    torch.cuda.reset_peak_memory_stats()
    for start in range(0, len(documents), 4):
        batch = documents[start:start + 4]
        t0 = perf_counter()
        states = encode(language, [document[1] for document in batch], pad_id, torch)
        for row, (_, tokens, sentences) in enumerate(batch):
            locations = torch.tensor([position for _, position in sentences], device="cuda")
            local = states[row].index_select(0, locations)
            global_state = states[row, len(tokens) - 1].expand(len(sentences), -1)
            features.append(torch.cat((local, global_state), dim=-1).float().cpu().numpy())
            feature_ids.extend(sentence_id for sentence_id, _ in sentences)
        torch.cuda.synchronize()
        timings.append({
            "documents": len(batch),
            "tokens": sum(len(document[1]) for document in batch),
            "padded_tokens": len(batch) * max(len(document[1]) for document in batch),
            "elapsed_s": perf_counter() - t0,
        })
        if start % 16 == 0 or start + len(batch) == len(documents):
            temporary = output / "features.partial.npz"
            np.savez_compressed(temporary, ids=np.array(feature_ids), features=np.concatenate(features))
            temporary.replace(output / "features.npz")
            print(f"Encoded {start + len(batch)}/{len(documents)} documents", flush=True)
        del states
    elapsed = sum(batch["elapsed_s"] for batch in timings)
    token_count = sum(batch["tokens"] for batch in timings)
    save_json(output / "extraction.json", {
        "documents": len(documents),
        "backbone_forward_calls": len(timings),
        "examples_per_forward_max": 4,
        "tokens": token_count,
        "synchronized_elapsed_s_excluding_checkpoint_writes": elapsed,
        "tokens_per_second": token_count / elapsed,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "batches": timings,
    })
    return feature_ids, np.concatenate(features)


def sample_gpu(path, done):
    with path.open("w", encoding="utf-8") as handle:
        command = [
            "nvidia-smi",
            "--query-gpu=timestamp,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        while not done.is_set():
            try:
                sample = subprocess.run(command, capture_output=True, text=True, check=True, timeout=4)
                timestamp, utilization, used, total = (value.strip() for value in sample.stdout.strip().split(","))
                record = {
                    "gpu_timestamp": timestamp,
                    "utilization_pct": int(utilization),
                    "used_mib": int(used),
                    "total_mib": int(total),
                }
            except (subprocess.SubprocessError, OSError, ValueError):
                record = {"status": "sample_unavailable"}
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            done.wait(2)


def run(data_dir, output, compute_dtype):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(f"Expected Transformers {TRANSFORMERS_VERSION}, got {transformers.__version__}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("One CUDA device is required")
    device = torch.cuda.get_device_properties(0)
    if not device.name.endswith(" L4") or not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError(f"Expected an L4 with native BF16, got {device.name}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[compute_dtype]
    save_json(output / "environment.json", {
        "gpu": device.name,
        "vram_bytes": device.total_memory,
        "compute_capability": [device.major, device.minor],
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "model": MODEL,
        "revision": REVISION,
        "checkpoint_load_dtype": "bfloat16",
        "compute_dtype": compute_dtype,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "attention": "sdpa",
        "causal_conv1d": optional_version("causal-conv1d"),
        "flash_linear_attention": optional_version("flash-linear-attention"),
        "readout_features": ["sentence_end_final_hidden", "article_end_final_hidden"],
        "full_text_sentence_rating": True,
        "visible_input": "article_only",
        "torch_cpu_threads": 2,
    })
    torch.set_num_threads(2)
    started = perf_counter()
    device_watch = Event()
    sampler = Thread(target=sample_gpu, args=(output / "telemetry.jsonl", device_watch))
    sampler.start()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
        if not tokenizer.is_fast or tokenizer.pad_token_id is None:
            raise RuntimeError("A fast tokenizer with a defined padding token is required")
        documents = tokenize_stimuli(tokenizer, data_dir)
        print(f"Tokenized {len(documents)} documents, longest {len(documents[-1][1])} tokens", flush=True)
        model, loading = Qwen3_5ForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16,
            attn_implementation="sdpa", use_safetensors=True, output_loading_info=True,
        )
        load_issues = {key: value for key, value in loading.items() if value}
        if load_issues:
            raise RuntimeError(f"Pinned Qwen weights did not load exactly: {load_issues}")
        if next(model.model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError("Qwen backbone weights did not load as BF16")
        language = model.model.eval().requires_grad_(False).to(device="cuda", dtype=dtype)
        del model
        if next(language.parameters()).dtype != dtype:
            raise RuntimeError(f"Qwen backbone did not move to the requested dtype: {dtype}")
        ready = perf_counter()
        checks = run_causal_checks(language, documents, tokenizer.pad_token_id, torch)
        save_json(output / "causality.json", {"tokenizer_class": type(tokenizer).__name__, "checks": checks})
        if not all(check["passed"] for check in checks):
            raise RuntimeError(f"Prefix invariance failed. See {output / 'causality.json'}")
        print("Causal checks passed for all three prefix lengths", flush=True)
        feature_ids, features = collect_features(language, documents, tokenizer.pad_token_id, output, torch, np)
        del language
        evaluation_start = perf_counter()
        evaluate(data_dir, feature_ids, features, output)
        save_json(output / "timings.json", {
            "model_and_tokenizer_setup_s": ready - started,
            "cpu_probe_and_evaluation_s": perf_counter() - evaluation_start,
            "total_script_s": perf_counter() - started,
        })
    finally:
        device_watch.set()
        sampler.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", required=True, choices=("bfloat16", "float32"))
    arguments = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    run(arguments.data, arguments.output, arguments.dtype)
