"""Extract the two Zhan Experiment 1 stopping-point states on a Colab L4."""

import argparse
import hashlib
from itertools import groupby
import os
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

from ..stage3.run import MODEL, REVISION, TRANSFORMERS_VERSION, difference, encode, optional_version, sample_gpu, save_json
from ..stage3.zhan_landmark_features import PREFIX_RMS_TOLERANCE, tokenize_pairs


def collect(language, pairs, pad_id, output, torch, np):
    vectors = {}
    checks = []
    timings = []
    torch.cuda.reset_peak_memory_stats()
    ordered = sorted(pairs, key=lambda pair: len(pair[3]))
    for length, same_length in groupby(ordered, key=lambda pair: len(pair[3])):
        group = list(same_length)
        for start in range(0, len(group), 4):
            batch = group[start:start + 4]
            began = perf_counter()
            full = encode(language, [pair[3] for pair in batch], pad_id, torch)
            if full.shape != (len(batch), length, 2048) or full.dtype != torch.float32:
                raise RuntimeError(f"Unexpected Qwen final states: {full.shape}, {full.dtype}")
            states = full[:, -2:].cpu().numpy()
            forward_s = perf_counter() - began

            began = perf_counter()
            truncated = encode(language, [pair[2] for pair in batch], pad_id, torch)
            for row, (family_id, ids, _, _, offsets) in enumerate(batch):
                metrics = difference(full[row, :-1], truncated[row])
                checks.append({
                    "family_id": family_id,
                    "stimulus_ids": ids,
                    "landmark_token_indices": [length - 2, length - 1],
                    "landmark_offsets": offsets[-2:],
                    "prefix_tokens": length - 1,
                    "pronoun_token_count": 1,
                    **metrics,
                    "passed": metrics["max_token_relative_rms"] <= PREFIX_RMS_TOLERANCE,
                })
                vectors.update(zip(ids, states[row], strict=True))
            timings.append({
                "families": len(batch),
                "main_tokens": len(batch) * length,
                "main_forward_and_readout_s": forward_s,
                "truncated_check_s": perf_counter() - began,
            })
            del full, truncated

    checks.sort(key=lambda check: check["family_id"])
    save_json(output / "causality.json", {
        "comparison": "full_vs_separately_truncated_batch_at_shared_prefix",
        "prefix_rms_tolerance": PREFIX_RMS_TOLERANCE,
        "checks": checks,
    })
    if not all(check["passed"] for check in checks):
        raise RuntimeError(f"Prefix invariance failed. See {output / 'causality.json'}")
    ids = sorted(vectors)
    hidden = np.stack([vectors[stimulus_id] for stimulus_id in ids])
    if hidden.shape != (2 * len(pairs), 2048) or not np.isfinite(hidden).all():
        raise ValueError("Expected two finite final hidden states per stimulus family")
    temporary = output / "features.partial.npz"
    np.savez_compressed(temporary, ids=np.array(ids), hidden=hidden)
    temporary.replace(output / "features.npz")
    main_s = sum(batch["main_forward_and_readout_s"] for batch in timings)
    main_tokens = sum(batch["main_tokens"] for batch in timings)
    save_json(output / "extraction.json", {
        "family_count": len(pairs),
        "landmark_count": len(ids),
        "max_prefix_token_relative_rms": max(check["max_token_relative_rms"] for check in checks),
        "main_forward_calls": len(timings),
        "main_forward_token_count": main_tokens,
        "main_forward_and_readout_s": main_s,
        "truncated_check_forward_calls": len(timings),
        "truncated_check_s": sum(batch["truncated_check_s"] for batch in timings),
        "tokens_per_second": main_tokens / main_s,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "batches": timings,
    })


def run(data, output):
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
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False

    output.mkdir(parents=True, exist_ok=True)
    stimuli = data / "stimuli.jsonl"
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
        "stimuli_sha256": hashlib.sha256(stimuli.read_bytes()).hexdigest(),
        "checkpoint_load_dtype": "bfloat16",
        "compute_dtype": "float32",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "attention": "sdpa",
        "causal_conv1d": optional_version("causal-conv1d"),
        "flash_linear_attention": optional_version("flash-linear-attention"),
        "visible_input": "stimulus_text_only",
    })
    started = perf_counter()
    stop = Event()
    sampler = Thread(target=sample_gpu, args=(output / "telemetry.jsonl", stop))
    sampler.start()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
        if not tokenizer.is_fast or tokenizer.pad_token_id is None:
            raise RuntimeError("A fast tokenizer with a defined padding token is required")
        pairs = tokenize_pairs(stimuli, tokenizer)
        model, loading = Qwen3_5ForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa",
            use_safetensors=True, output_loading_info=True,
        )
        if any(loading.values()):
            raise RuntimeError(f"Pinned Qwen weights did not load exactly: {loading}")
        if next(model.model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError("Qwen backbone weights did not load as BF16")
        language = model.model.eval().requires_grad_(False).to(device="cuda", dtype=torch.float32)
        del model
        if next(language.parameters()).dtype != torch.float32:
            raise RuntimeError("Qwen backbone did not move to FP32")
        ready = perf_counter()
        collect(language, pairs, tokenizer.pad_token_id, output, torch, np)
        save_json(output / "timings.json", {
            "model_and_tokenizer_setup_s": ready - started,
            "total_script_s": perf_counter() - started,
        })
        print(f"Extracted {2 * len(pairs)} landmark states and verified their shared prefixes", flush=True)
    finally:
        stop.set()
        sampler.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Directory containing stimuli.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    run(args.data, args.output)
