"""Extract frozen, terminal Qwen states from outcome-free OneStopQA prompts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

from ..stage3.run import (
    MODEL,
    REVISION,
    TRANSFORMERS_VERSION,
    encode,
    optional_version,
    run_causal_checks,
    sample_gpu,
    save_json,
)


HIDDEN_SIZE = 2048
PILOT_BATCHES = 8
MIN_HEADROOM_BYTES = 2 * 1024**3
CHECKPOINT_BATCHES = 32


def tokenize(tokenizer, path):
    ids = set()
    documents = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            row = json.loads(line)
            if set(row) != {"id", "text"}:
                raise ValueError(f"Expected only id and text at input line {line_number}")
            prompt_id, prompt = row["id"], row["text"]
            if not isinstance(prompt_id, str) or not prompt_id or prompt_id in ids:
                raise ValueError(f"Missing or duplicate prompt ID at input line {line_number}")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Expected a nonempty visible prompt in {prompt_id}")
            token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            if not token_ids:
                raise ValueError(f"No tokens in {prompt_id}")
            documents.append((prompt_id, token_ids))
            ids.add(prompt_id)
    return sorted(documents, key=lambda row: (len(row[1]), row[0]))


def _memory(torch):
    free, total = torch.cuda.mem_get_info()
    peak_reserved = torch.cuda.max_memory_reserved()
    reserved_now = torch.cuda.memory_reserved()
    return {
        "cuda_total_bytes": total,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": peak_reserved,
        "free_bytes_at_observed_peak_reserved": free - (peak_reserved - reserved_now),
        "minimum_headroom_bytes": MIN_HEADROOM_BYTES,
    }


def _checkpoint(path, ids, hidden, np):
    temporary = path.with_name(f"{path.stem}.tmp.npz")
    np.savez(temporary, ids=ids, hidden=hidden)
    temporary.replace(path)


def collect(language, documents, pad_id, output, torch, np, batch_size, max_seconds):
    starts = list(range(0, len(documents), batch_size))
    if not starts:
        raise ValueError("Expected at least one input")
    # Five eight-row batches or eight four-row batches cover the prompt-length range.
    pilot_count = min(len(starts), PILOT_BATCHES if batch_size == 4 else 5)
    pilot_starts = (
        [starts[-1]] if pilot_count == 1 else
        [starts[round(index * (len(starts) - 1) / (pilot_count - 1))] for index in range(pilot_count)]
    )
    piloted = set(pilot_starts)
    schedule = pilot_starts + [start for start in starts if start not in piloted]
    ids = np.array([prompt_id for prompt_id, _ in documents])
    hidden = np.empty((len(documents), HIDDEN_SIZE), dtype=np.float32)
    completed = np.zeros(len(documents), dtype=bool)
    timings = []
    began = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    partial = output / "features.partial.npz"

    for iteration, start in enumerate(schedule, 1):
        if perf_counter() - began >= max_seconds:
            _checkpoint(partial, ids[completed], hidden[completed], np)
            raise TimeoutError(f"Extraction exceeded {max_seconds}s after {int(completed.sum())} prompts")
        batch = documents[start:start + batch_size]
        tokens = [sequence for _, sequence in batch]
        lengths = [len(sequence) for sequence in tokens]
        started = perf_counter()
        states = encode(language, tokens, pad_id, torch)
        if states.shape != (len(batch), max(lengths), HIDDEN_SIZE) or states.dtype != torch.float32:
            raise RuntimeError(f"Unexpected frozen Qwen state: {states.shape}, {states.dtype}")
        rows = torch.arange(len(batch), device=states.device)
        ends = torch.as_tensor(lengths, device=states.device) - 1
        terminal = states[rows, ends].cpu().numpy()
        if not np.isfinite(terminal).all():
            raise ValueError(f"Nonfinite terminal states in batch starting at {start}")
        hidden[start:start + len(batch)] = terminal
        completed[start:start + len(batch)] = True
        del states
        timings.append({
            "prompts": len(batch),
            "tokens": sum(lengths),
            "padded_tokens": len(batch) * max(lengths),
            "max_length": max(lengths),
            "elapsed_s": perf_counter() - started,
        })

        if iteration == pilot_count:
            pilot_seconds = sum(record["elapsed_s"] for record in timings)
            pilot_tokens = sum(record["tokens"] for record in timings)
            pilot = {
                "prompts": int(completed.sum()),
                "tokens": pilot_tokens,
                "forward_and_readout_s": pilot_seconds,
                "tokens_per_second": pilot_tokens / pilot_seconds,
                "batches": timings.copy(),
                **_memory(torch),
            }
            save_json(output / "pilot.json", pilot)
            if pilot["free_bytes_at_observed_peak_reserved"] < MIN_HEADROOM_BYTES:
                _checkpoint(partial, ids[completed], hidden[completed], np)
                raise RuntimeError("L4 pilot left less than 2 GiB of device headroom")
            print(f"Pilot passed on {pilot['prompts']} prompts ({pilot_tokens} tokens)", flush=True)
        if iteration == pilot_count or iteration % CHECKPOINT_BATCHES == 0:
            _checkpoint(partial, ids[completed], hidden[completed], np)
            print(f"Encoded {int(completed.sum())}/{len(documents)} prompts", flush=True)

    if not completed.all():
        raise RuntimeError("Feature inventory is incomplete")
    elapsed = sum(record["elapsed_s"] for record in timings)
    tokens = sum(record["tokens"] for record in timings)
    save_json(output / "extraction.json", {
        "prompts": len(documents),
        "tokens": tokens,
        "padded_tokens": sum(record["padded_tokens"] for record in timings),
        "min_prompt_tokens": len(documents[0][1]),
        "max_prompt_tokens": len(documents[-1][1]),
        "backbone_forward_calls": len(timings),
        "examples_per_forward_max": batch_size,
        "forward_and_readout_s": elapsed,
        "tokens_per_second": tokens / elapsed,
        "extraction_wall_s": perf_counter() - began,
        **_memory(torch),
        "batches": timings,
    })
    _checkpoint(output / "features.npz", ids, hidden, np)
    partial.unlink(missing_ok=True)
    print(f"Encoded {len(documents)}/{len(documents)} prompts", flush=True)


def run(inputs, output, batch_size, max_seconds):
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
    if max_seconds <= 0:
        raise ValueError("Extraction time budget must be positive")
    if (output / "features.npz").exists() or (output / "features.partial.npz").exists():
        raise FileExistsError("Use a fresh output directory for each feature run")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
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
        "inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
        "checkpoint_load_dtype": "bfloat16",
        "compute_dtype": "float32",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "attention": "sdpa",
        "causal_conv1d": optional_version("causal-conv1d"),
        "flash_linear_attention": optional_version("flash-linear-attention"),
        "readout_feature": "last_visible_token_final_hidden",
        "visible_input": "one_complete_rendered_study_prompt_per_row",
        "max_extraction_seconds": max_seconds,
    })
    began = perf_counter()
    stop = Event()
    sampler = Thread(target=sample_gpu, args=(output / "telemetry.jsonl", stop))
    sampler.start()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
        if not tokenizer.is_fast or tokenizer.pad_token_id is None:
            raise RuntimeError("A fast tokenizer with a padding token is required")
        documents = tokenize(tokenizer, inputs)
        eligible = [row for row in documents if len(row[1]) > 105]
        if len(eligible) < 3:
            raise ValueError("At least three inputs over 105 tokens are needed for fixed causal checks")
        model, loading = Qwen3_5ForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa",
            use_safetensors=True, output_loading_info=True,
        )
        if any(loading.values()) or next(model.model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError(f"Pinned Qwen BF16 weights did not load exactly: {loading}")
        language = model.model.eval().requires_grad_(False).to(device="cuda", dtype=torch.float32)
        del model
        if next(language.parameters()).dtype != torch.float32:
            raise RuntimeError("Qwen backbone did not move to FP32")
        ready = perf_counter()
        checks = run_causal_checks(language, eligible, tokenizer.pad_token_id, torch)
        save_json(output / "causality.json", {
            "scope": "Study prompt prefixes under same-length replacement, truncation and right padding",
            "reference_prompt": eligible[len(eligible) // 2][0],
            "replacement_prompt": eligible[-1][0],
            "tokenizer_class": type(tokenizer).__name__,
            "checks": checks,
        })
        if not all(check["passed"] for check in checks):
            raise RuntimeError("Causal checks failed")
        checked = perf_counter()
        collect(language, documents, tokenizer.pad_token_id, output, torch, np, batch_size, max_seconds)
        save_json(output / "timings.json", {
            "model_and_tokenizer_setup_s": ready - began,
            "causal_checks_s": checked - ready,
            "total_script_s": perf_counter() - began,
        })
    finally:
        stop.set()
        sampler.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, choices=(4, 8), default=8)
    parser.add_argument("--max-seconds", type=int, default=780)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    run(args.inputs, args.output, args.batch_size, args.max_seconds)
