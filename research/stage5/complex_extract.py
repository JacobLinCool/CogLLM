"""Extract three frozen, causal Qwen readouts for outcome-free CompLex items."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

from ..stage3.probe import _jsonl
from ..stage3.run import (
    MODEL,
    REVISION,
    TRANSFORMERS_VERSION,
    difference,
    encode,
    optional_version,
    run_causal_checks,
    sample_gpu,
    save_json,
)
from .complex_tokenization import token_span


EXPECTED_OBSERVATIONS = 10_264
HIDDEN_SIZE = 2048
CHECKPOINT_BATCHES = 256
MIN_HEADROOM_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class Sentence:
    index: int
    id: str
    tokens: list[int]
    target_positions: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class TargetOnly:
    index: int
    tokens: list[int]


@dataclass(frozen=True)
class Inventory:
    ids: list[str]
    sentences: list[Sentence]
    targets: list[TargetOnly]
    sentence_index: list[int]
    target_index: list[int]


def read_rows(path, fields):
    rows = {}
    for row in _jsonl(path):
        if not isinstance(row, dict) or row.keys() != fields:
            raise ValueError(f"Unexpected fields in outcome-free CompLex input {path.name}")
        key = row["id"]
        if not isinstance(key, str) or not key or key in rows:
            raise ValueError(f"Missing or duplicate ID in {path.name}: {key!r}")
        rows[key] = row
    return rows


def tokenize(tokenizer, stimuli_path, targets_path):
    stimuli = read_rows(stimuli_path, {"id", "context_group", "language", "text", "split"})
    observations = read_rows(targets_path, {"id", "stimulus_id", "target_span"})
    sentences = {}
    for sentence_id, row in stimuli.items():
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty sentence: {sentence_id}")
        reading = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        tokens, offsets = reading["input_ids"], reading["offset_mapping"]
        if not tokens or len(tokens) != len(offsets):
            raise ValueError(f"Invalid native sentence tokenization: {sentence_id}")
        sentences[sentence_id] = (tokens, offsets)

    ids = sorted(observations)
    sentence_indexes = {key: index for index, key in enumerate(sorted(sentences))}
    target_indexes = {}
    positions = defaultdict(list)
    sentence_index, target_index, targets = [], [], []
    for index, obs_id in enumerate(ids):
        observation = observations[obs_id]
        sentence_id, span = observation["stimulus_id"], observation["target_span"]
        if sentence_id not in sentences or not isinstance(span, dict) or span.keys() != {"start", "end", "text"}:
            raise ValueError(f"Invalid target reference: {obs_id}")
        text = stimuli[sentence_id]["text"]
        start, end, target = span["start"], span["end"], span["text"]
        if (type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text) or text[start:end] != target):
            raise ValueError(f"Invalid target character span: {obs_id}")
        tokens, offsets = sentences[sentence_id]
        status, location = token_span(text, offsets, start, end)
        if location is None:
            raise ValueError(f"Unaudited CompLex target {obs_id}: {status}")
        _, last = location
        truncated = tokenizer(text[:end], add_special_tokens=False)["input_ids"]
        if tokens[:last + 1] != truncated:
            raise ValueError(f"Native target-end tokenization changes when removing the suffix: {obs_id}")
        if target not in target_indexes:
            target_tokens = tokenizer(target, add_special_tokens=False)["input_ids"]
            if not target_tokens:
                raise ValueError(f"Native tokenizer produced no isolated target tokens: {obs_id}")
            target_indexes[target] = len(targets)
            targets.append(TargetOnly(len(targets), target_tokens))
        positions[sentence_id].append((index, last))
        sentence_index.append(sentence_indexes[sentence_id])
        target_index.append(target_indexes[target])

    if positions.keys() != sentences.keys():
        raise ValueError("Outcome-free stimuli include sentences without an eligible target")
    return Inventory(
        ids,
        [Sentence(sentence_indexes[key], key, sentences[key][0], tuple(positions[key]))
         for key in sorted(sentences)],
        targets,
        sentence_index,
        target_index,
    )


def causal_checks(language, sentences, pad_id, torch):
    eligible = sorted(
        ((item.id, item.tokens) for item in sentences if len(item.tokens) > 105),
        key=lambda row: (len(row[1]), row[0]),
    )
    if len(eligible) < 3:
        raise ValueError("CompLex needs at least three sentences over 105 tokens for fixed causal checks")
    checks = run_causal_checks(language, eligible, pad_id, torch)
    # Direct checks at actual target ends, with short, medium and long visible prefixes.
    candidates = sorted(
        ((position, item.id, item.tokens) for item in sentences for _, position in item.target_positions
         if position + 8 < len(item.tokens)),
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    for index in (len(candidates) // 4, len(candidates) // 2, len(candidates) * 3 // 4):
        position, sentence_id, tokens = candidates[index]
        full = encode(language, [tokens], pad_id, torch)[0, position:position + 1]
        cut = encode(language, [tokens[:position + 1]], pad_id, torch)[0, -1:]
        metrics = difference(full, cut)
        checks.append({
            "comparison": "truncate_at_designated_target_end",
            "sentence_id": sentence_id,
            "prefix_tokens": position + 1,
            "original_tokens": len(tokens),
            "max_token_relative_rms_tolerance": 0.02,
            **metrics,
            "passed": metrics["max_token_relative_rms"] <= 0.02,
        })
    return {
        "reference_sentence": eligible[len(eligible) // 2][0],
        "replacement_sentence": eligible[-1][0],
        "checks": checks,
    }


def memory(torch):
    free, total = torch.cuda.mem_get_info()
    peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "cuda_total_bytes": total,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": peak_reserved,
        "free_bytes_at_observed_peak_reserved": free - (peak_reserved - torch.cuda.memory_reserved()),
    }


def write_features(path, values, np, *, compressed=False):
    temporary = path.with_name(f"{path.stem}.tmp.npz")
    writer = np.savez_compressed if compressed else np.savez
    writer(temporary, **values)
    temporary.replace(path)


def completed_features(values, done, np):
    # Partial archives expose only rows for which all three conditions finished.
    keep = done["target_end"] & done["target_only"][values["target_only_index"]]
    keep &= done["sentence_end"][values["sentence_end_index"]]
    kept = {key: values[key][keep] for key in ("ids", "target_end")}
    for key in ("target_only", "sentence_end"):
        references, index = np.unique(values[f"{key}_index"][keep], return_inverse=True)
        kept[key], kept[f"{key}_index"] = values[key][references], index
    return kept


def collect(language, inventory, pad_id, output, torch, np, batch_size, max_seconds):
    values = {
        "ids": np.array(inventory.ids),
        "target_only_index": np.array(inventory.target_index, dtype=np.int64),
        "sentence_end_index": np.array(inventory.sentence_index, dtype=np.int64),
    }
    sizes = {"target_only": len(inventory.targets), "target_end": len(inventory.ids),
             "sentence_end": len(inventory.sentences)}
    done = {key: np.zeros(count, dtype=bool) for key, count in sizes.items()}
    values.update({key: np.empty((count, HIDDEN_SIZE), dtype=np.float32) for key, count in sizes.items()})
    prompts = sorted([*inventory.sentences, *inventory.targets], key=lambda item: len(item.tokens))
    starts = list(range(0, len(prompts), batch_size))
    sample_count = min(len(starts), round(40 / batch_size))
    pilot_starts = [starts[-1]] if sample_count == 1 else [
        starts[round(i * (len(starts) - 1) / (sample_count - 1))] for i in range(sample_count)
    ]
    piloted = set(pilot_starts)
    schedule = pilot_starts + [start for start in starts if start not in piloted]
    timings = []
    partial = output / "features.partial.npz"
    began = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for batch_number, start in enumerate(schedule, 1):
        if perf_counter() - began >= max_seconds:
            write_features(partial, completed_features(values, done, np), np)
            raise TimeoutError(f"Extraction exceeded {max_seconds}s after {batch_number - 1} forwards")
        batch = prompts[start:start + batch_size]
        sequences = [item.tokens for item in batch]
        n_sentences = sum(isinstance(item, Sentence) for item in batch)
        locations, destinations = [], []
        for row, item in enumerate(batch):
            key = "sentence_end" if isinstance(item, Sentence) else "target_only"
            locations.append((row, len(item.tokens) - 1))
            destinations.append((key, item.index))
            if isinstance(item, Sentence):
                for index, position in item.target_positions:
                    locations.append((row, position))
                    destinations.append(("target_end", index))

        started = perf_counter()
        states = encode(language, sequences, pad_id, torch)
        rows, positions = torch.tensor(locations, dtype=torch.long, device="cuda").unbind(dim=1)
        readouts = states[rows, positions].cpu().numpy()
        if readouts.dtype != np.float32 or not np.isfinite(readouts).all():
            raise RuntimeError(f"Expected finite FP32 Qwen states in batch {batch_number}")
        for (key, index), state in zip(destinations, readouts, strict=True):
            values[key][index] = state
            done[key][index] = True
        del states

        timings.append({
            "prompts": len(batch), "sentence_prompts": n_sentences,
            "target_only_prompts": len(batch) - n_sentences,
            "tokens": sum(map(len, sequences)), "padded_tokens": len(batch) * max(map(len, sequences)),
            "elapsed_s": perf_counter() - started,
        })
        if batch_number == sample_count:
            pilot_seconds = sum(row["elapsed_s"] for row in timings)
            pilot_tokens = sum(row["tokens"] for row in timings)
            pilot = {
                "prompts": sum(row["prompts"] for row in timings),
                "tokens": pilot_tokens, "forward_and_readout_s": pilot_seconds,
                "tokens_per_second": pilot_tokens / pilot_seconds,
                "batches": timings.copy(), **memory(torch),
            }
            save_json(output / "pilot.json", pilot)
            if pilot["free_bytes_at_observed_peak_reserved"] < MIN_HEADROOM_BYTES:
                raise RuntimeError("L4 pilot left less than 2 GiB of device headroom")
            print(f"Pilot passed on {pilot['prompts']} prompts ({pilot_tokens} tokens)", flush=True)
        if batch_number == sample_count or batch_number % CHECKPOINT_BATCHES == 0:
            write_features(partial, completed_features(values, done, np), np)
            print(f"Completed {batch_number}/{len(schedule)} extraction batches", flush=True)

    if not all(mask.all() for mask in done.values()):
        raise RuntimeError("Incomplete extraction")
    archive = output / "features.npz"
    write_features(archive, values, np, compressed=True)
    partial.unlink()
    elapsed = sum(row["elapsed_s"] for row in timings)
    tokens = sum(row["tokens"] for row in timings)
    save_json(output / "extraction.json", {
        "observations": len(inventory.ids), "unique_sentences": len(inventory.sentences),
        "unique_targets": len(inventory.targets), "native_prefix_matches": len(inventory.ids),
        "backbone_forward_calls": len(timings), "max_examples_per_forward": batch_size,
        "tokens": tokens, "padded_tokens": sum(row["padded_tokens"] for row in timings),
        "forward_and_readout_s": elapsed, "tokens_per_second": tokens / elapsed,
        "extraction_wall_s": perf_counter() - began, "feature_archive_bytes": archive.stat().st_size,
        **memory(torch), "batches": timings,
    })
    print(f"Encoded {len(prompts)} distinct text inputs for {len(inventory.ids)} observations", flush=True)


def run(stimuli, targets, output, batch_size, max_seconds):
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
        raise RuntimeError(f"Expected a Colab L4 with native BF16 support, got {device.name}")
    if max_seconds <= 0:
        raise ValueError("Extraction time budget must be positive")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("features*.npz")):
        raise FileExistsError("Use a fresh output directory for each CompLex extraction run")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    save_json(output / "environment.json", {
        "gpu": device.name, "vram_bytes": device.total_memory,
        "compute_capability": [device.major, device.minor], "cuda": torch.version.cuda,
        "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
        "model": MODEL, "revision": REVISION,
        "stimuli_sha256": hashlib.sha256(stimuli.read_bytes()).hexdigest(),
        "targets_sha256": hashlib.sha256(targets.read_bytes()).hexdigest(),
        "checkpoint_load_dtype": "bfloat16", "compute_dtype": "float32", "storage_dtype": "float32",
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "attention": "sdpa",
        "causal_conv1d": optional_version("causal-conv1d"),
        "flash_linear_attention": optional_version("flash-linear-attention"),
        "readout": "final_normalized_hidden_state", "add_special_tokens": False,
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
        inventory = tokenize(tokenizer, stimuli, targets)
        if len(inventory.ids) != EXPECTED_OBSERVATIONS:
            raise ValueError(f"Expected {EXPECTED_OBSERVATIONS} audited CompLex targets, got {len(inventory.ids)}")
        model, loading = Qwen3_5ForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa",
            use_safetensors=True, output_loading_info=True,
        )
        if any(loading.values()) or next(model.model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError(f"Pinned Qwen BF16 weights did not load exactly: {loading}")
        language = model.model.eval().requires_grad_(False).to(device="cuda", dtype=torch.float32)
        del model
        if next(language.parameters()).dtype != torch.float32 or language.config.hidden_size != HIDDEN_SIZE:
            raise RuntimeError("Expected the pinned 2048-dimensional Qwen backbone in FP32")
        ready = perf_counter()
        causality = causal_checks(language, inventory.sentences, tokenizer.pad_token_id, torch)
        save_json(output / "causality.json", {"tokenizer_class": type(tokenizer).__name__, **causality})
        if not all(check["passed"] for check in causality["checks"]):
            raise RuntimeError("CompLex causal readout checks failed")
        checked = perf_counter()
        collect(language, inventory, tokenizer.pad_token_id, output, torch, np, batch_size, max_seconds)
        save_json(output / "timings.json", {
            "model_and_tokenizer_setup_s": ready - began,
            "causal_checks_s": checked - ready, "total_script_s": perf_counter() - began,
        })
    finally:
        stop.set()
        sampler.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stimuli", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--max-seconds", type=float, default=780)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    run(args.stimuli, args.targets, args.output, args.batch_size, args.max_seconds)
