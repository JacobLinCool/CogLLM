"""Extract frozen causal readouts for the Currie et al. coherence-monitoring materials."""

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
from time import perf_counter

from research.stage3.probe import _jsonl
from research.stage3.run import MODEL, REVISION, TRANSFORMERS_VERSION, difference, encode, save_json


def plan_readouts(stimuli, tokenizer):
    prompts = defaultdict(list)
    target_checks = []
    ids = []
    for index, row in enumerate(stimuli):
        text, segments = row["text"], row["segments"]
        if len(segments) != 8 or [s["index"] for s in segments] != list(range(8)):
            raise ValueError("Each displayed passage must have eight ordered sentence segments")
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        tokens, offsets = tuple(encoded["input_ids"]), encoded["offset_mapping"]
        if not tokens or len(tokens) != len(offsets):
            raise ValueError(f"Invalid native tokenization for {row['id']}")
        ids.append(row["id"])
        prompts[tokens].append(("full", index, len(tokens) - 1))
        final = segments[-1]
        final_text = text[final["start"]:final["end"]]
        final_tokens = tuple(tokenizer(final_text, add_special_tokens=False)["input_ids"])
        if not final_tokens or text[final["end"]:].strip():
            raise ValueError(f"Invalid final sentence for {row['id']}")
        prompts[final_tokens].append(("last_only", index, len(final_tokens) - 1))
        critical = row["critical_index"]
        support = row["support_index"]
        if (row["condition"] not in {"near", "far"} or type(critical) is not int
                or type(support) is not int or not 0 <= support < critical < len(segments)):
            raise ValueError(f"Invalid manipulation landmarks for {row['id']}")
        span = segments[critical]
        prefix_tokens = tuple(tokenizer(text[:span["end"]], add_special_tokens=False)["input_ids"])
        end_token = len(prefix_tokens) - 1
        if end_token < 0 or tokens[:end_token + 1] != prefix_tokens or offsets[end_token][1] != span["end"]:
            raise ValueError(f"Full and truncated sentence prefixes diverge for {row['id']}")
        prompts[tokens].append(("critical", index, end_token))
        critical_text = text[span["start"]:span["end"]]
        critical_tokens = tuple(tokenizer(critical_text, add_special_tokens=False)["input_ids"])
        if not critical_tokens:
            raise ValueError(f"Empty critical sentence for {row['id']}")
        prompts[critical_tokens].append(("critical_only", index, len(critical_tokens) - 1))
        target_checks.append((row["id"], tokens, prefix_tokens))
    return ids, prompts, target_checks


def run(stimuli_path, output):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    if transformers.__version__ != TRANSFORMERS_VERSION or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected the pinned Transformers version and one CUDA device")
    gpu = torch.cuda.get_device_name(0)
    if gpu != "NVIDIA L4":
        raise RuntimeError(f"Expected NVIDIA L4, got {gpu}")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False

    output.mkdir(parents=True, exist_ok=True)
    stimuli = sorted(_jsonl(stimuli_path), key=lambda row: row["id"])
    if len(stimuli) != 48 or len({row["id"] for row in stimuli}) != len(stimuli):
        raise ValueError("Expected the 48 uniquely joined Currie stimulus variants")
    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    if not tokenizer.is_fast or tokenizer.pad_token_id is None:
        raise RuntimeError("Expected Qwen's pinned fast tokenizer")
    ids, prompts, target_checks = plan_readouts(stimuli, tokenizer)
    if len(target_checks) != 48:
        raise ValueError("Expected the 48 near/far variants with defined landmarks")
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa",
        use_safetensors=True, output_loading_info=True,
    )
    if any(loading.values()) or next(model.model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError(f"Pinned Qwen weights did not load exactly: {loading}")
    language = model.model.eval().requires_grad_(False).to(device="cuda", dtype=torch.float32)
    del model
    setup_seconds = perf_counter() - started

    checks = []
    ordered_checks = sorted(
        (row for row in target_checks if len(row[1]) > len(row[2])), key=lambda row: len(row[2]),
    )
    if len(ordered_checks) < 4:
        raise ValueError("Not enough actual critical stops with a suffix for causal checks")
    for fraction in (0.25, 0.5, 0.75):
        item, full, prefix = ordered_checks[int(fraction * len(ordered_checks))]
        position = len(prefix) - 1
        full, prefix = list(full), list(prefix)
        complete = encode(language, [full], tokenizer.pad_token_id, torch)[0, position:position + 1]
        independent = encode(language, [prefix], tokenizer.pad_token_id, torch)[0, -1:]
        padded = encode(language, [prefix, full], tokenizer.pad_token_id, torch)[0, position:position + 1]
        for name, other in (("suffix_truncated", independent), ("right_padded", padded)):
            check = difference(complete, other)
            if check["max_token_relative_rms"] > 0.02:
                raise RuntimeError(f"Causal stop invariance failed for {item}: {check}")
            checks.append({"stimulus_id": item, "comparison": name, "prefix_tokens": position + 1, **check})

    keys = ("full", "last_only", "critical", "critical_only")
    states = {key: np.zeros((len(ids), language.config.hidden_size), dtype=np.float32) for key in keys}
    inputs = sorted(prompts, key=lambda seq: (len(seq), seq))
    extraction_started = perf_counter()
    torch.cuda.reset_peak_memory_stats()
    batch_size = 8
    for start in range(0, len(inputs), batch_size):
        sequences = inputs[start:start + batch_size]
        hidden = encode(language, list(map(list, sequences)), tokenizer.pad_token_id, torch)
        destinations = [(*dest, r) for r, tokens in enumerate(sequences) for dest in prompts[tokens]]
        vectors = hidden[[r for _, _, _, r in destinations], [p for _, _, p, _ in destinations]].cpu().numpy()
        if vectors.dtype != np.float32 or not np.isfinite(vectors).all() or not np.any(vectors, axis=-1).all():
            raise RuntimeError("The Qwen readouts must be finite nonzero FP32")
        for (key, index, _, _), state in zip(destinations, vectors, strict=True):
            states[key][index] = state
    np.savez_compressed(output / "features.npz", ids=np.asarray(ids), **states)
    save_json(output / "extraction.json", {
        "model": MODEL, "revision": REVISION, "gpu": gpu, "cuda": torch.version.cuda,
        "torch": torch.__version__, "transformers": transformers.__version__, "precision": "float32",
        "stimuli_sha256": hashlib.sha256(stimuli_path.read_bytes()).hexdigest(),
        "stimuli": len(ids), "token_aligned_critical_stops": len(target_checks),
        "unique_inputs": len(inputs), "nonpadding_tokens": sum(map(len, inputs)),
        "forward_batches": (len(inputs) + batch_size - 1) // batch_size,
        "setup_seconds": setup_seconds, "extraction_and_save_seconds": perf_counter() - extraction_started,
        "total_seconds": perf_counter() - started, "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "causal_stop_checks": checks,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stimuli", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.stimuli, args.output)
