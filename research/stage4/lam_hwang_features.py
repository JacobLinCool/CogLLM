"""Extract controlled-boundary and native Qwen states for the Lam–Hwang holdout."""

import argparse
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

from ..stage3.run import MODEL, REVISION, TRANSFORMERS_VERSION, encode, optional_version, sample_gpu, save_json


CONDITIONS = {"free": "", "overt_ta_m": "他", "overt_ta_f": "她"}
PREFIX_TOLERANCE = 0.02


def tokenize(tokenizer, path):
    families = {}
    ids = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            condition = row["metadata"]["prompt_type"]
            family = families.setdefault(row["family_id"], {})
            if condition not in CONDITIONS or condition in family or row["id"] in ids:
                raise ValueError("Unknown condition or duplicate stimulus")
            family[condition] = row
            ids.add(row["id"])
    if len(families) != 24 or any(family.keys() != CONDITIONS.keys() for family in families.values()):
        raise ValueError("Expected 24 families, each with the free prompt and two written pronoun variants")

    cue_tokens = {
        name: tokenizer(cue, add_special_tokens=False)["input_ids"]
        for name, cue in CONDITIONS.items() if cue
    }
    if any(len(tokens) != 1 for tokens in cue_tokens.values()):
        raise ValueError("Expected each standalone written pronoun to be one token")
    documents = []
    checks = []
    for family_id, family in sorted(families.items()):
        free = family["free"]["text"]
        if not free.endswith("，因为"):
            raise ValueError(f"Unexpected free-prompt boundary for {family_id}")
        encoded = {}
        for condition, cue in CONDITIONS.items():
            row = family[condition]
            if row["text"] != free + cue:
                raise ValueError(f"Mismatch at the native prompt boundary for {row['id']}")
            if cue and [s for s in row["segments"] if s["id"] == "pronoun"] != [
                {"id": "pronoun", "start": len(free), "end": len(free) + 1},
            ]:
                raise ValueError(f"Unexpected pronoun offset for {row['id']}")
            tokens = tokenizer(row["text"], add_special_tokens=False, return_offsets_mapping=True)
            token_ids, offsets = tokens["input_ids"], tokens["offset_mapping"]
            if not token_ids or len(token_ids) != len(offsets) or offsets[-1][1] != len(row["text"]):
                raise ValueError(f"The native final token fails to reach the prompt boundary for {row['id']}")
            documents.append((("native", row["id"]), token_ids))
            encoded[condition] = (row["id"], token_ids, offsets)

        free_id, free_tokens, free_offsets = encoded["free"]
        for condition in ("overt_ta_m", "overt_ta_f"):
            variant_id, variant_tokens, variant_offsets = encoded[condition]
            shared = 0
            for before, after in zip(zip(free_tokens, free_offsets), zip(variant_tokens, variant_offsets)):
                if before != after:
                    break
                shared += 1
            if shared == len(free_tokens) or free_offsets[-1] != (len(free) - 2, len(free)):
                raise ValueError(f"The native tokenizer no longer has the expected fused cue for {variant_id}")
            if variant_offsets[-1] != (len(free) - 2, len(free) + 1) or not shared:
                raise ValueError(f"Unable to validate the native fused cue range for {variant_id}")
            controlled = free_tokens + cue_tokens[condition]
            decoded = tokenizer.decode(controlled, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if decoded.encode("utf-8") != family[condition]["text"].encode("utf-8"):
                raise ValueError(f"Controlled tokens do not reproduce the authored text for {variant_id}")
            documents.append((("controlled", variant_id), controlled))
            checks.append({
                "family_id": family_id,
                "free_id": free_id,
                "variant_id": variant_id,
                "shared_tokens_before_fused_cue": shared,
                "free_final_token_offset": free_offsets[-1],
                "variant_final_token_offset": variant_offsets[-1],
                "controlled_prefix_tokens": len(free_tokens),
                "controlled_pronoun_token_id": cue_tokens[condition][0],
                "controlled_decode_utf8_exact": True,
            })
    return documents, checks


def collect(language, tokenizer, path, output, torch, np):
    documents, checks = tokenize(tokenizer, path)
    states = {}
    timings = []
    torch.cuda.reset_peak_memory_stats()
    for length, same_length in groupby(sorted(documents, key=lambda row: len(row[1])), key=lambda row: len(row[1])):
        group = list(same_length)
        for start in range(0, len(group), 4):
            batch = group[start:start + 4]
            began = perf_counter()
            hidden = encode(language, [tokens for _, tokens in batch], tokenizer.pad_token_id, torch)
            if hidden.shape != (len(batch), length, 2048) or hidden.dtype != torch.float32:
                raise RuntimeError(f"Unexpected Qwen final states: {hidden.shape}, {hidden.dtype}")
            states.update(zip((key for key, _ in batch), hidden.cpu().numpy(), strict=True))
            del hidden
            timings.append({
                "conditions": len(batch), "tokens": len(batch) * length, "elapsed_s": perf_counter() - began,
            })
    def relative_rms(left, right):
        error = np.linalg.norm(left - right, axis=1) / np.maximum(np.linalg.norm(left, axis=1), 1e-8)
        return float(error.max())

    for check in checks:
        end = check["shared_tokens_before_fused_cue"]
        free = states["native", check["free_id"]]
        native = states["native", check["variant_id"]]
        controlled = states["controlled", check["variant_id"]]
        check["native_pre_fusion_max_token_relative_rms"] = relative_rms(free[:end], native[:end])
        check["controlled_vs_standalone_prefix_max_token_relative_rms"] = relative_rms(free, controlled[:-1])
        check["passed"] = max(
            check["native_pre_fusion_max_token_relative_rms"],
            check["controlled_vs_standalone_prefix_max_token_relative_rms"],
        ) <= PREFIX_TOLERANCE
    prefix_by_family = {}
    for check in checks:
        prefix_by_family.setdefault(check["family_id"], []).append(states["controlled", check["variant_id"]][:-1])
    variant_prefix_comparisons = {
        family: relative_rms(first, second) for family, (first, second) in prefix_by_family.items()
    }
    save_json(output / "causality.json", {
        "scope": "Controlled byte-exact prefix/cue encoding; native conditions share only pre-fusion tokens",
        "prefix_rms_tolerance": PREFIX_TOLERANCE,
        "controlled_prefix_between_spellings_max_token_relative_rms": variant_prefix_comparisons,
        "checks": checks,
    })
    if not all(check["passed"] for check in checks) or max(variant_prefix_comparisons.values()) > PREFIX_TOLERANCE:
        raise RuntimeError("Prefix invariance failed")
    ids = sorted(key for mode, key in states if mode == "native")
    native_hidden = np.stack([states["native", key][-1] for key in ids])
    free_to_variant = {check["free_id"]: check["variant_id"] for check in checks[::2]}
    hidden = np.stack([
        states["controlled", free_to_variant[key]][-2] if key in free_to_variant else states["controlled", key][-1]
        for key in ids
    ])
    controlled_prefix_hidden = np.stack([
        states["controlled", free_to_variant.get(key, key)][-2] for key in ids
    ])
    if any(array.shape != (72, 2048) or not np.isfinite(array).all() for array in (hidden, native_hidden, controlled_prefix_hidden)):
        raise ValueError("Expected 72 finite states for each tokenization representation")
    temporary = output / "features.partial.npz"
    np.savez_compressed(
        temporary, ids=np.array(ids), hidden=hidden, native_hidden=native_hidden,
        controlled_prefix_hidden=controlled_prefix_hidden,
    )
    temporary.replace(output / "features.npz")
    elapsed = sum(batch["elapsed_s"] for batch in timings)
    tokens = sum(batch["tokens"] for batch in timings)
    save_json(output / "extraction.json", {
        "families": 24, "conditions": len(ids), "observed_pronoun_spellings": False,
        "native_cue_fusions": len(checks), "byte_exact_controlled_decodes": len(checks),
        "encoded_native_conditions": len(ids), "encoded_controlled_variants": len(checks),
        "controlled_free_reference": "Free row uses the masculine variant prefix; each overt row also has its own same-forward prefix.",
        "main_forward_calls": len(timings), "main_forward_token_count": tokens, "main_forward_and_readout_s": elapsed,
        "max_native_pre_fusion_prefix_token_relative_rms": max(check["native_pre_fusion_max_token_relative_rms"] for check in checks),
        "max_controlled_vs_standalone_prefix_token_relative_rms": max(check["controlled_vs_standalone_prefix_max_token_relative_rms"] for check in checks),
        "max_controlled_prefix_between_spellings_token_relative_rms": max(variant_prefix_comparisons.values()),
        "tokens_per_second": tokens / elapsed, "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(), "batches": timings,
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
    path = data / "stimuli.jsonl"
    save_json(output / "environment.json", {
        "gpu": device.name, "vram_bytes": device.total_memory, "cuda": torch.version.cuda,
        "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
        "model": MODEL, "revision": REVISION, "stimuli_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checkpoint_load_dtype": "bfloat16", "compute_dtype": "float32", "attention": "sdpa",
        "causal_conv1d": optional_version("causal-conv1d"),
        "flash_linear_attention": optional_version("flash-linear-attention"),
        "visible_input": "stimulus_text_only",
    })
    began = perf_counter()
    stop = Event()
    sampler = Thread(target=sample_gpu, args=(output / "telemetry.jsonl", stop))
    sampler.start()
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
        if not tokenizer.is_fast or tokenizer.pad_token_id is None:
            raise RuntimeError("A fast tokenizer with a defined padding token is required")
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
        collect(language, tokenizer, path, output, torch, np)
        save_json(output / "timings.json", {
            "model_and_tokenizer_setup_s": ready - began, "total_script_s": perf_counter() - began,
        })
        print("Extracted 72 controlled-boundary and native states from 24 stimulus families", flush=True)
    finally:
        stop.set()
        sampler.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    run(args.data, args.output)
