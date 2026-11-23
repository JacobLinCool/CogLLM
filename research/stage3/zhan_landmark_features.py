"""Extract sentence-boundary and supplied-pronoun states from Zhan Experiment 1."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

from .run import MODEL, REVISION, TRANSFORMERS_VERSION, optional_version, save_json


PREFIX_RMS_TOLERANCE = 0.02


def tokenize_pairs(stimuli_path, tokenizer):
    families = defaultdict(dict)
    stimulus_ids = set()
    for line in stimuli_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        family = families[row["family_id"]]
        condition = row["metadata"]["prompt_type"]
        if row["id"] in stimulus_ids or condition in family:
            raise ValueError(f"Duplicate Zhan stimulus {row['id']}")
        stimulus_ids.add(row["id"])
        family[condition] = row

    if len(families) != 36 or any(set(pair) != {"free", "pronoun"} for pair in families.values()):
        raise ValueError("Expected 36 paired free and pronoun prompts")

    pairs = []
    for family_id, pair in sorted(families.items()):
        free, pronoun = pair["free"], pair["pronoun"]
        short, full = free["text"], pronoun["text"]
        anchors = {segment["id"]: segment for segment in pronoun["segments"]}
        anchor = anchors["pronoun"]
        if (
            anchor["start"] != len(short) or anchor["end"] != len(full)
            or not short.endswith("。") or full[:-1] != short or full[-1] not in "他她"
        ):
            raise ValueError(f"Invalid pronoun landmark for {family_id}")

        free_tokens, full_tokens = (
            tokenizer(text, add_special_tokens=False, return_offsets_mapping=True) for text in (short, full)
        )
        shared, all_ids = free_tokens["input_ids"], full_tokens["input_ids"]
        free_offsets, offsets = free_tokens["offset_mapping"], full_tokens["offset_mapping"]
        if (
            not shared or len(all_ids) != len(shared) + 1
            or all_ids[:len(shared)] != shared or offsets[:len(shared)] != free_offsets
            or free_offsets[-1][1] != len(short)
            or any(not len(short) <= left < right <= len(full) for left, right in offsets[len(shared):])
            or offsets[-1][1] != len(full)
        ):
            raise ValueError(f"Shared tokenization or isolated pronoun failed for {family_id}")

        pairs.append((family_id, (free["id"], pronoun["id"]), shared, all_ids, offsets))
    return pairs


def extract(data, model_path, output):
    import mlx.core as mx
    from mlx_lm import load
    import numpy as np
    from transformers import AutoTokenizer

    if model_path.resolve().name != REVISION:
        raise ValueError(f"Expected local checkpoint directory for revision {REVISION}: {model_path}")
    if optional_version("transformers") != TRANSFORMERS_VERSION:
        raise ValueError(f"Expected Transformers {TRANSFORMERS_VERSION}")

    started = perf_counter()
    stimuli_path = data / "stimuli.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError("The pinned tokenizer must expose character offsets")
    pairs = tokenize_pairs(stimuli_path, tokenizer)

    device = mx.gpu
    mx.set_default_device(device)
    model, _ = load(str(model_path))
    encoder = model.language_model.model
    weight_dtype = str(encoder.embed_tokens.weight.dtype)
    encoder.set_dtype(mx.float32)
    mx.eval(encoder.parameters())
    if encoder.embed_tokens.weight.dtype != mx.float32:
        raise RuntimeError("Expected FP32 backbone weights")
    setup_s = perf_counter() - started

    vectors = {}
    anchors = []
    forward_s = 0.0
    check_s = 0.0
    for family_id, ids, prefix_ids, full_ids, offsets in pairs:
        before = perf_counter()
        states = encoder(mx.array([full_ids]))[0]
        if states.shape != (len(full_ids), 2048):
            raise ValueError(f"Unexpected Qwen state shape {states.shape}")
        full = np.array(states, copy=True)
        forward_s += perf_counter() - before

        before = perf_counter()
        prefix = np.array(encoder(mx.array([prefix_ids]))[0], copy=True)
        prefix_delta = prefix - full[:len(prefix_ids)]
        norm = np.linalg.norm(prefix, axis=1)
        if not np.isfinite(full).all() or not np.isfinite(prefix).all() or (norm == 0).any():
            raise ValueError(f"Invalid Qwen states in {family_id}")
        max_rms = float((np.linalg.norm(prefix_delta, axis=1) / norm).max())
        if max_rms > PREFIX_RMS_TOLERANCE:
            raise ValueError(f"Inconsistent prefix in {family_id}: {max_rms:.6g}")
        check_s += perf_counter() - before

        indices = (len(prefix_ids) - 1, len(full_ids) - 1)
        for stimulus_id, index in zip(ids, indices, strict=True):
            vectors[stimulus_id] = full[index]
        anchors.append({
            "family_id": family_id,
            "free_id": ids[0],
            "pronoun_id": ids[1],
            "landmark_indices": indices,
            "landmark_offsets": [offsets[i] for i in indices],
            "pronoun_token_count": len(full_ids) - len(prefix_ids),
            "full_token_count": len(full_ids),
            "prefix_max_token_relative_rms": max_rms,
        })

    output.mkdir(parents=True, exist_ok=True)
    ids = sorted(vectors)
    hidden = np.stack([vectors[stimulus_id] for stimulus_id in ids])
    np.savez_compressed(output / "features.npz", ids=np.array(ids), hidden=hidden)
    save_json(output / "extraction.json", {
        "model": MODEL,
        "revision": REVISION,
        "stimuli_sha256": hashlib.sha256(stimuli_path.read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "versions": {pkg: optional_version(pkg) for pkg in ("mlx", "mlx-lm", "numpy", "transformers")},
        "device": str(device),
        "checkpoint_embedding_dtype": weight_dtype,
        "compute_embedding_dtype": str(encoder.embed_tokens.weight.dtype),
        "cached_state_dtype": str(hidden.dtype),
        "family_count": len(pairs),
        "landmark_count": len(ids),
        "main_forward_calls": len(pairs),
        "main_forward_token_count": sum(len(pair[3]) for pair in pairs),
        "prefix_verification_forward_calls": len(pairs),
        "prefix_max_token_relative_rms": max(row["prefix_max_token_relative_rms"] for row in anchors),
        "prefix_rms_tolerance": PREFIX_RMS_TOLERANCE,
        "model_and_tokenizer_setup_s": setup_s,
        "main_forward_s": forward_s,
        "prefix_verification_s": check_s,
        "anchors": anchors,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    extract(args.data, args.model, args.output)
