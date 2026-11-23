"""Export a matched CompLex inventory with held-out ratings kept local."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ..stage3.probe import _jsonl
from ..stage3.run import REVISION
from .complex_tokenization import token_span


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def prepare(data, tokenizer_path, output):
    from tokenizers import Tokenizer

    if tokenizer_path.parent.name != REVISION or tokenizer_path.name != "tokenizer.json":
        raise ValueError("Expected tokenizer.json from the pinned Qwen3.5-2B revision")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    stimuli = {item["id"]: item for item in _jsonl(data / "stimuli.jsonl")}
    encodings = {
        key: tokenizer.encode(stimulus["text"], add_special_tokens=False) for key, stimulus in stimuli.items()
    }
    retained = set()
    targets, design = [], []
    counts, removed = Counter(), Counter()
    for obs in _jsonl(data / "observations.jsonl"):
        stimulus = stimuli[obs["stimulus_id"]]
        span = obs["target_span"]
        text = stimulus["text"]
        start, end = span["start"], span["end"]
        split = obs["split"]
        if not (0 <= start < end <= len(text)) or text[start:end] != span["text"] or split != stimulus["split"]:
            raise ValueError(f"Invalid fixed CompLex target for {obs['id']}")
        encoding = encodings[stimulus["id"]]
        status, indices = token_span(text, encoding.offsets, start, end)
        if indices is None:
            removed[(split, status)] += 1
            continue
        prefix = tokenizer.encode(text[:end], add_special_tokens=False)
        if encoding.ids[:indices[1] + 1] != prefix.ids:
            removed[(split, "changed_stopping_prefix")] += 1
            continue
        rating = obs["observations"]["lexical_difficulty"]
        if rating["status"] != "observed" or not 0 <= rating["value"] <= 1:
            raise ValueError(f"Invalid released CompLex rating for {obs['id']}")
        targets.append({"id": obs["id"], "stimulus_id": stimulus["id"], "target_span": span})
        design.append({
            "id": obs["id"], "stimulus_id": stimulus["id"], "context_group": stimulus["context_group"],
            "split": split, "kind": obs["kind"], "corpus": obs["corpus"], "target_text": span["text"],
            "value": None if split == "test" else rating["value"],
        })
        retained.add(stimulus["id"])
        counts[split] += 1

    if (
        counts != {"train": 8667, "trial": 503, "test": 1094}
        or sum(removed.values()) != 15 or len({row["id"] for row in targets}) != len(targets)
    ):
        raise ValueError("Token-aligned inventory differs from the audited Stage 5 inputs")
    retained_stimuli = [stimuli[key] for key in stimuli if key in retained]
    isolated_texts = {row["target_text"] for row in design}
    output.mkdir(parents=True, exist_ok=True)
    files = {"stimuli.jsonl": retained_stimuli, "targets.jsonl": targets, "design.jsonl": design}
    for name, rows in files.items():
        write_jsonl(output / name, rows)
    report = {
        "scope": "Same token-aligned inventory for all three conditions. No test ratings in the fit payload.",
        "rows_by_split": dict(counts),
        "excluded_by_split_and_reason": {f"{split}/{reason}": n for (split, reason), n in sorted(removed.items())},
        "sentence_inputs": len(retained_stimuli),
        "sentence_input_tokens": sum(len(encodings[row["id"]].ids) for row in retained_stimuli),
        "isolated_target_inputs": len(isolated_texts),
        "isolated_target_tokens": sum(len(tokenizer.encode(text, add_special_tokens=False).ids) for text in isolated_texts),
        "tokenizer_sha256": digest(tokenizer_path),
        "source_sha256": {name: digest(data / name) for name in ("stimuli.jsonl", "observations.jsonl")},
        "output_sha256": {name: digest(output / name) for name in files},
    }
    (output / "input-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("data", "tokenizer", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.data, args.tokenizer, args.output)
