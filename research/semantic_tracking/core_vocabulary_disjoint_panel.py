"""Freeze additional masked cue paths for the original held-out target families."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.semantic_tracking.beukers_extract import preflight as certify_prefixes
from research.semantic_tracking.core_vocabulary_marginal_prepare import (
    AUTHOR_REVISION,
    BIN_SIZE,
    DEFAULT_SEED,
    ORDINALS,
    SEPARATOR,
    SOURCE_SHA256,
    SUPPORT_SIZE,
    load_cue_bins,
    path_indices,
    render,
    sha256,
    target_family,
)


SPLIT_SHA256 = "2af0ba7e968d2fa3acfb2b6f774a09cd68819986232b82d5273724af36f3894b"
ORIGINAL_PREFLIGHT_SHA256 = "4fa6258798bc3f76410276491b985031f4c7e8a0d156063fd0908b1bb4451bdd"
PANEL_SEED = "cogllm-core-vocabulary-disjoint-panel-v1"
ORIGINAL_DRAWS = 8
PANEL_DRAWS = 32
STIMULI_FILE = "model_input/stimuli.jsonl"
MIXTURE_FILE = "mixture.jsonl"
PATH_MAP_FILE = "path-indices.jsonl"
CAUSAL_PREFLIGHT_FILE = "causal-preflight.json"
CONDITIONS = ("AoA", "Freq", "Instrength", "NonCore")


def stimulus_row(bins: tuple[tuple[str, ...], ...], index: int) -> dict:
    content, stops = render(bins, index)
    identifier = hashlib.sha256(b"core-vocabulary-stimulus-v1\x00" + content.encode()).hexdigest()[:32]
    return {"id": identifier, "text": content, "stops": stops}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def original_paths(source_bins: dict, original: Path) -> tuple[dict[str, list[int]], list[dict]]:
    preflight_path = original / "preflight.json"
    if sha256(preflight_path) != ORIGINAL_PREFLIGHT_SHA256:
        raise ValueError("Expected the frozen eight-draw preflight")
    receipt = json.loads(preflight_path.read_text(encoding="utf-8"))
    for filename in (STIMULI_FILE, MIXTURE_FILE):
        if sha256(original / filename) != receipt["file_sha256"][filename]:
            raise ValueError(f"Original packet digest differs for {filename}")

    original_draws = {}
    stimuli = []
    mixture = []
    for family, bins in source_bins.items():
        original_draws[family] = path_indices(family, DEFAULT_SEED, ORIGINAL_DRAWS)
        for draw, index in enumerate(original_draws[family], start=1):
            row = stimulus_row(bins, index)
            stimuli.append(row)
            mixture.append({"id": row["id"], "family": family, "draw": draw, "weight": 1 / ORIGINAL_DRAWS})
    if stimuli != read_jsonl(original / STIMULI_FILE) or mixture != read_jsonl(original / MIXTURE_FILE):
        raise ValueError("The pinned cue renderer and draw algorithm did not reconstruct the original packet")
    return original_draws, stimuli


def read_holdout(split: Path, families: set[str]) -> list[dict]:
    if sha256(split) != SPLIT_SHA256:
        raise ValueError("Expected the exact frozen grade-free family split")
    frozen = json.loads(split.read_text(encoding="utf-8"))
    if frozen["packet_preflight_sha256"] != ORIGINAL_PREFLIGHT_SHA256:
        raise ValueError("The frozen family split and original packet differ")
    rows = frozen["targets"]
    expected = Counter({(condition, part): count for condition in CONDITIONS
                        for part, count in (("train", 16), ("holdout", 8))})
    if (len(rows) != len(families) or {row["family"] for row in rows} != families
            or Counter((row["condition"], row["split"]) for row in rows) != expected):
        raise ValueError("Original packet families do not reconcile with the frozen split")
    return [row for row in rows if row["split"] == "holdout"]


def build_panel(source: Path, original: Path, split: Path, tokenizer: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    source_bins = {target_family(target): bins for target, bins in load_cue_bins(source).items()}
    previous_indices, original_stimuli = original_paths(source_bins, original)
    held_out = read_holdout(split, set(source_bins))
    seen_ids = {row["id"] for row in original_stimuli}
    seen_texts = {row["text"] for row in original_stimuli}
    if len(seen_ids) != len(original_stimuli) or len(seen_texts) != len(original_stimuli):
        raise ValueError("Original full cue paths are not unique")

    stimuli = []
    mixture = []
    local_path_map = []
    shared_prefix_counts = [0] * ORDINALS
    panel_prefixes = [set() for _ in range(ORDINALS)]
    for frozen in held_out:
        family = frozen["family"]
        bins = source_bins[family]
        excluded = set(previous_indices[family])
        # Drawing at most 40 unique full-support ranks guarantees 32 in the complement.
        ordered = path_indices(family, PANEL_SEED, PANEL_DRAWS + ORIGINAL_DRAWS)
        accepted = [index for index in ordered if index not in excluded][:PANEL_DRAWS]
        if len(set(accepted)) != PANEL_DRAWS:
            raise ValueError("Complement does not supply 32 unique paths")
        previous_prefixes = [set() for _ in range(ORDINALS)]
        for index in excluded:
            content, stops = render(bins, index)
            for ordinal, end in enumerate(stops):
                previous_prefixes[ordinal].add(content[:end])

        for draw, index in enumerate(accepted, start=1):
            row = stimulus_row(bins, index)
            identifier, content = row["id"], row["text"]
            if identifier in seen_ids or content in seen_texts:
                raise ValueError("Rendered panel repeats an original or a new full cue path")
            seen_ids.add(identifier)
            seen_texts.add(content)
            stimuli.append(row)
            mixture.append({"id": identifier, "family": family, "draw": draw, "weight": 1 / PANEL_DRAWS})
            local_path_map.append({"id": identifier, "family": family, "path_index": index})
            for ordinal, end in enumerate(row["stops"]):
                prefix = content[:end]
                panel_prefixes[ordinal].add(prefix)
                shared_prefix_counts[ordinal] += prefix in previous_prefixes[ordinal]

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix="core-vocab-disjoint-") as temporary:
        packet = Path(temporary) / "panel"
        (packet / "model_input").mkdir(parents=True)
        for filename, rows in ((STIMULI_FILE, stimuli), (MIXTURE_FILE, mixture), (PATH_MAP_FILE, local_path_map)):
            write_jsonl(packet / filename, rows)
        certify_prefixes(packet / STIMULI_FILE, tokenizer, packet / CAUSAL_PREFLIGHT_FILE)
        causal = json.loads((packet / CAUSAL_PREFLIGHT_FILE).read_text(encoding="utf-8"))
        receipt = {
            "schema_version": 1,
            "scope": "Grade-free held-out cue-path mixture sensitivity",
            "author_git_revision": AUTHOR_REVISION,
            "source_sha256": SOURCE_SHA256,
            "split_sha256": SPLIT_SHA256,
            "original_preflight_sha256": ORIGINAL_PREFLIGHT_SHA256,
            "prepare_code_sha256": sha256(Path(__file__)),
            "tokenizer_sha256": causal["tokenizer_sha256"],
            "seed": PANEL_SEED,
            "heldout_families": len(held_out),
            "heldout_families_by_condition": dict(Counter(row["condition"] for row in held_out)),
            "original_paths_verified": len(original_stimuli),
            "bins_per_target": ORDINALS,
            "cues_per_bin": BIN_SIZE,
            "author_paths_per_target": SUPPORT_SIZE,
            "author_path_probability": 1 / SUPPORT_SIZE,
            "excluded_paths_per_target": ORIGINAL_DRAWS,
            "conditional_paths_per_target": SUPPORT_SIZE - ORIGINAL_DRAWS,
            "conditional_path_probability": 1 / (SUPPORT_SIZE - ORIGINAL_DRAWS),
            "sampling": "First 32 SHA-256 rejection-sampled unique indices outside the original eight, by family",
            "draws_per_target": PANEL_DRAWS,
            "separator": SEPARATOR,
            "stimuli": len(stimuli),
            "certified_stops": causal["certified_stops"],
            "input_tokens": causal["input_tokens"],
            "max_input_tokens": causal["max_input_tokens"],
            "repeated_full_path_count": len(stimuli) - len(panel_prefixes[-1]),
            "new_path_count_with_original_prefix_by_ordinal": shared_prefix_counts,
            "distinct_new_prefixes_by_ordinal": [len(prefixes) for prefixes in panel_prefixes],
            "file_sha256": {
                filename: sha256(packet / filename)
                for filename in (STIMULI_FILE, MIXTURE_FILE, PATH_MAP_FILE, CAUSAL_PREFLIGHT_FILE)
            },
        }
        (packet / "preflight.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        packet.rename(output)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("source", "original", "split", "tokenizer", "output"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    print(json.dumps(build_panel(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
