"""Repack the locked PLABA and CompLex cohorts for shared causal extraction."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.stage2.__main__ import development_folds
from research.stage2.records import write_jsonl
from research.stage5.complex_prepare import canonical_context


RESEARCH = Path(__file__).resolve().parents[1]
RESULTS = RESEARCH / "results"
TARGETS = ("processing_ease", "perceived_understanding")
COMPLEX_SPLITS = ("train", "trial", "test")


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def checked_file(path, expected):
    actual = digest(path)
    if actual != expected:
        raise ValueError(f"Source digest changed for {path}")
    return actual


def index_rows(rows, key):
    indexed = {row[key]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate values in key {key}")
    return indexed


def reader_connected_groups(documents, trials, folds):
    reader_families = defaultdict(set)
    for trial in trials:
        reader = trial["participant_id"]
        if not reader:
            raise ValueError("A PLABA trial is missing a reader key")
        reader_families[reader].add(documents[trial["stimulus_id"]]["family_id"])
    adjacency = {family: {family} for family in folds}
    for families in reader_families.values():
        if len({folds[family] for family in families}) != 1:
            raise ValueError("A reader crossed the locked PLABA folds")
        for family in families:
            adjacency[family].update(families)

    groups = {}
    for first in sorted(adjacency):
        if first in groups:
            continue
        remaining = {first}
        while remaining:
            family = remaining.pop()
            if family not in groups:
                groups[family] = first
                remaining.update(adjacency[family] - groups.keys())
    return groups, len(reader_families)


def plaba_rows(cache):
    aligned = cache / "stage2" / "plaba"
    author_csv = cache / "raw" / "plaba" / "sentences.csv"
    stage2_receipt = RESULTS / "stage2-audit.json"
    ordinal_receipt = RESULTS / "causal-plaba-ordinal" / "manifest.json"
    stage2 = read_json(stage2_receipt)["plaba"]
    ordinal = read_json(ordinal_receipt)
    input_hashes = {}
    for filename in ("stimuli.jsonl", "trials.jsonl", "splits.json"):
        expected = ordinal["data"]["current_stage2_sha256"][filename]
        if expected != stage2["artifacts"][filename]["sha256"]:
            raise ValueError(f"Stage 2 and causal ordinal hashes disagree for {filename}")
        input_hashes[filename] = checked_file(aligned / filename, expected)
    input_hashes["sentences.csv"] = checked_file(author_csv, stage2["source"]["files"]["sentences.csv"]["sha256"])

    documents = index_rows(read_jsonl(aligned / "stimuli.jsonl"), "id")
    trials = read_jsonl(aligned / "trials.jsonl")
    split = read_json(aligned / "splits.json")
    if development_folds(documents, trials, group_readers=True) != split:
        raise ValueError("PLABA reader-connected folds no longer reproduce the lock")
    folds = {family: row["fold"] for row in split["folds"] for family in row["families"]}
    groups, reader_count = reader_connected_groups(documents, trials, folds)
    with author_csv.open(encoding="utf-8", newline="") as source:
        author_anchors = index_rows(list(csv.DictReader(source)), "sentence_uid")

    stimuli, layout = [], []
    rewrite_variants = defaultdict(set)
    for key, document in documents.items():
        text, family = document["text"], document["family_id"]
        variant = document["metadata"]["variant"]
        stops = []
        for span in document["segments"]:
            anchor_id, start, end = span["id"], span["start"], span["end"]
            source = author_anchors[anchor_id]
            if (
                not 0 <= (stops[-1] if stops else 0) <= start < end <= len(text)
                or (key, family, variant, text[start:end]) != (
                    source["doc_id"], source["base_text_id"], source["variant"], source["sentence_text"],
                )
                or variant not in {"orig", "simp"}
            ):
                raise ValueError(f"Invalid PLABA author sentence coordinates for {anchor_id}")
            rewrite_anchor = f"{family}:{int(source['mapped_orig_sentence_index'])}"
            layout.append({
                "id": key, "anchor_id": anchor_id, "start": start, "end": end,
                "group_id": groups[family], "family_id": family, "fold": folds[family],
                "variant": variant, "alignment_anchor_id": rewrite_anchor,
            })
            rewrite_variants[rewrite_anchor].add(variant)
            stops.append(end)
        if not stops:
            raise ValueError(f"PLABA document has no stops: {key}")
        stimuli.append({"id": key, "text": text, "stops": stops})

    by_anchor = index_rows(layout, "anchor_id")
    if by_anchor.keys() != author_anchors.keys():
        raise ValueError("The PLABA stop inventory differs from the author sentence inventory")
    histograms = defaultdict(lambda: {name: [0] * 5 for name in TARGETS})
    pair_count = 0
    for trial in trials:
        if trial["exposure"]["task"] != "sentence_rating":
            continue
        anchor = trial["target_segment_id"]
        key = by_anchor[anchor]["id"]
        if (
            trial["stimulus_id"] != key or trial["exposure"]["mode"] != "full_text"
            or trial["exposure"]["visible_end"] != len(documents[key]["text"])
        ):
            raise ValueError(f"Invalid original PLABA sentence rating exposure for {anchor}")
        for name in TARGETS:
            observation = trial["observations"][name]
            value = observation["value"]
            if observation["status"] != "observed" or type(value) is not int or not 1 <= value <= 5:
                raise ValueError(f"Invalid original PLABA rating for {anchor}")
            histograms[anchor][name][value - 1] += 1
        pair_count += 1
    if histograms.keys() != by_anchor.keys():
        raise ValueError("Some PLABA released sentences lack their ratings")

    outcomes = [{"anchor_id": anchor, **histograms[anchor]} for anchor in by_anchor]
    paired_alignments = sum(len(variants) == 2 for variants in rewrite_variants.values())
    fold_counts = Counter(row["fold"] for row in layout)
    expected_counts = (ordinal["sentences"], ordinal["rating_pairs"], ordinal["readers"], ordinal["reader_connected_groups"])
    if (
        (len(layout), pair_count, reader_count, len(set(groups.values()))) != expected_counts
        or [fold_counts[fold] for fold in range(5)] != ordinal["folds"]
        or paired_alignments != stage2["metrics"]["paired_sentence_alignments"]
    ):
        raise ValueError("Causal PLABA counts or alignment denominator differ from the original assay")

    return stimuli, layout, outcomes, {
        "source": stage2["source"]["source"], "license": stage2["source"]["license"],
        "input_sha256": input_hashes,
        "prior_receipt_sha256": {path.name if path != ordinal_receipt else "causal-plaba-ordinal/manifest.json":
                                 digest(path) for path in (stage2_receipt, ordinal_receipt)},
        "documents": len(stimuli), "sentences": len(layout), "rating_pairs": pair_count, "reader_count": reader_count,
        "reader_connected_groups": len(set(groups.values())), "segments_by_fold": dict(sorted(fold_counts.items())),
        "paired_rewrite_anchors": paired_alignments,
        "split_policy": "Original 5 development folds, grouped by article family and any shared reader.",
        "bootstrap_group_ids": "Stable smallest family ID within each original reader-connected component.",
        "human_exposure": "Whole published article before rating the designated sentence.",
        "rating_order": [1, 2, 3, 4, 5],
    }


def complex_rows(cache):
    aligned = cache / "stage5" / "complex-l4-input"
    original = cache / "stage5" / "complex"
    receipt_path = RESULTS / "stage5-complex" / "input-audit.json"
    receipt = read_json(receipt_path)
    source_preparation = read_json(original / "preparation.json")
    hashes = {
        source: {name: checked_file(directory / name, expected) for name, expected in values.items()}
        for source, directory, values in (
            ("pre_token_filter", original, receipt["source_sha256"]),
            ("stage5_qwen_aligned", aligned, receipt["output_sha256"]),
        )
    }
    if any(hashes["pre_token_filter"][name] != expected for name, expected in source_preparation["output_sha256"].items()):
        raise ValueError("Original CompLex span selection hashes disagree")
    documents = index_rows(read_jsonl(aligned / "stimuli.jsonl"), "id")
    targets = index_rows(read_jsonl(aligned / "targets.jsonl"), "id")
    design = index_rows(read_jsonl(aligned / "design.jsonl"), "id")
    observations = index_rows(read_jsonl(original / "observations.jsonl"), "id")
    if design.keys() != targets.keys() or targets.keys() - observations.keys():
        raise ValueError("CompLex target IDs disagree with the pinned Stage 5 design")
    excluded = Counter(observations[key]["split"] for key in observations.keys() - targets.keys())
    if excluded != {"train": 13, "test": 2}:
        raise ValueError("CompLex exclusions differ from the original tokenizer audit")

    ends, contexts, layout, outcomes = defaultdict(set), defaultdict(set), [], []
    for anchor, target in targets.items():
        observed, planned = observations[anchor], design[anchor]
        key = target["stimulus_id"]
        document, span = documents[key], target["target_span"]
        text = document["text"]
        start, end = span["start"], span["end"]
        split, corpus, kind = (planned[name] for name in ("split", "corpus", "kind"))
        rating = observed["observations"]["lexical_difficulty"]
        group = hashlib.sha256(canonical_context(text).encode("utf-8")).hexdigest()
        if (
            not 0 <= start < end <= len(text) or text[start:end] != span["text"]
            or key != observed["stimulus_id"] or span != observed["target_span"]
            or (split, corpus, kind) != (observed["split"], observed["corpus"], observed["kind"])
            or key != planned["stimulus_id"] or span["text"] != planned["target_text"]
            or group != document["context_group"] or group != planned["context_group"]
            or split != document["split"] or split not in COMPLEX_SPLITS
            or rating["status"] != "observed" or not 0 <= rating["value"] <= 1
            or planned["value"] != (None if split == "test" else rating["value"])
        ):
            raise ValueError(f"CompLex released target disagrees with the old alignment for {anchor}")
        contexts[split].add(group)
        ends[key].add(end)
        layout.append({
            "id": key, "anchor_id": anchor, "start": start, "end": end, "group_id": group,
            "split": split, "kind": kind, "corpus": corpus,
        })
        outcomes.append({"anchor_id": anchor, "value": rating["value"]})

    row_counts = Counter(row["split"] for row in layout)
    groups_by_split = {split: len(contexts[split]) for split in COMPLEX_SPLITS}
    if (
        row_counts != receipt["rows_by_split"] or len(documents) != receipt["sentence_inputs"]
        or ends.keys() != documents.keys() or sum(groups_by_split.values()) != len(set.union(*contexts.values()))
    ):
        raise ValueError("CompLex split coverage or context disjointness changed")

    stimuli = [{"id": key, "text": document["text"], "stops": sorted(ends[key])} for key, document in documents.items()]
    return stimuli, layout, outcomes, {
        "source": source_preparation["source"], "release": source_preparation["release"],
        "input_sha256": hashes, "prior_receipt_sha256": digest(receipt_path),
        "sentences": len(stimuli), "anchors_by_split": dict(row_counts), "groups_by_split": groups_by_split,
        "excluded_from_original_qwen_alignment_by_split": dict(sorted(excluded.items())),
        "original_full_sentence_qwen_tokens": receipt["sentence_input_tokens"],
        "split_policy": "Original Stage 5 normalized sentence-disjoint full train, trial, and holdout test.",
        "human_exposure": "Readers saw the whole sentence before rating the designated target at 0–1 difficulty.",
    }


def prepare(cache, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="backbone-packet-", dir=output.parent) as temporary:
        bundle = Path(temporary) / "prepared"
        bundle.mkdir()
        audit = {
            "coordinates": "Half-open Python Unicode codepoint spans. Stops contain sorted unique exclusive ends.",
            "model_input": "Only stimuli.jsonl enters the model. Layout supplies anchor lookup and split metadata.",
            "outcome_handling": "outcomes.jsonl stays local. Join on anchor ID after sealing models' extracted features.",
            "sampling": "All original rows enter this packet before alternative tokenizer checks.",
            "preparation_sha256": digest(Path(__file__)),
            "datasets": {},
        }
        for name, loader in (("plaba", plaba_rows), ("complex", complex_rows)):
            stimuli, layout, outcomes, receipt = loader(cache)
            dataset = bundle / name
            dataset.mkdir()
            files = {"stimuli.jsonl": stimuli, "layout.jsonl": layout, "outcomes.jsonl": outcomes}
            if name == "complex":
                spans = {(row["id"], row["start"], row["end"]) for row in layout}
                files["targets.jsonl"] = [
                    {"id": identifier, "start": start, "end": end} for identifier, start, end in sorted(spans)
                ]
            receipt["output_sha256"] = {}
            for filename, rows in files.items():
                path = dataset / filename
                write_jsonl(path, rows)
                receipt["output_sha256"][filename] = digest(path)
            audit["datasets"][name] = receipt
        (bundle / "source-audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        bundle.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True, help="Root of the existing CogLLM data cache")
    parser.add_argument("--output", type=Path, required=True, help="New prepared directory")
    prepare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
