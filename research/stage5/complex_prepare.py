"""Produce unique CompLex target spans with sentence-disjoint partitions."""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import unicodedata

from ..joint.complex_audit import COUNTS, LOCKS, RELEASE, SOURCE, SPLITS, _locations, _read


PRIORITY = ("test", "trial", "train")
EXACT = "unique_exact_isolated_match"


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _identifier(value):
    return _sha256(value.encode("utf-8"))


def canonical_context(sentence):
    return " ".join(unicodedata.normalize("NFKC", sentence).casefold().split())


def _read_release(source):
    for relative, expected in LOCKS.items():
        if _sha256((source / relative).read_bytes()) != expected:
            raise ValueError(f"Source SHA-256 differs from the fixed release: {relative}")

    inventories = {split: {} for split in SPLITS}
    for kind, counts in COUNTS.items():
        for split, count in zip(SPLITS[:2], counts[:2], strict=True):
            rows = _read(source / f"{split}/lcp_{kind}_{split}.tsv", True, count)
            inventories[split][kind] = rows
        prompts = _read(source / f"test/lcp_{kind}_test.tsv", False, counts[2])
        labels = _read(source / f"test-labels/lcp_{kind}_test.tsv", True, counts[2])
        if prompts != {key: replace(value, complexity=None) for key, value in labels.items()}:
            raise ValueError(f"The public test prompts and released labels do not match for {kind}")
        inventories["test"][kind] = labels
    return inventories


def unique_exact_span(sentence, target):
    _, status = _locations(sentence, target)
    if status != EXACT:
        return None, status
    match = re.search(rf"(?<!\w){re.escape(target)}(?!\w)", sentence)
    if match is None:
        raise ValueError("Audited unique exact target could not be retrieved")
    return match.span(), status


def _overlap(groups):
    return {
        f"{left}_vs_{right}": len(groups[left] & groups[right])
        for left, right in combinations(SPLITS, 2)
    }


def _rows(root, split):
    for kind, records in root[split].items():
        for key in sorted(records):
            yield kind, records[key]


def _jsonl(records):
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records).encode("utf-8")


def prepare(source, output):
    inventories = _read_release(source)
    raw_groups = {split: set() for split in SPLITS}
    exact_groups = {split: set() for split in SPLITS}
    retained_groups = {split: set() for split in SPLITS}
    retained_types = {split: set() for split in SPLITS}
    chosen_split = {}
    stimuli = {}
    observations = []
    counts = {}
    examples = {}
    for split in PRIORITY:
        for kind, row in _rows(inventories, split):
            group = _identifier(canonical_context(row.sentence))
            raw_groups[split].add(group)
            bucket = counts.setdefault((split, kind, row.corpus), Counter())
            bucket["source"] += 1
            span, status = unique_exact_span(row.sentence, row.token)
            if span is None:
                bucket[status] += 1
                continue
            bucket["unique_exact"] += 1
            exact_groups[split].add(group)
            assigned_split = chosen_split.setdefault(group, split)
            if assigned_split != split:
                bucket[f"purged_context_kept_in_{assigned_split}"] += 1
                continue

            start, end = span
            stimulus_id = f"complex-sentence-{_identifier(row.sentence)}"
            stimuli.setdefault(stimulus_id, {
                "id": stimulus_id,
                "context_group": group,
                "language": "en",
                "text": row.sentence,
                "split": split,
            })
            id_ = f"complex-{kind}-{split}-{row.id}"
            label_directory = "test-labels" if split == "test" else split
            observations.append({
                "id": id_,
                "source_id": row.id,
                "source_file": f"{label_directory}/lcp_{kind}_{split}.tsv",
                "stimulus_id": stimulus_id,
                "kind": kind,
                "corpus": row.corpus,
                "split": split,
                "exposure": {
                    "mode": "full_sentence",
                    "visible_characters": len(row.sentence),
                    "target_designated": True,
                },
                "target_span": {"start": start, "end": end, "text": row.token},
                "observations": {"lexical_difficulty": {"status": "observed", "value": row.complexity}},
            })
            bucket["retained"] += 1
            retained_groups[split].add(group)
            retained_types[split].add(canonical_context(row.token))
            examples.setdefault((split, kind), {"id": id_, "stimulus_id": stimulus_id})

    overlaps = _overlap(retained_groups)
    if any(overlaps.values()):
        raise ValueError("A canonical sentence group crosses retained partitions")
    for row in observations:
        stimulus = stimuli[row["stimulus_id"]]
        span = row["target_span"]
        if stimulus["split"] != row["split"] or stimulus["text"][span["start"]:span["end"]] != span["text"]:
            raise ValueError(f"Stimulus and target span disagree for {row['id']}")

    totals = Counter()
    split_totals = {}
    breakdown = []
    for (split, kind, corpus), bucket in sorted(counts.items()):
        totals.update(bucket)
        split_totals.setdefault(split, Counter()).update(bucket)
        breakdown.append({"split": split, "kind": kind, "corpus": corpus, **bucket})
    payloads = {
        "stimuli.jsonl": _jsonl(stimuli[key] for key in sorted(stimuli)),
        "observations.jsonl": _jsonl(sorted(observations, key=lambda row: row["id"])),
    }
    report = {
        "source": SOURCE,
        "release": RELEASE,
        "source_sha256": LOCKS,
        "scope": "Retrospective lexical difficulty of the designated target when the whole sentence is visible.",
        "unobserved_positions": "Undesignated targets and discarded ambiguous positions have no assessed label.",
        "span_coordinates": "Zero-based Unicode code points in unmodified sentence text, with exclusive end.",
        "context_grouping": "NFKC, then Unicode casefold, then collapse whitespace. Hash the resulting context.",
        "split_priority_for_unique_exact_spans": list(PRIORITY),
        "counts": dict(totals),
        "counts_by_split": {split: dict(split_totals[split]) for split in SPLITS},
        "counts_by_split_kind_corpus": breakdown,
        "retained_stimuli": len(stimuli),
        "retained_context_groups": {split: len(retained_groups[split]) for split in SPLITS},
        "raw_canonical_context_overlap": _overlap(raw_groups),
        "unique_exact_canonical_context_overlap_before_purge": _overlap(exact_groups),
        "retained_canonical_context_overlap": overlaps,
        "retained_canonical_target_type_overlap": _overlap(retained_types),
        "example_ids": {f"{split}_{kind}": examples[split, kind] for split in SPLITS for kind in COUNTS},
        "output_sha256": {name: _sha256(blob) for name, blob in payloads.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    for filename, blob in payloads.items():
        (output / filename).write_bytes(blob)
    (output / "preparation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
