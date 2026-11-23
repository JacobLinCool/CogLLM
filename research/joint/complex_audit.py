"""Audit target-word offsets and official partitions in the fixed CompLex release."""

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re


RELEASE = "4b31c2bd34da4c1265f9db3d727d95ee23099f14"
SOURCE = f"https://github.com/MMU-TDMLab/CompLex/tree/{RELEASE}"
LOCKS = {
    "train/lcp_single_train.tsv": "12142b2653edfbedf27452e9b0e9d14d2808a90950319a013b91245d567379e0",
    "train/lcp_multi_train.tsv": "8bd1e51343ac36e975deed6ee45926d0fd91a3fe2dd0f94cc37272b69441b2df",
    "trial/lcp_single_trial.tsv": "67bb6ef05cb1a690eb78a948592ef8693fbf8f5db451762522ecb02e8e47d9ab",
    "trial/lcp_multi_trial.tsv": "7ebaa82428381458c7e0abcd4c712099ccd435b1c4393c623e7b02bc308810a8",
    "test/lcp_single_test.tsv": "8eaa0a3985deaca4d15ae40b71c38052abc0a8cba9327cea3bed6c33e7a26bb1",
    "test/lcp_multi_test.tsv": "ab9170be99f67b3a3b7a8028893ffb14fb0b9d72b34c5cf1255916437ab9e866",
    "test-labels/lcp_single_test.tsv": "846ef133c72cfa0547a285dc9060e74e87a8e9ad3a0de149e2cf680c12f611ed",
    "test-labels/lcp_multi_test.tsv": "59669b0a546f2bd4cb7b23271e4c8d555e24c862948aae76aaa7b957db736372",
}
COUNTS = {"single": (7662, 421, 917), "multi": (1517, 99, 184)}
SPLITS = ("train", "trial", "test")


@dataclass(frozen=True)
class Observation:
    id: str
    corpus: str
    sentence: str
    token: str
    complexity: float | None


def _read(path, labeled, expected):
    corpus = "subcorpus" if path.parent.name == "trial" else "corpus"
    columns = ["id", corpus, "sentence", "token"]
    if labeled:
        columns.append("complexity")
    records = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames != columns:
            raise ValueError(f"Unexpected columns in {path}")
        for row in reader:
            if None in row or not all(row.values()) or row["id"] in records:
                raise ValueError(f"Malformed or duplicate TSV row at {path}:{reader.line_num}")
            score = float(row["complexity"]) if labeled else None
            if labeled and (not math.isfinite(score) or not 0 <= score <= 1):
                raise ValueError(f"Difficulty must be in [0, 1] at {path}:{reader.line_num}")
            records[row["id"]] = Observation(row["id"], row[corpus], row["sentence"], row["token"], score)
    if len(records) != expected:
        raise ValueError(f"Expected {expected} items in {path}, observed {len(records)}")
    return records


def _locations(sentence, token):
    escaped = re.escape(token)
    raw_matches = [match.span(1) for match in re.finditer(f"(?=({escaped}))", sentence)]
    isolated = [match.span(1) for match in re.finditer(f"(?<!\\w)(?=({escaped})(?!\\w))", sentence, re.IGNORECASE)]
    if not isolated:
        result = "no_isolated_match"
    elif len(isolated) > 1:
        result = "ambiguous_repeated_isolated_match_ignoring_case"
    else:
        start, end = isolated[0]
        result = "unique_exact_isolated_match" if sentence[start:end] == token else "unique_case_insensitive_only"
    return len(raw_matches), result


def _summarize(records):
    literal = Counter()
    location = Counter()
    scores = []
    for row in records.values():
        matches, result = _locations(row.sentence, row.token)
        literal["none" if matches == 0 else "one" if matches == 1 else "multiple"] += 1
        location[result] += 1
        scores.append(row.complexity)
    return {
        "items": len(records),
        "contexts": len({row.sentence for row in records.values()}),
        "items_by_corpus": dict(Counter(row.corpus for row in records.values())),
        "raw_case_sensitive_substring_matches": dict(literal),
        "isolated_location": dict(location),
        "observed_difficulty_range": [min(scores), max(scores)],
    }


def _overlap(partitions):
    return {
        f"{left}_vs_{right}": {
            "literal_context_overlap": len({row.sentence for row in a} & {row.sentence for row in b}),
            "literal_target_type_overlap": len({row.token for row in a} & {row.token for row in b}),
            "casefolded_target_type_overlap": len({row.token.casefold() for row in a} & {row.token.casefold() for row in b}),
        }
        for (left, a), (right, b) in combinations(partitions.items(), 2)
    }


def audit(source):
    for relative, expected in LOCKS.items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source SHA-256 differs from the fixed release: {relative}")
    inventories = {}
    tasks = {}
    for kind, counts in COUNTS.items():
        splits = {
            split: _read(source / f"{split}/lcp_{kind}_{split}.tsv", True, count)
            for split, count in zip(SPLITS[:2], counts[:2], strict=True)
        }
        test = _read(source / f"test/lcp_{kind}_test.tsv", False, counts[2])
        labels = _read(source / f"test-labels/lcp_{kind}_test.tsv", True, counts[2])
        if test != {key: replace(value, complexity=None) for key, value in labels.items()}:
            raise ValueError(f"The public test prompts and released labels do not match for {kind}")
        splits["test"] = labels
        inventories[kind] = splits
        tasks[kind] = {
            "splits": {split: _summarize(rows) for split, rows in splits.items()},
            "test_items_joined_by_id_and_exact_fields": len(test),
            "official_split_overlap": _overlap({split: rows.values() for split, rows in splits.items()}),
        }
    pooled = {
        split: [row for screens in inventories.values() for row in screens[split].values()]
        for split in SPLITS
    }
    all_locations = Counter()
    for task in tasks.values():
        for result in task["splits"].values():
            all_locations.update(result["isolated_location"])
    return {
        "source": SOURCE,
        "release": RELEASE,
        "files_sha256": LOCKS,
        "label_method": {
            "reference": "https://aclanthology.org/2021.semeval-1.1.pdf#page=3",
            "unit": "Human lexical difficulty judgment for a designated English word or multiword expression given its enclosing sentence.",
            "rating_values_before_aggregation": [0, 0.25, 0.5, 0.75, 1],
            "released_label": "Arithmetic mean of retained ratings on the five-point scale after the stated mapping.",
            "missing_observation": "Unrated tokens and phrases have no corresponding lexical-difficulty observation.",
            "position_scope": "The whole sentence was visible and a target was supplied. No reading-time or spontaneous confusion-onset label is present.",
        },
        "offset_audit_method": {
            "unit": "Zero-based Unicode code-point offsets with an exclusive end in the source sentence.",
            "exact": "Exactly one isolated case-insensitive candidate exists, and its surface equals the released target.",
            "ambiguous": "More than one isolated case-insensitive candidate exists. Do not assign its rating to a specific occurrence.",
            "boundary": "No Unicode word character just before or after the candidate. Search allows overlapping matches.",
            "annotation_interface": f"https://github.com/MMU-TDMLab/CompLex/blob/{RELEASE}/Annotation/mturk.html",
        },
        "licensing": {
            "dataset": {"stated": "CC BY 4.0", "source": f"https://github.com/MMU-TDMLab/CompLex/blob/{RELEASE}/Readme.md"},
            "bible": {"stated": "World English Bible text in the public domain", "source": "https://www.ebible.org/find/details.php?id=eng-web"},
            "europarl": {"stated": "Corpus provider reports unaware of copyright restrictions", "source": "https://www.statmt.org/europarl/"},
            "biomedical": {"stated": "CRAFT v2 and earlier annotations use CC BY 3.0, articles are PMC Open Access and row-level article provenance was not audited", "source": "https://bionlp-corpora.sourceforge.net/CRAFT/"},
        },
        "labeled_items": sum(map(len, pooled.values())),
        "isolated_location_totals": dict(all_locations),
        "tasks": tasks,
        "pooled_official_split_overlap": _overlap(pooled),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
