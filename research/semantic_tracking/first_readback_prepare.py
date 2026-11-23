"""Prepare a matched first-backward-keypress endpoint for two BSPR studies."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata

import rdata

from research.core_features.readback_prepare import (
    SOURCES,
    TEST_ITEMS,
    align_regions,
    digest,
    first_visit_outcomes,
    movements,
    opaque_id,
    write_jsonl,
)


ENGLISH_LOCK = {
    "audit.json": "017698e9d91b46cd03e563e7ee685067d37cd287b9cae4ed24d822f6fb1729f6",
    "stimuli.jsonl": "0f7bc8575532af166b7ff53dfc03205a1757ad2a487283daf710c5bc581d1f31",
    "outcomes.jsonl": "22f6286f44b6dc38f1beb0146fae5ba96c18964a18a0f749fa34f9cece100af6",
    "trials.jsonl": "5ed0d60c90d776d81b73a3b1c4a7603784f7bf14ab95dbac460c5cd6a33f021f",
}
GERMAN_LOCK = {
    "audit.json": "28c245b4199ed36ca2074a0429c9ed68437f1b34e85ba782dd198d749077c7a2",
    "stimuli.jsonl": "69a2ccaccb1977e1088abe22db26d00cd488ad9867baafc54d24b77d308cb231",
    "outcomes.jsonl": "57436ea3a1944c99a74f0eb82514fd93f74cd589400c27fce6f3f1d5ce7fe397",
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def unique_index(rows, field):
    index = {row[field]: row for row in rows}
    if len(index) != len(rows):
        raise ValueError(f"Expected one row per {field}")
    return index


def load_study(root, lock):
    for name, expected in lock.items():
        if digest(root / name) != expected:
            raise ValueError(f"The prepared source snapshot has changed: {root / name}")
    stimuli = unique_index(read_jsonl(root / "stimuli.jsonl"), "id")
    outcomes = read_jsonl(root / "outcomes.jsonl")
    audit = json.loads((root / "audit.json").read_text(encoding="utf-8"))
    return stimuli, outcomes, audit


def english_observations(events_path, stimuli, outcomes, trials, source_audit):
    event_sha256 = SOURCES["events"]["sha256"]
    if digest(events_path) != event_sha256:
        raise ValueError("The English events do not match the author's pinned OSF archive")
    published = defaultdict(list)
    for row in outcomes:
        published[row["trial_id"]].append(row)
    if not published.keys() <= trials.keys():
        raise ValueError("The English outcomes contain a trial missing from the trial manifest")
    raw = rdata.read_rda(events_path)["exp1_data"]
    discourse = raw.loc[raw.type == "strange"].sort_values(["trial", "fixno"])
    if discourse[["trial", "fixno"]].duplicated().any():
        raise ValueError("English discourse events must have unique within-trial positions in the log")
    if len(discourse) != source_audit["discourse_source_event_count"]:
        raise ValueError("The English discourse event count disagrees with the source receipt")
    seen_trials = set()
    used = Counter()
    excluded = 0
    derived = []
    for raw_trial, trial in discourse.groupby("trial", sort=True):
        trial_id = opaque_id("trial", str(raw_trial))
        if trial_id not in trials:
            excluded += 1
            continue
        seen_trials.add(trial_id)
        metadata = trials[trial_id]
        if any(trial[column].nunique() != 1 for column in ("itemno", "condition", "subj", "answer")):
            raise ValueError("Expected one English item, condition, reader and judgment per trial")
        first = trial.iloc[0]
        item = int(first.itemno)
        expected_metadata = {
            "group_id": f"bspr-item-{item}",
            "split": "test" if item in TEST_ITEMS else "train",
            "condition": str(first.condition),
            "reader_id": opaque_id("reader", str(first.subj)),
        }
        if any(metadata[field] != value for field, value in expected_metadata.items()):
            raise ValueError("The raw English trial disagrees with its published metadata")
        stimulus = stimuli[metadata["id"]]
        stops = stimulus["stops"]
        if align_regions(trial, len(stops)) != {"text": stimulus["text"], "stops": stops}:
            raise ValueError("The raw English trial disagrees with its published display")
        positions = trial.pos.astype(int).tolist()
        actions = list(movements(positions, trial.dir.tolist()))
        used.update(actions)
        prior = published[trial_id]
        recomputed = list(first_visit_outcomes(positions, actions, stops, Counter()))
        if len(prior) != len(recomputed):
            raise ValueError("The English first-visit counts disagree with the earlier preparation")
        for old, new in zip(prior, recomputed, strict=True):
            if any(old[field] != value for field, value in {**expected_metadata, **new}.items()):
                raise ValueError("An English first-visit row disagrees with its raw event replay")
            if old["id"] != stimulus["id"]:
                raise ValueError("An English first-visit row uses the wrong stimulus")

        seen_regions = set()
        for position, action in zip(positions, actions, strict=True):
            if position not in seen_regions:
                seen_regions.add(position)
                if position >= 2 and len(seen_regions.intersection(range(1, position))) == position - 1:
                    if action in {"backward", "next_region"}:
                        derived.append({
                            **metadata, "region": position, "stop": stops[position - 1],
                            "readback": action == "backward",
                        })
            if action == "backward":
                break
    if (
        seen_trials != trials.keys()
        or excluded != source_audit["exclusions"]["trial_without_exact_region_text"]
        or used != source_audit["movement_counts_in_aligned_trials"]
    ):
        raise ValueError("The English included trials or movement totals disagree with the source receipt")
    return derived, {
        "raw_events_sha256": event_sha256,
        "raw_events_url": SOURCES["events"]["url"],
        "aligned_trials": len(seen_trials),
        "excluded_incomplete_displays": excluded,
        "verified_first_visit_rows": len(outcomes),
    }


def material_design(stimuli, observations, study):
    variants = {}
    families = defaultdict(dict)
    for row in observations:
        identifier, family, split, condition = (row[key] for key in ("id", "group_id", "split", "condition"))
        if split not in {"train", "test"}:
            raise ValueError("Unexpected material split")
        if variants.setdefault(identifier, (family, split)) != (family, split):
            raise ValueError("A stimulus belongs to more than one study or split")
        if families[family].setdefault(condition, (identifier, split)) != (identifier, split):
            raise ValueError("A family condition contains more than one stimulus or split")
    for variants_by_condition in families.values():
        if len(variants_by_condition) != 2 or len({split for _, split in variants_by_condition.values()}) != 1:
            raise ValueError("A paired material family must have two conditions and one locked split")
    if variants.keys() != stimuli.keys():
        raise ValueError("The displays and observed material metadata disagree")
    targets = {}
    for identifier, (family, split) in variants.items():
        stimulus = stimuli[identifier]
        stops, text = stimulus["stops"], stimulus["text"]
        if set(stimulus) != {"id", "text", "stops"} or sorted(set(stops)) != stops or stops[-1] != len(text):
            raise ValueError("Invalid raw display stops")
        for previous_stop, stop in zip(stops, stops[1:]):
            region = text[previous_stop + 1:stop]
            if text[previous_stop] != " " or not region or region != region.strip():
                raise ValueError("A display region does not match its recorded text boundary")
            targets[identifier, stop] = {
                "id": identifier, "stop": stop, "study": study, "group_id": family,
                "split": split, "region_words": len(region.split()),
            }
    return targets


def aggregate(targets, observations):
    counts = defaultdict(Counter)
    for row in observations:
        key = row["id"], row["stop"]
        target = targets[key]
        if row["group_id"] != target["group_id"] or row["split"] != target["split"]:
            raise ValueError("An observation disagrees with its assigned material split")
        if "region_words" in row and row["region_words"] != target["region_words"]:
            raise ValueError("A source region word count disagrees with the display")
        if type(row["readback"]) is not bool:
            raise ValueError("Expected a boolean indicator of an observed backward keypress")
        counts[key][row["readback"]] += 1
    result = {"train": [], "test": []}
    for key in sorted(counts):
        labels = {field: value for field, value in targets[key].items() if field != "split"}
        partition = targets[key]["split"]
        labels.update(negative=counts[key][False], positive=counts[key][True])
        result[partition].append(labels)
    return result


def overlap(english_stimuli, german_stimuli):
    def terms(stimuli):
        documents = set()
        for row in stimuli.values():
            words = re.findall(r"\w+", unicodedata.normalize("NFKC", row["text"]).casefold())
            documents.add(tuple(words))
        shingles = {
            document[start:start + 5]
            for document in documents for start in range(len(document) - 4)
        }
        return documents, shingles

    english_docs, english_shingles = terms(english_stimuli)
    german_docs, german_shingles = terms(german_stimuli)
    return {
        "normalized_document_matches": len(english_docs & german_docs),
        "five_word_shingle_matches": len(english_shingles & german_shingles),
        "reader_overlap": "Study-local pseudonyms do not identify readers across studies.",
    }


def partition_summary(results, targets, study, split):
    rows = [row for row in results[split] if row["study"] == study]
    design = [row for row in targets.values() if row["study"] == study and row["split"] == split]
    single = [row for row in rows if row["region_words"] == 1]
    return {
        "families": len({row["group_id"] for row in design}),
        "stimuli": len({row["id"] for row in design}),
        "designed_stops": len(design),
        "observed_stops": len(rows),
        "negative": sum(row["negative"] for row in rows),
        "positive": sum(row["positive"] for row in rows),
        "single_word_observed_stops": len(single),
        "single_word_negative": sum(row["negative"] for row in single),
        "single_word_positive": sum(row["positive"] for row in single),
    }


def prepare(english_dir, english_events, german_dir, output):
    english, english_prior, english_audit = load_study(english_dir, ENGLISH_LOCK)
    german, german_prior, german_audit = load_study(german_dir, GERMAN_LOCK)
    english_trials = unique_index(read_jsonl(english_dir / "trials.jsonl"), "trial_id")
    english_observed, english_replay = english_observations(
        english_events, english, english_prior, english_trials, english_audit,
    )
    german_observed = [row for row in german_prior if row["initial_risk"] is True]
    if english.keys() & german.keys():
        raise ValueError("The two studies share a stimulus ID")
    english_targets = material_design(english, english_trials.values(), "english")
    german_targets = material_design(german, german_prior, "german")
    if {row["group_id"] for row in english_targets.values()} & {
        row["group_id"] for row in german_targets.values()
    }:
        raise ValueError("The two studies share a material family ID")
    targets = {**english_targets, **german_targets}
    counts = aggregate(targets, english_observed + german_observed)
    audit = {
        "endpoint": {
            "risk_set": "First visit to a recorded region after all earlier regions were displayed, from region "
            "two until and including the trial's first explicit backward keypress",
            "positive": "Next event is a one-region backward action or restart at region one",
            "negative": "Next event is a one-region advance",
            "excluded": "Terminal actions, forward skips and opportunities following an earlier backward action",
            "unit": "One trial opportunity at a displayed region; a region may span multiple words",
        },
        "english": {
            "article": english_audit["article"], "project": english_audit["project"],
            "license": english_audit["license"], "heldout_items": english_audit["material_split"]["test_items"],
            "prepared_source_sha256": ENGLISH_LOCK, "raw_replay": english_replay,
        },
        "german": {
            "article": german_audit["article"], "project": german_audit["data_project"],
            "license": german_audit["license"], "heldout_items": german_audit["split"]["heldout"],
            "prepared_source_sha256": GERMAN_LOCK, "events": german_audit["sources"]["events"],
        },
        "overlap": overlap(english, german),
        "counts": {
            study: {split: partition_summary(counts, targets, study, split) for split in ("train", "test")}
            for study in ("english", "german")
        },
    }
    tables = {
        "stimuli.jsonl": sorted((*english.values(), *german.values()), key=lambda row: row["id"]),
        "targets.jsonl": [targets[key] for key in sorted(targets)],
        "train.jsonl": counts["train"],
        "test.jsonl": counts["test"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent) as directory:
        staging = Path(directory) / "assay"
        staging.mkdir()
        for name, rows in tables.items():
            write_jsonl(staging / name, rows)
        audit["file_sha256"] = {name: digest(staging / name) for name in tables}
        (staging / "assay.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
        )
        staging.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--english", type=Path, required=True)
    parser.add_argument("--english-events", type=Path, required=True)
    parser.add_argument("--german", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.english, args.english_events, args.german, args.output)
