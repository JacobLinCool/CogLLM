"""Align German BSPR Experiment 2 movement to the actual display coordinates."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import rdata

from research.core_features.readback_prepare import digest, movements, write_jsonl


EVENTS = "https://osf.io/download/r5zwc/"
ANALYSIS = "https://osf.io/download/j5q6v/"
EVENTS_SHA256 = "16231d0c3eef7be7a07b9bdbe4bbeebb0d0d61b7f70bf30540357fa3bfbc6dcb"
PROJECT = "https://osf.io/j8cbh/"
ARTICLE = "https://doi.org/10.5070/G601139"
STOP_AND_REREAD_REGISTRATION = "https://osf.io/p56j8/"
MATERIAL_SEED = 43
FAMILY_HOLDOUT = 8


def item_order(item):
    return hashlib.sha256(f"{MATERIAL_SEED}:{item}".encode()).digest()


def displayed_regions(rows):
    by_position = {}
    for row in rows.itertuples(index=False):
        position = int(row.no)
        text = row.text
        if not isinstance(text, str) or not text or text.strip() != text or by_position.setdefault(position, text) != text:
            raise ValueError("A display position does not have one exact recorded wording")
    return by_position


def make_stimuli(events):
    endings = events.groupby("trial", sort=False).tail(1)
    natural = endings[endings.dir == 1]
    stimuli = {}
    for variant, rows in events.groupby(["item", "cond"], sort=True):
        material, condition = variant
        terminal = natural[(natural.item == material) & (natural.cond == condition)].no.unique()
        regions = displayed_regions(rows)
        if len(terminal) != 1 or set(regions) != set(range(1, int(terminal[0]) + 1)):
            raise ValueError(f"No unique complete display at {material}:{condition}")
        ordered = [regions[index] for index in range(1, len(regions) + 1)]
        text = " ".join(ordered)
        digest_id = hashlib.sha256(json.dumps(ordered, ensure_ascii=False).encode()).hexdigest()[:20]
        stops = [0]
        for region in ordered:
            stops.append(stops[-1] + len(region) + 1)
        stimuli[variant] = {
            "id": f"garden-path-exp2-{digest_id}", "text": text,
            "stops": [stop - 1 for stop in stops[1:]],
        }
    if len(stimuli) != 128 or len({row["id"] for row in stimuli.values()}) != 128:
        raise ValueError("Expected both distinct conditions for each of 64 materials")
    return stimuli


def observed_movements(positions, actions, display, critical, noun, audit):
    seen = set()
    no_preceding_readback = True
    for index, (position, action) in enumerate(zip(positions, actions, strict=True)):
        first_visit = position not in seen
        seen.add(position)
        initial_risk = no_preceding_readback
        if action == "backward":
            no_preceding_readback = False
        if not first_visit or position < 2:
            continue
        if not all(earlier in seen for earlier in range(1, position)):
            raise ValueError("A visited original prefix skips earlier display regions")
        if action not in {"backward", "next_region"}:
            audit[f"first_visit_{action}"] += 1
            continue
        readback = action == "backward"
        stops = display["stops"]
        start = stops[position - 2] + 1
        end = stops[position - 1]
        region_word_count = len(display["text"][start:end].split())
        if readback:
            last = index + 1
            while actions[last] == "backward":
                last += 1
            first_destination = stops[positions[index + 1] - 1]
            source = stops[positions[last] - 1]
            resumed = any(later > position for later in positions[last + 1:])
        else:
            first_destination = source = resumed = None
        yield {
            "stop": end, "region": position, "region_words": region_word_count,
            "critical": position == critical, "critical_offset": position - critical,
            "noun_region": noun, "readback": readback, "initial_risk": initial_risk,
            "first_destination_stop": first_destination, "source_stop": source,
            "resumed_beyond_stop": resumed,
        }


def prepare(path, output):
    if digest(path) != EVENTS_SHA256:
        raise ValueError("The raw file does not match the pinned OSF Experiment 2 events")
    raw = rdata.read_rda(path)["exp2_data"]
    events = raw[raw.gptype.isin([1, 2])].sort_values(["trial", "fixno"])
    if events[["trial", "fixno"]].duplicated().any() or set(events.cond.unique()) != {-1, 1}:
        raise ValueError("Non-unique event keys or unexpected experimental condition")
    stimuli = make_stimuli(events)
    family_by_type = events[["item", "gptype"]].drop_duplicates()
    if family_by_type.item.duplicated().any() or set(family_by_type.groupby("gptype").size()) != {31, 33}:
        raise ValueError("Each item must belong to a single verified construction type")
    heldout = {
        item for gptype in (1, 2)
        for item in sorted(family_by_type[family_by_type.gptype == gptype].item, key=item_order)[:FAMILY_HOLDOUT]
    }
    audit = Counter()
    outcomes = []
    observed_keypresses = Counter()
    for _, trial in events.groupby("trial", sort=True):
        if any(trial[column].nunique() != 1 for column in (
            "item", "cond", "subj", "gptype", "crit", "np1", "correct",
        )):
            raise ValueError("Inconsistent trial metadata")
        first = trial.iloc[0]
        material = str(first["item"])
        variant = stimuli[material, first.cond]
        positions = trial.no.astype(int).tolist()
        directions = trial.dir.astype(int).astype(str).tolist()
        actions = tuple(movements(positions, directions))
        observed_keypresses.update(actions)
        if actions[-1] != "terminal" or set(positions) != set(range(1, max(positions) + 1)):
            raise ValueError("Unexpected gap in experimentally displayed region indices")
        gptype = int(first.gptype)
        noun = int(first.und + 1 if gptype == 1 else first.np1)
        critical = int(first.crit)
        if not 1 < noun < critical <= len(variant["stops"]):
            raise ValueError("Unexpected noun and critical stoppers")
        metadata = {
            "id": variant["id"], "group_id": f"garden-path-exp2-{material.replace(' ', '-')}",
            "split": "test" if material in heldout else "train",
            "reader_id": f"reader-{hashlib.sha256(str(first.subj).encode()).hexdigest()[:16]}",
            "trial_id": f"trial-{int(first.trial)}",
            "condition": "late" if first.cond == 1 else "early",
            "construction": "coordination" if gptype == 1 else "subject_object",
            "question_correct": bool(first.correct),
        }
        outcomes.extend(
            {**metadata, **row}
            for row in observed_movements(positions, actions, variant, critical, noun, audit)
        )
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "stimuli.jsonl", sorted(stimuli.values(), key=lambda row: row["id"]))
    write_jsonl(output / "outcomes.jsonl", outcomes)
    summary = {
        "article": ARTICLE, "data_project": PROJECT, "analysis": ANALYSIS,
        "separate_stop_and_reread_registration": STOP_AND_REREAD_REGISTRATION,
        "license": "CC BY 4.0 per OSF",
        "sources": {"events": {"url": EVENTS, "sha256": EVENTS_SHA256}},
        "stimuli": len(stimuli), "readers": int(events.subj.nunique()),
        "trial_count": int(events.trial.nunique()), "event_log_rows": len(events),
        "key_actions": dict(observed_keypresses), "skipped": dict(audit),
        "families": {
            name: int((family_by_type.gptype == group).sum())
            for group, name in ((1, "coordination"), (2, "subject_object"))
        },
        "split": {
            "method": "First 8 families by SHA-256 in each construction, both versions held together.",
            "seed": MATERIAL_SEED, "heldout": sorted(heldout),
            "train_rows": sum(row["split"] == "train" for row in outcomes),
            "test_rows": sum(row["split"] == "test" for row in outcomes),
            "test_positive": sum(row["split"] == "test" and row["readback"] for row in outcomes),
            "test_single_word_rows": sum(row["split"] == "test" and row["region_words"] == 1 for row in outcomes),
            "test_critical_single_word_rows": sum(
                row["split"] == "test" and row["critical"] and row["region_words"] == 1 for row in outcomes
            ),
        },
        "observations": {
            "first_visit": "Explicit one-region forward or backward/restart action, excluding terminal and jump-to-question.",
            "destination": "Earliest region reached in the contiguous sequence of backward presses, conditional on readback.",
            "initial_risk": "The trial has no preceding backward movement at this anchor.",
            "words": "A single-word region has exactly one whitespace-delimited orthographic word.",
            "interpretation": "Action endpoints. Comprehension was rated only after the full trial.",
        },
        "rdata_version": rdata.__version__,
        "output_sha256": {name: digest(output / name) for name in ("stimuli.jsonl", "outcomes.jsonl")},
    }
    (output / "audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.events, args.output)
