"""Align BSPR Experiment 1 discourse keypresses to the displayed region ends."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote

import rdata


PROJECT = "https://osf.io/54bda/"
SOURCES = {
    "events": {
        "url": "https://osf.io/download/zrtgf/",
        "sha256": "59865b8468cad2bae33a29b5605d02cd21dadd1d136849942391cfddc447fb7a",
    },
    "items": {
        "url": "https://osf.io/download/fk8a4/",
        "sha256": "85c1dac31090a01925d62fbb7fdb705a9baf7db8ca6ea875dd5fc57998d630c2",
    },
}
EXPECTED_ITEMS = set(range(49, 69))
TEST_SEED = 43
TEST_ITEMS = set(sorted(EXPECTED_ITEMS, key=lambda item: hashlib.sha256(f"{TEST_SEED}:{item}".encode()).digest())[:5])


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_region_counts(path):
    decoder = json.JSONDecoder()
    counts = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith('["strange-'):
            continue
        (condition, item), split = decoder.raw_decode(line)
        if line[split] != "," or (condition, item) in counts:
            raise ValueError(f"Invalid stimulus script record for {condition} {item}")
        counts[condition, item] = len(json.loads(line[split + 1:]).split(" - "))
    if set(counts) != {(condition, item) for condition in ("strange-a", "strange-b") for item in EXPECTED_ITEMS}:
        raise ValueError("Expected 20 discourse material families with both experimental conditions")
    return counts


def align_regions(trial, region_count):
    wording_by_position = {}
    for row in trial.itertuples(index=False):
        position = int(row.pos)
        text = unquote(row.text, errors="strict")
        if text != text.strip() or not text:
            raise ValueError(f"Invalid displayed region boundaries for position {position}")
        if wording_by_position.setdefault(position, text) != text:
            return None
    if set(wording_by_position) != set(range(1, region_count + 1)):
        return None
    regions = [wording_by_position[position] for position in range(1, region_count + 1)]
    text = " ".join(regions)
    stops = []
    offset = 0
    for region in regions:
        offset += len(region)
        stops.append(offset)
        offset += 1
    if stops[-1] != len(text):
        raise AssertionError("Region offsets disagree with the reconstructed text")
    return {"text": text, "stops": stops}


def movements(positions, directions):
    if positions[0] != 1 or directions[-1] not in ("1", "2"):
        raise ValueError("Expected position one at first display and a terminal advance or exit")
    for index, (current, following) in enumerate(zip(positions, positions[1:])):
        direction = directions[index]
        step = following - current
        if direction == "-1" and step == -1:
            yield "backward"
        elif direction == "3" and following == 1 and step < 0:
            yield "backward"
        elif direction == "1" and step == 1:
            yield "next_region"
        elif direction == "1" and step > 1:
            yield "unmatched_forward"
        else:
            raise ValueError(f"Unexpected outgoing direction {direction} for {current} -> {following}")
    yield "terminal"


def first_visit_outcomes(positions, actions, stops, audit):
    visited = set()
    for index, (position, action) in enumerate(zip(positions, actions, strict=True)):
        if position in visited:
            continue
        visited.add(position)
        if position < 2 or any(earlier not in visited for earlier in range(1, position)):
            audit["no_observed_earlier_prefix"] += 1
            continue
        if action not in ("backward", "next_region"):
            audit[f"first_visit_{action}"] += 1
            continue
        readback = action == "backward"
        source_stop = None
        first_destination_stop = None
        resumed_beyond_stop = None
        if readback:
            first_destination_stop = stops[positions[index + 1] - 1]
            end = index + 1
            while actions[end] == "backward":
                end += 1
            source_stop = stops[positions[end] - 1]
            resumed_beyond_stop = any(later > position for later in positions[end + 1:])
        yield {
            "stop": stops[position - 1],
            "region": position,
            "readback": readback,
            "source_stop": source_stop,
            "first_destination_stop": first_destination_stop,
            "resumed_beyond_stop": resumed_beyond_stop,
        }


def opaque_id(namespace, value):
    return f"bspr-{namespace}-{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def prepare(events_path, items_path, output):
    for key, path in (("events", events_path), ("items", items_path)):
        if digest(path) != SOURCES[key]["sha256"]:
            raise ValueError(f"The publicly released {key} snapshot does not match the pinned SHA-256")
    data = rdata.read_rda(events_path)["exp1_data"]
    discourse = data.loc[data.type == "strange"].sort_values(["trial", "fixno"])
    if discourse[["trial", "fixno"]].duplicated().any():
        raise ValueError("BSPR Experiment 1 must have unique event indices within each discourse trial")
    region_counts = read_region_counts(items_path)
    stimuli = {}
    outcomes = []
    trials = []
    material_counts = Counter()
    movements_count = Counter()
    audit = Counter()
    for raw_trial, trial in discourse.groupby("trial", sort=True):
        if any(trial[column].nunique() != 1 for column in ("itemno", "condition", "subj", "answer")):
            raise ValueError("A trial must have one stimulus and one acceptability judgment")
        first = trial.iloc[0]
        item = int(first.itemno)
        group_id = f"bspr-item-{item}"
        condition = str(first.condition)
        stimulus = align_regions(trial, region_counts[condition, item])
        if stimulus is None:
            audit["trial_without_exact_region_text"] += 1
            continue
        stimulus_id = opaque_id("text", json.dumps(stimulus, ensure_ascii=False, sort_keys=True))
        stimulus_record = {"id": stimulus_id, **stimulus}
        if stimuli.setdefault(stimulus_id, stimulus_record) != stimulus_record:
            raise ValueError("Duplicate stimulus ID has inconsistent contents")
        metadata = {
            "id": stimulus_id,
            "group_id": group_id,
            "split": "test" if item in TEST_ITEMS else "train",
            "condition": condition,
            "trial_id": opaque_id("trial", str(raw_trial)),
            "reader_id": opaque_id("reader", str(first.subj)),
        }
        positions = trial.pos.astype(int).tolist()
        actions = list(movements(positions, trial.dir.tolist()))
        movements_count.update(actions)
        observed = list(first_visit_outcomes(positions, actions, stimulus["stops"], audit))
        outcomes.extend({**metadata, **row} for row in observed)
        judgment = {"Yes": True, "No": False, "Don't know": None}[first.answer]
        trials.append({**metadata, "acceptable": judgment})
        material_counts[group_id] += len(observed)
    if set(material_counts) != {f"bspr-item-{item}" for item in EXPECTED_ITEMS}:
        raise ValueError("Not every discourse item family contributes observed first-visit outcomes")
    split_stimuli = {split: {row["id"] for row in trials if row["split"] == split} for split in ("train", "test")}
    if split_stimuli["train"] & split_stimuli["test"]:
        raise ValueError("The same displayed stimulus appears in both material splits")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "stimuli.jsonl", (stimuli[key] for key in sorted(stimuli)))
    write_jsonl(output / "outcomes.jsonl", outcomes)
    write_jsonl(output / "trials.jsonl", trials)
    positive = [row for row in outcomes if row["readback"]]
    scope = {
        "project": PROJECT,
        "article": "https://doi.org/10.5070/g6011182",
        "license": "CC BY 4.0 (OSF project)",
        "sources": SOURCES,
        "rdata_version": rdata.__version__,
        "discourse_source_event_count": len(discourse),
        "discourse_source_trial_count": int(discourse.trial.nunique()),
        "readers_with_aligned_trials": len({row["reader_id"] for row in trials}),
        "aligned_trials": len(trials),
        "aligned_trials_with_binary_acceptability": sum(row["acceptable"] is not None for row in trials),
        "stimuli": len(stimuli),
        "movement_counts_in_aligned_trials": dict(movements_count),
        "first_visit_outcomes": len(outcomes),
        "first_visit_outcomes_by_item": dict(material_counts),
        "first_visit_positives": len(positive),
        "material_split": {
            "procedure": "Order item families by SHA-256 of 'seed:item'; first five held out with both conditions.",
            "seed": TEST_SEED,
            "test_items": sorted(TEST_ITEMS),
            "test_outcomes": sum(row["split"] == "test" for row in outcomes),
            "test_positives": sum(row["split"] == "test" for row in positive),
        },
        "immediate_destinations_before_previous_region": sum(
            row["first_destination_stop"] < stimuli[row["id"]]["stops"][row["region"] - 2] for row in positive
        ),
        "final_destinations_before_previous_region": sum(
            row["source_stop"] < stimuli[row["id"]]["stops"][row["region"] - 2] for row in positive
        ),
        "positive_followed_by_progress_beyond_origin": sum(row["resumed_beyond_stop"] for row in positive),
        "exclusions": dict(audit),
        "output_sha256": {
            name: digest(output / name) for name in ("stimuli.jsonl", "outcomes.jsonl", "trials.jsonl")
        },
    }
    (output / "audit.json").write_text(json.dumps(scope, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.events, args.items, args.output)
