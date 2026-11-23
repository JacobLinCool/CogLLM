"""Prepare the fixed Blott sentence-coherence holdout from the OSF release."""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import mean
import xml.etree.ElementTree as ET
from zipfile import ZipFile


TITLE = "Recovery from misinterpretations during online sentence processing"
PROJECT = "https://osf.io/hn3bu/"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
MATERIALS = (
    "https://osf.io/download/5a3f2/",
    "208f45919e2c6c2623d66c2275d11f409deb9644d720844be2b2c66f6e364580",
)
TRIALS = (
    "https://osf.io/download/tzxad/",
    "9acb103b64eb511d4e4c5430f49de5ff07ba850d86ba7f06a3aefcc3e5087e51",
)
AGGREGATE = (
    "https://osf.io/download/bwk5n/",
    "a18b9f44ef4eea17bd82a54db55831026d3f221d696dba1565347607d7fcab17",
)
CONDITIONS = (
    ("PlausUA", "Coh_UA"),
    ("PlausAmb", "Coh_Amb"),
    ("AnomUA", "Anom_UA"),
    ("AnomAmb", "Anom_Amb"),
)
WORD_XML = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


@dataclass(frozen=True)
class Stimulus:
    stimulus_id: str
    family: int
    item_label: str
    condition: str
    text: str
    noun_span: tuple[int, int]
    cue_span: tuple[int, int] | None
    spillover_span: tuple[int, int] | None


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(path, source):
    if sha256(path) != source[1]:
        raise ValueError(f"Unexpected OSF release for {path.name}")


def stimulus_table(path):
    with ZipFile(path) as document:
        root = ET.fromstring(document.read("word/document.xml"))
    table, = root.findall(".//w:tbl", WORD_XML)
    rows = table.findall("w:tr", WORD_XML)
    if len(rows) != 49:
        raise ValueError("Expected one heading and 48 published stimulus quartets")
    stimuli = {}
    for row in rows[1:]:
        family, alternatives, frame = (
            "".join(node.text or "" for node in cell.findall(".//w:t", WORD_XML)).strip()
            for cell in row.findall("w:tc", WORD_XML)
        )
        unambiguous, ambiguous, anomalous_unambiguous, anomalous_ambiguous = alternatives.split("/")
        if ambiguous == "bridge":
            continue
        for keyword, (condition, _) in zip(
            (unambiguous, ambiguous, anomalous_unambiguous, anomalous_ambiguous), CONDITIONS, strict=True,
        ):
            if frame.split("/")[1] != "KEYWORD" or frame.count("KEYWORD") != 1:
                raise ValueError(f"Unrecognized keyword assignment in family {family}")
            segmented = frame.replace("KEYWORD", keyword)
            parts = segmented.split("/")
            if len(parts) != (5 if ambiguous == "log" else 6) or any(not part for part in parts):
                raise ValueError(f"Unrecognized source regions in family {family}")
            start = 0
            spans = []
            for part in parts:
                spans.append((start, start + len(part)))
                start += len(part) + 1
            text = " ".join(parts)
            stimulus_id = hashlib.sha256(f"blott-2021:{text}".encode()).hexdigest()
            stimuli[segmented] = Stimulus(
                stimulus_id, int(family), ambiguous, condition, text, spans[1],
                None if ambiguous == "log" else spans[3],
                None if ambiguous == "log" else spans[4],
            )
    if len({item.family for item in stimuli.values()}) != 47 or len(stimuli) != 188 or len({
        item.stimulus_id for item in stimuli.values()
    }) != 188:
        raise ValueError("Expected 188 distinct sentences after the published exclusion")
    return stimuli


def trial_table(path, stimuli):
    trials = {}
    raw_accuracies = {}
    area_rows = Counter()
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            trial_key = row["RECORDING_SESSION_LABEL"], row["TRIAL_INDEX"]
            stimulus = stimuli[row["sentence_text"]]
            if row["item_label"] != stimulus.item_label or row["cond_label"] != stimulus.condition:
                raise ValueError(f"Trial {trial_key} has an inconsistent material assignment")
            accuracy = int(row["accuracy"])
            if accuracy not in (0, 1) or row["practice"] != "experiment":
                raise ValueError(f"Trial {trial_key} has an invalid endpoint")
            pressed = row["BUTTON_PRESSED"]
            expected = row["expected.button.press"]
            condition_button = "4" if stimulus.condition.startswith("Plaus") else "2"
            if pressed not in {"4", "2"} or expected != condition_button or accuracy != int(pressed == expected):
                raise ValueError(f"Trial {trial_key} cannot be reconstructed from the corrected button mapping")
            value = (stimulus, accuracy)
            if trials.setdefault(trial_key, value) != value:
                raise ValueError(f"Multiple area rows disagree for trial {trial_key}")
            raw_accuracy = int(row["BUTTON_ACCURACY"])
            if raw_accuracies.setdefault(trial_key, raw_accuracy) != raw_accuracy:
                raise ValueError(f"Multiple area rows disagree on raw button accuracy for {trial_key}")
            area_rows[trial_key] += 1
    if Counter(reader for reader, _ in trials) != {reader: 47 for reader, _ in trials} or len(trials) != 4512:
        raise ValueError("The released file should contain 96 readers with 47 eligible trials each")
    if Counter(stimulus.stimulus_id for stimulus, _ in trials.values()) != {
        stimulus.stimulus_id: 24 for stimulus in stimuli.values()
    }:
        raise ValueError("Each eligible variant should have 24 trials")
    if len({(reader, stimulus.family) for (reader, _), (stimulus, _) in trials.items()}) != len(trials):
        raise ValueError("A reader saw more than one variant in a stimulus quartet")
    if any(area_rows[key] != (5 if item.item_label == "log" else 6) for key, (item, _) in trials.items()):
        raise ValueError("Unexpected number of area rows per sentence")
    distinct_raw_outcomes = sum(accuracy != raw_accuracies[key] for key, (_, accuracy) in trials.items())
    return trials, area_rows, distinct_raw_outcomes


def summarize(trials, area_rows, aggregate):
    trial_totals = Counter()
    trial_correct = Counter()
    trial_coherent = Counter()
    reader_totals = Counter()
    reader_correct = Counter()
    reader_area_totals = Counter()
    reader_area_correct = Counter()
    for (reader, trial), (stimulus, accuracy) in trials.items():
        condition = stimulus.condition
        trial_totals[condition] += 1
        trial_correct[condition] += accuracy
        trial_coherent[condition] += accuracy if condition.startswith("Plaus") else 1 - accuracy
        reader_totals[reader, condition] += 1
        reader_correct[reader, condition] += accuracy
        reader_area_totals[reader, condition] += area_rows[reader, trial]
        reader_area_correct[reader, condition] += area_rows[reader, trial] * accuracy
    reader_accuracy = {
        key: reader_correct[key] / total for key, total in reader_totals.items()
    }
    group_accuracy = {}
    with aggregate.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["ID"], row["Condition"]
            if key in group_accuracy:
                raise ValueError(f"Duplicate released accuracy label {key}")
            group_accuracy[key] = float(row["Accuracy"])
    contrast = {}
    expected_keys = set()
    for condition, aggregate_condition in CONDITIONS:
        readers = [reader for reader, code in reader_accuracy if code == condition]
        references = []
        for reader in readers:
            trial_key = reader, condition
            summary_key = reader, aggregate_condition
            expected_keys.add(summary_key)
            published = group_accuracy[summary_key]
            area_weighted = reader_area_correct[trial_key] / reader_area_totals[trial_key]
            if abs(area_weighted - published) > 1e-8:
                raise ValueError(f"Aggregated area rows do not match the OSF reference for {summary_key}")
            references.append(published)
        contrast[aggregate_condition] = {
            "trials": trial_totals[condition],
            "correct": trial_correct[condition],
            "coherent_responses": trial_coherent[condition],
            "coherent_response_fraction": trial_coherent[condition] / trial_totals[condition],
            "reader_mean_accuracy": mean(reader_accuracy[reader, condition] for reader in readers),
            "author_area_weighted_reader_mean_accuracy": mean(references),
        }
    if len(expected_keys) != 384 or expected_keys != set(group_accuracy):
        raise ValueError("Expected exact match to all 384 released reader-by-condition summaries")
    return contrast


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def prepare(materials, trial_data, aggregate, output):
    for path, source in ((materials, MATERIALS), (trial_data, TRIALS), (aggregate, AGGREGATE)):
        verify(path, source)
    stimuli = stimulus_table(materials)
    trials, area_rows, raw_accuracy_disagreements = trial_table(trial_data, stimuli)
    reference = summarize(trials, area_rows, aggregate)
    inputs_dir = output / "model_input"
    scoring_dir = output / "scoring"
    inputs_dir.mkdir(exist_ok=True, parents=True)
    scoring_dir.mkdir(exist_ok=True, parents=True)
    input_path = inputs_dir / "stimuli.jsonl"
    design_path = scoring_dir / "design.jsonl"
    trials_path = scoring_dir / "human_trials.jsonl"
    ordered = sorted(stimuli.values(), key=lambda item: item.stimulus_id)
    write_jsonl(input_path, ({"id": item.stimulus_id, "text": item.text} for item in ordered))
    write_jsonl(design_path, ({
        "id": item.stimulus_id,
        "family": item.family,
        "condition": item.condition,
        "main_noun_span": item.noun_span,
        "coherence_cue_span": item.cue_span,
        "spillover_span": item.spillover_span,
    } for item in ordered))
    write_jsonl(trials_path, ({
        "reader_id": reader,
        "id": item.stimulus_id,
        "coherent": bool(accuracy if item.condition.startswith("Plaus") else 1 - accuracy),
    } for (reader, _), (item, accuracy) in sorted(trials.items())))
    return {
        "article": "https://doi.org/10.1037/xlm0000936",
        "authors": ["Lena M. Blott", "Jennifer M. Rodd", "Fernanda Ferreira", "Jane E. Warren"],
        "title": TITLE,
        "source_project": PROJECT,
        "source_license": LICENSE,
        "sources": {
            key: {"download": url, "sha256": digest}
            for key, (url, digest) in {
                "stimulus_table": MATERIALS, "experimental_trials": TRIALS, "aggregate_accuracy": AGGREGATE,
            }.items()
        },
        "readers": 96,
        "eligible_families": 47,
        "eligible_stimuli": 188,
        "trials": len(trials),
        "source_area_rows": sum(area_rows.values()),
        "published_excluded_family": "bridge",
        "published_excluded_regional_analysis_family": "log",
        "endpoint": "yes/no button judgment that the sentence made sense",
        "endpoint_source": "BUTTON_PRESSED with corrected expected.button.press, verified against analysis accuracy",
        "raw_button_accuracy_disagreements": raw_accuracy_disagreements,
        "area_weighted_reader_condition_rows_matching_author_aggregate_within_1e_8": 384,
        "condition_counts": reference,
        "coherent_ambiguous_minus_unambiguous_reader_mean": (
            reference["Coh_Amb"]["reader_mean_accuracy"] - reference["Coh_UA"]["reader_mean_accuracy"]
        ),
        "model_inputs_include_only": ["id", "text"],
        "artifact_sha256": {
            str(path.relative_to(output)): sha256(path) for path in (input_path, design_path, trials_path)
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.materials, args.trials, args.aggregate, args.output)
    args.report.parent.mkdir(exist_ok=True, parents=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
