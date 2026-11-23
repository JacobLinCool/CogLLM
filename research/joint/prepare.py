"""Build the first three-task design from locked Stage 2 data."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from ..stage3.issue_probe import _checklist
from ..stage3.probe import _jsonl, _load_plaba
from ..stage3.zhan_landmark_probe import load_trials as load_zhan


OPTIONS = "ABCD"


def prepare_plaba(source):
    stimuli, sentences, ratings, folds, _ = _load_plaba(source)
    checklist, groups, _ = _checklist(source, stimuli)
    ids = sorted(sentences)
    rating_counts = np.zeros((len(ids), 2, 5), dtype=np.int64)
    issue_counts = np.zeros((len(ids), 2, 2), dtype=np.int64)
    for i, sentence_id in enumerate(ids):
        for j, values in enumerate(np.asarray(ratings[sentence_id]).T):
            rating_counts[i, j] = np.bincount(values - 1, minlength=5)
        if sentence_id in checklist:
            for j, values in enumerate(np.asarray(checklist[sentence_id]).T):
                issue_counts[i, j] = np.bincount(values, minlength=2)
    if rating_counts.sum(axis=(0, 2)).tolist() != [1826, 1826] or issue_counts.sum(axis=(0, 2)).tolist() != [858, 858]:
        raise ValueError("PLABA survey denominators differ from the fixed pilot")
    families = [stimuli[sentences[key][0]]["family_id"] for key in ids]
    return {
        "plaba_ids": np.array(ids),
        "plaba_folds": np.array([folds[family] for family in families]),
        "plaba_groups": np.array([groups[family] for family in families]),
        "plaba_rating_counts": rating_counts,
        "plaba_issue_counts": issue_counts,
        "plaba_length": np.log1p([(len(sentences[key][1]), len(sentences[key][1].split())) for key in ids]),
    }


def prepare_zhan(source):
    families, ids, counts, verbs, folds = load_zhan(source)
    positives = counts[..., 0].ravel()
    return {
        "zhan_ids": np.array(ids),
        "zhan_folds": np.repeat(folds, 2),
        "zhan_counts": np.column_stack((counts[..., 1].ravel() - positives, positives)),
        "zhan_families": np.repeat(families, 2),
        "zhan_verb_type": np.repeat(verbs, 2),
    }


def prompt_id(prefix, text):
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def open_book_inputs(passage, question, displayed_options):
    option_text = "\n".join(f"{label}. {choice}" for label, choice in zip(OPTIONS, displayed_options, strict=True))
    question_input = f"Question:\n{question}\n\nOptions:\n{option_text}\n\nAnswer:"
    return f"Passage:\n{passage}\n\n{question_input}", question_input


def prepare_onestop(source):
    stimuli = {row["id"]: row for row in _jsonl(source / "stimuli.jsonl")}
    folds = json.loads((source / "splits.json").read_text(encoding="utf-8"))["folds"]
    family_fold = {group: row["fold"] for row in folds for group in row["families"]}
    if set(family_fold) != {row["family_id"] for row in stimuli.values()} or sum(len(row["families"]) for row in folds) != len(family_fold):
        raise ValueError("OneStopQA family split does not cover its passages exactly once")

    by_id = {}
    inputs = {}
    question_folds = defaultdict(set)
    seen = set()
    for row in _jsonl(source / "trials.jsonl"):
        stimulus = stimuli[row["stimulus_id"]]
        context = row["task_context"]
        observations = row["observations"]
        family = stimulus["family_id"]
        if (
            row["id"] in seen or row["exposure"]["task"] != "open_book_mcq"
            or row["exposure"]["visible_end"] != len(stimulus["text"])
            or row["metadata"]["question_visibility"] != "with_passage"
            or [option["id"] for option in context["options"]] != list(OPTIONS)
            or any(observations[name]["status"] != "observed" for name in (
                "mcq_correct", "mcq_correct_option", "mcq_selected_option",
            ))
        ):
            raise ValueError(f"Invalid OneStopQA exposure or response for {row['id']}")
        seen.add(row["id"])
        chosen, gold = (OPTIONS.index(observations[name]["value"]) for name in (
            "mcq_selected_option", "mcq_correct_option",
        ))
        if observations["mcq_correct"]["value"] != (chosen == gold):
            raise ValueError(f"Selected and gold display options disagree for {row['id']}")

        full_input, question_input = open_book_inputs(
            stimulus["text"], context["question"], [option["text"] for option in context["options"]],
        )
        full_id = prompt_id("full", full_input)
        question_id = prompt_id("question", question_input)
        inputs[full_id] = full_input
        inputs[question_id] = question_input
        item = by_id.setdefault(full_id, (question_id, family, gold, np.zeros(4, dtype=np.int64)))
        if item[:3] != (question_id, family, gold):
            raise ValueError(f"Identical OneStopQA input has inconsistent metadata for {row['id']}")
        item[3][chosen] += 1
        question_folds[question_id].add(family_fold[family])

    ids = sorted(by_id)
    questions, families, gold, counts = zip(*(by_id[key] for key in ids), strict=True)
    if len(ids) != 1872 or len(seen) != 1940:
        raise ValueError("OneStopQA pilot differs from the verified visible-input inventory")
    return {
        "onestop_ids": np.array(ids),
        "onestop_folds": np.array([family_fold[key] for key in families]),
        "onestop_counts": np.stack(counts),
        "onestop_question_only_ids": np.array(questions),
        "onestop_correct": np.array(gold),
        "onestop_families": np.array(families),
    }, inputs, sum(len(group) > 1 for group in question_folds.values())


def prepare(source, output):
    arrays = prepare_plaba(source / "plaba") | prepare_zhan(source / "zhan")
    onestop, visible, crossing_questions = prepare_onestop(source / "onestop")
    arrays.update(onestop)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "design.npz", **arrays)
    with (output / "inputs.jsonl").open("w", encoding="utf-8") as stream:
        for key, text in sorted(visible.items()):
            stream.write(json.dumps({"id": key, "text": text}, ensure_ascii=False) + "\n")
    report = {
        "unique_human_trials": 5417,
        "one_stop_full_inputs": len(onestop["onestop_ids"]),
        "one_stop_question_only_inputs": len(set(onestop["onestop_question_only_ids"])),
        "one_stop_question_only_inputs_crossing_folds": crossing_questions,
        "one_stop_actual_inputs": len(visible),
        "one_stop_input_source": "Visible passage, question and actual option text in display order. No outcome fields.",
        "source_sha256": {
            f"{corpus}/{name}": hashlib.sha256((source / corpus / name).read_bytes()).hexdigest()
            for corpus in ("plaba", "zhan", "onestop")
            for name in ("stimuli.jsonl", "trials.jsonl", "splits.json")
        },
        "outputs_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("design.npz", "inputs.jsonl")
        },
    }
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.output)
