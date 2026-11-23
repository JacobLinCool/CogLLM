"""Build RACE transfer prompts and keep the Prolific response ledger local."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from ..joint.prepare import OPTIONS, open_book_inputs, prompt_id
from .race_audit import RACE_ID, SOURCE_SHA256, SOURCE_URL, audit, normalize, visible_trials


RACE_REFERENCE_REVISION = "2fec9fd81f1dc971569a9b729c43f2f0e6436637"
RACE_REFERENCES = {
    "high": "97ef697f4c81e804eb43e7a667a6ac7bb06c11a99cfa537997ae5e2f6c5db2b7",
    "middle": "4ea3fb86ab6bbf8746dd0eb0ac1961a1fdbd1214d6d5a901c37b0cad5b5c5971",
}
EXPECTED_TRIALS = 1940
EXPECTED_PROMPTS_PER_MODE = 1851


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def model_rows(source):
    prompts = {}
    design = []
    source_rows = {}
    for index, row in visible_trials(source):
        if row["source"] != "RACE":
            continue
        order = [int(index) for index in row["new_options"].split("|")]
        choices = row["options"].split("|")
        if sorted(order) != list(range(len(OPTIONS))) or len(choices) != len(OPTIONS):
            raise ValueError(f"Invalid released option permutation at survey row {index}")
        full, question = open_book_inputs(row["paragraph"], row["question"], [choices[n] for n in order])
        full_id, question_id = prompt_id("full", full), prompt_id("question", question)
        for key, value in ((full_id, full), (question_id, question)):
            if prompts.setdefault(key, value) != value:
                raise ValueError(f"Prompt identity collision at survey row {index}")
        trial_id = f"race-row-{index:05d}"
        design.append({"id": trial_id, "full_id": full_id, "question_id": question_id, "displayed_order": order})
        source_rows[index] = row
    if len(design) != EXPECTED_TRIALS or Counter(key.split("-", 1)[0] for key in prompts) != {
        "full": EXPECTED_PROMPTS_PER_MODE, "question": EXPECTED_PROMPTS_PER_MODE,
    }:
        raise ValueError("RACE prompts do not match the audited Prolific inventory")
    return prompts, design, source_rows


def build_inputs(source, output):
    inventory = audit(source)
    if any(inventory["overlap"][field] for field in ("paragraph", "question", "item_id")):
        raise ValueError("RACE test and OneStop development materials overlap")
    prompts, design, source_rows = model_rows(source)
    output.mkdir(parents=True, exist_ok=True)
    prompt_path, design_path = output / "inputs.jsonl", output / "design.jsonl"
    write_jsonl(prompt_path, ({"id": key, "text": text} for key, text in sorted(prompts.items())))
    write_jsonl(design_path, design)
    write_json(output / "preparation.json", {
        "source": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "trials": len(design),
        "full_prompts": EXPECTED_PROMPTS_PER_MODE,
        "question_prompts": EXPECTED_PROMPTS_PER_MODE,
        "reader_keys": len({row["submission_ind"] for row in source_rows.values()}),
        "article_ids": len({(RACE_ID.fullmatch(row["item_id"])[1], row["difficulty"]) for row in source_rows.values()}),
        "material_overlap_with_onestop": {field: inventory["overlap"][field] for field in ("paragraph", "question", "item_id")},
        "same_reader_cohort_as_onestop": inventory["overlap"]["distinct_reader_keys"] == 970,
        "outcome_columns_in_inputs": [],
        "payload_sha256": {path.name: digest(path) for path in (prompt_path, design_path)},
    })


def original_race_keys(reference_dir, source_rows):
    from pyarrow.parquet import read_table

    originals = defaultdict(list)
    for level, expected in RACE_REFERENCES.items():
        path = reference_dir / f"{level}.parquet"
        if digest(path) != expected:
            raise ValueError(f"Unexpected RACE {level} reference release")
        for row in read_table(path, columns=["example_id", "article", "question", "options", "answer"]).to_pylist():
            originals[(level, row["example_id"], normalize(row["question"]))].append(row)

    gold_by_item = {}
    normalized_paragraph_matches = 0
    for row in source_rows.values():
        item = row["item_id"]
        if item in gold_by_item:
            continue
        level = row["difficulty"].lower()
        reference_item = f"{level}{RACE_ID.fullmatch(item)[1][1:]}.txt"
        choices = [normalize(choice) for choice in row["options"].split("|")]
        matching = [
            candidate for candidate in originals[(level, reference_item, normalize(row["question"]))]
            if sorted(map(normalize, candidate["options"])) == sorted(choices)
        ]
        if len(matching) != 1 or len(set(choices)) != len(OPTIONS):
            raise ValueError(f"No unique original RACE question and option join for {item}")
        original, = matching
        gold = normalize(original["options"][OPTIONS.index(original["answer"])])
        gold_by_item[item] = choices.index(gold)
        normalized_paragraph_matches += int(normalize(original["article"]) == normalize(row["paragraph"]))
    if len(gold_by_item) != 648 or set(gold_by_item.values()) != {0}:
        raise ValueError("Prolific source option keys disagree with the original RACE test answers")
    return gold_by_item, normalized_paragraph_matches


def released_answers(path, row_numbers):
    answers = {}
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream, delimiter="\t", strict=True)
        selected_column = next(rows).index("answer_response")
        for index, row in enumerate(rows, start=1):
            if index in row_numbers:
                value = row[selected_column]
                if value not in "0123" or len(value) != 1:
                    raise ValueError(f"Unexpected selected source option at survey row {index}")
                answers[index] = int(value)
    if answers.keys() != row_numbers:
        raise ValueError("Incomplete RACE respondent option inventory")
    return answers


def build_outcomes(source, design_path, reference_dir, output):
    if digest(source) != SOURCE_SHA256:
        raise ValueError("Expected the pinned author-released Prolific QA file")
    _, rebuilt, source_rows = model_rows(source)
    with design_path.open(encoding="utf-8") as stream:
        design = [json.loads(line) for line in stream]
    if rebuilt != design:
        raise ValueError("The prediction design does not agree with the verified source rows")
    gold_by_item, paragraph_matches = original_race_keys(reference_dir, source_rows)
    selected = released_answers(source, source_rows.keys())
    outcomes = []
    for trial, (index, row) in zip(design, source_rows.items(), strict=True):
        order = trial["displayed_order"]
        article = RACE_ID.fullmatch(row["item_id"])[1]
        outcomes.append({
            "id": trial["id"],
            "article_id": f"{row['difficulty'].lower()}-{article}",
            "reader_id": f"prolific-{row['submission_ind']}",
            "selected": order.index(selected[index]),
            "correct": order.index(gold_by_item[row["item_id"]]),
            "level": row["difficulty"],
        })
    output.mkdir(parents=True, exist_ok=True)
    outcomes_path = output / "outcomes.jsonl"
    write_jsonl(outcomes_path, outcomes)
    write_json(output / "outcomes-audit.json", {
        "source": SOURCE_URL,
        "source_sha256": SOURCE_SHA256,
        "design_sha256": digest(design_path),
        "outcomes_sha256": digest(outcomes_path),
        "trials": len(outcomes),
        "reference_join": "Original test article ID + unique NFKC/casefold/whitespace-normalized question and option multiset",
        "reference_items_matched": len(gold_by_item),
        "reference_gold_at_prolific_source_index_zero": len(gold_by_item),
        "reference_normalized_paragraph_matches": paragraph_matches,
        "displayed_text_source": "Prolific author release",
        "reference_revision": RACE_REFERENCE_REVISION,
        "reference_files": {
            level: {
                "source": f"https://huggingface.co/datasets/ehovy/race/resolve/{RACE_REFERENCE_REVISION}/{level}/test-00000-of-00001.parquet",
                "sha256": expected,
            }
            for level, expected in RACE_REFERENCES.items()
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser("inputs")
    outcomes = commands.add_parser("outcomes")
    for command in (inputs, outcomes):
        command.add_argument("--tsv", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    outcomes.add_argument("--design", type=Path, required=True)
    outcomes.add_argument("--reference-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inputs":
        build_inputs(args.tsv, args.output)
    else:
        build_outcomes(args.tsv, args.design, args.reference_dir, args.output)
