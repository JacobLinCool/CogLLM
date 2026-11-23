"""Audit the released RACE Prolific inputs without accessing answer responses."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata


SOURCE_URL = (
    "https://raw.githubusercontent.com/berzak/onestop-qa/"
    "7a058755e1ff0d3f835a67109c669c7e294d4239/human_experiments/prolific/qa.tsv"
)
SOURCE_SHA256 = "bc8af0a78819b918eb0ce97935bf8fcd82eeec2d9f99111762034817c1e56e0d"
VISIBLE_FIELDS = (
    "source", "trial_type", "submission_ind", "trial_number", "item_id", "difficulty",
    "paragraph", "question", "options", "new_options",
)
RACE_ID = re.compile(r"(r[0-9]+)_([1-9][0-9]*)_(High|Middle)")
ONESTOP_ID = re.compile(r"(os[1-9][0-9]*)_[1-9][0-9]*_[1-9][0-9]*_(Adv|Ele)")


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def visible_trials(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream, delimiter="\t", strict=True)
        header = next(rows)
        if len(set(header)) != len(header) or not set(VISIBLE_FIELDS).issubset(header):
            raise ValueError("Unexpected author-released trial columns")
        positions = [header.index(field) for field in VISIBLE_FIELDS]
        for row_number, row in enumerate(rows, start=1):
            if len(row) != len(header):
                raise ValueError(f"Incomplete field count at source row {row_number}")
            visible = dict(zip(VISIBLE_FIELDS, (row[index] for index in positions), strict=True))
            if visible["trial_type"] == "reading_comprehension":
                yield row_number, visible


def audit(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Expected the pinned author-released Prolific QA file")

    cohorts = defaultdict(Counter)
    articles = defaultdict(set)
    prompts = defaultdict(set)
    questions_only = defaultdict(set)
    item_counts = defaultdict(Counter)
    material_by_item = defaultdict(dict)
    item_by_full_prompt = defaultdict(dict)
    exact_match_fields = {field: defaultdict(set) for field in ("paragraph", "question", "item_id")}
    observed_trial_keys = set()
    race_text_by_article = {}
    for row_number, row in visible_trials(path):
        source, item = row["source"], row["item_id"]
        reader, trial = row["submission_ind"], row["trial_number"]
        if source not in {"RACE", "Onestop"} or not reader.isdecimal() or trial not in {"1", "2", "3", "4"}:
            raise ValueError(f"Unexpected experimental trial at row {row_number}")
        if (reader, trial) in observed_trial_keys:
            raise ValueError(f"Duplicate reader and trial position at row {row_number}")
        observed_trial_keys.add((reader, trial))

        identifier = (RACE_ID if source == "RACE" else ONESTOP_ID).fullmatch(item)
        if identifier is None or identifier.groups()[-1] != row["difficulty"]:
            raise ValueError(f"Invalid article and item identifiers at row {row_number}")
        paragraph, question = row["paragraph"], row["question"]
        options = tuple(row["options"].split("|"))
        order = row["new_options"].split("|")
        if len(options) != 4 or not all(part.strip() for part in (paragraph, question, *options)):
            raise ValueError(f"Missing displayed text at row {row_number}")
        if sorted(order) != list("0123"):
            raise ValueError(f"Invalid displayed choice permutation at row {row_number}")

        family = (identifier[1], row["difficulty"]) if source == "RACE" else identifier[1]
        if source == "RACE" and race_text_by_article.setdefault(family, paragraph) != paragraph:
            raise ValueError(f"RACE article key identifies different passages at row {row_number}")
        item_material = (paragraph, question, options)
        if material_by_item[source].setdefault(item, item_material) != item_material:
            raise ValueError(f"Item key identifies different versions at row {row_number}")

        displayed = tuple(options[int(index)] for index in order)
        short_key = (question, displayed)
        full_key = (paragraph, *short_key)
        if item_by_full_prompt[source].setdefault(full_key, item) != item:
            raise ValueError(f"An identical full input maps to different items at row {row_number}")
        prompts[source].add(full_key)
        questions_only[source].add(short_key)
        cohorts[source][reader] += 1
        articles[source].add(family)
        item_counts[source][item] += 1
        for field, sets in exact_match_fields.items():
            sets[source].add(normalize(row[field]))

    result_by_source = {}
    for source in ("OneStopQA", "RACE"):
        key = {"OneStopQA": "Onestop", "RACE": "RACE"}[source]
        result_by_source[source] = {
            "released_trials": sum(item_counts[key].values()),
            "distinct_reader_keys": len(cohorts[key]),
            "article_families": len(articles[key]),
            "distinct_items": len(item_counts[key]),
            "distinct_full_inputs_with_displayed_option_order": len(prompts[key]),
            "distinct_question_and_option_inputs_with_displayed_option_order": len(questions_only[key]),
            "items_by_released_trial_count": dict(Counter(item_counts[key].values())),
            "readers_by_released_trial_count": dict(Counter(cohorts[key].values())),
        }
    return {
        "source": SOURCE_URL,
        "source_sha256": digest,
        "source_bytes": path.stat().st_size,
        "protocol": "STARC §4.3: same Prolific QA experiment; passage, question and four displayed options concurrently visible.",
        "protocol_source": "https://aclanthology.org/2020.acl-main.507.pdf#page=6",
        "race_source_terms": "https://www.cs.cmu.edu/~glai1/data/race/",
        "race_source_terms_summary": "RACE releases the data for noncommercial research; source passages are third-party content.",
        "by_source": result_by_source,
        "structural_checks": {
            "displayed_option_permutations_valid": True,
            "cohort_trial_positions_unique": True,
            "item_material_mapping_unambiguous": True,
            "race_article_to_exact_text_mapping_unambiguous": True,
        },
        "overlap": {
            "distinct_reader_keys": len(cohorts["RACE"].keys() & cohorts["Onestop"].keys()),
            "comparison_normalization": "Unicode NFKC, casefold, collapse whitespace; exact equality only",
            **{field: len(sets["RACE"] & sets["Onestop"]) for field, sets in exact_match_fields.items()},
        },
        "scope": "Matched-protocol material transfer within the same experiment and cohort. Selected-response column omitted by parser; no labels or text exported.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.tsv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
