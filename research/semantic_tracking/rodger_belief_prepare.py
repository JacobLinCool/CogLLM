"""Prepare the published Rodger Study 1 text and paired belief ratings."""

import argparse
import csv
import json
import os
import re
from collections import Counter
from hashlib import file_digest
from html.parser import HTMLParser
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory


CSV_NAME = "demo2_study1_data_working.csv"
QSF_NAME = "demo2_study1_survey_qualtrics.qsf"
SOURCES = {
    CSV_NAME: ("https://osf.io/download/36bg8/", "2a203401066086560d96c58509632f82c9c38d0ee6303dfbfbc2c81e0071c8ca"),
    QSF_NAME: ("https://osf.io/download/xuqga/", "5bb404312a59b27bd19d9ae906f6427955c77a1b3fb29a7dd8ce91630d716cd5"),
}
POLICIES = ("mph", "quota", "tax", "merch", "label", "ban")
CONDITIONS = {
    1: "control", 2: "statistic", 3: "anecdote_for", 4: "anecdote_against",
    5: "combined_congruent", 6: "combined_incongruent",
}
READERS = 301
RATING_FIELDS = ("prior_effectiveness", "posterior_effectiveness", "prior_support", "posterior_support")
PAPER = "https://doi.org/10.1017/bpp.2025.10028"


class QuestionText(HTMLParser):
    """Project the pinned QSF question HTML to displayed plain text."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lines = [[]]

    def handle_starttag(self, tag, attrs):
        if tag == "br" and not attrs:
            self.lines.append([])
        elif not (
            (tag in {"b", "i"} and not attrs)
            or (tag == "span" and attrs == [("style", "font-weight: bolder;")])
        ):
            raise ValueError(f"Unrecognized markup in a retained question: {tag}")

    def handle_startendtag(self, tag, attrs):
        if tag != "br":
            raise ValueError(f"Unrecognized retained self-closing tag: {tag}")
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if tag not in {"b", "i", "span"}:
            raise ValueError(f"Unrecognized retained closing tag: {tag}")

    def handle_data(self, data):
        self.lines[-1].append(data)


def displayed_text(question):
    source = question["QuestionText"]
    if "${" in source:
        raise ValueError("Retained question requires a respondent-specific substitution")
    parser = QuestionText()
    parser.feed(source)
    parser.close()
    lines = [re.sub(r"[ \t\n\r\f\u00a0]+", " ", "".join(parts)).strip() for parts in parser.lines]
    text = "\n".join(lines).strip("\n")
    if not text:
        raise ValueError("Empty retained question")
    return text


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def qsf_pages(block, questions):
    if block["Options"]["RandomizeQuestions"] != "false":
        raise ValueError("The retained question pages must have fixed internal order")
    pages = [[]]
    for item in block["BlockElements"]:
        if item["Type"] == "Page Break":
            pages.append([])
        elif item["Type"] == "Question":
            pages[-1].append(questions[item["QuestionID"]])
        else:
            raise ValueError("Unsupported element in a retained survey block")
    return [[question for question in page if question["QuestionType"] != "Timing"] for page in pages]


def check_ratings(questions):
    for question in questions:
        config = question["Configuration"]
        if (
            question["QuestionType"] != "Slider"
            or (config["CSSliderMin"], config["CSSliderMax"], config["NumDecimals"]) != (0, 100, "0")
        ):
            raise ValueError("Expected the released 0–100 integer rating slider")


def block_id(policy, condition=None):
    suffix = "baseline" if condition is None else f"evidence-{condition}"
    return f"rodger-study1-{policy}-{suffix}"


def read_stimuli(qsf):
    elements = json.loads(qsf.read_text(encoding="utf-8"))["SurveyElements"]
    questions = {item["Payload"]["QuestionID"]: item["Payload"] for item in elements if item["Element"] == "SQ"}
    block_payload, = (item["Payload"] for item in elements if item["Element"] == "BL")
    blocks = {block["Description"]: block for block in block_payload.values()}
    stimuli, metadata = [], []

    for policy in POLICIES:
        before_name = f"baseline_{policy}"
        before, = qsf_pages(blocks[before_name], questions)
        if [q["DataExportTag"] for q in before] != [f"eff_pri_{policy}", f"sup_pri_{policy}"]:
            raise ValueError("The baseline questions changed")
        check_ratings(before)
        prior_text = "\n\n".join(map(displayed_text, before))
        prior_id = block_id(policy)
        prior_page = {"role": "before_evidence", "block": before_name, "page": 1, "start": 0, "stop": len(prior_text),
                      "question_ids": [q["QuestionID"] for q in before]}
        stimuli.append({"id": prior_id, "text": prior_text, "stops": [len(prior_text)]})
        metadata.append({"id": prior_id, "policy": policy, "condition": None, "pages": [prior_page]})

        for condition, condition_name in CONDITIONS.items():
            key = f"{condition}_{policy}"
            block_name = f"evidence_{key}"
            evidence, appraisal, posterior = qsf_pages(blocks[block_name], questions)
            evidence_question, = evidence
            appraisal_tags = [f"eval_{key}"] if condition <= 4 else [f"eval1_{key}", f"eval2_{key}"]
            if (
                evidence_question["DataExportTag"] != f"thought_{key}" or evidence_question["QuestionType"] != "TE"
                or [q["DataExportTag"] for q in appraisal] != appraisal_tags
                or [q["DataExportTag"] for q in posterior] != [f"eff_pos_{key}", f"effJus_{key}", f"sup_pos_{key}", f"supJus_{key}"]
            ):
                raise ValueError("Unexpected evidence, appraisal, or posterior question order")
            check_ratings(posterior[::2])
            text = prior_text + "\n\n" + displayed_text(evidence_question)
            uid = block_id(policy, condition)
            stimuli.append({"id": uid, "text": text, "stops": [len(prior_text), len(text)]})
            evidence_page = {"role": "evidence", "block": block_name, "page": 1, "start": len(prior_text) + 2,
                             "stop": len(text), "question_ids": [evidence_question["QuestionID"]]}
            metadata.append({
                "id": uid, "policy": policy, "condition": condition, "condition_name": condition_name,
                "baseline_id": prior_id, "pages": [prior_page, evidence_page],
                "omitted_appraisal_page": {"block": block_name, "page": 2, "question_ids": [q["QuestionID"] for q in appraisal]},
                "posterior_rating_page": {"block": block_name, "page": 3, "question_ids": [q["QuestionID"] for q in posterior]},
            })
    if len({row["text"] for row in stimuli}) != 42:
        raise ValueError("Expected six distinct baseline prompts and 36 distinct evidence inputs")
    return stimuli, metadata


def rating(value):
    if not re.fullmatch(r"(?:0|[1-9][0-9]?|100)", value):
        raise ValueError("Unexpected missing or out-of-range rating in a complete trial")
    return int(value)


def read_outcomes(csv_path):
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        csv_reader = csv.DictReader(stream)
        headers = csv_reader.fieldnames
        if not headers or len(set(headers)) != len(headers):
            raise ValueError("The response table has missing or repeated headers")
        rows = list(csv_reader)
    raw_ids = [row["ID"] for row in rows]
    if len(raw_ids) != READERS or len(set(raw_ids)) != READERS or not all(raw_ids):
        raise ValueError("Unexpected number of distinct source readers")
    shuffled_ids = sorted(raw_ids)
    Random(43).shuffle(shuffled_ids)
    reader_ids = {raw_id: f"rodger-reader-{index:03d}" for index, raw_id in enumerate(shuffled_ids, start=1)}

    outcomes, cells = [], Counter()
    for row in sorted(rows, key=lambda r: reader_ids[r["ID"]]):
        seen_conditions = set()
        for policy in POLICIES:
            post = {
                condition: (row[f"eff_pos_{condition}_{policy}_5"], row[f"sup_pos_{condition}_{policy}_1"])
                for condition in CONDITIONS
            }
            present = [condition for condition, (eff, sup) in post.items() if (eff, sup) != ("NA", "NA")]
            if len(present) != 1 or present[0] in seen_conditions:
                raise ValueError("Every reader must have exactly one condition per policy and vice versa")
            condition, = present
            seen_conditions.add(condition)
            eff_post, sup_post = post[condition]
            values = tuple(map(rating, (row[f"eff_pri_{policy}_1"], eff_post, row[f"sup_pri_{policy}_1"], sup_post)))
            reader_id = reader_ids[row["ID"]]
            outcomes.append({
                "id": block_id(policy, condition), "trial_id": f"{reader_id}-{policy}", "reader_id": reader_id,
                "policy": policy, "condition": condition, **dict(zip(RATING_FIELDS, values, strict=True)),
            })
            cells[(policy, condition)] += 1
    if len(outcomes) != 1806 or len(cells) != 36 or (min(cells.values()), max(cells.values())) != (48, 52):
        raise ValueError("Unexpected released trial counts by stimulus cell")
    if [sum(row[field] == 0 for row in outcomes) for field in RATING_FIELDS] != [133, 168, 185, 259]:
        raise ValueError("Reconstructed endpoint boundary counts disagree with the official responses")
    return outcomes, cells


def write_json(path, data, lines=False):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        if lines:
            for row in data:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        else:
            stream.write(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def prepare(raw, output):
    for name, (_, expected) in SOURCES.items():
        if digest(raw / name) != expected:
            raise ValueError(f"Unexpected bytes for the pinned public source {name}")
    stimuli, metadata = read_stimuli(raw / QSF_NAME)
    outcomes, cells = read_outcomes(raw / CSV_NAME)
    if {row["id"] for row in outcomes} != {row["id"] for row in metadata if row["condition"] is not None}:
        raise ValueError("Responses and reconstructed evidence inputs do not match")
    folds = [
        {
            "id": f"holdout-{test}", "train_policies": [policy for policy in POLICIES if policy != test],
            "test_policies": [test],
            "train_stimulus_ids": [row["id"] for row in metadata if row["policy"] != test],
            "test_stimulus_ids": [row["id"] for row in metadata if row["policy"] == test],
            "train_trials": 5 * READERS, "test_trials": READERS,
        }
        for test in POLICIES
    ]
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".rodger-preparation-") as directory:
        staged = Path(directory) / "prepared"
        staged.mkdir()
        write_json(staged / "stimuli.jsonl", stimuli, lines=True)
        write_json(staged / "stimulus_metadata.jsonl", metadata, lines=True)
        write_json(staged / "outcomes.jsonl", outcomes, lines=True)
        os.chmod(staged / "outcomes.jsonl", 0o600)
        write_json(staged / "folds.json", {"design": "within-study leave-one-policy-out with shared readers", "folds": folds})
        write_json(staged / "source-audit.json", {
            "paper": PAPER, "study": 1,
            "sources": {name: {"url": url, "sha256": sha} for name, (url, sha) in SOURCES.items()},
            "license": {"name": "CC BY 4.0", "data": "https://osf.io/z4wem/", "materials": "https://osf.io/h6ygr/"},
            "reader_count": READERS, "trials": len(outcomes), "baseline_prompts": 6, "evidence_prompts": 36,
            "counts_by_stimulus": {block_id(*key): value for key, value in sorted(cells.items())},
            "actual_slider_range": [0, 100], "paper_reported_range": [1, 100],
            "zero_ratings": {field: sum(row[field] == 0 for row in outcomes) for field in RATING_FIELDS},
            "rendering": "QSF QuestionText in page order, HTML entities decoded, br mapped to line break, inline styling removed, browser whitespace collapsed, blank line between questions and pages.",
            "offset_units": "exclusive Unicode character index in the prepared text",
            "scope": "Two retained static pages for the same policy. The reader's observed posterior rating follows a separate evidence appraisal page.",
            "omitted_context": ["survey-wide instructions", "other policies", "slider UI and scale labels", "appraisal and posterior pages", "participant answers, demographics, and typed text"],
            "output_sha256": {path.name: digest(path) for path in sorted(staged.iterdir())},
        })
        staged.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.raw, args.output)
