"""Check the Nahatame rating release against its TECO materials and source trials."""

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median, stdev
import subprocess
import unicodedata

from openpyxl import load_workbook


TEXT_IDS = set(range(1, 31))
RELEASE_SHA256 = {
    "df_individual.csv": "a1b02eb1133a3aaef06666c1f016ab1d8a54a94efd89f569999c3d133b6452ff",
    "df_text.csv": "b0440d8251cf0ba2688b47b18eec8beed4257ce76342cd1994d8256d0fe8453f",
    "teco_participant_info.csv": "873ed1afebe14991373ede612f1f1e2b27e1260dddd7e9660d12103dc5fa9867",
    "teco_reading_materials.pdf": "d59ec536a7306c455f4422105905203ae0cb0677b37da66a700a72adad7a3375",
    "teco_text_info.xlsx": "855e55d22195c6cf5da82ca060f7b337199e3c1238d9352110890b681d5202ad",
    "teco_textmeasure.csv": "6882faba674575c64f7086aab468541baf23850aa4f01ce9fcb3bf0f89e71044",
    "teco_word_info.csv": "25ea5891c269edd8efb1f29c27b233efb2837ded29f620b04a81eea1d5309f4b",
}


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def index(rows, *columns):
    indexed = {tuple(int(row[k]) for k in columns): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate key in {columns}")
    return indexed


def letters_and_numbers(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold() if c.isalnum())


def reading_materials(pdf):
    layout = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    body, questions = layout.split("Comprehension Questions")
    passages = defaultdict(list)
    text_id = 0
    for page in body.split("\f"):
        lines = page.splitlines()
        indents = Counter(len(line) - len(line.lstrip()) for line in lines if re.match(r"^ +[^\d \t]", line))
        if not indents:
            continue
        # The next passage can begin in the middle of a PDF page.
        body_column = indents.most_common(1)[0][0]
        for line in lines:
            labels, content = line[:body_column], line[body_column:].strip()
            header = re.fullmatch(r" *(\d{1,2})(?: {2,}([^\d\s].*))? *", labels)
            if header and int(header[1]) == text_id + 1:
                text_id += 1
            elif not text_id or labels.strip().startswith(("EIKEN", "*P =", "ID")):
                continue
            if content:
                passages[text_id].append(content)

    titles, trials = {}, defaultdict(list)
    for line in questions.splitlines():
        record = re.match(r"^\s*(?:(\d+)\s{2,}(.+?)\s{2,})?(\d+)\s{2,}(\d+)\s{2,}\S", line)
        if record is None:
            continue
        if record[1]:
            text_id = int(record[1])
            titles[text_id] = record[2]
        trials[text_id].append(int(record[4]))
    if set(passages) != TEXT_IDS or set(titles) != TEXT_IDS or set(trials) != TEXT_IDS:
        raise ValueError("The PDF does not contain all 30 passage and question sections")
    if any(numbers != list(range(1, len(numbers) + 1)) for numbers in trials.values()):
        raise ValueError("The PDF contains an incomplete or reordered question sequence")
    return {i: " ".join(lines) for i, lines in passages.items()}, titles, trials


def audit_materials(raw, text_rows):
    passages, titles, questions = reading_materials(raw / "teco_reading_materials.pdf")
    word_rows = index(csv_rows(raw / "teco_word_info.csv"), "textid", "position")
    words = defaultdict(list)
    for (text_id, position), row in sorted(word_rows.items()):
        if position != len(words[text_id]) + 1:
            raise ValueError(f"Non-consecutive word positions for passage {text_id}")
        words[text_id].append(row["ia"])

    with (raw / "teco_text_info.xlsx").open("rb") as stream:
        workbook = load_workbook(stream, read_only=True, data_only=True)
        try:
            columns = workbook.active.iter_rows(max_col=5, values_only=True)
            if next(columns) != ("id", "Grade", "Title", "Abb", "Subjective"):
                raise ValueError("Unexpected TECO workbook columns")
            workbook_rows = {
                int(i): (str(grade), title, name, float(rating))
                for i, grade, title, name, rating in columns
            }
        finally:
            workbook.close()
    if set(workbook_rows) != TEXT_IDS or set(words) != TEXT_IDS:
        raise ValueError("TECO metadata and word records must each cover all 30 passages")

    passages_report = []
    for (text_id,), row in sorted(text_rows.items()):
        grade, title, label, rating = workbook_rows[text_id]
        if (row["Grade"], row["short"], float(row["Subjective"])) != (grade, label, rating):
            raise ValueError(f"TECO workbook and published ratings disagree for passage {text_id}")
        title_matches = letters_and_numbers(titles[text_id]) == letters_and_numbers(title)
        if not title_matches and (text_id, titles[text_id], title) != (
            27, "Scottish Whiskey Industry", "The Rebirth of the Scottish Whiskey Industry",
        ):
            raise ValueError(f"Unexpected difference between PDF and workbook titles for passage {text_id}")
        if letters_and_numbers(passages[text_id]) != letters_and_numbers(" ".join(words[text_id])):
            raise ValueError(f"TECO PDF and ordered word records disagree for passage {text_id}")
        passages_report.append({
            "text_id": text_id,
            "grade": grade,
            "normalized_title_matches": title_matches,
            "author_word_count": int(row["number.words"]),
            "interest_area_token_count": len(words[text_id]),
            "comprehension_item_count": len(questions[text_id]),
        })
    return passages, passages_report


def audit(raw):
    for filename, expected in RELEASE_SHA256.items():
        with (raw / filename).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"{filename} differs from the audited release")

    texts = index(csv_rows(raw / "df_text.csv"), "id")
    readers = index(csv_rows(raw / "teco_participant_info.csv"), "subjectid")
    published = index(csv_rows(raw / "df_individual.csv"), "subject", "text")
    teco = index(csv_rows(raw / "teco_textmeasure.csv"), "subject", "text")
    expected_trials = {(reader, text) for (reader,) in readers for text in TEXT_IDS}
    if set(published) != expected_trials or set(teco) != expected_trials or set(texts) != {(i,) for i in TEXT_IDS}:
        raise ValueError("Incomplete or inconsistent reader × passage design")
    passages, materials = audit_materials(raw, texts)
    item_counts = {row["text_id"]: row["comprehension_item_count"] for row in materials}
    order = defaultdict(dict)
    ratings_by_text = defaultdict(list)
    missingness = Counter()
    accuracy = []
    for (reader, text), row in published.items():
        base = teco[reader, text]
        person = readers[reader,]
        if (row["acc"], row["Subjective"], row["pro"], base["book"]) != (
            base["acc"], base["subjective"], person["pro"], person["book"],
        ):
            raise ValueError("Trial endpoints, proficiency or counterbalancing set disagree across releases")
        visit = int(base["order"])
        if visit in order[reader]:
            raise ValueError("Duplicate presentation order")
        order[reader][visit] = text
        has_rating = row["Subjective"] != "NA"
        has_accuracy = row["acc"] != "NA"
        missingness[has_rating, has_accuracy] += 1
        if has_rating:
            score = int(row["Subjective"])
            if not 1 <= score <= 9:
                raise ValueError("Subjective rating lies outside the 1–9 scale")
            ratings_by_text[text].append(score)
        if has_accuracy:
            fraction = Decimal(row["acc"])
            n = item_counts[text]
            if min(abs(fraction - Decimal(k) / n) for k in range(n + 1)) > Decimal("0.0005"):
                raise ValueError(f"Accuracy is incompatible with the question count for passage {text}")
            accuracy.append(float(fraction))
    if any(visit_map.keys() != TEXT_IDS for visit_map in order.values()):
        raise ValueError("Presentation orders must cover consecutive trials 1–30 per reader")
    first_half_a = set(range(1, 6)) | set(range(11, 16)) | set(range(21, 26))
    expected_first_half = {"A": first_half_a, "B": TEXT_IDS - first_half_a}
    order_deviations = Counter(
        readers[reader,]["book"] for reader, visits in order.items()
        if {visits[visit] for visit in range(1, 16)} != expected_first_half[readers[reader,]["book"]]
    )

    rating_means = []
    for (text,), row in texts.items():
        rating_average = mean(ratings_by_text[text])
        if abs(float(row["Subjective"]) - rating_average) > 1e-8:
            raise ValueError(f"Cannot reproduce the published mean rating for passage {text}")
        rating_means.append(rating_average)

    levels = Counter(x for scores in ratings_by_text.values() for x in scores)
    proficiency = [int(row["pro"]) for row in readers.values()]
    return {
        "rating_source": "https://osf.io/t75nh/",
        "teco_source": "https://osf.io/wrvj3/",
        "files_sha256": RELEASE_SHA256,
        "participants": len(readers),
        "passages": len(texts),
        "trial_rows": len(published),
        "counterbalancing_sets_by_reader": dict(Counter(row["book"] for row in readers.values())),
        "counterbalancing_first_half_deviations_by_set": dict(order_deviations),
        "observed_ratings": sum(levels.values()),
        "rating_counts_by_level": dict(levels),
        "rating_mean_of_passage_means": mean(rating_means),
        "rating_sd_of_passage_means": stdev(rating_means),
        "observed_passage_accuracies": len(accuracy),
        "mean_of_observed_passage_accuracies": mean(accuracy),
        "median_of_observed_passage_accuracies": median(accuracy),
        "perfect_passage_accuracies": accuracy.count(1.0),
        "rating_and_accuracy": missingness[True, True],
        "rating_only": missingness[True, False],
        "accuracy_only": missingness[False, True],
        "both_missing": missingness[False, False],
        "proficiency_measured_score_range": [min(proficiency), max(proficiency)],
        "proficiency_mean": mean(proficiency),
        "proficiency_sd": stdev(proficiency),
        "passage_metadata": materials,
    }, passages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--passages-output", type=Path, help="Optional local JSONL of source passages")
    args = parser.parse_args()
    report, passages = audit(args.raw)
    if args.passages_output:
        with args.passages_output.open("w", encoding="utf-8") as stream:
            for text_id, text in sorted(passages.items()):
                stream.write(json.dumps({"text_id": text_id, "text": text}, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
