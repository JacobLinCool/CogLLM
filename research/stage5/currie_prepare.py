"""Rebuild Currie et al.'s released passage variants and final sense judgments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
from hashlib import file_digest
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from zipfile import ZipFile


PAPER = "https://doi.org/10.1080/10888438.2020.1831503"
PROJECT = "https://osf.io/sj28g/"
SOURCE_FILES = {
    "sense-judgments.csv": ("swjdf", "9f649f59e1eb517a562eb398979bb185d0cba81415da87ba8602682bddda2f85"),
    "reading-times.csv": ("qtcbv", "68b1b6eec799ff6dfbbd9ea06ccd5d9869789d32f66c81ac759dd583b3b22d5c"),
    "variable-descriptions.xlsx": ("tq36h", "c76c2cadf4d46eeb4b2b22d64ec242987dcb2cb1f6eb3e1447f1237155bef204"),
    "narrative-materials.docx": ("b73ks", "1c62b97572d0a671fd0f7780815328782553107b894bb356b07f7cac39400082"),
    "expository-materials.docx": ("3pswr", "27b085c6be95d165484a002f1b4b13ddd6322d104535dcb349406305fb3fae48"),
}
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
NUMBERED_HEADING = re.compile(r"(\d+)\.?\s+(.+)")
SENTENCES = 8
CONDITIONS = {"near", "far", "consistent"}
GROUPS = {"monolingual", "bilingual"}
RT_TOLERANCE_MS = Decimal("0.0001")
PUBLISHED_MEAN_ACCURACY = {
    ("narrative", "consistent"): (0.91, 0.80),
    ("narrative", "near"): (0.70, 0.60),
    ("narrative", "far"): (0.72, 0.58),
    ("expository", "consistent"): (0.90, 0.75),
    ("expository", "near"): (0.41, 0.39),
    ("expository", "far"): (0.42, 0.39),
}


def verify_release(source: Path) -> dict:
    files = {}
    for name, (identifier, expected) in SOURCE_FILES.items():
        with (source / name).open("rb") as handle:
            actual = file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Source revision mismatch for {name}: {actual}")
        files[name] = {"url": f"https://osf.io/download/{identifier}/", "sha256": actual}
    return files


def docx_paragraphs(path: Path) -> list[tuple[str, bool]]:
    def text(element):
        return "".join(part.text or "" for part in element.iter(WORD_NS + "t"))

    paragraphs = []
    with ZipFile(path) as document:
        xml = ET.fromstring(document.read("word/document.xml"))
    for paragraph in xml.findall(f"{WORD_NS}body/{WORD_NS}p"):
        content = text(paragraph).strip()
        if content:
            marked = any(
                run.find(f"{WORD_NS}rPr/{WORD_NS}u") is not None and text(run).strip()
                for run in paragraph.findall(WORD_NS + "r")
            )
            paragraphs.append((content, marked))
    return paragraphs


def stimulus(genre, item, title, condition, paragraphs, key):
    if len(paragraphs) != SENTENCES:
        raise ValueError(f"Expected eight stimulus sentences for {genre} item {item}")
    if re.fullmatch(r"SENSE QUESTION: .+\?\s+NO", key) is None:
        raise ValueError(f"Unexpected sense key for {genre} item {item}")
    anchors = [index for index, (_, marked) in enumerate(paragraphs) if marked]
    if len(anchors) != 2:
        raise ValueError(f"Unexpected underlining for {genre} item {item}, {condition}")

    sentences = [content for content, _ in paragraphs]
    if (genre, item, condition) == ("expository", 8, "far"):
        # The reading-time denominator check independently verifies this numbering removal.
        prefixes = [f"{index}. " for index in range(1, SENTENCES + 1)]
        if not all(s.startswith(p) for s, p in zip(sentences, prefixes, strict=True)):
            raise ValueError("Unexpected presentation numbering for expository item 8")
        sentences = [s.removeprefix(p) for s, p in zip(sentences, prefixes, strict=True)]

    segments, start = [], 0
    for index, sentence in enumerate(sentences):
        end = start + len(sentence)
        segments.append({"index": index, "start": start, "end": end})
        start = end + 1
    family = f"currie/{genre}/{item:02d}"
    return {
        "id": f"{family}/{condition}",
        "family_id": family,
        "item": item,
        "title": title,
        "genre": genre,
        "condition": condition,
        "text": " ".join(sentences),
        "segments": segments,
        "support_index": anchors[0],
        "critical_index": anchors[1],
    }


def parse_material(path: Path, genre: str) -> list[dict]:
    paragraphs = docx_paragraphs(path)
    test_start = next(i for i, (value, _) in enumerate(paragraphs) if value == "Test Items")
    controls_start = next(
        i for i, (value, _) in enumerate(paragraphs) if value.lower().rstrip(":") == "consistent items"
    )
    conflicts = paragraphs[test_start + 1:controls_start]
    variants = [(index, value) for index, (value, _) in enumerate(conflicts) if value in {"NEAR", "FAR"}]
    if [value for _, value in variants] != ["NEAR", "FAR"] * 12:
        raise ValueError(f"Expected 12 near/far pairs in {path.name}")
    stimuli = []
    scan_start = 0
    for item, pair_start in enumerate(range(0, len(variants), 2), start=1):
        near_start, far_start = (position for position, _ in variants[pair_start:pair_start + 2])
        heading = [match for value, _ in conflicts[scan_start:near_start]
                   if (match := NUMBERED_HEADING.fullmatch(value))]
        if len(heading) != 1 or int(heading[0][1]) != item or far_start - near_start != SENTENCES + 1:
            raise ValueError(f"Material pair {item} could not be aligned in {path.name}")
        sense_index = far_start + SENTENCES + 1
        for start, condition in ((near_start, "near"), (far_start, "far")):
            stimuli.append(stimulus(
                genre, item, heading[0][2], condition, conflicts[start + 1:start + SENTENCES + 1],
                conflicts[sense_index][0],
            ))
        if not conflicts[sense_index + 1][0].startswith(("COMP QUESTION:", "COMPREHENSION QUESTION:")):
            raise ValueError(f"Missing comprehension check for {genre} {item}")
        scan_start = sense_index + 2

    return stimuli


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def material_key(row: dict) -> tuple:
    return row["Text_Type"], int(row["Test"]), row["Condition"]


def trial_key(row: dict) -> tuple:
    return row["Subject"], *material_key(row)


def reading_time_audit(source: Path, judgments: dict, variants: dict) -> dict:
    roles, values = defaultdict(set), defaultdict(list)
    times = read_csv(source / "reading-times.csv")
    for row in times:
        key, role = trial_key(row), row["sentence.type"]
        if key not in judgments or key[-1] == "consistent" or role in roles[key]:
            raise ValueError(f"Unexpected or duplicate reading-time row for {key}")
        if row["text"] != judgments[key]["text"] or any(
            row[column] != judgments[key][column] for column in ("Question.ACC", "Language_Status")
        ):
            raise ValueError(f"Reading-time record disagrees with judgment for {key}")
        roles[key].add(role)
        duration = Decimal(row["msperword"])
        values[(material_key(row), role)].append(duration)
        if role == "INCONmsperword" and duration != Decimal(judgments[key]["CRITICALmsperword"]):
            raise ValueError(f"Critical-sentence reading-time columns disagree for {key}")
    if set(roles) != {key for key in judgments if key[-1] != "consistent"} or any(
        names != {"INCONmsperword", "COMPmsperword"} for names in roles.values()
    ) or len(times) != 8496:
        raise ValueError("Reading-time release does not cover the expected two rows per inconsistent trial")

    max_error = Decimal(0)
    for (key, role), durations in values.items():
        row = variants[key]
        segment = row["segments"][row["critical_index"] - (role == "COMPmsperword")]
        word_count = len(row["text"][segment["start"]:segment["end"]].split())
        errors = [max(abs(v * n - round(v * n)) for v in durations) for n in range(1, word_count + 1)]
        if errors[-1] > RT_TOLERANCE_MS or any(e <= RT_TOLERANCE_MS for e in errors[:-1]):
            raise ValueError(f"Reading-time denominator disagrees with published sentence word count for {key} {role}")
        max_error = max(errors[-1], max_error)
    return {
        "rows": len(times), "trials_exactly_linked": len(roles), "unmatched_rows": 0,
        "critical_times_match_judgment_table_exactly": True,
        "comparison_sentence": "Sentence immediately before the second underlined sentence (n−1).",
        "word_count_checks": len(values),
        "word_count_check": "Minimum integer denominator for recorded ms/word to recover whole milliseconds matches the whitespace word count, for both roles in each variant.",
        "max_integer_reconstruction_error_ms": float(max_error),
    }


def pair_audit(stimuli: list[dict]) -> dict:
    families = defaultdict(dict)
    for item in stimuli:
        families[item["family_id"]][item["condition"]] = item
    pure_reorder, differences, anchor_info = [], [], {}
    for family, variants in families.items():
        near, far = variants["near"], variants["far"]

        def sentences(row):
            return [row["text"][s["start"]:s["end"]] for s in row["segments"]]

        near_sentences, far_sentences = sentences(near), sentences(far)
        near_distance = near["critical_index"] - near["support_index"]
        far_distance = far["critical_index"] - far["support_index"]
        if near_distance >= far_distance:
            raise ValueError(f"Near/far anchor distance does not increase for {family}")
        if Counter(near_sentences) == Counter(far_sentences):
            pure_reorder.append(family)
        else:
            differences.append(family)
        anchor_info[family] = {
            "near_distance_in_sentences": near_distance,
            "far_distance_in_sentences": far_distance,
            "support_sentence_exact_match": near_sentences[near["support_index"]] == far_sentences[far["support_index"]],
            "critical_sentence_exact_match": near_sentences[near["critical_index"]] == far_sentences[far["critical_index"]],
        }
    if (len(pure_reorder), len(differences), len(families)) != (20, 4, 24):
        raise ValueError("Unexpected exact-wording coverage of matched near/far pairs")
    return {"pure_reorder_families": pure_reorder, "surface_change_families": differences, "anchors": anchor_info}


def rebuild(source: Path, output: Path, audit_path: Path) -> None:
    files = verify_release(source)
    stimuli = sorted(
        [item for genre in ("narrative", "expository")
         for item in parse_material(source / f"{genre}-materials.docx", genre)],
        key=lambda row: row["id"],
    )
    lookup = {(row["genre"], row["item"], row["condition"]): row for row in stimuli}
    if len(lookup) != len(stimuli) or len(stimuli) != 48:
        raise ValueError("Expected 48 explicitly numbered near/far variants")

    judgments, seen_reader_items = {}, set()
    tally = {row["id"]: dict.fromkeys(("yes", "no"), 0) for row in stimuli}
    grouped = defaultdict(lambda: defaultdict(lambda: dict.fromkeys(("yes", "no"), 0)))
    by_condition = defaultdict(lambda: dict.fromkeys(("yes", "no", "correct", "incorrect"), 0))
    by_group = defaultdict(Counter)
    rows = read_csv(source / "sense-judgments.csv")
    for row in rows:
        key = trial_key(row)
        _, genre, number, condition = key
        control = condition == "consistent"
        reader_item = key[:3]
        accuracy, group = row["Question.ACC"], row["Language_Status"]
        if (
            reader_item in seen_reader_items or row["text"] != f"{number}{genre}"
            or accuracy not in {"0", "1"} or group not in GROUPS or condition not in CONDITIONS
            or (control and not 13 <= number <= 18)
        ):
            raise ValueError(f"Unverified or duplicate released sense response {key}")
        seen_reader_items.add(reader_item)
        judgments[key] = row
        correct = accuracy == "1"
        made_sense = "yes" if correct == control else "no"
        if not control:
            item = lookup[material_key(row)]
            tally[item["id"]][made_sense] += 1
            grouped[item["id"]][group][made_sense] += 1
        by_condition[condition][made_sense] += 1
        by_condition[condition]["correct" if correct else "incorrect"] += 1
        by_group[genre, condition, group][correct] += 1
    if len(rows) != 6372 or len({key[0] for key in judgments}) != 179:
        raise ValueError("Released trial or reader count differs from the verified source")
    model_trials = sum(sum(counts.values()) for counts in tally.values())
    if model_trials != 4248:
        raise ValueError("Expected 4,248 released near/far responses in model payload")
    for variant, counts in tally.items():
        if not sum(counts.values()) or set(grouped[variant]) != GROUPS:
            raise ValueError(f"Incomplete response counts for {variant}")
        if sum(counts.values()) != sum(sum(count.values()) for count in grouped[variant].values()):
            raise ValueError(f"Group response counts differ for {variant}")
    reconstructed = {}
    for (genre, condition), expected in PUBLISHED_MEAN_ACCURACY.items():
        groups = ("monolingual", "bilingual")
        accuracies = [by_group[genre, condition, g][True] / by_group[genre, condition, g].total() for g in groups]
        if tuple(round(value, 2) for value in accuracies) != expected:
            raise ValueError(f"Released means disagree with paper Table 3 for {genre} {condition}")
        reconstructed[f"{genre}/{condition}"] = dict(zip(groups, accuracies, strict=True))

    timing = reading_time_audit(source, judgments, lookup)
    pairs = pair_audit(stimuli)
    observations = [{
        "stimulus_id": item["id"],
        "responses_made_sense": tally[item["id"]],
        "responses_made_sense_by_language_status": dict(sorted(grouped[item["id"]].items())),
    } for item in stimuli]
    output.mkdir(parents=True, exist_ok=True)
    for name, content in (("stimuli.jsonl", stimuli), ("observations.jsonl", observations)):
        with (output / name).open("w", encoding="utf-8") as handle:
            for data in content:
                handle.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "stimuli.jsonl").open("rb") as handle:
        stimuli_digest = file_digest(handle, "sha256").hexdigest()
    with (output / "observations.jsonl").open("rb") as handle:
        observations_digest = file_digest(handle, "sha256").hexdigest()
    report = {
        "paper": PAPER, "project": PROJECT, "data_component": "https://osf.io/g9pyc/",
        "materials_component": "https://osf.io/vp45x/", "data_and_materials_license": "CC BY 4.0",
        "source_files": files, "stimuli": len(stimuli), "passage_families": 24,
        "sentence_segments": sum(len(row["segments"]) for row in stimuli),
        "released_response_rows": len(rows), "model_input_response_rows": model_trials,
        "reader_ids": len({key[0] for key in judgments}),
        "response_counts": dict(sorted(by_condition.items())),
        "table_3_genre_condition_language_accuracy": reconstructed,
        "response_key": "Published binary accuracy records the final passage sense question. The material keys determine which Yes/No response was scored correct.",
        "visibility": "The eight sentences were displayed successively in self-paced reading. The sense question appeared after the final sentence.",
        "target_spans": "The first and second underlined sentences are the author-marked premise and critical sentence. The reader did not choose a sentence or report token-level onset.",
        "material_join": "DOCX item numbers 1–12 per genre align with CSV Test and Text_Type for both near and far conditions.",
        "control_item_linkage": "Consistent controls use IDs 13–18 in the judgment file, and their material headings lack these IDs. The audit reports their response totals and keeps model inputs to the precisely numbered near/far pairs.",
        "material_formatting": "Removed numbered-list prefixes on expository item 8 FAR. Its critical sentence has 16 stimulus words by the recorded reading-time denominator, with the listing number excluded.",
        "near_far_pairs": pairs, "reading_time_crosscheck": timing,
        "output_sha256": {"stimuli.jsonl": stimuli_digest, "observations.jsonl": observations_digest},
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "output", "audit"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    args = parser.parse_args()
    rebuild(args.source, args.output, args.audit)
