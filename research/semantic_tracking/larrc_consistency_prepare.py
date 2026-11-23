"""Prepare the matched Currie et al. reading texts and separate end-of-text responses."""

import argparse
from collections import Counter
import csv
from hashlib import file_digest, sha256
import json
from pathlib import Path
from random import Random
import re
from tempfile import TemporaryDirectory
import unicodedata
from xml.etree import ElementTree
from zipfile import ZipFile


PAPER = "https://doi.org/10.1080/10888438.2020.1831503"
SOURCES = {
    "narrative": ("https://osf.io/download/b73ks/", "1c62b97572d0a671fd0f7780815328782553107b894bb356b07f7cac39400082"),
    "expository": ("https://osf.io/download/3pswr/", "27b085c6be95d165484a002f1b4b13ddd6322d104535dcb349406305fb3fae48"),
    "accuracy": ("https://osf.io/download/swjdf/", "9f649f59e1eb517a562eb398979bb185d0cba81415da87ba8602682bddda2f85"),
}
GENRES = ("narrative", "expository")
CONDITIONS = ("near", "far")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SEPARATOR = "\n\n"
CHANGED_FAMILIES = {("expository", 6), ("expository", 11), ("narrative", 10), ("narrative", 12)}
CSV_FIELDS = ("Subject", "Test", "Text_Type", "Condition", "Question.ACC", "Language_Status",
              "CRITICALmsperword", "text", "TOWRE.SWE", "cTOWRE.SWE")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def paragraphs(path):
    with ZipFile(path) as document:
        root = ElementTree.fromstring(document.read("word/document.xml"))
    rows = []
    for paragraph in root.iter(W + "p"):
        text = "".join(node.text or "" for node in paragraph.iter(W + "t")).strip()
        if not text:
            continue
        marked = any(
            any(node.text and node.text.strip() for node in run.iter(W + "t"))
            and any(node.get(W + "val") == "single" for node in run.iter(W + "u"))
            for run in paragraph.iter(W + "r")
        )
        rows.append((text, marked))
    return rows


def read_versions(path, genre):
    rows = paragraphs(path)
    test, = (i for i, (text, _) in enumerate(rows) if text.casefold() == "test items")
    controls, = (i for i, (text, _) in enumerate(rows) if text.casefold().rstrip(":") == "consistent items")
    near = [i for i in range(test + 1, controls) if rows[i][0] == "NEAR"]
    if len(near) != 12:
        raise ValueError(f"Expected 12 explicitly numbered near/far pairs for {genre}")
    versions = {}
    for item, start in enumerate(near, 1):
        # The source includes an editorial note before the Portuguese man o' war variants.
        note_present = rows[start - 1][0].startswith("NOTE:")
        if note_present != (genre == "expository" and item == 11):
            raise ValueError("Unexpected stimulus material before a near block")
        heading = rows[start - 1 - note_present][0]
        match = re.fullmatch(r"([0-9]+)\.?\s+(.+)", heading)
        if (not match or int(match[1]) != item or rows[start + 9][0] != "FAR"
                or not re.fullmatch(r"SENSE QUESTION: .+\? NO", rows[start + 18][0])):
            raise ValueError(f"Unexpected near/far markup for {genre} item {item}")
        for condition, first in zip(CONDITIONS, (start + 1, start + 10), strict=True):
            block = rows[first:first + 8]
            if (genre, item, condition) == ("expository", 8, "far"):
                # The Birds of paradise far variant uses literal list numerals in the text runs.
                numbered = []
                for index, (text, marked) in enumerate(block, 1):
                    prefix = f"{index}. "
                    if not text.startswith(prefix):
                        raise ValueError("Expected eight explicit ordered list numerals")
                    numbered.append((text.removeprefix(prefix), marked))
                block = numbered
            sentences = [text for text, _ in block]
            anchors = [i for i, (_, marked) in enumerate(block, 1) if marked]
            if (len(sentences) != 8 or len(anchors) != 2 or any(
                    not sentence or sentence != sentence.strip() or any(c in sentence for c in "\n\r\t")
                    for sentence in sentences)):
                raise ValueError(f"Expected eight sentences and two text-anchored marks for {genre} item {item}")
            versions[(genre, item, condition)] = sentences, anchors
    return versions


def comparison_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("’", "'").split())


def build_stimuli(versions):
    stimuli, layout, lookup, changed = [], [], {}, set()
    for genre in GENRES:
        for item in range(1, 13):
            near, far = [versions[genre, item, condition] for condition in CONDITIONS]
            same_inventory = [Counter(map(comparison_text, sentences)) for sentences, _ in (near, far)]
            strict_pair = same_inventory[0] == same_inventory[1]
            same_target = comparison_text(near[0][near[1][1] - 1]) == comparison_text(far[0][far[1][1] - 1])
            if same_target != ((genre, item) != ("narrative", 10)):
                raise ValueError("The published target-sentence correspondence changed")
            if not strict_pair:
                changed.add((genre, item))
            for condition, (sentences, (antecedent, target)) in zip(CONDITIONS, (near, far), strict=True):
                distance = target - antecedent - 1
                if not (1 <= distance <= 2 if condition == "near" else 3 <= distance <= 5):
                    raise ValueError(f"Unexpected antecedent–target distance for {genre} item {item} {condition}")
                text = SEPARATOR.join(sentences)
                identifier = sha256(text.encode("utf-8")).hexdigest()
                start, spans = 0, []
                for sentence in sentences:
                    stop = start + len(sentence)
                    spans.append({"start": start, "stop": stop})
                    start = stop + len(SEPARATOR)
                stops = [span["stop"] for span in spans]
                stimuli.append({"id": identifier, "text": text, "stops": stops})
                layout.append({
                    "id": identifier, "family": f"{genre}-{item:02d}", "genre": genre, "item": item,
                    "condition": condition, "antecedent_sentence": antecedent, "target_sentence": target,
                    "same_sentence_inventory": strict_pair, "same_target_sentence": same_target, "sentences": spans,
                })
                lookup[(genre, item, condition)] = identifier
    if changed != CHANGED_FAMILIES or len({row["id"] for row in stimuli}) != 48:
        raise ValueError("Unexpected exact-order pair inventory or duplicate rendered text")
    return stimuli, layout, lookup


def read_outcomes(path, lookup):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS:
            raise ValueError("Unexpected public response header")
        rows = list(reader)
    participants = sorted({row["Subject"] for row in rows})
    if len(participants) != 179 or "" in participants:
        raise ValueError("Unexpected number of published readers")
    Random(43).shuffle(participants)
    ids = {raw: f"larrc-reader-{i:03d}" for i, raw in enumerate(participants, 1)}
    counts, retained_counts = Counter(), Counter()
    outcomes, seen = [], set()
    for row in rows:
        participant, test, genre, condition, correct = (row[key] for key in CSV_FIELDS[:5])
        key = (participant, genre, test)
        if (key in seen or genre not in GENRES or not test.isascii() or not test.isdecimal()
                or row["text"] != test + genre or correct not in {"0", "1"}):
            raise ValueError("Invalid or repeated participant/item outcome")
        seen.add(key)
        counts[genre, condition, correct] += 1
        stimulus_key = (genre, int(test), condition)
        if condition == "consistent" and 13 <= int(test) <= 18:
            continue
        if stimulus_key not in lookup:
            raise ValueError("Retained response does not reference a published near/far variant")
        # For either inconsistent condition, the correct answer grounds an explicit "No" report.
        outcomes.append({"id": lookup[stimulus_key], "reader": ids[participant], "report": int(correct)})
        retained_counts[stimulus_key, correct] += 1

    expected_yes = {("narrative", "near"): 696, ("narrative", "far"): 693, ("narrative", "consistent"): 910,
                    ("expository", "near"): 423, ("expository", "far"): 431, ("expository", "consistent"): 867}
    expected = {(genre, condition, "1"): count for (genre, condition), count in expected_yes.items()}
    expected.update({(genre, condition, "0"): {"narrative": 1068, "expository": 1056}[genre] - count
                     for (genre, condition), count in expected_yes.items()})
    if (counts != expected or len(rows) != 6372 or len(outcomes) != 4248
            or retained_counts.keys() != {(key, val) for key in lookup for val in {"0", "1"}}):
        raise ValueError("The response table does not match the audited release counts")
    outcomes.sort(key=lambda row: (row["reader"], row["id"]))
    by_condition = [{"genre": genre, "condition": condition, "no": expected_yes[genre, condition],
                     "trials": counts[genre, condition, "0"] + counts[genre, condition, "1"]}
                    for genre in GENRES for condition in CONDITIONS]
    return outcomes, by_condition


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def prepare(narrative, expository, accuracy, output):
    if output.exists():
        raise FileExistsError(output)
    for name, path in {"narrative": narrative, "expository": expository, "accuracy": accuracy}.items():
        if digest(path) != SOURCES[name][1]:
            raise ValueError(f"SHA-256 mismatch for the pinned {name} source")
    versions = read_versions(narrative, "narrative") | read_versions(expository, "expository")
    stimuli, layout, lookup = build_stimuli(versions)
    outcomes, by_condition = read_outcomes(accuracy, lookup)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".larrc-consistency-") as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir(mode=0o700)
        tables = {"stimuli.jsonl": stimuli, "layout.jsonl": layout, "outcomes.jsonl": outcomes}
        for name, table in tables.items():
            write_jsonl(stage / name, table)
        receipt = {
            "schema": 1, "paper": PAPER, "source_code_sha256": digest(Path(__file__)),
            "sources": {name: {"url": url, "sha256": checksum} for name, (url, checksum) in SOURCES.items()},
            "source_rights": {"license": "CC BY 4.0", "materials": "https://osf.io/vp45x/", "data": "https://osf.io/g9pyc/"},
            "counts": {"families": 24, "strict_sentence_inventory_pairs": 20, "inconsistent_variants": len(stimuli),
                       "sentence_stops": sum(len(row["stops"]) for row in stimuli), "published_trials": 6372,
                       "excluded_consistent_trials": 6372 - len(outcomes), "retained_trials": len(outcomes),
                       "retained_readers": len({row["reader"] for row in outcomes}), "by_condition": by_condition},
            "rendering": {"separator": SEPARATOR, "coordinates": "half-open Unicode code-point indices",
                          "sentence_ordinal_base": 1, "ids": "SHA-256 of rendered UTF-8 text",
                          "excluded": "practice, prompts, source headings, list markers, editorial notes, and consistent controls"},
            "outcome": {"when": "after the full text", "report": "1 for explicit No from Question.ACC=1 on inconsistent trials"},
            "output_sha256": {name: digest(stage / name) for name in tables},
        }
        (stage / "source-audit.json").write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
                                                 encoding="utf-8")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("narrative", "expository", "accuracy", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    prepare(**vars(parser.parse_args()))
