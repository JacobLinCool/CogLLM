"""Prepare Frick and Schüler Experiment 1 text and separate sentence judgments."""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

from openpyxl import load_workbook


ROOT = Path(".cache/cogllm/semantic-tracking/frick-consistency")
WORKBOOK = "data_final_experiment_1.xlsx"
SOURCES = {
    WORKBOOK: (
        "https://osf.io/download/hp9sg/",
        "439a63a228d06257661c76ed5b9b96ae3c8f57c4ac08b393844a9ef9cf517ba4",
    ),
    "Experimental_Texts_Experiment_1.pdf": (
        "https://osf.io/download/3p2d7/",
        "c249217876c2c46a98a706babd3bc9644592e6bdb47abc4ca72c4fc677e13b5d",
    ),
}
PAPER = "https://doi.org/10.3758/s13421-026-01900-7"
TEXT_ONLY = ("01", "02")
STORIES = (*range(1, 13), *range(14, 22))
READER_COUNT = 180
DISPLAY_ID = re.compile(r"([0-9]{2})(0[1-4])([0-9]{2})([ts]?)")
DISPLAY_FIELDS = (
    "Satz", "Bild", "ID", "LoopSentences.thisN", "participant", "StoryKey.keys", "StoryKey.rt",
)


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def read_workbook(path):
    variants = defaultdict(dict)
    readings = defaultdict(list)
    with path.open("rb") as stream:
        workbook = load_workbook(stream, read_only=True, data_only=True)
        try:
            sheet, = workbook.worksheets
            # The XLSX declares A1 as its used range even though later cells are present.
            sheet.reset_dimensions()
            rows = iter(sheet.values)
            header = next(rows)
            if len(header) != 133 or any(header.count(field) != 1 for field in DISPLAY_FIELDS):
                raise ValueError("Unexpected column layout in the pinned workbook")
            indexes = [header.index(field) for field in DISPLAY_FIELDS]

            for row_number, row in enumerate(rows, start=1):
                sentence, picture, raw_id, source_ordinal, reader, key, seconds = (row[i] for i in indexes)
                if sentence in (None, "", "NA"):
                    if key not in (None, "", "NA"):
                        raise ValueError("A reading judgment has no corresponding sentence")
                    continue
                match = DISPLAY_ID.fullmatch(str(raw_id).zfill(6))
                if match is None or type(reader) is not int or key not in ("k", "d"):
                    raise ValueError("A sentence row has an invalid display ID, reader, or judgment")
                story_code, condition, ordinal, role = match.groups()
                story, index = int(story_code), int(ordinal) - 1
                trajectory = readings[reader, story, condition]
                if story not in STORIES or index != len(trajectory) or index != int(source_ordinal):
                    raise ValueError("Sentence ordinals disagree with the workbook presentation order")
                if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("A sentence response has invalid source timing")
                if not isinstance(sentence, str) or sentence != sentence.strip() or any(
                    character in sentence for character in ("\n", "\r", "\t")
                ):
                    raise ValueError("The source sentence does not satisfy the exact-text whitespace contract")
                if condition in TEXT_ONLY and picture != "trans.png":
                    raise ValueError("A text-only condition contains a visible illustration")

                display = sentence, role
                if variants[story, condition].setdefault(index, display) != display:
                    raise ValueError("Readers in the same story condition received different sentence text or roles")
                trajectory.append((key, seconds))
        finally:
            workbook.close()

    if row_number != 72_723 or sum(map(len, readings.values())) != 56_160:
        raise ValueError("Unexpected row coverage in the pinned workbook")
    expected_variants = {(story, f"{condition:02d}") for story in STORIES for condition in range(1, 5)}
    if variants.keys() != expected_variants:
        raise ValueError("The workbook lacks a story and condition combination")
    reader_counts = defaultdict(Counter)
    for (reader, story, condition), trajectory in readings.items():
        reader_counts[reader][condition] += 1
        if len(trajectory) != len(variants[story, condition]):
            raise ValueError("A reading trajectory ended before its last displayed sentence")
    if len(reader_counts) != READER_COUNT or any(
        sorted(counts.items()) != [(f"{condition:02d}", 5) for condition in range(1, 5)]
        for counts in reader_counts.values()
    ) or len({(reader, story) for reader, story, _ in readings}) != len(readings):
        raise ValueError("Every reader must encounter each story once and five stories per condition")
    return variants, readings


def remember_stimulus(stimuli, text, stops):
    identifier = sha256(text.encode("utf-8")).hexdigest()
    row = {"id": identifier, "text": text, "stops": stops}
    if stimuli.setdefault(identifier, row) != row:
        raise ValueError("A literal text hash has conflicting stop coordinates")
    return identifier


def build_stimuli(variants):
    stimuli, metadata, pairs = {}, [], []
    for story in STORIES:
        first, second = (sorted(variants[story, condition].items()) for condition in TEXT_ONLY)
        if [(i, role) for i, (_, role) in first] != [(i, role) for i, (_, role) in second]:
            raise ValueError("Matched story versions have unequal sentence layouts")
        changed, = [i for (i, before), (_, after) in zip(first, second, strict=True) if before != after]
        target, = [i for i, (_, role) in first if role == "t"]
        spill, = [i for i, (_, role) in first if role == "s"]
        if not 7 <= target - changed <= 10 or spill != target + 1:
            raise ValueError("Matched versions must differ once before their identical target and spillover")

        pair_ids = []
        for condition, version in zip(TEXT_ONLY, (first, second), strict=True):
            sentences, exact_text, stops = [], [], []
            start = 0
            for _, (text, _) in version:
                stop = start + len(text)
                single_id = remember_stimulus(stimuli, text, [len(text)])
                sentences.append({"start": start, "stop": stop, "single_id": single_id})
                exact_text.append(text)
                stops.append(stop)
                start = stop + 2
            identifier = remember_stimulus(stimuli, "\n\n".join(exact_text), stops)
            pair_ids.append(identifier)
            metadata.append({
                "id": identifier, "story": story, "condition": condition,
                "target_index": target, "changed_index": changed, "sentences": sentences,
            })
        pairs.append({"story": story, "ids": pair_ids, "changed_index": changed, "target_index": target})

    if len({row["id"] for row in metadata}) != 40 or len(stimuli) - len(metadata) != 332:
        raise ValueError("Unexpected number of unique full stories or sentence controls")
    return [stimuli[key] for key in sorted(stimuli)], metadata, pairs


def build_outcomes(readings, metadata):
    source_readers = sorted({reader for reader, _, _ in readings})
    Random(43).shuffle(source_readers)
    pseudonyms = {reader: f"frick-reader-{index:03d}" for index, reader in enumerate(source_readers, start=1)}
    model_ids = {(row["story"], row["condition"]): row["id"] for row in metadata}
    outcomes, counts = [], defaultdict(Counter)
    for (reader, story, condition), trajectory in readings.items():
        if condition not in TEXT_ONLY:
            continue
        keys, times = map(list, zip(*trajectory, strict=True))
        outcomes.append({
            "reader_id": pseudonyms[reader], "id": model_ids[story, condition],
            "keys": keys, "rt_seconds": times,
        })
        counts[condition]["trajectories"] += 1
        counts[condition]["sentence_judgments"] += len(keys)
        counts[condition]["k"] += keys.count("k")
        counts[condition]["d"] += keys.count("d")

    if [(counts[c]["trajectories"], counts[c]["sentence_judgments"]) for c in TEXT_ONLY] != [
        (900, 14_066), (900, 14_074),
    ]:
        raise ValueError("Unexpected coverage of the text-only conditions")
    outcomes.sort(key=lambda row: (row["reader_id"], row["id"]))
    return outcomes, counts


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def prepare(raw, output):
    if output.exists():
        raise FileExistsError(output)
    for name, (_, expected) in SOURCES.items():
        if digest(raw / name) != expected:
            raise ValueError(f"Hash mismatch for the pinned public source {name}")
    variants, readings = read_workbook(raw / WORKBOOK)
    stimuli, metadata, pairs = build_stimuli(variants)
    outcomes, counts = build_outcomes(readings, metadata)
    pair_distances = [pair["target_index"] - pair["changed_index"] for pair in pairs]

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".frick-consistency-") as directory:
        staged = Path(directory) / "prepared"
        staged.mkdir(mode=0o700)
        tables = {"stimuli.jsonl": stimuli, "metadata.jsonl": metadata, "outcomes.jsonl": outcomes}
        for name, rows in tables.items():
            write_jsonl(staged / name, rows)
            os.chmod(staged / name, 0o600)
        audit = {
            "paper": PAPER, "experiment": 1,
            "sources": {name: {"url": url, "sha256": checksum} for name, (url, checksum) in SOURCES.items()},
            "source_rights": {
                "paper_license": "CC BY 4.0", "osf_metadata": "https://api.osf.io/v2/nodes/pg67h/",
                "osf_license": None, "workbook_and_stimulus_reuse": "unconfirmed",
            },
            "source_code_sha256": digest(Path(__file__)),
            "counts": {
                "readers": READER_COUNT, "story_pairs": len(pairs), "full_stories": len(metadata),
                "sentence_controls": len(stimuli) - len(metadata), "trajectories": len(outcomes),
                "source_worksheet_rows_excluding_header": 72_723,
                "source_sentence_judgments_all_conditions": sum(map(len, readings.values())),
                "by_condition": counts,
            },
            "pair_checks": {
                "conditions_in_id_order": TEXT_ONLY, "pairs": pairs,
                "changed_sentences_per_pair": 1, "target_and_spillover_identical": True,
                "changed_to_target_sentence_distance": [min(pair_distances), max(pair_distances)],
            },
            "rendering": {
                "separator": "\n\n", "coordinates": "half-open Unicode character offsets",
                "ordinal_base": 0, "ids": "SHA-256 hex of the exact UTF-8 text",
            },
            "responses": {
                "keys": {"k": "consistent", "d": "inconsistent"},
                "rt_seconds": "raw StoryKey.rt", "first_sentence_key": "retained as instructed trial response",
            },
            "output_sha256": {name: digest(staged / name) for name in tables},
        }
        audit_path = staged / "source-audit.json"
        with audit_path.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(audit, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.chmod(audit_path, 0o600)
        staged.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "prepared")
    args = parser.parse_args()
    prepare(args.raw, args.output)
