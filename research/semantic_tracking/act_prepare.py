"""Reconstruct the ACT stimulus paths and separate first-six historical outcomes."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


ARTICLE = "https://doi.org/10.5334/joc.254"
PROJECT = "https://osf.io/et5kb/"
PDF_NAME = "ACT_word_lists.pdf"
SAV_NAME = "ACT_exp.sav"
PDF_SHA256 = "2fcb40195f5dfa0370d1615c82fa1bb1621bb7350abe574df54775bd3fdac3ad"
SAV_SHA256 = "e10e2d971d9b9f89719e7409d49c9b7da1e79233186cd5e05d951c36b0fa9496"
PDF_URL = "https://files.us.osf.io/v1/resources/et5kb/providers/osfstorage/6284cc9c47df854ff26ef906"
SAV_URL = "https://files.us.osf.io/v1/resources/et5kb/providers/osfstorage/62666ba85bb1ac1b2d460e95"
LISTS = 20
READERS = 100
FULL_HORIZON = 15
HORIZON = 6
FOLDS = 5
FOLD_SEED = "cogllm-act-material-folds-v1"
SEPARATOR = "\n"
LIST_LINE = re.compile(r"\s*List\s+(\d+):\s*(\S+)\s+\(")
CUE_LINE = re.compile(r"\s*Clue\s+(\d+)\s+(.+?)\s+(\d+\.\d+)\s*")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def binding(path: Path, expected: str, url: str) -> dict:
    if file_sha256(path) != expected:
        raise ValueError(f"Unexpected SHA-256 for official source {path.name}")
    return {"version": 1, "url": url, "bytes": path.stat().st_size, "sha256": expected}


def key(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def identifier(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}\0{value}".encode()).hexdigest()[:32]


@dataclass(frozen=True)
class Material:
    number: int
    target: str
    cues: tuple[str, ...]

    @property
    def family(self) -> str:
        return identifier("cogllm-act-family-v1", self.target)

    def stimulus(self, horizon: int) -> dict:
        stops = []
        text = ""
        for cue in self.cues[:horizon]:
            text = f"{text}{SEPARATOR}{cue}" if text else cue
            stops.append(len(text))
        return {"id": identifier("cogllm-act-input-v1", text), "text": text, "stops": stops}


def parse_material(text: str) -> list[Material]:
    lists: list[tuple[int, str, list[str], list[Decimal]]] = []
    for line in text.splitlines():
        header = LIST_LINE.match(line)
        cue = CUE_LINE.fullmatch(line)
        if header:
            lists.append((int(header[1]), key(header[2]), [], []))
        elif cue:
            if not lists or int(cue[1]) != len(lists[-1][2]) + 1:
                raise ValueError("A clue is missing from its numbered source list")
            lists[-1][2].append(cue[2])
            lists[-1][3].append(Decimal(cue[3]))
        elif re.match(r"\s*(?:Clue|List)\s+\d", line):
            raise ValueError("Unrecognized official list or clue row")
    if [row[0] for row in lists] != list(range(1, LISTS + 1)):
        raise ValueError("Expected the complete numbered list inventory")
    if len({row[1] for row in lists}) != LISTS:
        raise ValueError("Repeated source solution keys")
    for _, _, cues, proximity in lists:
        if len(cues) != FULL_HORIZON or proximity != sorted(proximity):
            raise ValueError("Unexpected source cue count or sequence")
        if any(len(cue.split()) != 1 or cue != cue.upper() for cue in cues[:HORIZON]):
            raise ValueError("Expected original uppercase single-word clues through step six")
    materials = [Material(number, target, tuple(cues)) for number, target, cues, _ in lists]
    if len({row.stimulus(HORIZON)["id"] for row in materials}) != LISTS:
        raise ValueError("Repeated first-six body across material families")
    return materials


def read_material(source: Path) -> tuple[list[Material], dict]:
    pdf = source / PDF_NAME
    receipt = binding(pdf, PDF_SHA256, PDF_URL)
    rendered = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout
    return parse_material(rendered), receipt


def jsonl(rows) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n" for row in rows)


def dump(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def material_records(materials: list[Material], horizon: int) -> tuple[str, str, dict]:
    if horizon not in (HORIZON, FULL_HORIZON):
        raise ValueError("Expected the primary or full source horizon")
    stimuli = jsonl(row.stimulus(horizon) for row in materials)
    full_cues = jsonl(
        {"id": row.stimulus(horizon)["id"], "family": row.family, "list_number": row.number, "cues": row.cues}
        for row in materials
    )
    ordered = sorted(materials, key=lambda row: hashlib.sha256(f"{FOLD_SEED}:{row.family}".encode()).digest())
    fold = {row.family: index % FOLDS for index, row in enumerate(ordered)}
    split = {
        "seed": FOLD_SEED,
        "folds": FOLDS,
        "material": [
            {"id": row.stimulus(horizon)["id"], "family": row.family, "fold": fold[row.family]}
            for row in materials
        ],
    }
    return stimuli, full_cues, split


def prepare_stimuli(source: Path, output: Path, horizon: int = HORIZON) -> dict:
    if output.exists():
        raise FileExistsError(output)
    materials, pdf = read_material(source)
    stimuli, full_cues, split = material_records(materials, horizon)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".act-material-", dir=output.parent) as temporary:
        stage = Path(temporary) / "prepared"
        (stage / "model_input").mkdir(parents=True)
        (stage / "model_input" / "stimuli.jsonl").write_text(stimuli, encoding="utf-8")
        (stage / "full-cues.jsonl").write_text(full_cues, encoding="utf-8")
        dump(stage / "split.json", split)
        result = {
            "schema_version": 1, "article": ARTICLE, "osf_project": PROJECT, "source": {PDF_NAME: pdf},
            "prepare_source_sha256": file_sha256(Path(__file__)),
            "documents": LISTS, "primary_horizon": HORIZON, "model_input_horizon": horizon,
            "certification_candidates": LISTS * horizon,
            "private_full_horizon": FULL_HORIZON, "private_full_clues": LISTS * FULL_HORIZON,
            "separator": SEPARATOR, "model_input": "model_input/stimuli.jsonl",
            "file_sha256": {
                name: file_sha256(stage / name)
                for name in ("model_input/stimuli.jsonl", "full-cues.jsonl", "split.json")
            },
        }
        dump(stage / "preflight.json", result)
        stage.rename(output)
    return result


def verify_material_directory(prepared: Path, materials: list[Material]) -> None:
    receipt = json.loads((prepared / "preflight.json").read_text(encoding="utf-8"))
    stimuli, full_cues, split = material_records(materials, receipt["model_input_horizon"])
    emitted = {
        "model_input/stimuli.jsonl": stimuli.encode(),
        "full-cues.jsonl": full_cues.encode(),
        "split.json": (json.dumps(split, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    }
    expected = {name: hashlib.sha256(data).hexdigest() for name, data in emitted.items()}
    if receipt["file_sha256"] != expected:
        raise ValueError("Material receipt does not match the official stimuli and fixed folds")
    for name, checksum in expected.items():
        if file_sha256(prepared / name) != checksum:
            raise ValueError(f"Material isolation or frozen fold binding changed for {name}")


def released_ordinal(value) -> int | None:
    if value is None or math.isnan(value):
        return None
    if not math.isfinite(value) or int(value) != value or not 1 <= value <= FULL_HORIZON:
        raise ValueError("Unexpected original clue ordinal")
    return int(value)


def outcome_reader(source: Path, material: list[Material]) -> tuple[list[dict], dict]:
    import pyreadstat

    sav = source / SAV_NAME
    receipt = binding(sav, SAV_SHA256, SAV_URL)
    _, layout = pyreadstat.read_sav(sav, metadataonly=True, output_format="dict")
    columns = {}
    for prefix in ("clues", "unsolved"):
        columns[prefix] = {
            key(name[len(prefix) + 1:]): name for name in layout.column_names
            if name.startswith(f"{prefix}_") and name != f"{prefix}_total"
        }
        # The author also releases aggregate percentages, which have no cue-level label.
        if prefix == "unsolved":
            columns[prefix].pop("total_pct", None)
        if set(columns[prefix]) != {row.target for row in material}:
            raise ValueError(f"Official list keys do not exactly reconcile with {prefix} columns")
    names = ["subject", *(col for mapping in columns.values() for col in mapping.values())]
    records, metadata = pyreadstat.read_sav(sav, usecols=names, output_format="dict")
    for column in columns["unsolved"].values():
        if metadata.variable_value_labels.get(column) != {0: "list solved", 1: "list unsolved"}:
            raise ValueError("Unexpected author outcome coding")
    readers = records["subject"]
    if len(readers) != READERS or len(set(readers)) != READERS or any(
        reader is None or (isinstance(reader, float) and not math.isfinite(reader)) for reader in readers
    ):
        raise ValueError("Unexpected reader-key inventory in official outcomes")
    rows = []
    identities = [identifier("cogllm-act-reader-v1", json.dumps(reader)) for reader in readers]
    for item in material:
        original_flags = records[columns["unsolved"][item.target]]
        original_clues = records[columns["clues"][item.target]]
        for reader, flag, clues in zip(identities, original_flags, original_clues, strict=True):
            original = released_ordinal(clues)
            if flag not in (0, 1) or flag == 1 and original is not None:
                raise ValueError("Inconsistent official ACT outcome")
            if flag == 0 and original is None:
                status = "unknown"
            elif original is not None and original <= HORIZON:
                status = "first_correct"
            else:
                status = "right_censored"
            rows.append({
                "family": item.family, "reader_id": reader, "status": status,
                "first_correct": original if status == "first_correct" else None,
                "original_solved": flag == 0, "original_first_correct": original,
            })
    return rows, receipt


def prepare_assessment(source: Path, prepared: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    materials, pdf = read_material(source)
    verify_material_directory(prepared, materials)
    rows, sav = outcome_reader(source, materials)
    statuses = collections.Counter(row["status"] for row in rows)
    if len(rows) != LISTS * READERS or statuses["unknown"] != 1:
        raise ValueError("Unexpected original historical trial coverage")
    original_events = collections.Counter(row["original_first_correct"] for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".act-assessment-", dir=output.parent) as temporary:
        stage = Path(temporary) / "assessment"
        stage.mkdir()
        outcomes = stage / "outcomes.jsonl"
        outcomes.write_text(jsonl(rows), encoding="utf-8")
        result = {
            "schema_version": 1, "article": ARTICLE, "osf_project": PROJECT,
            "source": {PDF_NAME: pdf, SAV_NAME: sav}, "prepare_source_sha256": file_sha256(Path(__file__)),
            "material_preflight_sha256": file_sha256(prepared / "preflight.json"),
            "split_sha256": file_sha256(prepared / "split.json"),
            "outcomes_sha256": file_sha256(outcomes),
            "readers": READERS, "targets": LISTS, "trials": len(rows),
            "primary_horizon": HORIZON, "first_correct": [original_events[i] for i in range(1, HORIZON + 1)],
            "primary_status": dict(statuses), "full_horizon": FULL_HORIZON,
            "full_first_correct": [original_events[i] for i in range(1, FULL_HORIZON + 1)],
            "full_flagged_solved": sum(row["original_solved"] for row in rows),
            "full_flagged_unsolved": sum(not row["original_solved"] for row in rows),
            "unknown_original_ordinal": statuses["unknown"],
        }
        dump(stage / "assessment.json", result)
        stage.rename(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("stimuli", "assessment"):
        command = commands.add_parser(name)
        command.add_argument("--source", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "assessment":
            command.add_argument("--prepared", required=True, type=Path)
        else:
            command.add_argument("--horizon", type=int, choices=(HORIZON, FULL_HORIZON), default=HORIZON)
    args = parser.parse_args()
    if args.command == "stimuli":
        result = prepare_stimuli(args.source, args.output, args.horizon)
    else:
        result = prepare_assessment(args.source, args.prepared, args.output)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
