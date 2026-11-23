"""Align original human responses with their causal Qwen stopping positions."""

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np
from wordfreq import zipf_frequency

from research.reading_validation.external_input import (
    digest_string,
    material_key,
    workbook_rows,
)


TASKS = {
    "plaba": "plaba_processing_ease",
    "clear": "clear_excerpt_easiness",
    "students": "studentannotations_comprehension_difficulty",
}
WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
DEFAULT_CACHE = Path(".cache/cogllm")


def read(path):
    return json.loads(path.read_text())


def lines(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def digest(path):
    with path.open("rb") as f:
        return __import__("hashlib").file_digest(f, "sha256").hexdigest()


def surface(text, lang):
    words = WORD.findall(text.casefold())
    if not words:
        return np.zeros(12, dtype=np.float64)
    sizes = np.array([len(w) for w in words])
    freqs = np.array([zipf_frequency(w, lang) for w in words])
    return np.array(
        [
            np.log1p(len(words)),
            sizes.mean(),
            sizes.max(),
            (sizes >= 7).mean(),
            freqs.mean(),
            freqs.min(),
            np.quantile(freqs, 0.2),
            (freqs < 3).mean(),
            np.std(freqs),
            sum(c.isdigit() for c in text) / len(text),
            sum(c in ",:;()—" for c in text) / len(words),
            len(set(words)) / len(words),
        ]
    )


@dataclass
class Study:
    name: str
    ids: list
    groups: list
    role: np.ndarray
    hidden: np.ndarray
    surface: np.ndarray
    response: np.ndarray
    auxiliary_scale: float = 1.0
    auxiliary_center: float = 0.0

    @property
    def continuous(self):
        return self.name == "clear"


def auxiliary(cache, external_metadata):
    observations = cache / "adaptive-joint/inputs/observations"
    receipt = read(observations.parent / "receipt.json")
    target_keys = {r["text_key"] for r in external_metadata["rows"]}
    text_file = cache / "union-training/observations-2/stimuli.jsonl"
    texts = {r["id"]: r["text"] for r in lines(text_file)}
    parts = {
        digest_string(row["text"]): {p["end"]: p["start"] for p in row["segments"]}
        for row in lines(cache / "stage2/plaba/stimuli.jsonl")
    }
    raw, needed = {}, set()
    duplicates = Counter()
    for name, task in TASKS.items():
        path = observations / f"{task}.jsonl"
        if digest(path) != receipt["files_sha256"][f"observations/{task}.jsonl"]:
            raise ValueError(f"Training observations changed for {name}")
        records = []
        for r in lines(path):
            key, stop = r["coordinates"][0]
            start = parts[key][stop] if name == "plaba" else 0
            target = texts[key][start:stop]
            if not target.strip() or not 0 <= start < stop <= len(texts[key]):
                raise ValueError("Invalid observed text span")
            if material_key(target) in target_keys:
                duplicates[name] += 1
                continue
            if r["role"] not in (0, 1):
                raise ValueError("Unexpected historical split")
            needed.add((key, stop))
            outcome = r["outcome"][::-1] if name == "plaba" else r["outcome"]
            records.append(((key, stop), r["family"], r["role"], outcome, target))
        raw[name] = records
    shards = (
        cache
        / "union-training/checkpointed/cogllm-resume-20261121T185154Z-8ade72/shards"
    )
    features = {}
    for reference in receipt["raw_shards"]:
        folder = shards / reference["id"]
        seal = read(folder / "seal.json")
        if (
            digest(folder / "seal.json") != reference["seal_sha256"]
            or digest(folder / "coordinates.jsonl")
            != seal["file_sha256"]["coordinates.jsonl"]
            or digest(folder / "states.npy") != reference["raw_state_sha256"]
            or seal["revision"] != receipt["revision"]
        ):
            raise ValueError(
                "Previously extracted features do not match their training receipt"
            )
        matrix = np.load(folder / "states.npy", mmap_mode="r", allow_pickle=False)
        coords = (
            (doc["id"], s)
            for doc in lines(folder / "coordinates.jsonl")
            for s in doc["stops"]
        )
        for i, coord in enumerate(coords):
            if coord in needed:
                if coord in features:
                    raise ValueError("Repeated causal position")
                features[coord] = matrix[i].copy()
        if i + 1 != len(matrix):
            raise ValueError("Feature and coordinate count differ")
    if needed != features.keys():
        raise ValueError("Missing human-observation features")
    output = {}
    for name, records in raw.items():
        ids, families, roles, outcomes, spans = zip(*records, strict=True)
        roles = np.array(roles)
        if set(np.array(families)[roles == 0]) & set(np.array(families)[roles == 1]):
            raise ValueError("An auxiliary family crosses the training boundary")
        y = np.array(outcomes, dtype=np.float64)
        center, scale = 0.0, 1.0
        if name == "clear":
            center = -y[roles == 0, 0].mean()
            scale = y[roles == 0, 0].std()
            if scale <= 0:
                raise ValueError("Degenerate continuous human training responses")
            y[:, 0] = (-y[:, 0] - center) / scale
            y[:, 1] /= scale
        output[name] = Study(
            name,
            list(ids),
            list(families),
            roles,
            np.stack([features[i] for i in ids]),
            np.stack([surface(s, "de" if name == "students" else "en") for s in spans]),
            y,
            scale,
            center,
        )
    info = dict(
        exclusions=dict(duplicates),
        source_receipt_sha256=digest(observations.parent / "receipt.json"),
        revision=receipt["revision"],
    )
    return output, info


def external(cache, extraction, metadata, with_test=False):
    base = cache / "reading-validation"
    receipt = read(extraction / "extraction.json")
    if receipt["packet_sha256"] != metadata["stimulus_sha256"]:
        raise ValueError("Extraction used a different packet")
    for key, expected in receipt["files_sha256"].items():
        if digest(extraction / key) != expected:
            raise ValueError("GPU output digest mismatch")
    ids = read(extraction / "ids.json")
    matrix = np.load(extraction / "states.npy", mmap_mode="r", allow_pickle=False)
    lookup = dict(zip(ids, range(len(ids)), strict=True))
    if len(lookup) != len(matrix):
        raise ValueError("Duplicate or missing external feature")
    texts = {r["id"]: r["text"] for r in lines(base / "external-stimuli.jsonl")}
    labels = {}
    for split in ("train", "val", "test") if with_test else ("train", "val"):
        if (
            digest(base / "readme" / f"{split}.xlsx")
            != metadata["workbook_sha256"][split]
        ):
            raise ValueError("Workbook changed")
        for number, row in workbook_rows(base / "readme", split):
            value = float(row[5])
            if not value.is_integer() or not 1 <= value <= 6:
                raise ValueError("Invalid CEFR rating")
            labels[(split, number)] = digest_string(row[4]), int(value)
    rows = metadata["rows"]
    roles = np.array([{"train": 0, "val": 1, "test": 2}[r["split"]] for r in rows])
    y = np.full((len(rows), 6), np.nan)
    for i, r in enumerate(rows):
        if r["split"] == "test" and not with_test:
            continue
        uid, rating = labels[r["split"], r["row"]]
        if uid != r["id"]:
            raise ValueError("Misaligned CEFR release row")
        y[i] = np.eye(6)[rating - 1]
    return Study(
        "readme",
        [r["id"] for r in rows],
        [r["domain"] for r in rows],
        roles,
        np.stack([matrix[lookup[r["id"]]] for r in rows]),
        np.stack([surface(texts[r["id"]], "en") for r in rows]),
        y,
    )
