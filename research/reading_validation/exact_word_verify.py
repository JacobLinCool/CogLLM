"""Verify exact spelling and grouped word probabilities from a single native pass per paragraph."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import (
    digest,
    read_json,
    stream_rows,
    write_json,
)
from research.reading_validation.cloze_population_data import OUT as POP
from research.reading_validation.cloze_population_data import ROOT
from research.reading_validation.spelling_agreement import OUT as SPELLING
from research.reading_validation.spelling_inference import METHODS
from research.reading_validation.word_population_dispersion import (
    OUT as DISPERSION,
)
from research.reading_validation.word_population_dispersion import (
    components,
)

FIELDS = (
    "conditional_spelling",
    "direct_spelling",
    "grouped",
    "choices_grouped",
    "joint_grouped",
)
TOLERANCE = 0.001


def require(condition, message):
    if not condition:
        raise ValueError(message)


def take_unique(rows, key):
    result = {}
    for row in rows:
        k = key(row)
        if k in result:
            raise ValueError("Duplicated native material or word coordinate")
        result[k] = row
    return result


def errors(actual, expected):
    require(
        actual.shape == expected.shape, "Different counts of aligned word predictions"
    )
    delta = abs(actual - expected)
    if len(delta) == 0 or not np.isfinite(delta).all() or delta.max() >= TOLERANCE:
        raise ValueError(f"Word probability parity failed at tolerance {TOLERANCE}")
    return {
        "n": len(delta),
        "maximum_absolute_error": float(delta.max()),
        "mean_absolute_error": float(delta.mean()),
    }


def grade(native):
    receipt = read_json(native / "receipt.json")
    runtime = read_json(native.parent / "causality.json")
    job = native.parent.parent
    plan = read_json(job / "plan.json")
    serving = read_json(POP / "serving-export.json")
    spelled = read_json(SPELLING / "fit-lock.json")
    grouped = read_json(DISPERSION / "forecast-lock.json")
    old_score = read_json(DISPERSION / "scores.json")
    native_file = native / "word-forecasts.jsonl"
    original_file = ROOT / "serving-materials.jsonl"
    references_file = ROOT / "serving-reference.jsonl"
    positions_file = ROOT / "external-scored-word-rows.jsonl"
    full = SPELLING / "provo-forecasts.npz"
    gates_path = DISPERSION / "held-gates.npz"
    base_path = ROOT / "dispersion-external-logp.npy"
    old_table = DISPERSION / "provo-word-forecasts.npz"
    checks = {
        native_file: receipt["output"],
        original_file: receipt["input"],
        references_file: serving["references"],
        positions_file: read_json(POP / "word-export.json")["file"],
        full: spelled["studies"]["provo"]["forecast"],
        gates_path: grouped["held_gates"],
        base_path: grouped["held_logp"],
        old_table: old_score["forecast"],
        DISPERSION / "forecast-lock.json": old_score["lock"],
        job / "worker.py": plan["worker_sha256"],
    }
    deps = (
        "research/reading_validation/exact_word_l4.py",
        "research/reading_validation/spelling_inference.py",
        "research/results/reading-validation-spelling-agreement/fit-lock.json",
        "research/results/reading-validation-word-dispersion/forecast-lock.json",
    )
    for name in deps:
        checks[Path(name)] = plan["payload_members_sha256"][name]
    for file, expected in checks.items():
        require(digest(file) == expected, f"Changed signed input: {file.name}")
    require(
        receipt["input"]
        == serving["stimuli"]
        == plan["payload_members_sha256"]["stimuli.jsonl"]
        == runtime["studies"]["exact-word"]["input_sha256"]
        and runtime["device"] == "NVIDIA L4"
        and not receipt["reader_labels_in_input"]
        and not plan["heldout_outcomes_in_worker"]
        and runtime["studies"]["exact-word"]["completed"]
        and receipt["revision"] == runtime["model_revision"] == plan["revision"],
        "Native exposure, backbone, hardware or outcomes contract differs",
    )
    docs = take_unique(stream_rows(original_file), lambda r: r["id"])
    prediction_docs = take_unique(stream_rows(native_file), lambda r: r["id"])
    require(
        set(docs) == set(prediction_docs)
        and len(docs) == receipt["documents"] == receipt["forwards"] == 54,
        "Native encoder hook must count exactly one call per complete input",
    )
    all_words = {}
    for ident, doc in docs.items():
        words = prediction_docs[ident]["words"]
        require(
            len(words) == len(doc["ends"]), "Some printed words lack a native forecast"
        )
        for j, (word, bounds) in enumerate(zip(words, doc["ends"], strict=True)):
            start, end, _, last = bounds
            require(
                (j, start, end, last)
                == (word["index"], word["start"], word["end"], word["token"])
                and 0 <= start < end <= len(doc["text"]),
                "Changed printed-word/token geometry",
            )
            require(
                set(word["calibrated_grouped"]) == {"choices", "joint"},
                "Different calibrated grouping methods",
            )
            all_words[ident, j] = word
    refs = take_unique(stream_rows(references_file), lambda r: r["source_id"])
    positions = take_unique(stream_rows(positions_file), lambda r: r["id"])
    with np.load(full, allow_pickle=False) as a:
        ids = a["ids"].tolist()
        methods = a["methods"].tolist()
        require(
            tuple(methods) == METHODS, "Published spelling method identifiers changed"
        )
        spelling = a["forecast"][
            :, [methods.index(x) for x in ("conditional", "direct", "grouped")]
        ]
    require(
        len(refs) == len(positions) == len(set(ids)) == len(ids) == 2500
        and set(refs) == set(ids) == set(positions),
        "Not all source QIDs are uniquely matched",
    )
    paired = [(refs[k]["entity"], refs[k]["word"]) for k in ids]
    require(len(set(paired)) == len(paired), "Two norming rows share an article word")
    actual = np.empty((len(ids), len(FIELDS)))
    for i, (k, (doc, index)) in enumerate(zip(ids, paired, strict=True)):
        row, w = positions[k], all_words[doc, index]
        require(
            row["entity"] == doc
            and row["target_ordinal_zero_based"] - 1 == index
            and row["preceding_last_token"] == w["token"]
            and row["preceding_chars"] == [w["start"], w["end"]]
            and docs[doc]["text"][w["start"] : w["end"]] == row["preceding_print"],
            "Serving index is not the unchanged observed prefix before the authentic next word",
        )
        actual[i] = [
            w["full_spelling"],
            w["direct_spelling"],
            w["grouped"],
            w["calibrated_grouped"]["choices"],
            w["calibrated_grouped"]["joint"],
        ]
    require(
        np.isfinite(actual).all() and np.all((0 <= actual) & (actual <= 1)),
        "Invalid native estimate",
    )
    base = np.load(base_path, mmap_mode="r", allow_pickle=False)
    grouped_expected = np.empty((len(ids), 2))
    with np.load(gates_path, allow_pickle=False) as g:
        require(
            np.array_equal(g["ids"], ids) and base.shape == (len(ids), 8193),
            "Gates, original word distributions and spelling forecasts are misaligned",
        )
        gates = {m: g[m] for m in ("choices", "joint")}
    for j in range(0, len(ids), 24):
        sl = slice(j, min(j + 24, len(ids)))
        variants = components(base[sl])
        for m, method in enumerate(("choices", "joint")):
            p = np.einsum("ijk,ik->ij", variants, gates[method][sl])
            require(
                np.max(abs(p.sum(1) - 1)) < 2e-6 and np.min(p) >= 0,
                "Invalid frozen word law",
            )
            grouped_expected[sl, m] = 1 - np.sum(p * p, axis=1)
    expected = np.column_stack((spelling, grouped_expected))
    eligible = np.array(
        [positions[k]["normative_denominator_verified"] for k in ids], dtype=bool
    )
    require(
        eligible.sum() == old_score["word_ends"] == 1791,
        "Outcome-grade denominator differs",
    )
    comparison = {
        scope: {
            name: errors(actual[rows, m], expected[rows, m])
            for m, name in enumerate(FIELDS)
        }
        for scope, rows in (
            ("all_certified_norming_positions", np.ones(len(ids), dtype=bool)),
            ("outcome_grade_subset", eligible),
        )
    }
    lookup = {k: i for i, k in enumerate(ids)}
    with np.load(old_table, allow_pickle=False) as old:
        require(
            set(old["id"]) == set(np.array(ids)[eligible])
            and len(old["id"]) == int(eligible.sum()),
            "Registered outcome table and denominator check disagree",
        )
        rows = np.array([lookup[k] for k in old["id"]])
        stored = np.column_stack([old["D_original"], old["D_choices"], old["D_joint"]])
    frozen_replay = {
        name: errors(expected[rows, j + 2], stored[:, j])
        for j, name in enumerate(FIELDS[2:])
    }
    require(
        max(r["maximum_absolute_error"] for r in frozen_replay.values()) < 1e-10,
        "Frozen group baselines failed to reproduce original outcome-grade forecasts",
    )
    report = {
        "device": runtime["device"],
        "revision": receipt["revision"],
        "native_seconds": receipt["seconds"],
        "original_articles": len(docs),
        "raw_word_ends": len(all_words),
        "registered_norming_positions": len(ids),
        "outcome_grade_positions": int(eligible.sum()),
        "encoder_forward_calls": receipt["forwards"],
        "additional_encoder_calls": receipt["forwards"] - len(docs),
        "reader_labels_in_native_packet": False,
        "strict_maximum_error_threshold": TOLERANCE,
        "comparisons": comparison,
        "recovered_original_outcome_grade_table": frozen_replay,
        "fingerprints": {
            str(file.resolve().relative_to(Path.cwd())): value
            for file, value in checks.items()
        },
        "native_job_plan": digest(job / "plan.json"),
        "native_receipt": digest(native / "receipt.json"),
        "native_hardware_receipt": digest(native.parent / "causality.json"),
        "norming_coordinate_seal": digest(POP / "word-export.json"),
        "serving_export": digest(POP / "serving-export.json"),
        "estimator": digest(Path(__file__)),
    }
    write_json(SPELLING / "native-serving-verification.json", report)
    print(
        {
            name: vals["maximum_absolute_error"]
            for name, vals in comparison["all_certified_norming_positions"].items()
        }
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    grade(parser.parse_args().native)
