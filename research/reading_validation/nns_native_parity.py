"""Verify exported native probabilities against all locked, outcome-free word forecasts."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.reading_validation.local_forecast import digest
from research.reading_validation.nns_features import geometry
from research.reading_validation.nns_inference import WordReport
from research.reading_validation.nns_material import OUT


def run(extraction):
    model = WordReport.load()
    docs, _ = geometry("dense")
    source = json.loads((OUT / "forecast-seal.json").read_bytes())
    forecast = OUT / "word-forecasts.npz"
    if digest(forecast) != source["outcome_free_forecast_sha256"]:
        raise ValueError("Forecast mismatch")
    receipt = json.loads((extraction / "extraction.json").read_bytes())
    checksum = source["extractions"]["dense"]["files_sha256"]
    if receipt["files_sha256"] != checksum or any(
        digest(extraction / k) != v for k, v in checksum.items()
    ):
        raise ValueError("Native feature mismatch")
    ids = json.loads((extraction / "ids.json").read_bytes())
    num = 0
    maximum = 0.0
    actual = 0
    with (
        np.load(forecast, allow_pickle=False) as pred,
        np.load(extraction / "signals.npz", allow_pickle=False) as z,
    ):
        keys = pred["doc"]
        coords = pred["coords"]
        pp = np.column_stack([pred["text.adaptive"], pred["cefr.adaptive"]])
        offsets = z["offsets"]
        positions = z["word_spans"]
        ok = z["native"] >= 0
        v = z["lexical"]
        p = z["plaba"]
        x = z["context"]
        for i, key in enumerate(ids):
            if key not in docs:
                continue
            sl = slice(offsets[i], offsets[i + 1])
            keep = keys == key
            expected = coords[keep]
            n = 0
            for col, level in enumerate([None, *model.levels]):
                out = model._packed(
                    docs[key], positions[sl], ok[sl], v[sl], p[sl], x[sl], level
                )
                if not np.array_equal(
                    [(row["start"], row["end"]) for row in out], expected
                ):
                    raise ValueError("Scored word offsets differ")
                pr = np.array([row["reported_unknown_probability"] for row in out])
                maximum = max(maximum, float(np.max(abs(pr - pp[keep, col]))))
                n = len(out)
            num += n
            actual += 1
    if maximum > 5e-7 or actual != 200:
        raise ArithmeticError(
            f"Inference disagreement on registered English material, max {maximum}"
        )
    score = {
        "sentences": actual,
        "certified_word_ends": num,
        "reader_conditions": 7,
        "predictions_compared": 7 * num,
        "max_absolute_probability_difference": maximum,
        "input_sha256": digest(extraction / "extraction.json"),
        "native_manifest_sha256": model.checksum,
        "source_predictions_sha256": digest(forecast),
        "native_code_sha256": digest(Path(__file__).with_name("nns_inference.py")),
        "used_held_human_responses": False,
    }
    (OUT / "serving-parity.json").write_text(json.dumps(score, indent=2) + "\n")
    print(score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, required=True)
    run(parser.parse_args().extraction)
