"""Decompose the sealed Frick target-onset contrast without fitting or inference."""

import argparse
from collections import defaultdict
from fractions import Fraction
from hashlib import file_digest
import json
from pathlib import Path
from statistics import mean

import numpy as np


CONDITIONS = ("01", "02")
SEED, DRAWS, CHUNK = 735972, 10_000, 200
SOURCE_AUDIT = "18275a8edaccdac253a1cc1057debbcc11adf82af69dc6efbdd982cd7079a6e4"
SCORE_SHA256 = "d991259e5b7b55acaa889b4c0ca67fb06f957035dabf3a69d28f7e16b4cd0efd"


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def read_material(assay, results):
    source = assay / "source-audit.json"
    raw_score = results / "results.json"
    seal_path = results / "forecast-seal.json"
    if digest(source) != SOURCE_AUDIT or digest(raw_score) != SCORE_SHA256:
        raise ValueError("Expected the published Frick source and mass-calibration score")
    audit = json.loads(source.read_bytes())
    score = json.loads(raw_score.read_bytes())
    seal = json.loads(seal_path.read_bytes())
    if (score["forecast_seal_sha256"] != digest(seal_path)
            or seal["provenance"]["source_audit_sha256"] != SOURCE_AUDIT):
        raise ValueError("Forecast, source and score seals do not agree")
    for name, expected in audit["output_sha256"].items():
        if digest(assay / name) != expected:
            raise ValueError(f"Prepared Frick source no longer matches: {name}")
    forecast_path = results / "forecasts.jsonl"
    if digest(forecast_path) != seal["file_sha256"]["forecasts.jsonl"]:
        raise ValueError("Frozen model logits no longer match the forecast seal")
    metadata = {row["id"]: row for row in jsonl(assay / "metadata.jsonl")}
    stimuli = {row["id"]: row for row in jsonl(assay / "stimuli.jsonl")}
    observed = jsonl(assay / "outcomes.jsonl")
    logits = defaultdict(dict)
    for row in jsonl(forecast_path):
        sequence = logits[row["id"]]
        if row["stop"] in sequence:
            raise ValueError("Repeated sealed forecast stop")
        sequence[row["stop"]] = row["first_logits"]["calibrated"]
    positions = {(metadata[key]["story"], metadata[key]["condition"]) for key in logits}
    stories = sorted({key[0] for key in positions})
    if (len(stories) != 20 or len(logits) != 40 or len(observed) != 1800
            or positions != {(family, condition) for family in stories for condition in CONDITIONS}):
        raise ValueError("Expected the original 20 story pairs and 1,800 observed trajectories")
    return metadata, stimuli, observed, logits, score, {
        "source_audit_sha256": SOURCE_AUDIT,
        "outcomes_sha256": audit["output_sha256"]["outcomes.jsonl"],
        "score_sha256": SCORE_SHA256,
        "forecast_seal_sha256": digest(seal_path),
        "forecasts_sha256": digest(forecast_path),
    }


def factorize(metadata, stimuli, observed, logits):
    stories = {name: i for i, name in enumerate(sorted({metadata[k]["story"] for k in logits}))}
    readers = {name: i for i, name in enumerate(sorted({row["reader_id"] for row in observed}))}
    if len(readers) != 180:
        raise ValueError("The original 180 reader groups are required")
    shape = (len(stories), len(CONDITIONS))
    total, risk, event = (np.zeros((*shape, len(readers)), dtype=np.int8) for _ in range(3))
    model_survival, model_hazard = (np.empty(shape, dtype=np.float64) for _ in range(2))
    positions = {}
    for key, by_stop in logits.items():
        info = metadata[key]
        s, c = stories[info["story"]], CONDITIONS.index(info["condition"])
        target = info["target_index"] - 1
        if list(by_stop) != stimuli[key]["stops"][1:] or not 0 < target < len(by_stop):
            raise ValueError("The sealed forecasts and exact target sentence do not align")
        z = np.asarray(list(by_stop.values()), dtype=np.float64)
        if not np.isfinite(z).all():
            raise ValueError("Expected finite sentence hazard logits")
        model_survival[s, c] = np.exp(-np.logaddexp(0, z[:target]).sum())
        model_hazard[s, c] = np.exp(-np.logaddexp(0, -z[target]))
        positions[key] = s, c, target
    for row in observed:
        s, c, target = positions[row["id"]]
        reader = readers[row["reader_id"]]
        keys = row["keys"][1:]
        if (len(keys) != len(logits[row["id"]]) or any(key not in ("k", "d") for key in keys)
                or total[s, c, reader]):
            raise ValueError("Expected one complete trajectory per reader, story and presented variant")
        first = next((i for i, response in enumerate(keys) if response == "d"), len(keys))
        total[s, c, reader] = 1
        risk[s, c, reader] = first >= target
        event[s, c, reader] = first == target
    return total, risk, event, model_survival, model_hazard


def estimates(tables, reader_weights, story_weights):
    total, risk, event, model_survival, model_hazard = tables
    draws = len(reader_weights)
    n_story, n_condition, n_reader = total.shape

    def reweighted(matrix):
        return (reader_weights @ matrix.reshape(-1, n_reader).T).reshape(draws, n_story, n_condition)

    n, r, e = map(reweighted, (total, risk, event))
    if np.any(r <= 0):
        raise ValueError("A bootstrap draw did not retain an at-risk reader in one story variant")
    human_survival, human_hazard = r / n, e / r
    weights = story_weights / story_weights.sum(axis=1, keepdims=True)

    def contrast(values):
        return np.einsum("bs,bs->b", weights, values[..., 1] - values[..., 0])

    human = contrast(human_survival * human_hazard)
    model = contrast(np.broadcast_to(model_survival * model_hazard, human_survival.shape))
    oracle_survival = contrast(human_survival * model_hazard)
    oracle_hazard = contrast(model_survival * human_hazard)
    ceiling = weights @ model_hazard[:, 1]
    return {
        "observed_target_first_report_contrast": human,
        "calibrated_target_first_report_contrast": model,
        "replace_survival_only_with_human": oracle_survival,
        "replace_hazard_only_with_human": oracle_hazard,
        "maximum_contrast_with_fixed_model_target_hazards": ceiling,
        "human_contrast_minus_survival_only_ceiling": human - ceiling,
        "contrast_change_after_replacing_survival_only": oracle_survival - model,
        **{
            f"human_target_hazard_{condition}": (weights * human_hazard[:, :, c]).sum(axis=1)
            for c, condition in enumerate(CONDITIONS)
        },
        **{f"model_target_hazard_{condition}": weights @ model_hazard[:, c]
           for c, condition in enumerate(CONDITIONS)},
    }


def report(assay, results):
    metadata, stimuli, observed, logits, score, provenance = read_material(assay, results)
    tables = factorize(metadata, stimuli, observed, logits)
    total, _, event, model_survival, model_hazard = tables
    stories, _, readers = total.shape
    point = estimates(tables, np.ones((1, readers)), np.ones((1, stories)))
    human_points = [mean(Fraction(int(y), int(n)) for y, n in zip(
        event.sum(axis=2)[:, c], total.sum(axis=2)[:, c], strict=True,
    )) for c in range(len(CONDITIONS))]
    model_points = (model_survival * model_hazard).mean(axis=0)
    scalar_truth = float(human_points[1] - human_points[0])
    check_gaps = [abs(scalar_truth - point["observed_target_first_report_contrast"].item())]
    for c, condition in enumerate(CONDITIONS):
        by_condition = score["first_d_by_condition"][condition]
        check_gaps.extend((
            abs(float(human_points[c]) - by_condition["human_probabilities"]["at_target"]["estimate"]),
            abs(float(model_points[c]) - by_condition["methods"]["calibrated"]["probabilities"]["at_target"]["estimate"]),
        ))
    if max(check_gaps) > 1e-13:
        raise ValueError("Factorization differs from exact human counts or the sealed onset scores")

    rng = np.random.default_rng(SEED)
    rw = rng.multinomial(readers, np.full(readers, 1 / readers), size=DRAWS)
    sw = rng.multinomial(stories, np.full(stories, 1 / stories), size=DRAWS)
    samples = [estimates(tables, rw[i:i + CHUNK], sw[i:i + CHUNK]) for i in range(0, DRAWS, CHUNK)]
    return {
        "schema": 1,
        "scope": "Retrospective diagnostic on existing story pairs, with no fitting or new model inference.",
        "units": "Probability or probability contrast; multiply by 100 for percent or percentage points.",
        "trial_count": int(total.sum()),
        "story_families": stories,
        "reader_groups": readers,
        "forecast_stops": sum(len(row) for row in logits.values()),
        "bootstrap": {"draws": DRAWS, "seed": SEED, "clusters": ["reader", "story_family"], "frozen_model": True},
        "provenance": {**provenance, "diagnostic_code_sha256": digest(Path(__file__))},
        "independent_max_absolute_check_error": max(check_gaps),
        "statistics": {
            key: {"estimate": float(value.item()),
                  "crossed_story_reader_ci95": np.quantile(
                      np.concatenate([batch[key] for batch in samples]), (0.025, 0.975),
                  ).tolist()}
            for key, value in point.items()
        },
    }


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("assay", "results", "output"):
        cli.add_argument(f"--{name}", type=Path, required=True)
    args = cli.parse_args()
    document = json.dumps(report(args.assay, args.results), sort_keys=True, indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(document)
