"""Fit and score first-correct ordinal distributions over sampled cue paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from research.semantic_tracking.core_vocabulary_marginal_prepare import (
    ORDINALS,
    sha256,
    target_family,
)


TARGET_TABLE_SHA256 = "de963d23c2fb9a118a30f76ac3acfe6d4121c3554f010695f06bae8d4525ae49"
TRIAL_TABLE_SHA256 = "c1743cc61a320830b24450d0f2b6067042e5e46e8ff75c59cc55ff2cfed07966"
SPLIT_SEED = "cogllm-core-vocabulary-target-split-v1"
CONDITIONS = ("AoA", "Freq", "Instrength", "NonCore")
TRAIN_PER_CONDITION = 16
TARGETS = 96
DRAWS = 8


def log_sigmoid(value: float) -> float:
    return -math.log1p(math.exp(-value)) if value >= 0 else value - math.log1p(math.exp(value))


def path_log_masses(logits) -> tuple[float, ...]:
    """Return log probabilities for first correct at exposures 1–6 and censoring."""
    if len(logits) != ORDINALS or any(
        type(value) not in (int, float) or not math.isfinite(value) for value in logits
    ):
        raise ValueError("Expected six finite hazard logits")
    survival = 0.0
    events = []
    for value in logits:
        events.append(survival + log_sigmoid(value))
        survival += log_sigmoid(-value)
    return (*events, survival)


def logsumexp(values) -> float:
    peak = max(values)
    if peak == -math.inf:
        return peak
    return peak + math.log(math.fsum(math.exp(value - peak) for value in values))


def marginal_log_masses(path_logits) -> tuple[float, ...]:
    """Average path-level outcome masses with equal path weights."""
    paths = [path_log_masses(logits) for logits in path_logits]
    if not paths:
        raise ValueError("Expected at least one cue path")
    scale = math.log(len(paths))
    mixture = tuple(logsumexp(column) - scale for column in zip(*paths, strict=True))
    if not math.isclose(logsumexp(mixture), 0.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Mixture outcome probabilities do not sum to one")
    return mixture


def position_log_masses(training_counts) -> tuple[float, ...]:
    """Maximum-likelihood position-only masses under equal target weights."""
    if not training_counts:
        raise ValueError("Position-only fitting requires training targets")
    rates = [tuple(count / sum(row) for count in row) for row in training_counts.values()]
    masses = [math.fsum(column) / len(rates) for column in zip(*rates, strict=True)]
    if any(mass <= 0.0 for mass in masses) or len(masses) != ORDINALS + 1:
        raise ValueError("All seven position outcomes need support in training")
    return tuple(map(math.log, masses))


def macro_nll(counts, log_masses) -> float:
    if not counts or counts.keys() != log_masses.keys():
        raise ValueError("The outcome and forecast target groups must align exactly")
    losses = []
    for family, row in counts.items():
        losses.append(
            -math.fsum(count * log_mass for count, log_mass in zip(row, log_masses[family], strict=True) if count)
            / sum(row)
        )
    score = math.fsum(losses) / len(losses)
    if not math.isfinite(score):
        raise ValueError("A realized outcome has no finite log probability")
    return score


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_json(output: Path, payload: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def read_packet(packet: Path) -> dict[str, tuple[str, ...]]:
    receipt = json.loads((packet / "preflight.json").read_text())
    if (
        receipt["targets"] != TARGETS
        or receipt["draws_per_target"] != DRAWS
        or receipt["stimuli"] != TARGETS * DRAWS
        or receipt["stops"] != TARGETS * DRAWS * ORDINALS
    ):
        raise ValueError("Expected the frozen 96-target, eight-draw pilot")
    for name in ("model_input/stimuli.jsonl", "mixture.jsonl"):
        if sha256(packet / name) != receipt["file_sha256"][name]:
            raise ValueError(f"Pilot digest mismatch for {name}")
    families: dict[str, dict[int, str]] = {}
    seen = set()
    for row in read_jsonl(packet / "mixture.jsonl"):
        if set(row) != {"id", "family", "draw", "weight"} or row["weight"] != 1 / DRAWS:
            raise ValueError("Unexpected mixture row schema or weight")
        draws = families.setdefault(row["family"], {})
        if row["draw"] in draws or row["id"] in seen:
            raise ValueError("Duplicate path in the mixture")
        draws[row["draw"]] = row["id"]
        seen.add(row["id"])
    if len(families) != TARGETS or any(set(draws) != set(range(1, DRAWS + 1)) for draws in families.values()):
        raise ValueError("Incomplete sampled path support")
    stimuli_ids = [row["id"] for row in read_jsonl(packet / "model_input/stimuli.jsonl")]
    if len(stimuli_ids) != len(seen) or set(stimuli_ids) != seen:
        raise ValueError("Stimulus IDs do not match the local mixture map")
    return {family: tuple(draws[draw] for draw in sorted(draws)) for family, draws in families.items()}


def split_rows(conditions: dict[str, str]) -> list[dict[str, str]]:
    if set(conditions.values()) != set(CONDITIONS):
        raise ValueError("Both splits require the four author target conditions")
    rows = []
    for condition in CONDITIONS:
        families = [family for family, group in conditions.items() if group == condition]
        families.sort(key=lambda family: hashlib.sha256(f"{SPLIT_SEED}:{family}".encode()).digest())
        if len(families) != 24:
            raise ValueError("Unexpected target inventory within a condition")
        rows.extend(
            {"family": family, "condition": condition, "split": "train" if index < TRAIN_PER_CONDITION else "holdout"}
            for index, family in enumerate(families)
        )
    return rows


def freeze_split(targets: Path, packet: Path) -> dict:
    if sha256(targets) != TARGET_TABLE_SHA256:
        raise ValueError("Expected the pinned grade-free author target table")
    with targets.open(encoding="utf-8", newline="") as f:
        conditions = {target_family(row["target"]): row["condition"] for row in csv.DictReader(f)}
    if conditions.keys() != read_packet(packet).keys():
        raise ValueError("Material split does not cover the packet")
    return {
        "schema_version": 1,
        "seed": SPLIT_SEED,
        "packet_preflight_sha256": sha256(packet / "preflight.json"),
        "target_table_sha256": TARGET_TABLE_SHA256,
        "targets": split_rows(conditions),
    }


def read_split(path: Path, packet: Path, families) -> dict:
    receipt = json.loads(path.read_text())
    rows = receipt["targets"]
    conditions = {row["family"]: row["condition"] for row in rows}
    if (
        receipt["schema_version"] != 1
        or receipt["seed"] != SPLIT_SEED
        or receipt["packet_preflight_sha256"] != sha256(packet / "preflight.json")
        or receipt["target_table_sha256"] != TARGET_TABLE_SHA256
        or conditions.keys() != families.keys()
        or rows != split_rows(conditions)
    ):
        raise ValueError("Unexpected sealed material split")
    return {row["family"]: row for row in rows}


def aggregate_trials(trials: Path, split: dict, role: str) -> tuple[dict, int]:
    counts = {family: [0] * (ORDINALS + 1) for family, row in split.items() if row["split"] == role}
    readers = set()
    keys = set()
    with trials.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row["exp"], row["stim_set"]) != ("1", "Original"):
                continue
            family = target_family(row["target"])
            if family not in counts:
                continue
            key = (row["subject"], family)
            ordinal = int(row["guesses"])
            correct = row["correct"]
            if (
                key in keys
                or row["targetWordCondition"] != split[family]["condition"]
                or not 1 <= ordinal <= ORDINALS
                or correct not in {"0", "1"}
                or (correct == "0" and ordinal != ORDINALS)
            ):
                raise ValueError("Invalid historical trial outcome")
            keys.add(key)
            readers.add(row["subject"])
            counts[family][ordinal - 1 if correct == "1" else ORDINALS] += 1
    if any(not sum(row) for row in counts.values()):
        raise ValueError("A material group has no historical observations")
    return counts, len(readers)


def forecast_mixtures(forecasts: Path, families: dict) -> dict:
    rows = read_jsonl(forecasts)
    by_id = {row["id"]: row["methods"] for row in rows}
    expected = {identifier for ids in families.values() for identifier in ids}
    if (
        len(rows) != len(expected)
        or by_id.keys() != expected
        or any(set(row) != {"id", "methods"} for row in rows)
    ):
        raise ValueError("Forecast rows must cover exactly the grade-free paths")
    methods = set(rows[0]["methods"])
    if not methods or "position" in methods or any(set(row["methods"]) != methods for row in rows):
        raise ValueError("Forecast methods must match and use distinct names")
    return {
        name: {
            family: marginal_log_masses(by_id[identifier][name] for identifier in identifiers)
            for family, identifiers in families.items()
        }
        for name in sorted(methods)
    }


def inventory(counts, readers: int) -> dict:
    totals = [sum(column) for column in zip(*counts.values(), strict=True)]
    return {"targets": len(counts), "readers": readers, "trials": sum(totals), "outcome_counts": totals}


def evaluate(trials: Path, packet: Path, split_file: Path, forecasts: Path | None = None) -> dict:
    families = read_packet(packet)
    split = read_split(split_file, packet, families)
    log_masses = {} if forecasts is None else forecast_mixtures(forecasts, families)
    if sha256(trials) != TRIAL_TABLE_SHA256:
        raise ValueError("Expected the pinned Experiment 1 trial release")
    train, train_readers = aggregate_trials(trials, split, "train")
    position = position_log_masses(train)
    log_masses["position"] = dict.fromkeys(families, position)
    holdout, holdout_readers = aggregate_trials(trials, split, "holdout")
    if sum(map(sum, train.values())) + sum(map(sum, holdout.values())) != 4680:
        raise ValueError("Retained trial inventory changed")
    scores = {
        name: {
            "train_nll": macro_nll(train, {family: masses[family] for family in train}),
            "holdout_nll": macro_nll(holdout, {family: masses[family] for family in holdout}),
        }
        for name, masses in log_masses.items()
    }
    for score in scores.values():
        score["holdout_nll_minus_position"] = score["holdout_nll"] - scores["position"]["holdout_nll"]
    masses = tuple(map(math.exp, position))
    return {
        "schema_version": 1,
        "trial_table_sha256": TRIAL_TABLE_SHA256,
        "packet_preflight_sha256": sha256(packet / "preflight.json"),
        "split_sha256": sha256(split_file),
        "forecasts_sha256": sha256(forecasts) if forecasts is not None else None,
        "outcome_order": ["first_correct_1", "first_correct_2", "first_correct_3", "first_correct_4",
                          "first_correct_5", "first_correct_6", "censored_after_6"],
        "train": inventory(train, train_readers),
        "holdout": inventory(holdout, holdout_readers),
        "position_train_fit": {
            "masses": masses,
            "hazards": [masses[i] / math.fsum(masses[i:]) for i in range(ORDINALS)],
        },
        "target_macro_log_scores_nats": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split")
    split.add_argument("--targets", required=True, type=Path)
    for name in ("baseline", "score"):
        command = commands.add_parser(name)
        command.add_argument("--trials", required=True, type=Path)
        command.add_argument("--split", required=True, type=Path)
        if name == "score":
            command.add_argument("--forecasts", required=True, type=Path)
    for command in commands.choices.values():
        command.add_argument("--packet", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "split":
        result = freeze_split(args.targets, args.packet)
    else:
        result = evaluate(args.trials, args.packet, args.split, getattr(args, "forecasts", None))
    write_json(args.output, result)


if __name__ == "__main__":
    main()
