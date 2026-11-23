"""Decide when a single word pointer is reliable enough to offer an editor."""

from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_pointer_data import OUT, ROOT
from research.reading_validation.transition_pointer_gap_data import OUT as GAP
from research.reading_validation.transition_pointer_score import ARMS
from research.reading_validation.transition_score import dump, sha


PROTOCOL = Path("research/reading_validation/transition-pointer-selective-protocol.md")
REGISTERED = "7464b8f63ca9eaa1c1a1005e6ff455a2d716f57c2bd3a8c32afce5cd4f6e1415"
METHODS = ("lexical", "headwise")
GRID = np.arange(1, 10) / 10


def prediction(logits, clicks):
    """Return maximum word probability and expected correctness over readers and score ties."""
    logits = np.asarray(logits, dtype=np.float64)
    top = logits == logits.max()
    confidence = 1 / np.exp(logits - logits.max()).sum()
    votes = np.zeros(len(logits), dtype=float)
    n = 0
    for start, stop, count, *_ in clicks:
        if not 0 <= start < stop <= len(votes) or count <= 0:
            raise ValueError("Click falls outside the displayed prefix")
        votes[start:stop] += count
        n += count
    if n == 0:
        raise ValueError("An observed query needs at least one eligible click")
    return float(confidence), float(votes[top].mean() / n), n


def evaluate(rows, threshold):
    by_group = defaultdict(list)
    for group, confidence, correct, voters in rows:
        selected = confidence >= threshold
        by_group[group].append((selected, selected * correct, voters))
    groups = sorted(by_group)
    matrix = np.stack([np.mean(by_group[g], axis=0) for g in groups])
    offered, correct, _ = matrix.mean(0)
    _, _, draws = grouped_distribution(groups)
    boot = draws @ matrix / len(groups)
    boot_precision = np.divide(boot[:, 1], boot[:, 0], out=np.zeros(len(boot)), where=boot[:, 0] > 0)

    def interval(point, estimates):
        return {"estimate": float(point), "whole_group_95_interval": np.quantile(estimates, [.025, .975]).tolist()}

    total = sum(len(items) for items in by_group.values())
    linked = sum(int(confidence >= threshold) for _, confidence, _, _ in rows)
    result = {
        "threshold": float(threshold),
        "groups": len(groups),
        "eligible_queries": total,
        "reader_responses": sum(v for *_, v in rows),
        "linked_queries": linked,
        "linked_reader_responses": sum(n for _, p, _, n in rows if p >= threshold),
        "groups_with_links": int((matrix[:, 0] > 0).sum()),
        "coverage": interval(offered, boot[:, 0]),
        "precision": interval(correct / offered, boot_precision[boot[:, 0] > 0]) if offered else None,
        "false_suggestions_per_100_queries": interval(100 * (offered - correct), 100 * (boot[:, 0] - boot[:, 1])),
        "bootstrap_draws_without_links": int((boot[:, 0] == 0).sum()),
    }
    return result, boot_precision


def calibrate(rows):
    grid = []
    eligible = []
    for threshold in GRID:
        item, boot_precision = evaluate(rows, threshold)
        grid.append(item)
        if item["linked_queries"] >= 15 and item["groups_with_links"] >= 4:
            eligible.append((item, boot_precision))
    if eligible:
        max_errors = np.stack([item["precision"]["estimate"] - boot for item, boot in eligible]).max(axis=0)
        adjustment = float(np.quantile(max_errors, .95))
        for item, _ in eligible:
            item["simultaneous_precision_lower"] = max(0.0, item["precision"]["estimate"] - adjustment)
    else:
        adjustment = None
    accepted = [item for item, _ in eligible if item["simultaneous_precision_lower"] >= .70]
    best = max(accepted, key=lambda x: (x["coverage"]["estimate"], x["threshold"])) if accepted else None
    return {"grid": grid, "simultaneous_adjustment": adjustment, "selected": best}


def append(rows, group, logits, counts):
    for arm in METHODS:
        vector = logits[ARMS.index(arm) - 1]
        rows[arm].append((group, *prediction(vector, counts)))


def score_policy(rows, rule):
    results = {key: evaluate(rows, threshold)[0] for key, threshold in (("always", 0.), ("fixed_half", .50))}
    selected = rule["selected"]
    results["source_selected"] = evaluate(rows, selected["threshold"])[0] if selected else None
    return results


def validated_predictions(folder, seal, task_protocol_hash):
    receipt = json.loads((folder / "receipt.json").read_text())
    p = folder / "candidate-word-logits.npz"
    if (sha(p) != receipt["predictions_sha256"] or receipt["input_sha256"] != seal["input_sha256"]
            or receipt["protocol_sha256"] != task_protocol_hash):
        raise ValueError("The locked prior-word forecast changed")
    return np.load(p), sha(p)


def run(source, external=None):
    if sha(PROTOCOL) != REGISTERED:
        raise ValueError("Selective pointer protocol changed")
    seal = json.loads((OUT / "source-seal.json").read_text())
    packet = ROOT / "pointer-stimuli.jsonl"
    if sha(packet) != seal["input_sha256"]:
        raise ValueError("A reading-game input changed")
    rows = {r["id"]: r for r in map(json.loads, packet.read_text().splitlines())}
    results_dir = OUT / "selective"
    results_dir.mkdir(exist_ok=True)
    src, psha = validated_predictions(source, seal, seal["protocol_sha256"])
    dev = defaultdict(list)
    for article, row in rows.items():
        if row["fold"] == "dev":
            for q in row["queries"]:
                append(dev, row["group"], src[f"{article}/{q}"], row["clicks"][str(q)])
    policy = {method: calibrate(dev[method]) for method in METHODS}
    lock = results_dir / "source-selection.json"
    dump(lock, {"protocol_sha256": REGISTERED, "source_forecasts_sha256": psha, "methods": policy})
    audit = OUT / "external-clicks.json"
    if sha(audit) != seal["gold_sha256"]:
        raise ValueError("Human reader clicks changed")
    gold = defaultdict(list)
    works = defaultdict(list)
    for article, queries in json.loads(audit.read_text())["articles"].items():
        for q, human in queries.items():
            logits = src[f"{article}/{q}"]
            append(gold, article, logits, human["spans"])
            append(works, rows[article]["group"], logits, human["spans"])
    result = {"selection_sha256": sha(lock), "game": {m: score_policy(gold[m], policy[m]) for m in METHODS},
              "game_distinct_works": {m: score_policy(works[m], policy[m]) for m in METHODS}}
    dump(results_dir / "game.json", result)
    if external:
        gseal = json.loads((GAP / "source-seal.json").read_text())
        g, h = validated_predictions(external, gseal, gseal["protocol_sha256"])
        gp = GAP / "external-regions.json"
        if sha(gp) != gseal["gold_sha256"]:
            raise ValueError("Independent name regions changed")
        observations = defaultdict(list)
        for row in json.loads(gp.read_text())["rows"]:
            indices = row["B"] if row["target"] else row["A"]
            if indices != list(range(min(indices), max(indices) + 1)):
                raise ValueError("Verified name is not a contiguous source-word region")
            append(observations, row["article"], g[f"{row['id']}/{row['q']}"], [[min(indices), max(indices) + 1, 1]])
        dump(results_dir / "gap.json", {"selection_sha256": sha(lock), "forecast_sha256": h,
             "gap": {m: score_policy(observations[m], policy[m]) for m in METHODS}})
    print({method: (v["selected"] or {}).get("threshold") for method, v in policy.items()}, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--external", type=Path)
    args = parser.parse_args()
    run(args.source, args.external)
