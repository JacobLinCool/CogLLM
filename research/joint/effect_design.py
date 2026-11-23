"""Prepare fold-local direct supervision and disjoint literature effect targets."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .prepare import prepare_plaba, prepare_zhan
from ..stage3.probe import TARGETS, _jsonl
from ..stage4.zhan_weak import LABELS_PER_CELL, _contrast_coefficients


FOLDS = range(5)
SEEDS = np.arange(5, dtype=np.int64)
RATING_SCALE = np.arange(1, 6)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plaba_trials(path):
    rows = defaultdict(list)
    for trial in _jsonl(path):
        if trial["exposure"]["task"] != "sentence_rating":
            continue
        observations = trial["observations"]
        ratings = tuple(observations[name]["value"] for name in TARGETS)
        checklist = observations["issue_tags"]
        votes = None
        if checklist["status"] == "observed":
            tags = checklist["value"]
            votes = (int(bool(tags)), int("unclear_term" in tags))
        rows[trial["target_segment_id"]].append((ratings, votes))
    return rows


def plaba_alignments(path, ids, folds):
    lookup = {identifier: index for index, identifier in enumerate(ids)}
    anchors = defaultdict(lambda: defaultdict(list))
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as source:
        for item in csv.DictReader(source):
            identifier = item["sentence_uid"]
            variant = item["variant"]
            if identifier in seen or identifier not in lookup or variant not in {"orig", "simp"}:
                raise ValueError(f"Unexpected PLABA alignment row for {identifier}")
            seen.add(identifier)
            key = (item["base_text_id"], int(item["mapped_orig_sentence_index"]))
            anchors[key][variant].append(lookup[identifier])

    alignments = [
        (np.asarray(item["orig"]), np.asarray(item["simp"]))
        for item in anchors.values() if {"orig", "simp"} <= item.keys()
    ]
    if seen != set(lookup) or len(alignments) != 265:
        raise ValueError("PLABA alignments differ from the verified Stage 2 contrast")
    for original, simplified in alignments:
        if len(set(folds[np.r_[original, simplified]])) != 1:
            raise ValueError("A PLABA alignment crosses the locked folds")
    return alignments


def plaba_fold_draw(ids, rows, ratings, issues, alignments, train, fold, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, fold, 2]))
    direct_ratings = np.zeros_like(ratings)
    direct_issues = np.zeros_like(issues)
    for index in np.flatnonzero(train):
        candidates = rows[ids[index]]
        values, votes = candidates[rng.integers(len(candidates))]
        for endpoint, value in enumerate(values):
            direct_ratings[index, endpoint, value - 1] = 1
        if votes is not None:
            for endpoint, value in enumerate(votes):
                direct_issues[index, endpoint, value] = 1

    if (direct_ratings > ratings).any() or (direct_issues > issues).any():
        raise ValueError("PLABA direct labels are inconsistent with the full counts")
    remainder = (ratings - direct_ratings)[:, 1]
    remaining_trials = remainder.sum(axis=-1)
    contrast = np.zeros(len(ids), dtype=np.float64)
    usable = train & (remaining_trials > 0)
    retained = []
    for original, simplified in alignments:
        if usable[original].all() and usable[simplified].all():
            retained.append((original, simplified))
    if not retained:
        raise ValueError(f"No complete residual PLABA alignments in fold {fold}, seed {seed}")

    for original, simplified in retained:
        contrast[original] -= 1 / (len(retained) * len(original))
        contrast[simplified] += 1 / (len(retained) * len(simplified))
    selected = contrast != 0
    means = remainder[selected] @ RATING_SCALE / remaining_trials[selected]
    target = float(contrast[selected] @ means)
    audit = {
        "direct_rating_trials": int(direct_ratings[:, 1].sum()),
        "direct_checklist_trials": int(direct_issues[:, 0].sum()),
        "effect_alignment_positions": len(retained),
        "effect_sentence_rows": int(selected.sum()),
        "effect_disjoint_rating_trials": int(remaining_trials[selected].sum()),
        "effect_target_rating_points": target,
    }
    return direct_ratings, direct_issues, contrast, target, len(retained), audit


def zhan_fold_draw(payload, train, counts, verbs, seed):
    train_rows = np.flatnonzero(train)
    totals = counts[train].sum(axis=-1)
    positives = counts[train, 1]
    contrast = _contrast_coefficients(verbs[train][::2])
    if (
        payload["train_cells"] != train_rows.tolist()
        or payload["full_counts"] != np.column_stack((positives, totals)).tolist()
        or not np.array_equal(payload["contrast_coefficients"], contrast)
    ):
        raise ValueError(f"Zhan fold {payload['fold']} differs from the locked Stage 4 contrast")
    draw = payload["draws"][seed]
    if draw["seed"] != seed:
        raise ValueError("Unexpected Zhan draw seed")

    chosen = np.asarray(draw["limited_positives"], dtype=np.int64)
    direct = np.zeros_like(counts)
    direct[train] = np.column_stack((LABELS_PER_CELL - chosen, chosen))
    if (direct < 0).any() or (direct > counts).any() or (totals <= LABELS_PER_CELL).any():
        raise ValueError("Zhan direct labels are inconsistent with the full counts")
    target = float(contrast @ ((positives - chosen) / (totals - LABELS_PER_CELL)))
    if not np.isclose(target, draw["effect_target"], atol=1e-15, rtol=0):
        raise ValueError("Zhan target must be computed only from non-direct records")
    coefficients = np.zeros(len(counts), dtype=np.float64)
    coefficients[train] = contrast
    audit = {
        "direct_choice_trials": int(direct.sum()),
        "effect_disjoint_choice_trials": int(totals.sum() - direct.sum()),
        "effect_target_probability": target,
    }
    return direct, coefficients, target, audit


def prepare(data, design_path, plaba_sentences, zhan_design, output):
    with np.load(design_path, allow_pickle=False) as source:
        plaba = prepare_plaba(data / "plaba")
        zhan = prepare_zhan(data / "zhan")
        if any(not np.array_equal(source[name], value) for name, value in (plaba | zhan).items()):
            raise ValueError("Joint design does not match the locked Stage 2 records")
        onestop_counts = source["onestop_counts"]
        onestop_folds = source["onestop_folds"]
    ids = plaba["plaba_ids"]
    plaba_folds = plaba["plaba_folds"]
    rows = plaba_trials(data / "plaba/trials.jsonl")
    if set(rows) != set(ids):
        raise ValueError("Missing direct PLABA trial candidates")
    alignments = plaba_alignments(plaba_sentences, ids, plaba_folds)

    zhan_source = json.loads(zhan_design.read_text(encoding="utf-8"))
    zhan_folds = zhan["zhan_folds"]
    zhan_sources = {
        name: digest(data / "zhan" / name)
        for name in ("stimuli.jsonl", "trials.jsonl", "splits.json")
    }
    if (
        zhan_source["stimulus_ids"] != zhan["zhan_ids"].tolist()
        or np.repeat(zhan_source["family_folds"], 2).tolist() != zhan_folds.tolist()
        or zhan_source["trials_per_limited_stimulus"] != LABELS_PER_CELL
        or zhan_source["source_sha256"] != zhan_sources
        or [row["fold"] for row in zhan_source["fold_designs"]] != list(FOLDS)
    ):
        raise ValueError("Zhan inputs no longer match the Stage 4 sampling design")

    keys = (
        "plaba_rating_counts", "plaba_issue_counts", "plaba_effect_coefficients",
        "plaba_effect_targets", "plaba_alignment_n", "zhan_counts",
        "zhan_effect_coefficients", "zhan_effect_targets",
    )
    stored = {name: [] for name in keys}
    diagnostics = []
    for fold, payload in zip(FOLDS, zhan_source["fold_designs"], strict=True):
        fold_draws = []
        report = []
        for seed in SEEDS.tolist():
            *plaba_draw, plaba_report = plaba_fold_draw(
                ids, rows, plaba["plaba_rating_counts"], plaba["plaba_issue_counts"],
                alignments, plaba_folds != fold, fold, seed,
            )
            *zhan_draw, zhan_report = zhan_fold_draw(
                payload, zhan_folds != fold, zhan["zhan_counts"], zhan["zhan_verb_type"], seed,
            )
            fold_draws.append(plaba_draw + zhan_draw)
            report.append({"seed": seed, "plaba": plaba_report, "zhan": zhan_report})
        for name, values in zip(keys, zip(*fold_draws, strict=True), strict=True):
            stored[name].append(np.stack(values))
        diagnostics.append({
            "fold": fold,
            "onestop_full_training_choice_trials": int(onestop_counts[onestop_folds != fold].sum()),
            "draws": report,
        })

    output.mkdir(parents=True, exist_ok=True)
    archive = output / "effect-design.npz"
    np.savez_compressed(archive, seeds=SEEDS, **{name: np.stack(values) for name, values in stored.items()})
    report = {
        "scope": "Fold-local aggregate counts. Held-out rows are zero in all direct and effect arrays. OneStop uses all training-fold choices and no effect.",
        "sampling": {
            "seeds": SEEDS.tolist(),
            "plaba": "One entire sentence trial per training sentence. Ratings and available checklist come from the same trial. Remaining understanding ratings are used only at complete orig/simp alignment positions.",
            "zhan": "Reuses the locked Stage 4 draw seeds and four direct choices per training prompt. Remaining choices form the effect target.",
        },
        "plaba_effect": "Mean simplified-minus-original understanding rating with equal weight to each complete alignment position and equal weight to sentences within each variant.",
        "zhan_effect": "IC-2 minus IC-1 interaction in the pronoun-minus-free probability to choose NP1, equal weight per family within verb type.",
        "input_sha256": {
            "joint_design.npz": digest(design_path),
            "plaba_sentences.csv": digest(plaba_sentences),
            "stage4_design.json": digest(zhan_design),
            **{
                f"{task}/{name}": digest(data / task / name)
                for task in ("plaba", "zhan") for name in ("stimuli.jsonl", "trials.jsonl", "splits.json")
            },
        },
        "output_sha256": digest(archive),
        "folds": diagnostics,
    }
    (output / "effect-preparation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "design", "plaba-sentences", "zhan-design", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.data, args.design, args.plaba_sentences, args.zhan_design, args.output)
