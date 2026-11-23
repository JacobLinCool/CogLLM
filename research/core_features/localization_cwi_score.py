"""Score sealed CWI span probabilities and annotator background on held-out HITs."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np

from ..stage3.probe import _jsonl
from ..stage5.complex_inputs import digest


METHODS = ("length_frequency", "qwen_pooled", "background_only", "qwen_group_offset", "qwen_group_interaction")
CONTRASTS = (
    ("qwen_pooled", "length_frequency"),
    ("qwen_group_offset", "qwen_pooled"),
    ("qwen_group_offset", "background_only"),
    ("qwen_group_interaction", "qwen_group_offset"),
    ("qwen_group_interaction", "qwen_pooled"),
)
TOTAL_KEYS = ("native_total", "nonnative_total")
MARK_KEYS = ("native_marked", "nonnative_marked")
OUTCOME_KEYS = {"id", "group_id", "start", "stop", "split", *TOTAL_KEYS, *MARK_KEYS}
PREDICTION_KEYS = {"id", "start", "stop", *METHODS}
SPLITS = {"train", "dev", "test"}
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 43
TIE_ATOL = 1e-12
LEXICAL_UNIT = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def load_inputs(outcomes_path, predictions_path, seal_path, extraction_path):
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    prediction_sha = digest(predictions_path)
    if seal["predictions_sha256"] != prediction_sha:
        raise ValueError("Pre-scoring prediction seal differs from predictions")
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    rejected = {(item["id"], item["stop"]) for item in extraction["excluded_stops"]}

    outcomes = {}
    splits_by_group = {}
    groups_by_stimulus = {}
    for item in _jsonl(outcomes_path):
        candidate = span_key(item)
        if (
            item.keys() != OUTCOME_KEYS or candidate in outcomes or item["split"] not in SPLITS
            or any(not isinstance(item[key], str) or not item[key] for key in ("id", "group_id"))
            or type(item["start"]) is not int or type(item["stop"]) is not int
            or not 0 <= item["start"] < item["stop"]
            or any(type(item[key]) is not int for key in (*TOTAL_KEYS, *MARK_KEYS))
            or not all(0 <= item[marked] <= item[total] and item[total] > 0
                       for marked, total in zip(MARK_KEYS, TOTAL_KEYS, strict=True))
        ):
            raise ValueError(f"Invalid released candidate: {item.get('id')}")
        if splits_by_group.setdefault(item["group_id"], item["split"]) != item["split"]:
            raise ValueError("A released HIT family crosses splits")
        if groups_by_stimulus.setdefault(item["id"], item["group_id"]) != item["group_id"]:
            raise ValueError("An exact stimulus appears in multiple analysis groups")
        outcomes[candidate] = item

    predictions = {}
    for row in _jsonl(predictions_path):
        candidate = span_key(row)
        if row.keys() != PREDICTION_KEYS or candidate in predictions or candidate not in outcomes:
            raise ValueError("Unexpected or repeated prediction identifier")
        outcome = outcomes[candidate]
        if outcome["split"] == "train":
            raise ValueError(f"Prediction does not match a held-out span: {row['id']}")
        probabilities = np.array([row[method] for method in METHODS], dtype=np.float64)
        if probabilities.shape != (len(METHODS), 2) or not np.all((0 < probabilities) & (probabilities < 1)):
            raise ValueError("Expected two finite, interior probabilities for every method")
        for name in ("length_frequency", "qwen_pooled"):
            pair = probabilities[METHODS.index(name)]
            if pair[0] != pair[1]:
                raise ValueError(f"A background-independent prediction varies by group: {name}")
        predictions[candidate] = probabilities

    rows, probabilities, exclusions = {}, {}, {}
    for split in ("dev", "test"):
        original = [item for item in outcomes.values() if item["split"] == split]
        excluded = [r for r in original if (r["id"], r["stop"]) in rejected]
        expected = {span_key(item) for item in original} - {span_key(item) for item in excluded}
        actual = {key for key in predictions if outcomes[key]["split"] == split}
        if not expected or actual != expected:
            raise ValueError(f"Unexpected prediction/extraction coverage for {split}")
        rows[split] = [outcomes[key] for key in sorted(actual)]
        probabilities[split] = np.stack([predictions[span_key(r)] for r in rows[split]], axis=0)
        exclusions[split] = {"released": len(original), "excluded_exact_stops": len(excluded)}
    return rows, probabilities, exclusions, prediction_sha


def span_key(row):
    return row["id"], row["start"], row["stop"]


class ClusterBootstrap:
    def __init__(self, groups):
        self.groups, self.indices = np.unique(groups, return_inverse=True)
        if len(self.groups) < 2:
            raise ValueError("At least two held-out clusters are needed")
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        size = len(self.groups)
        self.draw_weights = rng.multinomial(size, np.full(size, 1.0 / size), size=BOOTSTRAP_DRAWS)

    def mean(self, numerators, denominators, indices=None):
        index = self.indices if indices is None else indices
        numerator = np.bincount(index, weights=numerators, minlength=len(self.groups))
        denominator = np.bincount(index, weights=denominators, minlength=len(self.groups))
        counts = self.draw_weights @ denominator
        weighted = self.draw_weights @ numerator
        replicates = np.divide(weighted, counts, out=np.full(counts.shape, np.nan), where=counts > 0)
        return float(numerator.sum() / denominator.sum()), replicates


def interval(replicates):
    finite = replicates[np.isfinite(replicates)]
    if finite.size < BOOTSTRAP_DRAWS * 0.95:
        return None
    return np.quantile(finite, [0.025, 0.975]).tolist()


def observed(rows):
    votes = np.array([[row[k] for k in MARK_KEYS] for row in rows], dtype=np.float64)
    totals = np.array([[row[k] for k in TOTAL_KEYS] for row in rows], dtype=np.float64)
    return votes, totals


def threshold_f1(scores, gold):
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    valid = np.r_[sorted_scores[1:] < sorted_scores[:-1], True]
    positives = gold[order].cumsum()
    f1 = 2.0 * positives / (gold.sum() + np.arange(1, len(scores) + 1))
    if not gold.any():
        raise ValueError("No validation candidate has a positive mark")
    best = np.flatnonzero(valid)[np.argmax(f1[valid])]
    # Candidates are sorted from high to low score, so the first tied optimum is most selective.
    return float(sorted_scores[best]), float(f1[best])


def evaluate_masks(validation_rows, validation_probs, rows, probabilities, bootstrap):
    validation_votes, _ = observed(validation_rows)
    test_votes, _ = observed(rows)
    gold = test_votes.sum(axis=1) > 0
    validation_gold = validation_votes.sum(axis=1) > 0
    masks, samples = {}, {}
    for name in ("length_frequency", "qwen_pooled"):
        index = METHODS.index(name)
        validation_scores = validation_probs[:, index, 0]
        test_scores = probabilities[:, index, 0]
        threshold, validation_f1 = threshold_f1(validation_scores, validation_gold)
        selected = test_scores >= threshold
        hit = selected & gold
        count = gold.astype(int) + selected
        f1, samples[name] = bootstrap.mean(2 * hit, count)
        masks[name] = {
            "threshold": threshold, "validation_f1": validation_f1, "f1": f1,
            "true_positive": int(hit.sum()), "false_positive": int((selected & ~gold).sum()),
            "false_negative": int((~selected & gold).sum()),
            "true_negative": int((~selected & ~gold).sum()),
            "precision": float(hit.sum() / selected.sum()) if selected.any() else None,
            "recall": float(hit.sum() / gold.sum()) if gold.any() else None,
        }
    return {
        "gold": "At least one released native or non-native annotator marked the candidate.",
        "threshold_rule": "Choose validation F1 maximum. Break ties with the higher threshold.",
        "positive_test_candidates": int(gold.sum()), "methods": masks,
        "qwen_minus_length_frequency": {
            "test_f1_difference": masks["qwen_pooled"]["f1"] - masks["length_frequency"]["f1"],
            "article_family_bootstrap_95_interval": interval(samples["qwen_pooled"] - samples["length_frequency"]),
        },
    }


def evaluate_ranking(rows, probabilities, bootstrap):
    votes, totals = observed(rows)
    rating = votes.sum(axis=1) / totals.sum(axis=1)
    pooled = (probabilities * totals[:, None, :]).sum(axis=2) / totals.sum(axis=1)[:, None]
    by_excerpt = defaultdict(list)
    for i, row in enumerate(rows):
        by_excerpt[row["id"]].append(i)
    left, right = [], []
    for candidates in by_excerpt.values():
        for a, b in combinations(candidates, 2):
            span_a, span_b = rows[a], rows[b]
            if max(span_a["start"], span_b["start"]) < min(span_a["stop"], span_b["stop"]):
                continue
            if abs(rating[a] - rating[b]) > TIE_ATOL:
                left.append(a)
                right.append(b)
    if not left:
        raise ValueError("No unequal non-overlapping rated target pairs within released excerpts")
    left, right = np.array(left), np.array(right)
    delta = pooled[left] - pooled[right]
    direction = np.sign(rating[left] - rating[right])
    correct = np.where(np.abs(delta) <= TIE_ATOL, 0.5, delta * direction[:, None] > 0)
    selected_pairs = {
        "non_overlapping_unequal_selection_rates": np.ones(left.size, dtype=bool),
        "observed_positive_vs_zero_selection": (rating[left] == 0) ^ (rating[right] == 0),
    }
    result = {}
    for label, include in selected_pairs.items():
        if not include.any():
            raise ValueError(f"No pairs for {label}")
        pairs = int(include.sum())
        groups = bootstrap.indices[left[include]]
        base, _ = bootstrap.mean(correct[include, 0], np.ones(pairs), groups)
        qwen, _ = bootstrap.mean(correct[include, 1], np.ones(pairs), groups)
        difference, draws = bootstrap.mean(correct[include, 1] - correct[include, 0], np.ones(pairs), groups)
        result[label] = {
            "pairs": pairs, "excerpts": len({rows[i]["id"] for i in left[include]}),
            "article_families": len(set(groups)), "length_frequency": base, "qwen_pooled": qwen,
            "qwen_minus_length_frequency": difference, "article_family_bootstrap_95_interval": interval(draws),
        }
    return result


def group_scores(rows, probabilities, bootstrap):
    votes, totals = observed(rows)
    targets = votes / totals
    actual_delta = targets[:, 1] - targets[:, 0]
    unequal = np.abs(actual_delta) > TIE_ATOL
    participant_decisions = totals.sum(axis=1)
    denominators = {"selection_nll": participant_decisions, "selection_brier": participant_decisions, "gap_mae": np.ones(len(rows))}
    per_method, sampler = {}, {}
    for i, method in enumerate(METHODS):
        p = probabilities[:, i]
        if method == "background_only" and not np.all(p == p[:1]):
            raise ValueError("The background-only comparison varies across input spans")
        nll = -(votes * np.log(p) + (totals - votes) * np.log1p(-p))
        brier = votes * np.square(1 - p) + (totals - votes) * np.square(p)
        gap = p[:, 1] - p[:, 0]
        gap_error = np.abs(actual_delta - gap)
        metrics = {"selection_nll": nll.sum(axis=1), "selection_brier": brier.sum(axis=1), "gap_mae": gap_error}
        per_method[method], sampler[method] = {}, {}
        for metric, values in metrics.items():
            per_method[method][metric], sampler[method][metric] = bootstrap.mean(values, denominators[metric])
        per_method[method]["by_annotator_background"] = {
            label: {
                "selection_nll": float(nll[:, index].sum() / totals[:, index].sum()),
                "selection_brier": float(brier[:, index].sum() / totals[:, index].sum()),
                "observed_selection_rate": float(votes[:, index].sum() / totals[:, index].sum()),
                "predicted_selection_rate": float((p[:, index] * totals[:, index]).sum() / totals[:, index].sum()),
            } for index, label in enumerate(("native", "nonnative"))
        }
        direction = np.where(np.abs(gap[unequal]) <= TIE_ATOL, 0.5, gap[unequal] * actual_delta[unequal] > 0)
        per_method[method]["observed_gap_direction_concordance"] = float(direction.mean()) if unequal.any() else None
        per_method[method]["mean_predicted_nonnative_minus_native"] = float(gap.mean())

    contrasts = {}
    for name, base in CONTRASTS:
        contrasts[f"{name}_minus_{base}"] = {
            metric: {
                "difference": per_method[name][metric] - per_method[base][metric],
                "article_family_bootstrap_95_interval": interval(sampler[name][metric] - sampler[base][metric]),
            } for metric in denominators
        }
    return {
        "definition": "Selection NLL (nats) and Brier are per-annotator scores. Gap compares non-native minus native selection fractions per candidate.",
        "candidate_annotator_decisions": totals.sum(axis=0).astype(int).tolist(),
        "annotator_order": ["native", "nonnative"],
        "candidates_with_unequal_observed_fractions": int(unequal.sum()),
        "mean_observed_nonnative_minus_native": float(actual_delta.mean()),
        "methods": per_method, "paired_differences": contrasts,
    }


def write_masks(path, rows, probabilities, threshold):
    by_excerpt = defaultdict(list)
    for row, methods in zip(rows, probabilities, strict=True):
        pooled = float(methods[METHODS.index("qwen_pooled"), 0])
        native, nonnative = map(float, methods[METHODS.index("qwen_group_interaction")])
        by_excerpt[row["id"]].append({
            "start": row["start"], "end": row["stop"], "selection_probability": pooled,
            "highlight": pooled >= threshold,
            "background_conditioned_selection_probability": {"native": native, "nonnative": nonnative},
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for excerpt, spans in by_excerpt.items():
            record = {
                "id": excerpt, "offset_convention": "Unicode code point, half open",
                "scope": "Released CWI candidates certified at their exact causal end.",
                "pooled_threshold": threshold, "spans": spans,
            }
            stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")


def lexical_coverage(stimuli_path, rows):
    documents = {row["id"]: row for row in _jsonl(stimuli_path)}
    result = {}
    for split, candidates in rows.items():
        by_excerpt = defaultdict(set)
        for row in candidates:
            document = documents[row["id"]]
            if row["stop"] not in document["stops"] or row["stop"] > len(document["text"]):
                raise ValueError(f"Unknown candidate boundary in {row['id']}")
            by_excerpt[row["id"]].add((row["start"], row["stop"]))
        total = exact = included = 0
        for key, spans in by_excerpt.items():
            for unit in LEXICAL_UNIT.finditer(documents[key]["text"]):
                total += 1
                exact += unit.span() in spans
                included += any(start <= unit.start() and unit.end() <= stop for start, stop in spans)
        result[split] = {
            "excerpts_with_certified_candidates": len(by_excerpt), "lexical_units": total,
            "exact_word_candidates": exact, "words_enclosed_in_a_candidate": included,
        }
    return {
        "definition": "Lexical units are letter spans with internal apostrophes. Coverage is measured over excerpts with certified candidates.",
        "stimuli_sha256": digest(stimuli_path), "counts": result,
    }


def evaluate(outcomes, predictions, seal, extraction, output, masks_output=None, stimuli=None):
    rows, probabilities, exclusions, prediction_sha = load_inputs(outcomes, predictions, seal, extraction)
    test = rows["test"]
    bootstrap = ClusterBootstrap([r["group_id"] for r in test])
    result = {
        "scope": "CWI released candidates with exact Qwen prefix stops. Annotator background conditions prospective lexical highlight judgments.",
        "input_sha256": {"outcomes": digest(outcomes), "predictions": prediction_sha, "seal": digest(seal), "extraction": digest(extraction)},
        "coverage": exclusions,
        "test_candidates": len(test),
        "test_article_families": len(bootstrap.groups),
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "article family"},
        "span_masks": evaluate_masks(rows["dev"], probabilities["dev"], test, probabilities["test"], bootstrap),
        "within_excerpt_ranking": evaluate_ranking(test, probabilities["test"], bootstrap),
        "annotator_background": group_scores(test, probabilities["test"], bootstrap),
    }
    if stimuli is not None:
        result["lexical_unit_coverage"] = lexical_coverage(stimuli, rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if masks_output is not None:
        threshold = result["span_masks"]["methods"]["qwen_pooled"]["threshold"]
        write_masks(masks_output, test, probabilities["test"], threshold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("outcomes", "predictions", "seal", "extraction", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--masks-output", type=Path)
    parser.add_argument("--stimuli", type=Path)
    evaluate(**vars(parser.parse_args()))
