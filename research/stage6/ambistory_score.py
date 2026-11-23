"""Score sealed AmbiStory predictions on the original whole-family holdout."""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from statistics import stdev

import numpy as np


DESIGN_PROTOCOL = Path(__file__).with_name("ambistory-context-update.md")
FIT_CODE = Path(__file__).with_name("ambistory_fit.py")
PREPARATION_SHA256 = "0364a1942796740435c060da2d86a26774d6c794827dfd0cecae8037fafd3253"
METHODS = ("form_control", "common_prefix", "ending")
GRADES = (1, 2, 3, 4, 5)
FAMILIES, HOMONYMS, STORIES, SENSES, VOTES, FLAGGED = 155, 86, 465, 930, 4663, 91
DRAWS, SEED = 10_000, 43
MARGIN = 0.5
TIE_ATOL = 1e-12
SEM_EVAL = "https://github.com/Janosch-Gehring/semeval26-05-scripts/blob/362fd1c51fca99bd329bdc9a84fdf509a6ff6c3b/scoring.py"
PREP_FILES = {
    "train_design": "train/design/stops.jsonl", "train_outcomes": "train/scoring/ordinal-ratings.jsonl",
    "test_design": "test/design/stops.jsonl", "test_outcomes": "test/scoring/ordinal-ratings.jsonl",
}
FEATURE_FILES = {
    "feature_manifest": "extraction.json", "story_features": "story-states.npz",
    "sense_features": "sense-states.npz", "causality": "causality.json", "extraction_seal": "seal.json",
}


@dataclass(frozen=True)
class Holdout:
    keys: tuple[tuple[str, str], ...]
    families: tuple[str, ...]
    pair_clusters: np.ndarray


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        yield from map(json.loads, stream)


def read_holdout(path):
    families = defaultdict(list)
    seen = set()
    for row in jsonl(path):
        if (row["id"] in seen or row["homonym"] != row["homonym"].casefold()
                or type(row["has_ending"]) is not bool or len(row["senses"]) != 2):
            raise ValueError("Invalid story variant, sense count or homonym in the sealed design")
        seen.add(row["id"])
        families[row["family"]].append(row)
    homonyms = sorted({row["homonym"] for variants in families.values() for row in variants})
    if len(families) != FAMILIES or len(seen) != STORIES or len(homonyms) != HOMONYMS:
        raise ValueError("Unexpected test story inventory")
    index = {homonym: i for i, homonym in enumerate(homonyms)}
    keys, family_ids, pair_clusters = [], [], []
    for family, variants in sorted(families.items()):
        base = [row for row in variants if not row["has_ending"]]
        endings = sorted((row for row in variants if row["has_ending"]), key=lambda row: row["id"])
        if len(base) != 1 or len(endings) != 2 or len({row["homonym"] for row in variants}) != 1:
            raise ValueError("Every family needs two endings, one base and a common homonym")
        senses = {sense["id"] for sense in base[0]["senses"]}
        if len(senses) != 2 or any({sense["id"] for sense in row["senses"]} != senses for row in endings):
            raise ValueError("Expected the two original sense identities in all three versions")
        for sense in sorted(senses):
            keys.extend((row["id"], sense) for row in (*base, *endings))
            family_ids.append(family)
            pair_clusters.append(index[base[0]["homonym"]])
    return Holdout(tuple(keys), tuple(family_ids), np.array(pair_clusters))


def verify_fit(args, plan):
    if read_json(args.score_protocol) != plan or digest(args.preparation) != PREPARATION_SHA256:
        raise ValueError("Score policy or preparation differs from the frozen receipts")
    prep = read_json(args.preparation)
    fit = read_json(args.seal)
    expected = {
        "preparation": PREPARATION_SHA256,
        **{key: prep["files_sha256"][filename] for key, filename in PREP_FILES.items()},
        **{key: digest(args.extraction / filename) for key, filename in FEATURE_FILES.items()},
    }
    if (fit["status"] != "fit_and_forecasts_locked_before_scoring"
            or fit["protocol_sha256"] != plan["fit_protocol_sha256"]
            or fit["fit_code_sha256"] != plan["fit_code_sha256"]
            or fit["input_sha256"] != expected or fit["predictions_sha256"] != digest(args.forecasts)
            or tuple(fit["grades"]) != GRADES or tuple(fit["methods"]) != METHODS
            or fit["fit_split"] != "train" or fit["forecast_split"] != "test" or fit["forecast_samples"] != SENSES):
        raise ValueError("Unexpected fit identity, partition, feature or forecast bytes")
    for key, filename in PREP_FILES.items():
        if digest(args.prepared / filename) != expected[key]:
            raise ValueError(f"Unexpected edition of {filename}")
    return read_holdout(args.prepared / PREP_FILES["test_design"])


def forecasts_at_stops(path, holdout):
    predictions = {}
    for row in jsonl(path):
        key = row["id"], row["sense"]
        if row.keys() != {"id", "sense", *METHODS} or key in predictions:
            raise ValueError("Duplicate sense item or unexpected forecast schema")
        predictions[key] = [row[method] for method in METHODS]
    if predictions.keys() != set(holdout.keys):
        raise ValueError("Every sense in the held-out whole families needs one forecast")
    p = np.array([predictions[key] for key in holdout.keys], dtype=np.float64)
    if (p.shape != (SENSES, len(METHODS), len(GRADES))
            or not np.all(np.isfinite(p) & (p > 0) & (p <= 1))
            or not np.allclose(p.sum(axis=-1), 1, rtol=0, atol=1e-7)):
        raise ValueError("Expected complete strictly positive ordered plausibility distributions")
    variants = p.reshape(-1, 3, len(METHODS), len(GRADES))
    if (not np.all(variants[:, 1:, :2, :] == variants[:, :1, :2, :])
            or not np.array_equal(variants[:, 0, 2, :], variants[:, 0, 1, :])):
        raise ValueError("Form and prefix must be exactly shared; the base cannot contain an ending update")
    return p


def official_votes(path, holdout):
    ratings = {}
    samples = set()
    for row in jsonl(path):
        key = row["id"], row["sense_id"]
        grades, flags = row["ratings"], row["nonsensical"]
        if (key in ratings or row["author_sample"] in samples or len(grades) < 5 or len(flags) != len(grades)
                or any(type(grade) is not int or grade not in GRADES for grade in grades)
                or any(type(flag) is not bool for flag in flags)):
            raise ValueError("Unexpected author sense-sample votes")
        ratings[key] = grades, flags
        samples.add(row["author_sample"])
    if ratings.keys() != set(holdout.keys):
        raise ValueError("Human vote coverage differs from the fixed holdout")
    if (sum(len(grades) for grades, _ in ratings.values()) != VOTES
            or sum(sum(flags) for _, flags in ratings.values()) != FLAGGED):
        raise ValueError("Vote totals differ from the frozen edition")
    return [ratings[key] for key in holdout.keys]


def midranks(values):
    order = np.argsort(values, kind="stable")
    unique, counts = np.unique(values, return_counts=True)
    starts = np.cumsum(counts) - counts
    ranks = np.empty(len(values))
    ranks[order] = np.repeat(starts + (counts + 1) / 2, counts)
    return ranks, len(unique)


def semeval_metrics(observations, expected):
    # The published task requests integers. The official threshold is strict.
    eligible = np.array([bool(votes) for votes in observations])
    means = np.array([sum(votes) / len(votes) for votes in observations if votes])
    gold_ranks, gold_unique = midranks(means)
    eligible_sd = np.array([len(votes) >= 2 for votes in observations if votes])
    sd = np.array([stdev(votes) for votes in observations if len(votes) >= 2])
    result = {}
    for index, method in enumerate(METHODS):
        point = np.rint(expected[eligible, index])
        ranks, unique = midranks(point)
        rho = float(np.corrcoef(gold_ranks, ranks)[0, 1]) if gold_unique > 1 and unique > 1 else None
        p, m = point[eligible_sd], means[eligible_sd]
        accuracy = float(np.mean(((m - sd < p) & (p < m + sd)) | (np.abs(m - p) < 1)))
        result[method] = {"spearman": rho, "accuracy_within_sd_or_one": accuracy}
    return {"spearman_items": len(means), "within_sd_items": len(sd), "methods": result}


def cluster_means(totals, counts, clusters, weights):
    numerator = np.array([np.bincount(clusters, weights=col, minlength=HOMONYMS) for col in totals.T]).T
    denominator = np.bincount(clusters, weights=counts, minlength=HOMONYMS)
    n = weights @ denominator
    if np.any(n == 0):
        raise ValueError("A bootstrap draw contains no eligible observations")
    return (weights @ numerator) / n[:, None]


def interval(draws, confidence=95):
    alpha = (100 - confidence) / 200
    return {"estimate": float(draws[0]), f"ci{confidence}": np.quantile(draws[1:], (alpha, 1 - alpha)).tolist()}


def exact_updates(holdout, grades, include):
    cases = np.array(holdout.keys).reshape(-1, 3, 2)
    labels = ("human", *METHODS)
    for index, family in enumerate(holdout.families):
        base, first, second = cases[index]
        current = grades[index]
        values = (current[2] - current[1], current[1] - current[0], current[2] - current[0])
        names = ("second_minus_first", "first_minus_base", "second_minus_base")
        if include[index]:
            yield {"family": family, "sense": str(base[1]), "base_id": str(base[0]),
                   "first_ending_id": str(first[0]), "second_ending_id": str(second[0]),
                   "update": {name: {
                       key: float(value) if np.isfinite(value) else None for key, value in zip(labels, difference, strict=True)
                   } for name, difference in zip(names, values, strict=True)}}


def summarize(holdout, probabilities, observations, weights):
    clusters = np.repeat(holdout.pair_clusters, 3)
    selected = np.array([len(votes) for votes in observations])
    count_by_grade = np.array([np.bincount(votes, minlength=6)[1:] for votes in observations])
    losses = -np.einsum("ig,img->im", count_by_grade, np.log(probabilities))
    nll = cluster_means(losses, selected, clusters, weights)
    ended = np.arange(SENSES) % 3 != 0
    ended_nll = cluster_means(losses[ended], selected[ended], clusters[ended], weights)
    expected = probabilities @ GRADES
    observed = np.divide(count_by_grade @ GRADES, selected, out=np.full(SENSES, np.nan), where=selected > 0)
    grades = np.column_stack((observed, expected)).reshape(-1, 3, len(METHODS) + 1)
    include = np.isfinite(grades[:, 1:, 0]).all(axis=1)
    delta = grades[include, 2, :] - grades[include, 1, :]
    pair_cluster = holdout.pair_clusters[include]
    count = np.ones(len(pair_cluster))
    mae = cluster_means(np.abs(delta[:, 1:] - delta[:, :1]), count, pair_cluster, weights)
    amplitude = cluster_means(np.abs(delta), count, pair_cluster, weights)
    sign = np.sign(np.where(np.abs(delta) <= TIE_ATOL, 0, delta))
    direction = (1 + sign[:, :1] * sign[:, 1:]) / 2
    direction_agreement = cluster_means(direction, count, pair_cluster, weights)
    posterior_gain = interval(nll[:, 2] - nll[:, 1])
    update_gain = interval(mae[:, 2] - mae[:, 1])
    magnitude_gap = interval(amplitude[:, 3] - amplitude[:, 0], 90)
    improvement = posterior_gain["ci95"][1] < 0 and update_gain["ci95"][1] < 0
    equivalent = -MARGIN < magnitude_gap["ci90"][0] and magnitude_gap["ci90"][1] < MARGIN
    return {
        "counts": {"rated_items": int(np.count_nonzero(selected)), "votes": int(sum(selected)),
                   "ending_pairs": len(delta), "ending_pair_homonyms": len(set(pair_cluster))},
        "methods": {
            method: {"all_variant_vote_nll_nats": interval(nll[:, i]),
                     "with_ending_vote_nll_nats": interval(ended_nll[:, i]),
                     "ending_pair_update_mae_grades": interval(mae[:, i]),
                     "mean_absolute_ending_pair_update_grades": interval(amplitude[:, i + 1]),
                     "mean_absolute_update_minus_human_grades": interval(amplitude[:, i + 1] - amplitude[:, 0], 90),
                     "ending_pair_direction_agreement": interval(direction_agreement[:, i])}
            for i, method in enumerate(METHODS)
        },
        "human_mean_absolute_ending_pair_update_grades": interval(amplitude[:, 0]),
        "ending_minus_common_prefix": {"all_variant_vote_nll_nats": posterior_gain,
                                       "ending_pair_update_mae_grades": update_gain},
        "candidate_gate": {"two_primary_improvement_intervals_below_zero": improvement,
                           "mean_absolute_update_magnitude_equivalent": equivalent,
                           "primary_improvement_and_magnitude_equivalence": improvement and equivalent},
        "semeval_secondary": semeval_metrics(observations, expected),
        "signed_ending_updates": list(exact_updates(holdout, grades, include)),
    }


def score_protocol(fit_protocol):
    training = read_json(fit_protocol)
    design_hash, fit_hash = digest(DESIGN_PROTOCOL), digest(FIT_CODE)
    if (training["input_preparation_sha256"] != PREPARATION_SHA256
            or training["design_protocol_sha256"] != design_hash
            or training["fit_code_sha256"] != fit_hash
            or tuple(training["methods"]) != METHODS or tuple(training["grades"]) != GRADES
            or training["fit_split"] != "train" or training["forecast_split"] != "test"):
        raise ValueError("Cannot freeze scoring against a different fit or holdout protocol")
    return {
        "status": "scoring_rules_locked_before_real_holdout_scoring",
        "design_protocol_sha256": design_hash,
        "fit_protocol_sha256": digest(fit_protocol),
        "fit_code_sha256": fit_hash,
        "preparation_sha256": PREPARATION_SHA256,
        "scorer_sha256": digest(Path(__file__)),
        "fit_split": "train",
        "forecast_split": "test",
        "counts": {"story_families": FAMILIES, "stories": STORIES, "sense_items": SENSES, "votes": VOTES,
                   "flagged_votes": FLAGGED, "casefold_homonym_groups": HOMONYMS},
        "methods": list(METHODS),
        "grades": list(GRADES),
        "primary": ["all-variant, per-vote NLL in nats", "mean absolute error of same-sense signed ending changes"],
        "primary_contrast": "ending minus common_prefix, paired 95% upper bounds below zero for both measures",
        "signed_update": "later-sorting opaque ending ID minus earlier-sorting opaque ending ID, identical order across "
                         "human means and model expectations; ending minus base reported separately",
        "direction_agreement": {"tie_credit": 0.5, "tie_grade_atol": TIE_ATOL},
        "equivalence": {"statistic": "model minus human, mean absolute signed ending change in grade units",
                        "paired_ci": 90, "strict_margin_grades": MARGIN},
        "bootstrap": {"unit": "casefolded homonym with all families, versions and votes kept together",
                      "draws": DRAWS, "seed": SEED, "paired": True},
        "semeval_secondary": {"reference": SEM_EVAL,
                              "point_readout": "round expected grade to nearest integer, half ties to even",
                              "rank": "Spearman with tied ranks averaged; constant input undefined",
                              "accuracy": "strictly inside human mean ± sample SD or absolute difference < 1"},
        "nonsensical_votes": "include all votes in primary analysis; repeat with flagged votes removed for sensitivity",
        "sensitivity_missing": "retain each nonempty rating item; require both endings for paired update and more than one "
                               "rating for within-SD accuracy",
    }


def write_json(path, payload):
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as output:
        output.write(text)


def score(args):
    plan = score_protocol(args.fit_protocol)
    holdout = verify_fit(args, plan)
    p = forecasts_at_stops(args.forecasts, holdout)
    observed = official_votes(args.prepared / PREP_FILES["test_outcomes"], holdout)
    weights = np.vstack((np.ones(HOMONYMS), np.random.default_rng(SEED).multinomial(
        HOMONYMS, np.full(HOMONYMS, 1 / HOMONYMS), size=DRAWS,
    )))
    expanded = summarize(holdout, p, [votes for votes, _ in observed], weights)
    filtered = summarize(holdout, p, [[vote for vote, flag in zip(votes, flags, strict=True) if not flag]
                                      for votes, flags in observed], weights)
    write_json(args.output, {
        "status": "scored_against_original_held_out_human_votes",
        "score_protocol_sha256": digest(args.score_protocol),
        "fit_seal_sha256": digest(args.seal),
        "forecasts_sha256": digest(args.forecasts),
        "primary": expanded,
        "excluding_nonsensical_votes_sensitivity": filtered,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    freeze = actions.add_parser("freeze")
    for name in ("fit-protocol", "output"):
        freeze.add_argument(f"--{name}", type=Path, required=True)
    evaluate = actions.add_parser("score")
    for name in ("prepared", "preparation", "extraction", "fit-protocol", "score-protocol", "forecasts", "seal", "output"):
        evaluate.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "freeze":
        write_json(args.output, score_protocol(args.fit_protocol))
    else:
        score(args)


if __name__ == "__main__":
    main()
