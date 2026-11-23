"""Score a second five-judge panel on strict changes at identical candidate pairs."""

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from research.reading_validation.local_forecast import digest
from research.reading_validation.nns_score import ci
from research.reading_validation.semlex_forecast import METHODS, read_ranks
from research.reading_validation.semlex_material import DATA, OUT

SEED = 420113
DRAWS = 10000
CHANCE = "context_invariant"


def summarize(cluster, methods):
    if not cluster:
        return {"independent_lemmas": 0, "scores": {}, "contrasts": {}}
    m = np.asarray([np.mean(v, axis=0) for _, v in sorted(cluster.items())])
    n = len(m)
    w = np.random.default_rng(SEED).multinomial(n, np.ones(n) / n, size=DRAWS)
    avg = m.mean(0)
    b = (w @ m) / n
    main = 0
    return {
        "independent_lemmas": n,
        "scores": {k: ci(avg[i], b[:, i]) for i, k in enumerate(methods)},
        "contrasts": {
            "mix_minus_" + k: ci(avg[main] - avg[i], b[:, main] - b[:, i])
            for i, k in enumerate(methods)
            if i != main
        },
    }


def test(keys, pred, human, sets):
    byword = defaultdict(list)
    bycontext = defaultdict(list)
    for i, (k, l, a, b) in enumerate(keys):
        if human[i] == 0:
            continue
        byword[l, a, b].append(i)
        bycontext[l, k].append(i)
    ordinary = defaultdict(list)
    for (l, k), idx in bycontext.items():
        hit = (1 + np.sign(pred[idx]) * human[idx, None]) / 2
        ordinary[l].append(np.mean(hit, axis=0))
    out = {
        "ordinary_pair_direction": summarize(ordinary, METHODS),
        "strictly_human_ordered_pairs": int((human != 0).sum()),
        "tied_aggregate_human_rank": int((human == 0).sum()),
        "total_ordinary_contexts": len(bycontext),
    }
    for label, same in (
        ("all_shared_candidate_pairs", False),
        ("identical_full_candidate_lists", True),
    ):
        pooled = {
            k: defaultdict(list)
            for k in ("relative_confidence", "relative_choice", "both_correct")
        }
        distinct = set()
        choices = 0
        comparisons = 0
        for (lemma, a, b), indices in byword.items():
            pair = [
                (i, j)
                for i, j in combinations(indices, 2)
                if human[i] * human[j] < 0
                and (not same or sets[keys[i, 0]] == sets[keys[j, 0]])
            ]
            if not pair:
                continue
            i, j = np.asarray(pair).T
            for g in np.r_[i, j]:
                distinct.add(keys[g, 0])
            choices += 1
            comparisons += len(i)
            hit_i = (1 + np.sign(pred[i]) * human[i, None]) / 2
            hit_j = (1 + np.sign(pred[j]) * human[j, None]) / 2
            rel = (1 + np.sign(pred[i] - pred[j]) * human[i, None]) / 2
            choice = (
                1 + np.sign(np.sign(pred[i]) - np.sign(pred[j])) * human[i, None]
            ) / 2
            for name, val, baseline in (
                ("relative_confidence", rel, 0.5),
                ("relative_choice", choice, 0.5),
                ("both_correct", hit_i * hit_j, 0.25),
            ):
                pooled[name][lemma].append(np.r_[val.mean(0), baseline])
        out[label] = {
            n: summarize(mat, (*METHODS, CHANCE)) for n, mat in pooled.items()
        }
        out[label].update(
            ordered_context_comparisons=comparisons,
            distinct_contexts=len(distinct),
            distinct_lemma_candidate_pairs=choices,
        )
    return out


def run():
    lock = json.loads((OUT / "analysis-lock.json").read_bytes())
    src = Path(__file__).parent
    if any(digest(src / k) != h for k, h in lock["methods_sha256"].items()):
        raise ValueError("Unexpected change to registered methods")
    seal = json.loads((OUT / "forecasts-seal.json").read_bytes())
    material = json.loads((OUT / "material.json").read_bytes())
    if (
        digest(OUT / "forecasts.npz") != seal["forecast_sha256"]
        or digest(DATA / "outcomes-reserve.txt") != material["held_ranking_sha256"]
    ):
        raise ValueError("Unsealed human targets or model comparison")
    with np.load(OUT / "forecasts.npz") as f:
        keys = f["keys"]
        pred = f["signed_a_preference"]
        names = tuple(f["methods"])
    if names != METHODS:
        raise ValueError("Comparator order changed")
    original = json.loads((DATA / "geometry.json").read_bytes())
    sets = {k: tuple(v["candidates"]) for k, v in original.items()}
    panel = read_ranks(DATA / "outcomes-reserve.txt")
    mask = np.array([a in panel[k] and b in panel[k] for k, _, a, b in keys])
    h = np.asarray([np.sign(panel[k][b] - panel[k][a]) for k, _, a, b in keys[mask]])
    result = test(keys[mask], pred[mask], h, sets)
    result.update(
        no_unique_consensus_rank=int((~mask).sum()),
        matched_prediction_pairs=int(mask.sum()),
        code_sha256=digest(Path(__file__)),
        forecast_seal_sha256=digest(OUT / "forecasts-seal.json"),
        analysis_sha256=digest(OUT / "analysis-lock.json"),
        bootstrap_draws=DRAWS,
    )
    primary = result["all_shared_candidate_pairs"]["relative_confidence"]
    result["primary_context_effect"] = bool(
        primary["independent_lemmas"] >= 10
        and primary["contrasts"]["mix_minus_context_invariant"]["ci95"][0] > 0
    )
    result["primary_beyond_native_reference"] = bool(
        primary["independent_lemmas"] >= 10
        and primary["contrasts"]["mix_minus_lexical_plus_surprisal"]["ci95"][0] > 0
    )
    (OUT / "scores.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(primary["scores"], flush=True)


if __name__ == "__main__":
    run()
