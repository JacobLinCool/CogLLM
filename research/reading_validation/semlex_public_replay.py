"""Reproduce direction of preference change using public SemEval ranks and NumPy."""

import argparse
import json
import re
from collections import defaultdict
from hashlib import file_digest
from itertools import combinations
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1] / "results/reading-validation-semlex"


def sha(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def load_ranks(path):
    table = {}
    for i, line in enumerate(path.read_text().splitlines(), 1):
        header, sep, body = line.partition(" rankings: ")
        if header != f"Sentence {i}" or not sep:
            raise ValueError(f"Unrecognized published rank {i}")
        groups = re.findall(r"\{([^{}]*)\}", body)
        if not groups or re.sub(r"\{[^{}]*\}", "", body).strip():
            raise ValueError(f"Bad rank levels {i}")
        assignments = defaultdict(set)
        for rank, group in enumerate(groups):
            for w in group.split(","):
                assignments[w.strip()].add(rank)
        table[str(i + 300)] = {
            w: next(iter(v)) for w, v in assignments.items() if len(v) == 1
        }
    return table


def score(keys, signed, ranks):
    instances = defaultdict(list)
    for i, (ctx, lemma, a, b) in enumerate(keys):
        ordering = ranks[ctx]
        if a not in ordering or b not in ordering or ordering[a] == ordering[b]:
            continue
        human_a = ordering[a] < ordering[b]
        instances[lemma, a, b].append((i, human_a))
    families = defaultdict(list)
    contexts = set()
    comparisons = 0
    for (lemma, _, _), group in instances.items():
        evidence = []
        for (i, prefer_i), (j, prefer_j) in combinations(group, 2):
            if prefer_i == prefer_j:
                continue
            good, poor = (i, j) if prefer_i else (j, i)
            evidence.append(0.5 + 0.5 * np.sign(signed[good] - signed[poor]))
            contexts.update((keys[i, 0], keys[j, 0]))
        if evidence:
            comparisons += len(evidence)
            families[lemma].append(np.mean(evidence, axis=0))
    matrix = np.array([np.mean(families[k], axis=0) for k in sorted(families)])
    if not len(matrix):
        raise ValueError("No evidence of an observed preference reversal")
    return matrix, len(contexts), comparisons


def run(path):
    material = json.loads((OUT / "material.json").read_text())
    seal = json.loads((OUT / "forecasts-seal.json").read_text())
    forecast = OUT / "forecasts.npz"
    if (
        sha(path) != material["held_ranking_sha256"]
        or sha(forecast) != seal["forecast_sha256"]
    ):
        raise ValueError("Different external study or model forecast")
    with np.load(forecast, allow_pickle=False) as f:
        keys, names, signals = (
            f["keys"],
            f["methods"].tolist(),
            f["signed_a_preference"],
        )
    ranks = load_ranks(path)
    matrix, contexts, pairs = score(keys, signals, ranks)
    n = len(matrix)
    baseline = (
        np.zeros(n) + 0.5,
        *[
            matrix[:, names.index(k)]
            for k in ("lexical_plus_surprisal", "paper_adaptive_ngram")
        ],
    )
    main = matrix[:, names.index("joint_native_mix")]
    values = np.column_stack((main, *[main - b for b in baseline]))
    draw = np.random.default_rng(420113).multinomial(n, np.ones(n) / n, size=10000)
    interval = np.quantile(draw @ values / n, [0.025, 0.975], axis=0)
    report = {
        "human_rank_sha256": sha(path),
        "forecast_sha256": sha(forecast),
        "lemmas": n,
        "contexts": contexts,
        "observed_context_reversals": pairs,
    }
    result = json.loads((OUT / "scores.json").read_text())["all_shared_candidate_pairs"]
    estimates = result["relative_confidence"]
    if (
        result["ordered_context_comparisons"] != pairs
        or result["distinct_contexts"] != contexts
        or estimates["independent_lemmas"] != n
    ):
        raise ValueError("Reversal counts differ")
    for i, key in enumerate(
        (
            "joint_native_mix",
            "mix_minus_context_invariant",
            "mix_minus_lexical_plus_surprisal",
            "mix_minus_paper_adaptive_ngram",
        )
    ):
        entry = estimates["scores" if i == 0 else "contrasts"][key]
        numbers = [values[:, i].mean(), *interval[:, i]]
        if not np.allclose(
            numbers, [entry["estimate"], *entry["ci95"]], rtol=0, atol=1e-12
        ):
            raise ValueError("Report diverged on " + key)
        report[key] = {"estimate": numbers[0], "ci95": numbers[1:]}
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ranks", type=Path)
    print(json.dumps(run(parser.parse_args().ranks), indent=2, allow_nan=False))
