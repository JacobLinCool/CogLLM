"""Check prefix invariance, cluster-bootstrap midranks and the official Italian reference independently."""

import json

import numpy as np
from scipy.stats import pearsonr, spearmanr

from research.reading_validation.coherence_order_data import OUT, WORD
from research.reading_validation.coherence_order_features import (
    BASE_COLUMNS,
    PRED_COLUMNS,
    SEM_COLUMNS,
    form,
    neural,
)
from research.reading_validation.coherence_order_genre_score import metrics
from research.reading_validation.coherence_order_reference import sha


def rank_check():
    truth = np.array([2, 3, 2, 7, 1, 4, 3])
    predictions = np.array([3.0, 5.0, 6.0, 2.0, 6.0, 4.0, 4.0])
    rng = np.random.default_rng(3374)
    counts = rng.multinomial(len(truth), np.full(len(truth), 1 / len(truth)), 300)
    matrix = np.vstack([np.ones(len(truth), int), counts])
    values = metrics(truth, predictions, matrix)
    errs = []
    for i, c in enumerate(counts):
        probes = (np.repeat(truth, c), np.repeat(predictions, c))
        outcomes = (
            np.mean((probes[0] - probes[1]) ** 2),
            np.mean(abs(probes[0] - probes[1])),
            pearsonr(*probes).statistic,
            spearmanr(*probes).statistic,
        )
        for k, v in zip(("mse", "mae", "pearson", "spearman"), outcomes, strict=True):
            got = values[k][1][i]
            if np.isnan(got) and np.isnan(v):
                continue
            errs.append(abs(got - v))
    e = max(errs)
    if not e < 1e-12:
        raise ValueError(
            "Weighted cluster ranks differ from expanding the actual replicated sample"
        )
    return float(e)


def prefix_check():
    prefix = "Heavy snow had fallen. The road was difficult to cross."
    suffixes = (
        " Guests ate a late meal.",
        " Xylophones accompanied zebras and canoes.",
    )
    rng = np.random.default_rng(123)
    hidden = rng.normal(size=(50, 64))
    attention = rng.uniform(size=(50, 3, 2))
    surprisal = rng.uniform(size=50)
    records = []
    for suffix in suffixes:
        t = prefix + suffix
        matches = list(WORD.finditer(t))
        ws = [m.group().casefold() for m in matches]
        coords = np.array([m.span() for m in matches])
        first = t.index(".") + 1
        last = len(prefix)
        spans = ((0, first), (first + 1, last), (last + 1, len(t)))
        form_result = form(t, ws, coords, first + 1, last, spans)
        valid = np.ones(len(ws), bool)
        valid[0] = False
        u = (
            coords,
            valid,
            surprisal[: len(ws)].copy(),
            hidden[: len(ws)].copy(),
            attention[: len(ws)].copy(),
        )
        # Perturb every future word and the attention channel to a sentence not yet seen.
        future = np.flatnonzero(coords[:, 0] >= last)
        if records:
            for x in u[2:]:
                x[future] += 7.5
            u[4][:, 2] += 31.3
        records.append(
            np.r_[form_result[0], neural(u, *form_result[1:], spans, first + 1)]
        )
    exp = len(BASE_COLUMNS) + len(PRED_COLUMNS) + len(SEM_COLUMNS)
    if len(records[0]) != exp or not np.array_equal(*records):
        raise ValueError(
            "A future word or invisible sentence changed an earlier boundary"
        )
    return exp


if __name__ == "__main__":
    result = {
        "prefix_invariant_width": prefix_check(),
        "weighted_midrank_max_abs_error": rank_check(),
        "official_sparse_reproduction": json.loads(
            (OUT / "reference-verification.json").read_text()
        ),
        "source_outcomes_opened": False,
    }
    file = OUT / "verification.json"
    file.write_text(json.dumps(result, indent=2) + "\n")
    print(sha(file), result)
