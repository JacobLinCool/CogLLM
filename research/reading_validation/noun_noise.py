"""Bound effective cloze cell sizes and their measurement variance from released rational means."""

import json
from pathlib import Path

import numpy as np

from research.reading_validation.local_forecast import digest
from research.reading_validation.noun_data import OUT, design

TRIALS = 11949
MAX_CELL = 35
TOL = 1e-6


def feasible_counts(sets, total):
    """Exact integer totals with every prefix and suffix; cellwise support conditional on the sum."""
    n = len(sets)
    before = np.zeros((n + 1, total + 1), bool)
    after = np.zeros_like(before)
    before[0, 0] = True
    after[-1, 0] = True
    for i, choices in enumerate(sets):
        for k in choices:
            before[i + 1, k:] |= before[i, :-k]
    for i in reversed(range(n)):
        for k in sets[i]:
            after[i, k:] |= after[i + 1, :-k]
    if not before[-1, total]:
        return None
    possible = []
    for i, choices in enumerate(sets):
        support = [
            k
            for k in choices
            if np.any(before[i, : total - k + 1] & after[i + 1, total - k :: -1])
        ]
        if not support:
            raise ArithmeticError("No denominator in a feasible joint assignment")
        possible.append(support)
    return possible


def extrema(sets, error, is_split):
    """Exact min/max errors and totals for each feasible split group sample count."""
    selected = [
        (candidate, e)
        for candidate, e, m in zip(sets, error, is_split, strict=True)
        if m
    ]
    other = [candidate for candidate, m in zip(sets, is_split, strict=True) if not m]
    minimum = np.full(TRIALS + 1, np.inf)
    maximum = np.full_like(minimum, -np.inf)
    minimum[0] = maximum[0] = 0
    for options, ratio in selected:
        lo = np.full_like(minimum, np.inf)
        hi = np.full_like(minimum, -np.inf)
        for k in options:
            errors = int(round(float(ratio) * k))
            lo[k:] = np.minimum(lo[k:], minimum[:-k] + errors)
            hi[k:] = np.maximum(hi[k:], maximum[:-k] + errors)
        minimum, maximum = lo, hi
    complement = np.zeros(TRIALS + 1, bool)
    complement[0] = True
    for options in other:
        nxt = np.zeros_like(complement)
        for k in options:
            nxt[k:] |= complement[:-k]
        complement = nxt
    valid = np.flatnonzero(np.isfinite(minimum) & complement[::-1])
    if not len(valid) or min(valid) == 0:
        raise ArithmeticError("No nonzero eligible trial denominator")
    return {
        "num_trials": [int(min(valid)), int(max(valid))],
        "weighted_error_bounds": [
            float(min(minimum[valid] / valid)),
            float(max(maximum[valid] / valid)),
        ],
    }


def main():
    rows = design()
    supports = []
    for r in rows:
        candidates = []
        for n in range(1, MAX_CELL + 1):
            x, y = n * r["difficulty"], n * r["error"]
            if (
                abs(x - round(x)) <= TOL
                and abs(y - round(y)) <= TOL
                and n <= round(x) <= 5 * n
                and 0 <= round(y) <= n
            ):
                candidates.append(n)
        supports.append(candidates)
    locked = feasible_counts(supports, TRIALS) if all(supports) else None
    payload = {
        "protocol_sha256": digest(Path(__file__).with_name("noun-noise-protocol.md")),
        "code_sha256": digest(Path(__file__)),
        "cells": len(rows),
        "source_trials": TRIALS,
        "locally_compatible": sum(bool(s) for s in supports),
        "jointly_possible": locked is not None,
    }
    if locked:
        p = np.array([r["error"] for r in rows])
        split = np.array([r["split"] for r in rows])
        payload.update(
            {
                "cell_counts_uniquely_identified": sum(len(s) == 1 for s in locked),
                "cell_counts_ambiguous": sum(len(s) > 1 for s in locked),
                "global_minimum_possible_per_cell": min(min(s) for s in locked),
                "global_maximum_possible_per_cell": max(max(s) for s in locked),
                "unsplit": extrema(locked, p, ~split),
                "split": extrema(locked, p, split),
            }
        )
        if min(min(s) for s in locked) > 1:
            low = np.asarray([max(s) for s in locked]) - 1
            high = np.asarray([min(s) for s in locked]) - 1
            payload["cloze_noise_mse_bounds"] = [
                float(np.mean(p * (1 - p) / low)),
                float(np.mean(p * (1 - p) / high)),
            ]
    (OUT / "human-measurement.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(payload)


if __name__ == "__main__":
    main()
