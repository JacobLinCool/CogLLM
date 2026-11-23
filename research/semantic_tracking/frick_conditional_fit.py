"""Fit Frick hazards with an exact matched conditional count likelihood."""

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

from research.semantic_tracking import frick_mass_calibration as phase


class TargetPairs(NamedTuple):
    left: np.ndarray
    right: np.ndarray
    support: np.ndarray
    log_multiplicity: np.ndarray
    observed_slot: np.ndarray
    pair_sizes: np.ndarray


def log_choose(n, count):
    return gammaln(n + 1) - gammaln(count + 1) - gammaln(n - count + 1)


def target_pairs(left, right, hits, totals):
    left, right = np.asarray(left), np.asarray(right)
    if (left.ndim != 1 or left.shape != right.shape or not len(left)
            or not np.issubdtype(left.dtype, np.integer) or not np.issubdtype(right.dtype, np.integer)):
        raise ValueError("Expected one pair of target stops per training story")
    joined = np.r_[left, right]
    if (np.any(joined < 0) or np.any(joined >= len(hits)) or len(set(joined)) != len(joined)
            or np.any(totals[joined] <= 0) or np.any(hits[joined] < 0) or np.any(hits[joined] > totals[joined])):
        raise ValueError("Matched target counts and positive risk sets must be valid")
    left_sizes, right_sizes = totals[left], totals[right]
    observed = hits[right]
    total_events = hits[left] + observed
    lower = np.maximum(0, total_events - left_sizes)
    upper = np.minimum(total_events, right_sizes)
    width = int(np.max(upper - lower)) + 1
    support = lower[:, None] + np.arange(width)
    active = support <= upper[:, None]
    baseline = np.full(support.shape, -np.inf, dtype=np.float64)
    x = support[active]
    repeats = active.sum(axis=1)
    baseline[active] = log_choose(np.repeat(right_sizes, repeats), x) + log_choose(
        np.repeat(left_sizes, repeats), np.repeat(total_events, repeats) - x,
    )
    if np.any(observed < lower) or np.any(observed > upper):
        raise ValueError("Observed conditional count lies outside its feasible support")
    # Center the support at the observation. This makes the objective a single
    # stable log-sum-exp even when a fitted odds ratio is far from one.
    support = np.where(active, support - observed[:, None], 0)
    return TargetPairs(left, right, support, baseline, observed - lower, left_sizes + right_sizes)


def conditional_loss(logits, pairs):
    difference = logits[pairs.right] - logits[pairs.left]
    tilted = pairs.log_multiplicity + difference[:, None] * pairs.support
    normalizer = logsumexp(tilted, axis=1)
    observed = tilted[np.arange(len(difference)), pairs.observed_slot]
    loss = np.mean((normalizer - observed) / pairs.pair_sizes)
    expected_centered = (np.exp(tilted - normalizer[:, None]) * pairs.support).sum(axis=1)
    derivative = expected_centered / (pairs.pair_sizes * len(difference))
    gradient = np.zeros_like(logits)
    gradient[pairs.right] += derivative
    gradient[pairs.left] -= derivative
    return float(loss), gradient


def objective(parameters, design, hit, miss, phases, phase_weight, pairs, conditional_weight):
    nll, gradient = phase.objective(parameters, design, hit, miss, phases, phase_weight)
    if conditional_weight == 0:
        return nll, gradient
    logits = parameters[0] + design @ parameters[1:]
    loss, slope = conditional_loss(logits, pairs)
    gradient[0] += conditional_weight * slope.sum()
    gradient[1:] += conditional_weight * (design.T @ slope)
    return nll + conditional_weight * loss, gradient


def fit(design, hits, totals, phases, phase_weight, pairs, conditional_weight, phase_only):
    """Use an independently fitted training-only phase solution as the fixed restart."""
    if conditional_weight <= 0:
        raise ValueError("Use the phase-only comparator when the conditional weight is zero")
    standardized = (design - phase_only["center"]) / phase_only["scale"]
    total = totals.sum()
    initial = np.r_[phase_only["intercept"], phase_only["coefficients"]]
    result = minimize(
        objective, initial, jac=True, method="L-BFGS-B",
        args=(standardized, hits / total, (totals - hits) / total, phases, phase_weight, pairs, conditional_weight),
        options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9},
    )
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f"Matched conditional Frick fit failed to converge: {result.message}")
    return {
        "center": phase_only["center"], "scale": phase_only["scale"], "intercept": float(result.x[0]),
        "coefficients": result.x[1:], "iterations": int(result.nit), "objective": float(result.fun),
        "train_at_risk_decisions": int(total), "matched_training_pairs": len(pairs.left),
    }
