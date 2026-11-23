"""Fit Frick retrospective readouts with shared scales and fixed objectives."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.semantic_tracking.frick_consistency_score import RIDGE, binary_loss
from research.semantic_tracking.rodger_belief_readout import feature_matrix, standardization


TASKS = ('current', 'first')
METHODS = ('global', 'target_weighted', 'paired_effect')
PAIR_WEIGHT = 4.0


def training_pairs(targets, size):
    pairs = []
    for left, right, delta in targets:
        paths = []
        for path in (left, right):
            indices = np.asarray(path)
            if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in 'iu'
                    or np.any(indices < 0) or np.any(indices >= size)
                    or np.any(indices[1:] <= indices[:-1])):
                raise ValueError('Each target path must have distinct increasing training row indices')
            paths.append(indices.astype(np.intp, copy=False))
        if np.intersect1d(*paths, assume_unique=True).size:
            raise ValueError('The two versions of a story must have disjoint training path rows')
        effect = float(delta)
        if not np.isfinite(effect) or not -1 <= effect <= 1:
            raise ValueError('An observed paired contrast must be finite and in [-1,1]')
        pairs.append((*paths, effect))
    if not pairs:
        raise ValueError('At least one training story pair is required')
    return pairs


def trial_weights(totals, pairs, method):
    if method != 'target_weighted':
        return np.full(len(totals), 1 / totals.sum(dtype=np.float64))
    target = np.zeros(len(totals), dtype=bool)
    for left, right, _ in pairs:
        target[[left[-1], right[-1]]] = True
    target_total = totals[target].sum(dtype=np.float64)
    other_total = totals[~target].sum(dtype=np.float64)
    if min(target_total, other_total) <= 0:
        raise ValueError('A target-weighted fit needs observed trials in both sentence strata')
    return np.where(target, 0.5 / target_total, 0.5 / other_total)


def target_mass(path, task, hazards, survivals, log_hazards, log_survivals):
    """Return target event mass and its derivatives for only the path it uses."""
    if task == 'current':
        last = path[-1:]
        mass = hazards[last[0]]
        return mass, last, mass * survivals[last]
    mass = np.exp(log_hazards[path[-1]] + log_survivals[path[:-1]].sum())
    slope = -mass * hazards[path]
    slope[-1] = mass * survivals[path[-1]]
    return mass, path, slope


def pair_loss(logits, pairs, task):
    """Compute mean paired squared effect error and the exact logit gradient."""
    hazards, survivals = expit(logits), expit(-logits)
    log_hazards, log_survivals = -np.logaddexp(0, -logits), -np.logaddexp(0, logits)
    gradient, squared_error = np.zeros_like(logits), 0.0
    masses = (hazards, survivals, log_hazards, log_survivals)
    for left, right, human_effect in pairs:
        p_left, i_left, d_left = target_mass(left, task, *masses)
        p_right, i_right, d_right = target_mass(right, task, *masses)
        error = p_right - p_left - human_effect
        squared_error += error * error
        gradient[i_left] -= 2 * error * d_left
        gradient[i_right] += 2 * error * d_right
    scale = 1.0 / len(pairs)
    return squared_error * scale, gradient * scale


def objective(parameters, design, hit, miss, pairs, task, method):
    loss, gradient = binary_loss(parameters, design, hit, miss)
    if method == 'paired_effect':
        logits = parameters[0] + design @ parameters[1:]
        error, derivative = pair_loss(logits, pairs, task)
        loss += PAIR_WEIGHT * error
        gradient[0] += PAIR_WEIGHT * derivative.sum()
        gradient[1:] += PAIR_WEIGHT * (design.T @ derivative)
    return loss, gradient


def fit(design, hits, totals, targets, task, method):
    """Fit one training fold. Target paths use row indices into this training design."""
    if task not in TASKS or method not in METHODS:
        raise ValueError(f'Unsupported Frick task or readout method: {task}, {method}')
    matrix = feature_matrix(design)
    hits, totals = map(np.asarray, (hits, totals))
    if (hits.shape != (len(matrix),) or totals.shape != hits.shape or hits.dtype.kind not in 'iu'
            or totals.dtype.kind not in 'iu' or np.any(totals < 0) or np.any(hits < 0) or np.any(hits > totals)):
        raise ValueError('Group counts must be aligned integers with 0 <= hits <= totals')
    pairs = training_pairs(targets, len(matrix))
    total = totals.sum(dtype=np.float64)
    positives = hits.sum(dtype=np.float64)
    if not 0 < positives < total:
        raise ValueError('A Bernoulli fit with an unpenalized intercept needs both observed outcomes')
    keep = totals > 0
    if np.all(keep):
        standardized, center, scale = standardization(matrix, totals / total, ())
    else:
        # Zero at-risk rows do not set moments, but can still enter an unconditional target path.
        _, center, scale = standardization(matrix[keep], totals[keep] / total, ())
        standardized = (matrix - center) / scale
    weights = trial_weights(totals, pairs, method)
    hit, miss = hits * weights, (totals - hits) * weights
    initial = np.zeros(matrix.shape[1] + 1, dtype=np.float64)
    initial[0] = np.log(positives) - np.log(total - positives)
    result = minimize(objective, initial, args=(standardized, hit, miss, pairs, task, method),
                      method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-9})
    if not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise RuntimeError(f'Frick paired readout did not converge for {task}/{method}: {result.message}')
    intercept, coefficients = float(result.x[0]), result.x[1:]
    regularized, _ = binary_loss(result.x, standardized, hit, miss)
    error, _ = pair_loss(intercept + standardized @ coefficients, pairs, task)
    return {'center': center, 'scale': scale, 'intercept': intercept, 'coefficients': coefficients,
            'iterations': int(result.nit), 'train_rows': int(keep.sum()), 'train_trials': int(total),
            'train_hits': int(positives), 'train_pairs': len(pairs), 'regularized_nll_nats': regularized,
            'paired_mse': float(error), 'objective': float(result.fun), 'method': method, 'task': task,
            'ridge': RIDGE, 'pair_weight': PAIR_WEIGHT if method == 'paired_effect' else 0.0}
