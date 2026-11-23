"""Fit the Rodger Study 1 ordinal readout to grouped training observations."""

from numbers import Integral

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


LEVELS = 101
GRID = (np.arange(LEVELS, dtype=np.float64) - 50) / 50
PSEUDOCOUNT = 0.5
RIDGE = 0.10


def rating_counts(ratings):
    counts = np.asarray(ratings)
    if (counts.ndim != 2 or counts.shape[1] != LEVELS or counts.dtype.kind not in 'iu'
            or not len(counts) or np.any(counts < 0)):
        raise ValueError('Ratings must be nonnegative integer counts by row for levels 0 through 100')
    return counts.astype(np.float64, copy=False)


def rating_histogram(ratings):
    """Return the pooled training base distribution, with 0.5 added per level."""
    count = rating_counts(ratings).sum(axis=0)
    if not np.any(count):
        raise ValueError('At least one training rating is required')
    smoothed = count + PSEUDOCOUNT
    return smoothed / smoothed.sum()


def feature_matrix(design):
    matrix = np.asarray(design)
    if matrix.ndim != 2 or not len(matrix) or matrix.dtype.kind not in 'iuf' or not np.isfinite(matrix).all():
        raise ValueError('Expected a nonempty matrix of finite numerical design rows')
    return matrix.astype(np.float64, copy=False)


def block_ranges(width, blocks):
    ranges = tuple(tuple(block) for block in blocks)
    previous_stop = 0
    for span in ranges:
        if (len(span) != 2 or any(not isinstance(index, Integral) or isinstance(index, bool) for index in span)
                or not previous_stop <= span[0] < span[1] <= width):
            raise ValueError('Vector blocks must be ordered, disjoint (start, stop) columns')
        previous_stop = span[1]
    return tuple((int(start), int(stop)) for start, stop in ranges)


def standardization(matrix, weights, blocks):
    """Scale scalar features by their SD, and each vector block by its RMS norm."""
    center = weights @ matrix
    constant = np.all(matrix == matrix[0], axis=0)
    center[constant] = matrix[0, constant]
    centered = matrix - center
    variance = weights @ (centered * centered)
    for start, stop in blocks:
        variance[start:stop] = variance[start:stop].sum()
    scale = np.sqrt(variance)
    scale[scale == 0] = 1
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError('Feature standardization exceeds the finite numerical range')
    centered /= scale
    return centered, center, scale


def tilt_log_probabilities(tilt, log_base):
    scores = log_base + tilt[:, None] * GRID
    return scores - logsumexp(scores, axis=1, keepdims=True)


def objective(parameters, design, weights, observed_moments, log_base, fixed_nll):
    intercept, coefficients = parameters[0], parameters[1:]
    tilt = intercept + design @ coefficients
    scores = log_base + tilt[:, None] * GRID
    normalizer = logsumexp(scores, axis=1)
    moment = np.exp(scores - normalizer[:, None]) @ GRID
    derivative = weights * moment - observed_moments
    nll = fixed_nll + weights @ normalizer - observed_moments @ tilt
    loss = nll + 0.5 * RIDGE * (coefficients @ coefficients)
    gradient = np.concatenate(([derivative.sum()], design.T @ derivative + RIDGE * coefficients))
    return float(loss), gradient


def fit(design, ratings, blocks=()):
    """Fit on integer rating counts aligned to unique or repeated design rows.

    Each individual rating has equal weight. Aggregation of observations with
    identical design rows preserves the histogram and objective exactly.
    """
    matrix = feature_matrix(design)
    counts = rating_counts(ratings)
    if len(matrix) != len(counts):
        raise ValueError('Design and rating-count rows must align')
    ranges = block_ranges(matrix.shape[1], blocks)
    totals = counts.sum(axis=1)
    observed = totals > 0
    if not np.any(observed):
        raise ValueError('At least one training rating is required')
    if not np.all(observed):
        matrix, counts, totals = matrix[observed], counts[observed], totals[observed]
    total = totals.sum()
    pooled = counts.sum(axis=0)
    smoothed = pooled + PSEUDOCOUNT
    base = smoothed / smoothed.sum()
    log_base = np.log(base)
    weights = totals / total
    standardized, center, scale = standardization(matrix, weights, ranges)
    observed_moments = (counts @ GRID) / total
    fixed_nll = -(pooled @ log_base) / total
    result = minimize(
        objective, np.zeros(standardized.shape[1] + 1, dtype=np.float64),
        args=(standardized, weights, observed_moments, log_base, fixed_nll),
        method='L-BFGS-B', jac=True, options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-9},
    )
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError(f'Rodger ordinal readout did not converge: {result.message}')
    return {
        'schema': 1, 'center': center.tolist(), 'scale': scale.tolist(), 'blocks': [list(block) for block in ranges],
        'intercept': float(result.x[0]), 'coefficients': result.x[1:].tolist(), 'base_probabilities': base.tolist(),
        'train_rating_counts': pooled.astype(np.int64).tolist(), 'train_ratings': int(total), 'train_rows': len(matrix),
        'regularized_nll_nats': float(result.fun), 'iterations': int(result.nit),
        'ridge': RIDGE, 'pseudocount': PSEUDOCOUNT,
    }


def log_probabilities(design, fitted):
    """Return normalized log probability for every ordered rating level."""
    matrix = feature_matrix(design)
    center, scale, coefficients, base = (np.asarray(fitted[name], dtype=np.float64)
                                         for name in ('center', 'scale', 'coefficients', 'base_probabilities'))
    intercept = float(fitted['intercept'])
    if (any(vector.shape != (matrix.shape[1],) or not np.isfinite(vector).all()
            for vector in (center, scale, coefficients)) or np.any(scale <= 0)
            or base.shape != (LEVELS,) or not np.isfinite(base).all() or np.any(base <= 0)
            or not np.isclose(base.sum(), 1, rtol=0, atol=1e-12) or not np.isfinite(intercept)):
        raise ValueError('Frozen readout and numerical design do not agree')
    tilt = intercept + ((matrix - center) / scale) @ coefficients
    if not np.isfinite(tilt).all():
        raise ValueError('Frozen readout produced nonfinite tilt values')
    log_p = tilt_log_probabilities(tilt, np.log(base))
    if not np.isfinite(log_p).all():
        raise ValueError('Frozen readout produced nonfinite log probabilities')
    return log_p


def probabilities(design, fitted):
    """Return probabilities in level order 0 through 100 for each design row."""
    distribution = np.exp(log_probabilities(design, fitted))
    if not np.allclose(distribution.sum(axis=1), 1, rtol=0, atol=1e-12):
        raise ValueError('Frozen readout probabilities do not sum to one')
    return distribution
