"""Fit Kerwer's eight-grade proportional-odds readout to grouped trial counts."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, log_expit


LEVELS = 8
CUTS = LEVELS - 1
L2 = 0.01
OPTIONS = {'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-7}


def _design(values):
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError('Expected a finite two-dimensional design without an added intercept')
    return matrix


def _thresholds(parameters):
    return parameters[0] + np.r_[0., np.cumsum(np.logaddexp(0, parameters[1:CUTS]))]


def _log_probabilities(cut_logits, log_gap_mass):
    log_cdf, log_survival = log_expit(cut_logits), log_expit(-cut_logits)
    # For adjacent shifted cuts a < b, F(b) - F(a) = F(b) F(-a) (1 - exp(a - b)).
    inner = log_cdf[:, 1:] + log_survival[:, :-1] + log_gap_mass
    return np.column_stack((log_cdf[:, 0], inner, log_survival[:, -1]))


def _objective(parameters, matrix, frequencies):
    weights = parameters[CUTS:]
    gap_parameters = parameters[1:CUTS]
    cut_logits = _thresholds(parameters) - (matrix @ weights)[:, None]
    # A softplus gap turns log(1 - exp(-gap)) into log(sigmoid(raw_gap)).
    log_p = _log_probabilities(cut_logits, log_expit(gap_parameters))
    nll = float(-np.sum(frequencies * log_p))
    penalty = float(L2 * (weights @ weights))

    shifted_gradient = frequencies[:, 1:] * expit(cut_logits) - frequencies[:, :-1] * expit(-cut_logits)
    threshold_gradient = np.cumsum(shifted_gradient.sum(axis=0)[::-1])[::-1]
    gap_gradient = (threshold_gradient[1:] * expit(gap_parameters)
                    - frequencies[:, 1:-1].sum(axis=0) * expit(-gap_parameters))
    weight_gradient = -(matrix.T @ shifted_gradient.sum(axis=1)) + (2 * L2) * weights
    return (nll + penalty, np.r_[threshold_gradient[0], gap_gradient, weight_gradient]), (nll, penalty)


def _starting_point(class_counts, width):
    if np.any(class_counts <= 0):
        raise ValueError('The empirical-logit start requires trials in each of the eight grades')
    lower = np.cumsum(class_counts)[:-1]
    upper = np.cumsum(class_counts[::-1])[::-1][1:]
    thresholds = np.log(lower) - np.log(upper)
    gaps = np.diff(thresholds)
    if not np.all(gaps > 0):
        raise ValueError('The empirical cutpoints are indistinguishable at float64 precision')
    inverse_softplus = gaps + np.log(-np.expm1(-gaps))
    return np.r_[thresholds[0], inverse_softplus, np.zeros(width)]


def fit(design, counts):
    """Return seven increasing thresholds, slope weights, and optimizer diagnostics."""
    matrix = _design(design)
    histogram = np.asarray(counts, dtype=np.float64)
    if (histogram.shape != (len(matrix), LEVELS) or not np.isfinite(histogram).all()
            or np.any(histogram < 0) or np.any(histogram != np.floor(histogram))):
        raise ValueError('Expected one histogram of eight nonnegative integer grade counts per design row')
    class_counts = histogram.sum(axis=0)
    total = class_counts.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Expected a finite positive number of observed trials')
    start = _starting_point(class_counts, matrix.shape[1])
    frequencies = histogram / total

    def loss(parameters):
        return _objective(parameters, matrix, frequencies)[0]

    result = minimize(loss, start, method='L-BFGS-B', jac=True, options=OPTIONS)
    if not (result.success and np.isfinite(result.fun) and np.isfinite(result.x).all()
            and np.isfinite(result.jac).all()):
        raise ArithmeticError(f'Kerwer proportional-odds fit failed: {result.message}')
    thresholds, weights = _thresholds(result.x), result.x[CUTS:]
    if not (np.isfinite(thresholds).all() and np.all(np.diff(thresholds) > 0)):
        raise ArithmeticError('Fitted cutpoints are not strictly ordered at float64 precision')
    _, (nll, penalty) = _objective(result.x, matrix, frequencies)
    diagnostics = {'success': True, 'message': str(result.message), 'iterations': int(result.nit),
                   'evaluations': int(result.nfev), 'objective': float(result.fun), 'mean_trial_nll': nll,
                   'penalty': penalty, 'max_abs_gradient': float(np.max(np.abs(result.jac))),
                   'trials': int(total), 'groups': len(matrix)}
    return thresholds, weights, diagnostics


def predict(design, thresholds, weights):
    """Return one row of eight probabilities per design row in grade order."""
    matrix = _design(design)
    thresholds, weights = (np.asarray(value, dtype=np.float64) for value in (thresholds, weights))
    if (thresholds.shape != (CUTS,) or weights.shape != (matrix.shape[1],)
            or not np.isfinite(thresholds).all() or not np.isfinite(weights).all()
            or not np.all(np.diff(thresholds) > 0)):
        raise ValueError('Expected seven finite increasing thresholds and one finite weight per design column')
    cut_logits = thresholds - (matrix @ weights)[:, None]
    log_gap_mass = np.log(-np.expm1(-np.diff(thresholds)))
    probability = np.exp(_log_probabilities(cut_logits, log_gap_mass))
    if not (np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1, atol=1e-12, rtol=0)):
        raise ArithmeticError('Proportional-odds predictions do not form finite probability distributions')
    return probability
