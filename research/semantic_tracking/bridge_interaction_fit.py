"""Fit a document-weighted marginal-softmax reranker for annotated bridge locations."""

from collections import Counter
from math import isfinite, log
from operator import index

import numpy as np
from scipy.optimize import minimize


PENALTY = 0.05
VARIANCE_FLOOR = 1e-12


def pair_features(anchor, candidates):
    """Return cosine and its zero-sum aspect residual from frozen endpoint states."""
    cue = np.asarray(anchor, dtype=np.float32)
    endpoints = np.asarray(candidates, dtype=np.float32)
    if (cue.ndim != 1 or not len(cue) or endpoints.ndim != 2 or not len(endpoints)
            or endpoints.shape[1] != len(cue) or not np.isfinite(cue).all() or not np.isfinite(endpoints).all()):
        raise ValueError('Expected finite aligned cue and candidate endpoint vectors')
    cue_norm = np.linalg.norm(cue)
    norms = np.linalg.norm(endpoints, axis=1)
    if not np.isfinite(cue_norm) or cue_norm <= 0 or not np.all(np.isfinite(norms) & (norms > 0)):
        raise ValueError('Cosine and interaction features require nonzero endpoint vectors')
    products = endpoints * (cue / cue_norm)
    products /= norms[:, None]
    cosine = products.sum(axis=1)
    products -= cosine[:, None] / products.shape[1]
    return cosine, products


def column_indices(width, columns):
    if columns is None:
        selected = tuple(range(width))
    elif isinstance(columns, slice):
        selected = tuple(range(width)[columns])
    else:
        selected = tuple(index(column) for column in columns)
    if not selected or len(set(selected)) != len(selected) or any(not 0 <= column < width for column in selected):
        raise ValueError('Choose nonempty distinct feature columns within the original design')
    return selected


def selector(columns):
    if columns == tuple(range(columns[0], columns[-1] + 1)):
        return slice(columns[0], columns[-1] + 1)
    return np.asarray(columns, dtype=np.intp)


def prepare_cases(examples):
    cases, counts, width, candidates = [], Counter(), None, 0
    for row in examples:
        design, gold = np.asarray(row.design), np.asarray(row.gold)
        if (type(row.document) is not str or not row.document or design.ndim != 2 or not design.shape[1]
                or design.dtype not in (np.float32, np.float64) or gold.dtype != bool or gold.shape != (len(design),)
                or not np.any(gold) or np.all(gold) or not np.isfinite(design).all()):
            raise ValueError('Each bridge case needs finite candidate rows, document ID, and a nontrivial positive set')
        if width is None:
            width = design.shape[1]
        if design.shape[1] != width:
            raise ValueError('Feature width must be shared across all bridge cases')
        cases.append((design, np.flatnonzero(gold), row.document))
        counts[row.document] += 1
        candidates += len(design)
    if not cases:
        raise ValueError('No training bridge cases')
    weighted = [(design, gold, 1 / (len(counts) * counts[document])) for design, gold, document in cases]
    return weighted, width, len(counts), candidates


def weighted_scaler(cases, selection, width):
    center = np.zeros(width)
    for design, _, weight in cases:
        center += weight * design[:, selection].mean(axis=0, dtype=np.float64)
    variance = np.zeros(width)
    for design, _, weight in cases:
        centered = design[:, selection] - center
        variance += (weight / len(design)) * np.einsum('ij,ij->j', centered, centered)
    if not np.isfinite(center).all() or not np.isfinite(variance).all():
        raise ValueError('Training features exceed the finite standardization range')
    return center, np.sqrt(np.maximum(variance, VARIANCE_FLOOR))


def marginal_loss(beta, cases, selection, inverse_scale, penalty):
    loss = 0.5 * penalty * (beta @ beta)
    gradient = penalty * beta
    effective = beta * inverse_scale
    # A shared centering term cancels within every candidate softmax and its gradient.
    for design, positive, weight in cases:
        features = design[:, selection]
        scores = features @ effective
        gold = scores[positive]
        all_max, gold_max = scores.max(), gold.max()
        posterior = np.exp(scores - all_max)
        positive_posterior = np.exp(gold - gold_max)
        all_sum, gold_sum = posterior.sum(), positive_posterior.sum()
        loss += weight * (all_max - gold_max + log(all_sum) - log(gold_sum))
        posterior /= all_sum
        posterior[positive] -= positive_posterior / gold_sum
        gradient += weight * (posterior @ features) * inverse_scale
    return loss, gradient


def fit(examples, columns=None, *, penalty=PENALTY):
    """Fit on original article groups; every article and its eligible cases get equal weight."""
    penalty = float(penalty)
    if not isfinite(penalty) or penalty <= 0:
        raise ValueError('Expected a strictly positive finite ridge penalty')
    cases, width, documents, candidates = prepare_cases(examples)
    selected = column_indices(width, columns)
    selection = selector(selected)
    center, scale = weighted_scaler(cases, selection, len(selected))
    result = minimize(marginal_loss, np.zeros(len(selected)), args=(cases, selection, 1 / scale, penalty),
                      method='L-BFGS-B', jac=True, options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise RuntimeError(f'Bridge interaction fit did not converge: {result.message}')
    return {'input_features': width, 'columns': list(selected), 'center': center.tolist(), 'scale': scale.tolist(),
            'coefficients': result.x.tolist(), 'penalty': penalty, 'fit_articles': documents,
            'fit_cases': len(cases), 'fit_candidates': candidates, 'iterations': result.nit,
            'regularized_nll_nats': float(result.fun)}


def predict(design, fitted):
    """Score an original full-width candidate design in its original row order."""
    design = np.asarray(design)
    if (design.ndim != 2 or design.shape[1] != fitted['input_features'] or len(design) == 0
            or design.dtype not in (np.float32, np.float64) or not np.isfinite(design).all()):
        raise ValueError('Prediction requires the original finite candidate design and feature width')
    columns = column_indices(design.shape[1], fitted['columns'])
    center, scale, coefficients = (np.asarray(fitted[name]) for name in ('center', 'scale', 'coefficients'))
    if (any(values.shape != (len(columns),) or not np.isfinite(values).all()
            for values in (center, scale, coefficients)) or np.any(scale <= 0)):
        raise ValueError('Invalid frozen feature scale or coefficients')
    effective = coefficients / scale
    scores = design[:, selector(columns)] @ effective - center @ effective
    if not np.isfinite(scores).all():
        raise ValueError('The frozen bridge readout produced nonfinite scores')
    return scores
