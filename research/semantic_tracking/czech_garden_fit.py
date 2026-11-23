"""Frozen-feature multinomial readouts for three explicit Czech qmis responses."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


ARMS = {'constant': (), 'surface': ('length', 'form'), 'terminal': ('length', 'form', 'terminal'),
        'unmarked': ('length', 'form', 'terminal', 'pooled'),
        'marked': ('length', 'form', 'terminal', 'cue'), 'author_labels': ('labels',)}
RESPONSES = ('yes', 'no', 'unsure')
RIDGE = 0.01


@dataclass(frozen=True)
class Head:
    blocks: tuple[str, ...]
    centers: tuple[np.ndarray, ...]
    scales: tuple[np.ndarray, ...]
    endings: tuple[str, ...]
    coefficients: np.ndarray
    diagnostics: dict

    def predict(self, rows):
        design = matrix(rows, self.blocks, self.endings, self.centers, self.scales)
        return probabilities(design, self.coefficients)

    def receipt(self):
        return {'blocks': self.blocks, 'centers': [row.tolist() for row in self.centers],
                'scales': [row.tolist() for row in self.scales], 'endings': self.endings,
                'coefficients': self.coefficients.tolist(), 'diagnostics': self.diagnostics}


def features(rows, block, endings):
    if block == 'form':
        return np.asarray([[row['word4_ending'] == ending for ending in endings] for row in rows], dtype=np.float64)
    return np.asarray([row[block] for row in rows], dtype=np.float64)


def standardize(values, block):
    if block in ('labels', 'form'):
        return np.zeros(values.shape[1]), np.ones(values.shape[1])
    center = values.mean(axis=0)
    variance = np.square(values - center).mean(axis=0)
    scale = np.sqrt(variance if block == 'length' else variance.sum())
    return center, np.where(scale > 1e-12, scale, 1.)


def matrix(rows, blocks, endings, centers, scales):
    columns = [(features(rows, name, endings) - center) / scale
               for name, center, scale in zip(blocks, centers, scales, strict=True)]
    return np.column_stack((np.ones(len(rows)), *columns))


def logits_for(design, coefficients):
    modeled = design @ coefficients
    return np.column_stack((modeled[:, 0], np.zeros(len(design)), modeled[:, 1]))


def probabilities(design, coefficients):
    logits = logits_for(design, coefficients)
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def objective(parameters, design, counts, totals, size):
    coefficients = parameters.reshape(design.shape[1], 2)
    logits = logits_for(design, coefficients)
    partition = logsumexp(logits, axis=1)
    log_mass = logits - partition[:, None]
    weights = coefficients.copy()
    weights[0] = 0.
    loss = -np.sum(counts * log_mass) / size + RIDGE / 2 * np.square(weights).sum()
    residual = np.exp(log_mass) * totals[:, None] - counts
    gradient = design.T @ residual[:, [0, 2]] / size + RIDGE * weights
    return float(loss), gradient.ravel()


def fit(rows, counts, arm):
    counts = np.asarray(counts, dtype=np.float64)
    if (counts.shape != (len(rows), len(RESPONSES)) or not np.isfinite(counts).all() or np.any(counts < 0)
            or np.any(counts.sum(axis=1) <= 0) or np.any(counts.sum(axis=0) <= 0)):
        raise ValueError('Every training sentence and each Czech response category require observed counts')
    blocks = ARMS[arm]
    endings = tuple(sorted({row['word4_ending'] for row in rows})) if 'form' in blocks else ()
    parameters = [standardize(features(rows, name, endings), name) for name in blocks]
    centers = tuple(row[0] for row in parameters)
    scales = tuple(row[1] for row in parameters)
    design = matrix(rows, blocks, endings, centers, scales)
    totals = counts.sum(axis=1)
    if not np.isfinite(design).all():
        raise ValueError('Training design must contain finite Czech readout inputs')
    args = design, counts, totals, totals.sum()
    result = minimize(objective, np.zeros(design.shape[1] * 2), args=args, jac=True, method='L-BFGS-B',
                      options={'maxiter': 500, 'maxcor': 20, 'ftol': 1e-15, 'gtol': 1e-10})
    loss, gradient = objective(result.x, *args)
    error = float(np.max(np.abs(gradient)))
    if not np.isfinite(loss) or not error <= 3e-7:
        raise ArithmeticError(f'Czech multinomial readout not stationary: {arm} {error:.3g} {result.message}')
    return Head(blocks, centers, scales, endings, result.x.reshape(design.shape[1], 2),
                {'training_sentences': len(rows), 'training_responses': int(totals.sum()),
                 'training_class_counts': counts.sum(axis=0).astype(int).tolist(), 'objective': loss,
                 'max_absolute_gradient': error, 'iterations': int(result.nit), 'optimizer_status': int(result.status)})
