"""Fit a causal prefix update readout on the Blott 2021 behavior only."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


RIDGE = 0.025
ARM_BLOCKS = {
    'surface': ('length',),
    'terminal': ('length', 'terminal'),
    'update': ('length', 'update'),
    'joint': ('length', 'terminal', 'update'),
    'joint_effect': ('length', 'terminal', 'update'),
}
ARMS = tuple(ARM_BLOCKS)
FOLDS = 5


@dataclass(frozen=True)
class Fit:
    blocks: tuple[str, ...]
    centers: tuple[np.ndarray, ...]
    scales: tuple[np.ndarray, ...]
    coefficient: np.ndarray
    diagnostics: dict

    def predict(self, features):
        design = design_matrix(features, self.blocks, self.centers, self.scales)
        return design @ self.coefficient

    def receipt(self):
        return {'blocks': self.blocks, 'centers': [c.tolist() for c in self.centers],
                'scales': [s.tolist() for s in self.scales], 'coefficients': self.coefficient.tolist(),
                'diagnostics': self.diagnostics}


def design_matrix(features, blocks, centers, scales):
    columns = [(features[name] - center) / scale
               for name, center, scale in zip(blocks, centers, scales, strict=True)]
    return np.column_stack((np.ones(len(columns[0])), *columns))


def binomial_loss(logits, rates, weights):
    losses = rates * np.logaddexp(0, -logits) + (1 - rates) * np.logaddexp(0, logits)
    slope = (1 - rates) * expit(logits) - rates * expit(-logits)
    return float(weights @ losses), weights * slope


def loss(beta, design, observed, pair):
    rates, weights = observed
    value, slope = binomial_loss(design @ beta, rates, weights)
    gradient = design.T @ slope
    if pair is not None:
        contrast, rates, weights = pair
        extra, slope = binomial_loss(contrast @ beta, rates, weights)
        value += extra
        gradient += contrast.T @ slope
    value += RIDGE / 2 * (beta[1:] @ beta[1:])
    gradient[1:] += RIDGE * beta[1:]
    return value, gradient


def paired_targets(design, hits, totals):
    if (len(hits) % 2 or np.any(hits < 0) or np.any(totals <= 0) or np.any(hits > totals)):
        raise ValueError('Expected paired Amb/UA cells and exact observed trial counts')
    wins = hits[1::2] * (totals - hits)[::2]
    losses = hits[::2] * (totals - hits)[1::2]
    contested = wins + losses > 0
    if not contested.any():
        raise ValueError('No family has discordant cross-reader support for the pair loss')
    contrast = (design[1::2] - design[::2])[contested]
    return contrast, wins[contested] / (wins[contested] + losses[contested]), np.full(len(contrast), 1 / len(contrast))


def fit_model(features, hits, totals, arm):
    blocks = ARM_BLOCKS[arm]
    centers = tuple(features[name].mean(axis=0) for name in blocks)
    variances = (np.square(features[name] - c).mean(axis=0) for name, c in zip(blocks, centers, strict=True))
    scales = tuple(np.sqrt(v if name == 'length' else v.sum()) for name, v in zip(blocks, variances, strict=True))
    if any(np.any(~np.isfinite(scale)) or np.any(scale <= 0) for scale in scales):
        raise ValueError('All training feature blocks must have finite nonzero scale')
    design = design_matrix(features, blocks, centers, scales)
    if (hits.shape != totals.shape or hits.shape != (len(design),) or np.any(hits < 0)
            or np.any(totals <= 0) or np.any(hits > totals) or not 0 < hits.sum() < totals.sum()
            or not np.isfinite(design).all()):
        raise ValueError('Invalid grouped human Sense observations')
    observed = hits / totals, totals / totals.sum()
    pair = paired_targets(design, hits, totals) if arm == 'joint_effect' else None
    initial = np.zeros(design.shape[1])
    result = minimize(loss, initial, args=(design, observed, pair), method='L-BFGS-B', jac=True,
                      options={'maxiter': 600, 'ftol': 1e-15, 'gtol': 1e-10})
    objective, gradient = loss(result.x, design, observed, pair)
    error = float(np.max(np.abs(gradient)))
    if not np.isfinite(result.x).all() or not np.isfinite(objective) or not np.isfinite(error) or error > 2e-7:
        raise ArithmeticError(f'Could not solve convex Blott readout for {arm}: {result.message}; gradient {error:g}')
    diagnostics = {'objective': objective, 'max_absolute_gradient': error, 'iterations': int(result.nit),
                   'solver_status': int(result.status),
                   'training_versions': len(design), 'training_responses': int(totals.sum()),
                   'effect_supervision_families': len(pair[0]) if pair is not None else 0}
    return Fit(blocks, centers, scales, result.x, diagnostics)


def restrict(features, index):
    return {name: values[index] for name, values in features.items()}


def crossfit(features, hits, totals, families):
    if len(families) * 2 != len(hits) or tuple(sorted(set(families))) != tuple(families):
        raise ValueError('Pass sorted unique training family IDs with adjacent Amb/UA variants')
    assignments = np.repeat(np.arange(len(families)) % FOLDS, 2)
    results = np.full((len(hits), len(ARMS)), np.nan)
    receipts = []
    for fold in range(FOLDS):
        train, test = assignments != fold, assignments == fold
        training_features, test_features = restrict(features, train), restrict(features, test)
        fits = [fit_model(training_features, hits[train], totals[train], arm) for arm in ARMS]
        results[test] = np.column_stack([fitted.predict(test_features) for fitted in fits])
        receipts.append({'fold': fold, 'held_out_families': list(families[fold::FOLDS]),
                         'fits': {arm: fit.diagnostics for arm, fit in zip(ARMS, fits, strict=True)}})
    if not np.isfinite(results).all():
        raise ValueError('Every eligible 2021 family must have a prediction from another fold')
    return results, receipts
