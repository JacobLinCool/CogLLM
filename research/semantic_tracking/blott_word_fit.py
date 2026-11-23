"""Fit a shared conditional ranker on grade-free causal word transitions."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


RIDGE = 0.025
STATIONARITY = 2e-7
BLOCK_NAMES = ('position', 'surface', 'change_norm', 'delta')
DELTA_WIDTH = 2048


@dataclass(frozen=True)
class RankFit:
    blocks: tuple[str, ...]
    centers: tuple[np.ndarray, ...]
    scales: tuple[np.ndarray, ...]
    coefficients: np.ndarray
    diagnostics: dict

    def score(self, features):
        """Compute one raw logit per candidate without a sentence-wide softmax."""
        matrices = feature_blocks(features, self.blocks)
        standardized = []
        for block, values, center, scale in zip(self.blocks, matrices, self.centers, self.scales, strict=True):
            if values.shape[1:] != center.shape:
                raise ValueError(f'Unexpected fitted Blott width for {block}')
            standardized.append((values - center) / scale)
        design = np.column_stack(standardized)
        values = np.einsum('ij,j->i', design, self.coefficients, optimize=False)
        if not np.isfinite(values).all():
            raise ValueError('Blott candidate scores must remain finite')
        return values

    def receipt(self):
        return {
            'blocks': list(self.blocks),
            'centers': [center.tolist() for center in self.centers],
            'scales': [scale.tolist() for scale in self.scales],
            'coefficients': self.coefficients.tolist(),
            'ridge': RIDGE,
            'diagnostics': self.diagnostics,
        }


def feature_blocks(features, blocks):
    matrices = []
    for block in blocks:
        raw = np.asarray(features[block])
        expected_width = {'delta': DELTA_WIDTH, 'change_norm': 1}.get(block)
        if (raw.ndim != 2 or raw.shape[1] == 0 or raw.dtype.kind not in 'bifu'
                or (expected_width is not None and raw.shape[1] != expected_width)):
            raise ValueError(f'Expected a numeric Blott candidate matrix for {block}')
        values = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite Blott inputs in block {block}')
        matrices.append(values)
    if len({len(matrix) for matrix in matrices}) != 1:
        raise ValueError('Each block must describe the same visible candidate rows')
    return matrices


def training_matrix(candidates, blocks, row_weights):
    centers, scales = [], []
    first = candidates[0]
    width = sum(matrix.shape[1] for matrix in first)
    design = np.empty((len(row_weights), width), dtype=np.float64)
    column = 0
    for block, sentence_blocks in zip(blocks, zip(*candidates, strict=True), strict=True):
        if len({matrix.shape[1] for matrix in sentence_blocks}) != 1:
            raise ValueError(f'Inconsistent training widths in Blott block {block}')
        values = np.vstack(sentence_blocks)
        width = values.shape[1]
        centered = design[:, column:column + width]
        center = row_weights @ values
        constant = values.max(axis=0) == values.min(axis=0)
        center[constant] = values[0, constant]
        np.subtract(values, center, out=centered)
        if block == 'delta':
            variance = row_weights @ np.einsum('ij,ij->i', centered, centered, optimize=False)
            scale = np.asarray(np.sqrt(variance) if variance > 0 else 1.)
        else:
            variance = np.einsum('i,ij,ij->j', row_weights, centered, centered, optimize=False)
            scale = np.sqrt(variance)
            scale[constant] = 1.
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError(f'Blott block {block} has an invalid training scale')
        centered /= scale
        centers.append(center)
        scales.append(scale)
        column += width
    return design, tuple(centers), tuple(scales)


def conditional_loss(coefficients, design, starts, groups, positive):
    logits = design @ coefficients
    maxima = np.maximum.reduceat(logits, starts)
    centered = logits - maxima[groups]
    numerators = np.exp(centered)
    totals = np.add.reduceat(numerators, starts)
    nll = float(np.mean(np.log(totals) - centered[positive]))
    probabilities = numerators / totals[groups]
    probabilities[positive] -= 1
    residual = probabilities / len(starts)
    ridge = RIDGE / 2 * (coefficients @ coefficients)
    return nll + ridge, design.T @ residual + RIDGE * coefficients


def fit_ranker(candidates, gold, *, blocks=('position',)):
    """Fit equal-sentence conditional likelihood with one annotated row per sentence."""
    blocks = tuple(blocks)
    if not blocks or len(set(blocks)) != len(blocks) or any(name not in BLOCK_NAMES for name in blocks):
        raise ValueError('Select distinct registered Blott candidate feature blocks')
    matrices = [feature_blocks(sentence, blocks) for sentence in candidates]
    counts = np.array([len(sentence[0]) for sentence in matrices], dtype=np.intp)
    targets = np.asarray(gold)
    if (not len(counts) or np.any(counts < 1) or not np.any(counts > 1)
            or targets.shape != counts.shape or targets.dtype.kind not in 'iu'
            or np.any(targets < 0) or np.any(targets >= counts)):
        raise ValueError('Expected valid per-sentence gold row indices and observed candidate sets')
    starts = np.r_[0, np.cumsum(counts[:-1])]
    groups = np.repeat(np.arange(len(counts)), counts)
    positive = starts + targets.astype(np.intp, copy=False)
    row_weights = 1 / (len(counts) * counts[groups])
    design, centers, scales = training_matrix(matrices, blocks, row_weights)

    args = design, starts, groups, positive
    result = minimize(conditional_loss, np.zeros(design.shape[1]), args=args, method='L-BFGS-B', jac=True,
                      options={'maxiter': 500, 'maxcor': 20, 'ftol': 1e-15, 'gtol': 1e-10})
    coefficients = result.x
    objective, gradient = conditional_loss(coefficients, *args)
    stationarity = float(np.max(np.abs(gradient)))
    if (not np.isfinite(coefficients).all() or not np.isfinite(objective)
            or not np.isfinite(stationarity) or stationarity > STATIONARITY):
        raise ArithmeticError(f'Blott conditional rank fit exceeded stationarity: {stationarity:.3g}; {result.message}')
    return RankFit(blocks, centers, scales, coefficients, {
        'training_sentences': len(counts),
        'training_candidates': len(design),
        'objective': objective,
        'training_conditional_nll': float(objective - RIDGE / 2 * (coefficients @ coefficients)),
        'max_absolute_gradient': stationarity,
        'stationarity_tolerance': STATIONARITY,
        'iterations': int(result.nit),
        'optimizer_status': int(result.status),
        'optimizer_success': bool(result.success),
    })
