"""Fit source-specific observation models with a shared direction of difficulty."""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import expit


RIDGE = 0.01


def normalized_hidden(array):
    state = np.asarray(array, dtype=np.float64)
    centered = state - state.mean(axis=1, keepdims=True)
    return centered / np.sqrt((centered**2).mean(axis=1, keepdims=True) + 1e-5)


@dataclass
class Basis:
    hidden_mean: np.ndarray
    projection: np.ndarray
    surface_mean: np.ndarray
    surface_scale: np.ndarray

    def transform(self, study, include_hidden):
        surface = (study.surface - self.surface_mean) / self.surface_scale
        if not include_hidden:
            return surface
        hidden = (normalized_hidden(study.hidden) - self.hidden_mean) @ self.projection
        return np.column_stack((surface, hidden))

    @classmethod
    def train(cls, studies):
        # Unlabeled covariance gives each training study equal weight.
        subsets = [
            (normalized_hidden(s.hidden[s.role == 0]), s.surface[s.role == 0])
            for s in studies
        ]
        mean = np.mean([h.mean(axis=0) for h, _ in subsets], axis=0)
        lexical_mean = np.mean([x.mean(axis=0) for _, x in subsets], axis=0)
        lexical_var = np.mean(
            [np.mean((x - lexical_mean) ** 2, axis=0) for _, x in subsets], axis=0
        )
        matrix = np.vstack(
            [(h - mean) / np.sqrt(len(h) * len(subsets)) for h, _ in subsets]
        )
        n_components = min(128, matrix.shape[1], len(matrix) - 1)
        eigen, vectors = eigh(
            matrix.T @ matrix,
            subset_by_index=[matrix.shape[1] - n_components, matrix.shape[1] - 1],
        )
        if eigen.min() <= 1e-8:
            raise ArithmeticError("Readout basis has a degenerate component")
        return cls(
            mean,
            vectors / np.sqrt(eigen),
            lexical_mean,
            np.where(lexical_var > 1e-12, np.sqrt(lexical_var), 1),
        )


def observations(study, rows):
    y = study.response[rows]
    if not np.isfinite(y).all():
        raise ValueError("Attempted to use a hidden response")
    if study.continuous:
        if np.any(y[:, 1] < 0):
            raise ValueError("Expected nonnegative published standard errors")
        return y[:, :1], 1 / (1 + y[:, 1] ** 2) / len(y)
    count = y.sum(axis=1)
    if np.any(count <= 0) or np.any(y < 0):
        raise ValueError("A rated position has zero or negative response counts")
    # Each column is the observed probability of a rating above its threshold.
    exceed = 1 - np.cumsum(y[:, :-1], axis=1) / count[:, None]
    return exceed, count / count.sum()


@dataclass
class Fitted:
    slope: np.ndarray
    offsets: dict
    convergence: dict

    def predict(self, study, features):
        linear = (features @ self.slope)[:, None] + self.offsets[study.name]
        if study.continuous:
            return linear
        values = expit(linear)
        if np.any(np.diff(values, axis=1) > 1e-9):
            raise ArithmeticError("Ordered probability model crossed thresholds")
        return values


def fit(studies, inputs):
    width = inputs[studies[0].name].shape[1]
    locations, blocks = {}, []
    end = width
    intercepts = []
    for s in studies:
        train = s.role == 0
        y, weight = observations(s, train)
        m = y.shape[1]
        locations[s.name] = slice(end, end + m)
        end += m
        mean = np.average(y, axis=0, weights=weight)
        if s.continuous:
            intercepts.extend(mean)
        else:
            if np.any((mean <= 0) | (mean >= 1)):
                raise ValueError("An ordinal threshold has no variance in training")
            intercepts.extend(np.log(mean / (1 - mean)))
        blocks.append((s, inputs[s.name][train], y, weight))

    def objective(values):
        slope = values[:width]
        grad = np.zeros_like(values)
        grad[:width] = RIDGE * slope
        loss = RIDGE * (slope @ slope) / 2
        for study, x, y, weight in blocks:
            indices = locations[study.name]
            pred = (x @ slope)[:, None] + values[indices]
            if study.continuous:
                residual = pred - y
                point = 0.5 * residual**2
            else:
                residual = expit(pred) - y
                point = np.logaddexp(0, pred) - y * pred
            scale = weight[:, None] / (len(studies) * y.shape[1])
            delta = scale * residual
            loss += (scale * point).sum()
            grad[indices] = delta.sum(axis=0)
            grad[:width] += x.T @ delta.sum(axis=1)
        return loss, grad

    initial = np.r_[np.zeros(width), intercepts]
    opt = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options=dict(maxiter=2500, maxls=60, gtol=5e-9, ftol=1e-15),
    )
    loss, grad = objective(opt.x)
    if not opt.success or not np.isfinite(loss) or np.max(np.abs(grad)) > 1e-6:
        raise ArithmeticError(
            f"Nonconverged fit: {opt.message}, max gradient {np.max(np.abs(grad))}"
        )
    audit = dict(
        iterations=int(opt.nit),
        objective=float(loss),
        gradient_inf=float(np.max(np.abs(grad))),
        solver=str(opt.message),
    )
    return Fitted(opt.x[:width], {k: opt.x[v] for k, v in locations.items()}, audit)


def mixture_weight(study, independent, shared):
    mask = study.role == 1
    y, weight = observations(study, mask)
    diff = shared[mask] - independent[mask]
    residual = y - independent[mask]
    denominator = (weight[:, None] * diff**2).sum()
    if denominator == 0.0:
        # The two candidates make identical development predictions.
        return 0.0
    return float(
        np.clip((weight[:, None] * residual * diff).sum() / denominator, 0.0, 1.0)
    )


def archive_model(fields, name, model):
    fields[f"{name}.slope"] = model.slope
    for source, offset in model.offsets.items():
        fields[f"{name}.offset.{source}"] = offset
