"""Fit discrete first-correct hazards for direct or latent cue paths."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


RIDGE = 0.10
METHODS = ("position", "surface", "qwen_independent", "qwen_shared")


@dataclass(frozen=True)
class HazardTraining:
    """Feature axes are family, path, exposure, feature; counts include end censoring."""

    surface: np.ndarray
    representation: np.ndarray
    counts: np.ndarray


@dataclass(frozen=True)
class _Study:
    surface: np.ndarray
    representation: np.ndarray
    target_probability: np.ndarray
    trials: float


@dataclass(frozen=True)
class _Layout:
    intercept: slice
    surface: slice
    representation: slice


def _feature_arrays(surface: np.ndarray, representation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    surface = np.asarray(surface, dtype=np.float64, order="C")
    representation = np.asarray(representation, dtype=np.float64, order="C")
    if (surface.ndim != 4 or representation.ndim != 4 or surface.shape[:3] != representation.shape[:3]
            or not all(surface.shape[:3]) or not np.isfinite(surface).all() or not np.isfinite(representation).all()):
        raise ValueError("Surface and representation must be finite, aligned family-by-path-by-exposure arrays")
    return surface, representation


def _prepare(data: HazardTraining) -> _Study:
    surface, representation = _feature_arrays(data.surface, data.representation)
    counts = np.asarray(data.counts, dtype=np.float64, order="C")
    if (counts.shape != (surface.shape[0], surface.shape[2] + 1)
            or not np.isfinite(counts).all() or (counts < 0).any()):
        raise ValueError("Counts must give a nonnegative, finite first-correct or censored frequency for every family")
    family_totals = counts.sum(axis=1, keepdims=True)
    total = float(family_totals.sum())
    if (family_totals <= 0).any() or not math.isfinite(total):
        raise ValueError("Every family must have positive observed trials and the study total must be finite")
    return _Study(surface, representation, counts / family_totals, total)


def _path_log_masses(logits: np.ndarray) -> np.ndarray:
    log_survival = np.cumsum(-np.logaddexp(0, logits), axis=-1)
    prior = np.concatenate((np.zeros_like(log_survival[..., :1]), log_survival[..., :-1]), axis=-1)
    event = prior - np.logaddexp(0, -logits)
    return np.concatenate((event, log_survival[..., -1:]), axis=-1)


def _marginal_term(logits: np.ndarray, target_probability: np.ndarray) -> tuple[float, np.ndarray]:
    paths = _path_log_masses(logits)
    log_sum = logsumexp(paths, axis=1)
    log_marginal = log_sum - math.log(logits.shape[1])
    weight = target_probability / target_probability.shape[0]
    loss = -np.sum(weight * log_marginal)
    # A path's responsibility is conditional on the observed terminal outcome.
    responsibility = np.exp(paths - log_sum[:, None, :]) * weight[:, None, :]
    total_remaining = np.cumsum(responsibility[..., ::-1], axis=-1)[..., :0:-1]
    derivative = expit(logits) * total_remaining - responsibility[..., :-1]
    return float(loss), derivative


def _linear_logits(surface: np.ndarray, representation: np.ndarray, intercept: np.ndarray,
                   surface_slopes: np.ndarray, qwen_slopes: np.ndarray) -> np.ndarray:
    logits = np.broadcast_to(intercept, surface.shape[:-1]).copy()
    if surface_slopes.size:
        logits += surface @ surface_slopes
    if qwen_slopes.size:
        logits += representation @ qwen_slopes
    return logits


def forecast_log_masses(model: dict, surface: np.ndarray, representation: np.ndarray) -> np.ndarray:
    """Return one normalized log distribution over event ordinals and censoring per family."""
    surface, representation = _feature_arrays(surface, representation)
    intercept, surface_slopes, qwen_slopes = (
        np.asarray(model[key], dtype=np.float64) for key in ("intercepts", "surface_slopes", "qwen_slopes")
    )
    if (intercept.shape != (surface.shape[2],)
            or surface_slopes.shape not in ((0,), (surface.shape[3],))
            or qwen_slopes.shape not in ((0,), (representation.shape[3],))
            or not all(np.isfinite(block).all() for block in (intercept, surface_slopes, qwen_slopes))):
        raise ValueError("Forecast feature axes do not agree with the frozen hazard model")
    logits = _linear_logits(surface, representation, intercept, surface_slopes, qwen_slopes)
    log_marginal = logsumexp(_path_log_masses(logits), axis=1) - math.log(surface.shape[1])
    if (not np.isfinite(log_marginal).all()
            or not np.allclose(logsumexp(log_marginal, axis=-1), 0, rtol=0, atol=1e-12)):
        raise ValueError("The fitted hazard model produced invalid outcome masses")
    return log_marginal


def _parameter_layout(studies: dict[str, _Study], method: str) -> tuple[dict[str, _Layout], slice, np.ndarray]:
    layouts = {}
    initial = []
    cursor = 0
    for name, data in studies.items():
        steps = data.surface.shape[2]
        surface_width = data.surface.shape[3] if method != "position" else 0
        representation_width = data.representation.shape[3] if method == "qwen_independent" else 0
        surface_start = cursor + steps
        representation_start = surface_start + surface_width
        stop = representation_start + representation_width
        layouts[name] = _Layout(slice(cursor, surface_start), slice(surface_start, representation_start),
                                slice(representation_start, stop))
        position = data.target_probability.mean(axis=0)
        future = np.cumsum(position[::-1])[::-1][1:]
        virtual_half = 0.5 / data.trials
        initial.extend(np.log(position[:-1] + virtual_half) - np.log(future + virtual_half))
        initial.extend(np.zeros(surface_width + representation_width))
        cursor = stop
    shared_width = next(iter(studies.values())).representation.shape[3] if method == "qwen_shared" else 0
    initial.extend(np.zeros(shared_width))
    return layouts, slice(cursor, cursor + shared_width), np.asarray(initial, dtype=np.float64)


def _objective(theta: np.ndarray, studies: dict[str, _Study], layouts: dict[str, _Layout],
               shared: slice) -> tuple[float, np.ndarray]:
    value = 0.0
    gradient = np.zeros_like(theta)
    shared_slope = theta[shared]
    for name, data in studies.items():
        layout = layouts[name]
        intercept = theta[layout.intercept]
        surface = theta[layout.surface]
        representation_slot = shared if shared_slope.size else layout.representation
        representation = theta[representation_slot]
        logits = _linear_logits(data.surface, data.representation, intercept, surface, representation)
        event_loss, derivative = _marginal_term(logits, data.target_probability)
        pseudo_loss = np.logaddexp(0, intercept).sum() - 0.5 * intercept.sum()
        value += event_loss + pseudo_loss / data.trials
        gradient[layout.intercept] = derivative.sum(axis=(0, 1)) + (expit(intercept) - 0.5) / data.trials
        if surface.size:
            value += RIDGE / 2 * (surface @ surface)
            gradient[layout.surface] = np.einsum("fpks,fpk->s", data.surface, derivative) + RIDGE * surface
        if representation.size:
            gradient[representation_slot] += np.einsum("fpkr,fpk->r", data.representation, derivative)
            if not shared_slope.size:
                value += RIDGE / 2 * (representation @ representation)
                gradient[representation_slot] += RIDGE * representation

    value /= len(studies)
    gradient /= len(studies)
    value += RIDGE / 2 * (shared_slope @ shared_slope)
    gradient[shared] += RIDGE * shared_slope
    return float(value), gradient


def _fit(studies: dict[str, _Study], method: str) -> tuple[dict[str, dict], dict]:
    layouts, shared, initial = _parameter_layout(studies, method)
    result = minimize(
        _objective, initial, args=(studies, layouts, shared), method="L-BFGS-B", jac=True,
        options={"maxiter": 750, "ftol": 1e-13, "gtol": 1e-9},
    )
    if (not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun)
            or not np.isfinite(result.jac).all()):
        raise RuntimeError(f"Joint cue hazard optimization failed: {result.message}")
    models = {
        name: {
            "intercepts": result.x[layout.intercept].tolist(),
            "surface_slopes": result.x[layout.surface].tolist(),
            "qwen_slopes": result.x[shared if method == "qwen_shared" else layout.representation].tolist(),
        }
        for name, layout in layouts.items()
    }
    return models, {"iterations": int(result.nit), "function_evaluations": int(result.nfev),
                    "penalized_loss": float(result.fun), "max_absolute_gradient": float(np.max(np.abs(result.jac)))}


def fit_hazards(studies: dict[str, HazardTraining], method: str) -> tuple[dict[str, dict], dict]:
    """Fit local controls separately and the shared Qwen slope jointly across studies."""
    if not studies or any(type(name) is not str or not name for name in studies) or method not in METHODS:
        raise ValueError("Expected named training studies and a registered hazard method")
    data = {name: _prepare(value) for name, value in studies.items()}
    widths = {value.representation.shape[-1] for value in data.values()}
    if method in {"qwen_independent", "qwen_shared"} and 0 in widths:
        raise ValueError("Qwen hazard methods require representation features")
    if method == "qwen_shared":
        if len(widths) != 1:
            raise ValueError("A shared Qwen slope requires one common feature width across studies")
        models, report = _fit(data, method)
        optimizations = {"joint": report}
    else:
        models, optimizations = {}, {}
        for name, study in data.items():
            fitted, optimizations[name] = _fit({name: study}, method)
            models[name] = fitted[name]
    return models, {"method": method, "ridge": RIDGE, "optimizations": optimizations}
