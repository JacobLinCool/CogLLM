"""FP64 readouts for the shared-adapter load-curve pilot."""

from dataclasses import dataclass
import hashlib
from typing import Literal

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


WIDTH = 24
L2 = 0.01
MAX_ITERATIONS = 65
FTOL = 1e-12
GTOL = 1e-7


@dataclass(frozen=True)
class TaskData:
    """Feature rows with binary vote counts or first-hit interval traces.

    Binary `y` rows are [negative votes, positive votes]. Interval `y` rows
    are [risk start, first eligible hit, exclusive end], indexing feature rows.
    Every interval row represents a trace with an observed hit in that interval.
    """

    x: np.ndarray
    extra: np.ndarray
    kind: Literal['binary', 'interval']
    y: np.ndarray

    def __post_init__(self):
        x = np.ascontiguousarray(self.x, dtype=np.float64)
        extra = np.ascontiguousarray(self.extra, dtype=np.float64)
        if x.ndim != 2 or not min(x.shape) or extra.shape != (len(x), 2):
            raise ValueError('Expected nonempty (N,D) features with (N,2) fixed covariates')
        if not np.isfinite(x).all() or not np.isfinite(extra).all():
            raise ValueError('Readout features must be finite')
        y = np.asarray(self.y)
        if self.kind == 'binary':
            y = np.ascontiguousarray(y, dtype=np.float64)
            if (y.shape != (len(x), 2) or not np.isfinite(y).all() or
                    np.any(y < 0) or np.any(y.sum(axis=1) <= 0)):
                raise ValueError('Each binary record needs nonnegative counts and at least one vote')
        elif self.kind == 'interval':
            if y.ndim != 2 or y.shape[1] != 3 or not len(y) or y.dtype.kind not in 'iu':
                raise ValueError('Expected nonempty integer first-hit intervals with shape (M,3)')
            y = np.ascontiguousarray(y, dtype=np.intp)
            start, first, end = y.T
            if np.any(start < 0) or np.any(start > first) or np.any(first >= end) or np.any(end > len(x)):
                raise ValueError('First-hit intervals must satisfy 0 <= start <= first < end <= N')
        else:
            raise ValueError(f'Unknown observation kind: {self.kind}')
        object.__setattr__(self, 'x', x)
        object.__setattr__(self, 'extra', extra)
        object.__setattr__(self, 'y', y)


def _binary_loss(counts, logits):
    negative, positive = counts.T
    return negative * np.logaddexp(0, logits) + positive * np.logaddexp(0, -logits)


def _row_indices(starts, ends):
    lengths = ends - starts
    offsets = np.r_[0, np.cumsum(lengths)]
    trace_ids = np.repeat(np.arange(len(starts)), lengths)
    rows = np.arange(offsets[-1]) + (starts - offsets[:-1])[trace_ids]
    return rows, trace_ids, offsets[:-1]


class _FirstHit:
    def __init__(self, intervals):
        start, first, end = intervals.T
        self.trace_count = len(intervals)
        self.before, self.before_trace, _ = _row_indices(start, first)
        self.inside, self.inside_trace, self.inside_offsets = _row_indices(first, end)

    def loss(self, logits):
        hazards = np.logaddexp(0, logits)
        window_logits = logits[self.inside]
        window_hazards = hazards[self.inside]
        log_hazards = window_logits.copy()
        np.log(window_hazards, out=log_hazards, where=window_logits >= -36)
        log_window = np.logaddexp.reduceat(log_hazards, self.inside_offsets)
        window = np.exp(log_window)
        log_hit = log_window.copy()
        representable = log_window >= -36
        small = representable & (window <= np.log(2))
        large = representable & ~small
        log_hit[small] = np.log(-np.expm1(-window[small]))
        log_hit[large] = np.log1p(-np.exp(-window[large]))
        before = np.bincount(self.before_trace, weights=hazards[self.before], minlength=self.trace_count)
        return before - log_hit, window + log_hit

    def objective(self, logits):
        losses, log_expm1_window = self.loss(logits)
        log_hit_gradient = -np.logaddexp(0, -logits[self.inside]) - log_expm1_window[self.inside_trace]
        gradient = np.zeros_like(logits)
        np.add.at(gradient, self.before, expit(logits[self.before]))
        np.add.at(gradient, self.inside, -np.exp(log_hit_gradient))
        return float(losses.mean()), gradient / self.trace_count


def per_observation_nll(dataset, logits):
    """Return one mean vote NLL per binary record or one NLL per first-hit trace."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != (len(dataset.x),) or not np.isfinite(logits).all():
        raise ValueError('Expected one finite logit per feature row')
    if dataset.kind == 'binary':
        return _binary_loss(dataset.y, logits) / dataset.y.sum(axis=1)
    return _FirstHit(dataset.y).loss(logits)[0]


def _objective_for(dataset):
    if dataset.kind == 'interval':
        return _FirstHit(dataset.y).objective
    negative, positive = dataset.y.T
    total = dataset.y.sum()

    def objective(logits):
        nll = _binary_loss(dataset.y, logits).sum() / total
        gradient = (negative * expit(logits) - positive * expit(-logits)) / total
        return float(nll), gradient

    return objective


def _head_initialization(task, data, seed):
    size = data.x.shape[1] + 2 + WIDTH + 1
    weights = np.zeros(size, dtype=np.float64)
    rng_seed = int.from_bytes(hashlib.sha256(f'{seed}:{task}'.encode()).digest()[:8], 'big')
    rng = np.random.default_rng(rng_seed)
    weights[-WIDTH - 1:-1] = rng.normal(scale=0.03, size=WIDTH)
    if data.kind == 'binary':
        negative, positive = data.y.sum(axis=0) + 0.5
        weights[-1] = np.log(positive) - np.log(negative)
    else:
        weights[-1] = -np.log((data.y[:, 2] - data.y[:, 0]).mean())
    return weights


def _components(parameters, dimensions):
    adapter_size = (dimensions + 1) * WIDTH
    adapter = parameters[:adapter_size].reshape(dimensions + 1, WIDTH)
    heads = parameters[adapter_size:].reshape(-1, dimensions + 2 + WIDTH + 1)
    return adapter, heads


def _logits(x, extra, hidden, head):
    dimensions = x.shape[1]
    return x @ head[:dimensions] + extra @ head[dimensions:dimensions + 2] + hidden @ head[-WIDTH - 1:-1] + head[-1]


def _joint_objective(tasks, dimensions):
    datasets = list(tasks.values())
    objectives = [_objective_for(data) for data in datasets]
    task_count = len(datasets)

    def objective(parameters):
        adapter, heads = _components(parameters, dimensions)
        gradient = np.zeros_like(parameters)
        adapter_gradient, head_gradients = _components(gradient, dimensions)
        loss = 0.0
        for data, score, head, head_gradient in zip(datasets, objectives, heads, head_gradients, strict=True):
            hidden = np.tanh(data.x @ adapter[:-1] + adapter[-1])
            task_nll, logit_gradient = score(_logits(data.x, data.extra, hidden, head))
            loss += task_nll
            head_gradient[:dimensions] = data.x.T @ logit_gradient
            head_gradient[dimensions:dimensions + 2] = data.extra.T @ logit_gradient
            head_gradient[-WIDTH - 1:-1] = hidden.T @ logit_gradient
            head_gradient[-1] = logit_gradient.sum()
            hidden_gradient = logit_gradient[:, None] * head[-WIDTH - 1:-1] * (1 - hidden**2)
            adapter_gradient[:-1] += data.x.T @ hidden_gradient
            adapter_gradient[-1] += hidden_gradient.sum(axis=0)

        loss /= task_count
        gradient /= task_count
        loss += L2 / 2 * (np.square(adapter[:-1]).sum() + np.square(heads[:, :-1]).sum() / task_count)
        adapter_gradient[:-1] += L2 * adapter[:-1]
        head_gradients[:, :-1] += (L2 / task_count) * heads[:, :-1]
        return float(loss), gradient

    return objective


def _solve(tasks, seed):
    dimensions = next(iter(tasks.values())).x.shape[1]
    rng = np.random.default_rng(seed)
    initial_adapter = np.zeros((dimensions + 1, WIDTH), dtype=np.float64)
    initial_adapter[:-1] = rng.normal(scale=1 / np.sqrt(dimensions), size=(dimensions, WIDTH))
    initial_heads = [_head_initialization(task, data, seed) for task, data in tasks.items()]
    initial_values = np.concatenate([initial_adapter.ravel(), *initial_heads])
    result = minimize(
        _joint_objective(tasks, dimensions), initial_values, method='L-BFGS-B', jac=True,
        options={'maxiter': MAX_ITERATIONS, 'ftol': FTOL, 'gtol': GTOL},
    )
    if (not np.isfinite(result.x).all() or not np.isfinite(result.fun) or not np.isfinite(result.jac).all()
            or (not result.success and not (result.status == 1 and result.nit == MAX_ITERATIONS))):
        raise RuntimeError(f'Load-curve L-BFGS-B failed for {list(tasks)}: {result.message}')
    adapter, heads = _components(result.x, dimensions)
    info = {
        'success': bool(result.success), 'status': int(result.status), 'message': str(result.message),
        'iterations': int(result.nit), 'evaluations': int(result.nfev), 'objective': float(result.fun),
        'max_abs_gradient': float(np.abs(result.jac).max()),
    }
    return adapter.copy(), dict(zip(tasks, (head.copy() for head in heads), strict=True)), info


def fit(tasks, seed, shared):
    """Fit matching heads with one shared adapter or separate optimization per task.

    The objective is the mean of task-level losses. Binary loss averages eligible votes
    and interval loss averages traces. Weight L2 is 0.01 times half their squared norm,
    averaged across tasks, with biases exempt. Each solver gets the same 65-step budget.
    """
    ordered = dict(sorted(tasks.items()))
    if not ordered or len({data.x.shape[1] for data in ordered.values()}) != 1:
        raise ValueError('Expected tasks with a common feature dimension')
    if shared:
        adapter, heads, info = _solve(ordered, seed)
        return {'shared': True, 'adapters': {'shared': adapter}, 'heads': heads, 'fit_info': {'shared': info}}

    adapters, heads, fit_info = {}, {}, {}
    for task, data in ordered.items():
        adapters[task], task_heads, fit_info[task] = _solve({task: data}, seed)
        heads[task] = task_heads[task]
    return {'shared': False, 'adapters': adapters, 'heads': heads, 'fit_info': fit_info}


def predict_logits(model, task, x, extra):
    """Apply a fitted task head without fitting or reweighting evaluation rows."""
    adapter = model['adapters']['shared' if model['shared'] else task]
    head = model['heads'][task]
    x = np.asarray(x, dtype=np.float64)
    extra = np.asarray(extra, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] + 1 != len(adapter) or extra.shape != (len(x), 2):
        raise ValueError('Prediction features must match the fitted dimensions')
    if not np.isfinite(x).all() or not np.isfinite(extra).all():
        raise ValueError('Prediction features must be finite')
    return _logits(x, extra, np.tanh(x @ adapter[:-1] + adapter[-1]), head)
