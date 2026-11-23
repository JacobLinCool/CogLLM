"""Bounded source weights from lagged training measurements."""

from dataclasses import dataclass
import math
from statistics import median


_MIN_GRADIENT = 1e-12
_MIN_LOSS_SCALE = 0.1


def _bounded_mean_one(values, bases, lower, upper):
    """Find a common multiplier so clipping preserves the total base mass."""
    minimum = 1.0 / max(values)
    maximum = 1.0 / min(values)
    target = math.fsum(bases)

    def clipped(scale):
        return tuple(min(upper, max(lower, scale * value)) for value in values)

    # The weighted sum is affine between clipping breakpoints. Interpolating
    # on the crossing interval needs at most two breakpoints per microbatch.
    breaks = {bound / value for bound in (lower, upper) for value in values if minimum < bound / value < maximum}
    previous_scale = minimum
    previous_mass = math.fsum(base * weight for base, weight in zip(bases, clipped(minimum)))
    if math.isclose(previous_mass, target, rel_tol=1e-12):
        return clipped(minimum)
    for scale in sorted(breaks | {maximum}):
        weights = clipped(scale)
        mass = math.fsum(base * weight for base, weight in zip(bases, weights))
        if math.isclose(mass, target, rel_tol=1e-12):
            return weights
        if mass > target:
            fraction = (target - previous_mass) / (mass - previous_mass)
            return clipped(previous_scale + fraction * (scale - previous_scale))
        previous_scale, previous_mass = scale, mass
    raise ArithmeticError('Could not normalize source weights within tolerance')


@dataclass
class _ObjectiveHistory:
    observations: int = 0
    reference_loss: float = 0.0
    reference_loss_m2: float = 0.0
    reference_observations: int = 0
    loss_ema: float = 0.0
    log_gradient_ema: float | None = None
    zero_gradients: int = 0

    def add(self, loss, norm, warmup, decay):
        self.observations += 1
        if self.observations <= warmup:
            delta = loss - self.reference_loss
            self.reference_observations += 1
            self.reference_loss += delta / self.reference_observations
            self.reference_loss_m2 += delta * (loss - self.reference_loss)
        if self.observations == 1:
            self.loss_ema = loss
        else:
            self.loss_ema = decay * self.loss_ema + (1 - decay) * loss
        if norm <= _MIN_GRADIENT:
            self.zero_gradients += 1
        else:
            value = math.log(norm)
            if self.log_gradient_ema is None:
                self.log_gradient_ema = value
            else:
                self.log_gradient_ema = decay * self.log_gradient_ema + (1 - decay) * value

    def relative_loss(self):
        # Changes in a density's units can add a constant to its NLL. The
        # warmup spread and additive loss change are invariant to that shift.
        scale = max(self.reference_scale(), _MIN_LOSS_SCALE)
        return min(2.0, max(-2.0, (self.loss_ema - self.reference_loss) / scale))

    def reference_scale(self):
        return math.sqrt(max(0.0, self.reference_loss_m2 / self.reference_observations))


@dataclass
class _WeightHistory:
    draws: int = 0
    total: float = 0.0
    least: float = math.inf
    greatest: float = -math.inf
    base_mass: float = 0.0
    weighted_mass: float = 0.0

    def add(self, value, base):
        self.draws += 1
        self.total += value
        self.least = min(self.least, value)
        self.greatest = max(self.greatest, value)
        self.base_mass += base
        self.weighted_mass += base * value


class SourceWeights:
    """Preserve the base mass of a group of training microbatches.

    The runner calls ``weights`` before backpropagating that group and calls
    ``observe`` once per microbatch after the optimizer step. Observations
    are unweighted loss and norm of its gradient on the shared adapter.
    Give different likelihood blocks within a ``source`` their stable
    ``endpoint`` names. Their loss changes are scaled separately before
    constructing the source weight.
    """

    def __init__(self, mode, *, warmup=4, decay=0.85, lower=0.35, upper=2.5):
        if mode not in ('fixed', 'adaptive'):
            raise ValueError(f'Unknown weight policy: {mode!r}')
        if type(warmup) is not int or warmup < 1 or not 0 <= decay < 1:
            raise ValueError('Expected positive warmup and EMA decay in [0, 1)')
        if not math.isfinite(upper) or not 0 < lower < 1 < upper:
            raise ValueError('Weight bounds must strictly contain the neutral weight 1')
        self.mode = mode
        self.warmup = warmup
        self.decay = decay
        self.lower = lower
        self.upper = upper
        self._objectives = {}
        self._applied = {}

    def observe(self, source, loss, gradient_norm, *, endpoint=None):
        """Record finite measurements from a completed training microbatch."""
        loss, gradient_norm = float(loss), float(gradient_norm)
        if not math.isfinite(loss) or not math.isfinite(gradient_norm) or gradient_norm < 0:
            raise ValueError(f'Invalid training loss or gradient norm for {source!r}')
        objectives = self._objectives.setdefault(source, {})
        history = objectives.setdefault(endpoint, _ObjectiveHistory())
        history.add(loss, gradient_norm, self.warmup, self.decay)

    def weights(self, sources, *, endpoints=None, base=None):
        """Return one multiplier per microbatch in the supplied order."""
        sources = tuple(sources)
        endpoints = (None,) * len(sources) if endpoints is None else tuple(endpoints)
        bases = (1.0,) * len(sources) if base is None else tuple(map(float, base))
        if not sources or len(sources) != len(endpoints) or len(sources) != len(bases):
            raise ValueError('A weight group needs aligned sources, endpoints and base masses')
        if any(not math.isfinite(value) or value <= 0 for value in bases):
            raise ValueError('Base masses must be finite and positive')
        keys = tuple(zip(sources, endpoints))
        result = (1.0,) * len(keys)
        if self.mode == 'adaptive' and len(set(sources)) > 1:
            result = self._adaptive_weights(sources, bases)
        for key, value, mass in zip(keys, result, bases):
            self._applied.setdefault(key, _WeightHistory()).add(value, mass)
        return result

    def _adaptive_weights(self, sources, bases):
        ready = {}
        for source in dict.fromkeys(sources):
            endpoints = self._objectives.get(source, {}).values()
            eligible = [item for item in endpoints if item.observations >= self.warmup
                        and item.log_gradient_ema is not None]
            if eligible:
                ready[source] = (median(item.log_gradient_ema for item in eligible),
                                 math.fsum(item.relative_loss() for item in eligible) / len(eligible))
        if len(ready) < 2:
            return (1.0,) * len(sources)
        centered_norm = median(value[0] for value in ready.values())
        centered_loss = median(value[1] for value in ready.values())
        raw = {}
        for source, (norm, relative_loss) in ready.items():
            log_weight = 0.5 * ((relative_loss - centered_loss) - (norm - centered_norm))
            raw[source] = math.exp(min(4.0, max(-4.0, log_weight)))
        active = [(raw[source], base) for source, base in zip(sources, bases) if source in raw]
        values, masses = zip(*active)
        scale = max(masses)
        relative_masses = tuple(mass / scale for mass in masses)
        normalized = iter(_bounded_mean_one(values, relative_masses, self.lower, self.upper))
        return tuple(next(normalized) if source in raw else 1.0 for source in sources)

    def summary(self):
        """Return JSON-compatible coverage and realized weight statistics."""
        objectives = []
        observed = {(source, endpoint) for source, values in self._objectives.items() for endpoint in values}
        for source, endpoint in sorted(self._applied.keys() | observed, key=repr):
            history = self._objectives.get(source, {}).get(endpoint)
            applied = self._applied.get((source, endpoint))
            objectives.append({
                'source': source,
                'endpoint': endpoint,
                'observations': 0 if history is None else history.observations,
                'zero_gradients': 0 if history is None else history.zero_gradients,
                'reference_loss': None if history is None else history.reference_loss,
                'reference_loss_scale': None if history is None else history.reference_scale(),
                'loss_ema': None if history is None else history.loss_ema,
                'log_gradient_ema': None if history is None else history.log_gradient_ema,
                'draws': 0 if applied is None else applied.draws,
                'mean_weight': None if applied is None else applied.total / applied.draws,
                'min_weight': None if applied is None else applied.least,
                'max_weight': None if applied is None else applied.greatest,
                'base_mass': 0.0 if applied is None else applied.base_mass,
                'weighted_mass': 0.0 if applied is None else applied.weighted_mass,
            })
        return {'mode': self.mode, 'warmup': self.warmup, 'decay': self.decay,
                'weight_bounds': [self.lower, self.upper], 'objectives': objectives}
