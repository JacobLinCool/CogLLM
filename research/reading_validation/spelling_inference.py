"""Read population-level full-spelling agreement from a single native causal exposure."""

from pathlib import Path

import numpy as np
from scipy.special import expit, logit, logsumexp, softmax

from research.backbone_transfer.extract import digest, read_json
from research.reading_validation.population_inference import (
    DEFAULT,
    PopulationWordInference,
)

METHODS = ("grouped", "midpoint", "constant", "direct", "conditional")
SPELLING_FIT = "44d7cbf5b8d4dc6dde725b6519ba30195294300adbb4c9b83e5e77f5f167491a"
DISPERSION_FIT = "b9e49f6789e54994475cd56760144837439876e42e9653a6f5111e9157ffaad1"


def statistics(probabilities):
    """Residual mass, agreement conditional on two named words, and grouped disagreement."""
    p = np.asarray(probabilities, dtype=float)
    if (
        p.ndim != 2
        or p.shape[1] != 8193
        or np.any(p < 0)
        or np.max(abs(p.sum(axis=1) - 1)) > 2e-6
    ):
        raise ValueError(
            "Expected normalized next-word probabilities in the original vocabulary"
        )
    named = p[:, :-1]
    named_mass = named.sum(axis=1)
    if np.any(named_mass <= 0):
        raise ValueError(
            "Conditional spelling agreement requires some named probability"
        )
    named_agreement = np.sum((named / named_mass[:, None]) ** 2, axis=1)
    grouped = 1 - np.sum(p * p, axis=1)
    return np.column_stack((p[:, -1], named_agreement, grouped))


def design(stats):
    residual, named, grouped = logit(np.clip(stats, 1e-8, 1 - 1e-8)).T
    ones = np.ones(len(stats))
    relation = np.column_stack((ones, named, residual))
    return {
        "residual": np.column_stack((ones, residual)),
        "named": relation,
        "unlisted": relation,
        "direct": np.column_stack((ones, grouped)),
    }


def predict(stats, fit):
    x = design(stats)
    quantities = {k: expit(v @ fit[k]["coefficients"]) for k, v in x.items()}
    r = quantities["residual"]
    exact = 1 - (1 - r) ** 2 * quantities["named"] - r**2 * quantities["unlisted"]
    result = np.column_stack(
        (
            stats[:, 2],
            stats[:, 2] + stats[:, 0] ** 2 / 2,
            np.full(len(stats), fit["constant"]),
            quantities["direct"],
            exact,
        )
    )
    if np.any((result < -2e-6) | (result > 1 + 2e-6)) or not np.isfinite(result).all():
        raise ArithmeticError("Invalid exact-spelling prediction")
    return result


class ReadingWordForecast:
    """Apply exact-spelling and response-choice readouts to the same next-token probabilities."""

    def __init__(self, base, spelling, dispersion):
        self.base = base
        self.spelling = spelling
        self.dispersion = dispersion

    @classmethod
    def load(cls, base_directory=DEFAULT):
        base = PopulationWordInference.load(base_directory)
        results = Path(base_directory).resolve().parents[1]
        spelling_file = results / "reading-validation-spelling-agreement/fit-lock.json"
        dispersion_file = (
            results / "reading-validation-word-dispersion/forecast-lock.json"
        )
        if (
            digest(spelling_file) != SPELLING_FIT
            or digest(dispersion_file) != DISPERSION_FIT
        ):
            raise ValueError(
                "Spelling or dispersion estimates differ from the scored release"
            )
        spelling = read_json(spelling_file)
        dispersion = read_json(dispersion_file)
        spec = read_json(Path(base_directory) / "spec.json")
        if (
            spec["source"] != dispersion["frozen_base_fits"][-1]
            or dispersion["native"] != spelling["training_native"]
        ):
            raise ValueError(
                "Readouts were fitted on a different source feature distribution"
            )
        return cls(base, spelling["weights"], dispersion)

    def distributions(self, p):
        """The two individually normalized word-choice forecasts, both trained only on UCL."""
        logp = np.log(p)
        features = np.column_stack(
            (p.max(1), p[:, -1], -(p * logp).sum(1) / np.log(p.shape[1]))
        )
        features = (features - self.dispersion["source_center"]) / self.dispersion[
            "source_scale"
        ]
        x = np.column_stack((np.ones(len(p)), features))
        logits = logp[:, :, None] * np.array([2 / 3, 1.0, 1.5])
        variants = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        result = {}
        for method, fit in self.dispersion["models"].items():
            eta = x @ np.array(fit["coefficients"]).reshape(4, 2)
            weights = softmax(
                np.column_stack((eta[:, 0], np.zeros(len(x)), eta[:, 1]))
                + np.log([0.1, 0.8, 0.1]),
                axis=1,
            )
            result[method] = np.einsum("ijk,ik->ij", variants, weights)
        return result

    def analyze(self, text, encoded, runtime):
        """Word-aligned probabilities. The caller encodes the printed article exactly once."""
        _, positions = self.base.positions(text, encoded, runtime.tokenizer)
        if not positions:
            raise ValueError("No prefix-certified printed word end")
        rows = []
        for p in self.base.probabilities(runtime, encoded, [x[3] for x in positions]):
            learned = predict(statistics(p), self.spelling)
            alternate = self.distributions(p)
            for j, values in enumerate(learned):
                index, start, end, token, _ = positions[len(rows)]
                calibrated = {k: float(1 - (q[j] @ q[j])) for k, q in alternate.items()}
                rows.append(
                    {
                        "index": index,
                        "start": start,
                        "end": end,
                        "token": token,
                        "full_spelling": float(values[-1]),
                        "direct_spelling": float(values[-2]),
                        "grouped": float(values[0]),
                        "calibrated_grouped": calibrated,
                    }
                )
        if len(rows) != len(positions):
            raise ArithmeticError("Incomplete word-alignment replay")
        return rows
