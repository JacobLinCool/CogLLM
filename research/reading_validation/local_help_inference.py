"""Fixed Guo information-need probabilities for an observed or unknown new reader."""

import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from research.reading_validation.joint_data import digest

DEFAULT = (
    Path(__file__).resolve().parents[1] / "results/reading-validation-local/help-choice"
)
COLUMNS = (
    "frozen_lexical",
    "form_score",
    "term_words",
    "term_characters",
    "relative_position",
    "deft_surface",
    "local_phrase",
    "deft_proposal",
    "deft_semantic",
)
DESIGNS = {
    "lexical": (5, False),
    "lexical_prior": (5, True),
    "lexical_deft_prior": (6, True),
    "article": (9, False),
    "article_prior": (9, True),
}


def design(features, prior, name):
    n, use_prior = DESIGNS[name]
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] < n or not np.isfinite(x[:, :n]).all():
        raise ValueError(f"Expected {n} finite input columns for {name}")
    x = x[:, :n]
    if not use_prior:
        return x
    pre = np.asarray(prior, dtype=float)
    if pre.shape != (len(x),) or not np.isin(pre, [0, 1, 2, 3, 4]).all():
        raise ValueError(
            "A known-reader model requires one prereading rating per term on the 0–4 scale"
        )
    pre = (pre - 2) / 2
    return np.column_stack([x, pre, x * pre[:, None]])


class HelpReadout:
    def __init__(self, path=DEFAULT):
        path = Path(path)
        seal = json.loads((path / "forecast-seal.json").read_bytes())
        if digest(path / "weights.json") != seal["weights_sha256"]:
            raise ValueError("Guo serving coefficients differ from forecast seal")
        self.weights = json.loads((path / "weights.json").read_bytes())

    def predict(self, features, *, endpoint, arm, familiarity_before=None):
        """Return log-odds and probabilities for new readers; no encoder or decoder is invoked."""
        matrix = design(features, familiarity_before, arm)
        p = self.weights[endpoint][arm]
        z = ((matrix - p["center"]) / p["scale"]) @ p["coef"] + p["intercept"]
        return {"log_odds": z, "probability": expit(z)}
