"""Estimate agreement of two actual next-word spellings from one causal probability vector."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.cloze_population_data import OUT as POP
from research.reading_validation.cloze_population_data import ROOT
from research.reading_validation.cloze_population_fit import assign, data
from research.reading_validation.cloze_population_model import distribution, project
from research.reading_validation.dispersion_transfer import BASE
from research.reading_validation.spelling_inference import (
    METHODS,
    design,
    predict,
    statistics,
)
from research.reading_validation.word_population_dispersion import OUT as DISPERSION

OUT = (
    Path(__file__).resolve().parents[1]
    / "results/reading-validation-spelling-agreement"
)
PLAN = Path(__file__).with_name("spelling-agreement-protocol.md")


def fit_fraction(x, successes, opportunities):
    if (
        np.any(successes < 0)
        or np.any(successes > opportunities + 1e-8)
        or opportunities.sum() <= 0
    ):
        raise ValueError("Invalid weighted observation support")

    def objective(w):
        z = x @ w
        denominator = opportunities.sum()
        value = (
            opportunities @ np.logaddexp(0, z) - successes @ z + 0.5 * w @ w
        ) / denominator
        gradient = (x.T @ (opportunities * expit(z) - successes) + w) / denominator
        return value, gradient

    fit = minimize(
        objective, np.zeros(x.shape[1]), method="BFGS", jac=True, options={"gtol": 1e-9}
    )
    value, gradient = objective(fit.x)
    if not np.isfinite(value) or np.max(abs(gradient)) > 1e-7:
        raise ArithmeticError("Spelling-agreement fit did not converge")
    return {
        "coefficients": fit.x.tolist(),
        "gradient": float(np.max(abs(gradient))),
        "loss": float(value),
        "effective_observations": float(opportunities.sum()),
    }


def evidence(counts, lexicon):
    named, unlisted = [], []
    for word, frequency in counts.items():
        (named if word in lexicon else unlisted).append(frequency)
    n, u = sum(named), sum(unlisted)
    total = n + u
    if total < 2:
        raise ValueError("Two different people are required for a pair")
    collisions = [sum(k * (k - 1) for k in group) for group in (named, unlisted)]
    return {
        "residual": (u, total),
        "named": (collisions[0] / (total - 1), n * (n - 1) / (total - 1)),
        "unlisted": (collisions[1] / (total - 1), u * (u - 1) / (total - 1)),
        "direct": (total - sum(collisions) / (total - 1), total),
    }


def training(native):
    raw, states, u, v, _, meta = data(native)
    source = np.flatnonzero(meta["source"])
    sentence = meta["articles"][source]
    folds = assign(sentence)
    stats = np.empty((len(source), 3))
    for fold in range(5):
        spec = read_json(POP / f"fold-{fold}.json")
        locations = np.flatnonzero([folds[s] == fold for s in sentence])
        if set(meta["ids"][source[locations]]) & set(spec["training_ids"]):
            raise ValueError("Training source received an in-sentence probability")
        for k in range(0, len(locations), 32):
            indices = locations[k : k + 32]
            native_indices = source[indices]
            lp = distribution(
                raw[native_indices],
                project(states[native_indices], spec["geometry"]),
                spec["models"]["context"],
                u,
                v,
            )
            stats[indices] = statistics(np.exp(lp))
    source_counts = read_json(ROOT / "source-counts.json")
    e = [evidence(source_counts[uid], meta["words"]) for uid in meta["ids"][source]]
    fit = {}
    for key, x in design(stats).items():
        successes, opportunities = np.asarray([row[key] for row in e]).T
        fit[key] = fit_fraction(x, successes, opportunities)
    fit["constant"] = sum(r["direct"][0] for r in e) / sum(r["direct"][1] for r in e)
    return fit


def lock(native):
    OUT.mkdir(exist_ok=True)
    fit = training(native)
    input_files = {
        "provo": (
            ROOT / "dispersion-external-logp.npy",
            "held-gates.npz",
            "forecast-lock.json",
            "held_logp",
            "held_gates",
        ),
        "peelle": (
            BASE,
            "transfer-gates.npz",
            "transfer-lock.json",
            "probabilities",
            "gates",
        ),
    }
    sealed = {}
    for study, (base, archive, receipt, prob_key, gate_key) in input_files.items():
        existing = read_json(DISPERSION / receipt)
        if (
            digest(base) != existing[prob_key]
            or digest(DISPERSION / archive) != existing[gate_key]
        ):
            raise ValueError("Frozen base probabilities changed")
        logs = np.load(base, mmap_mode="r")
        with np.load(DISPERSION / archive) as meta:
            ids = meta["ids"]
        if len(logs) != len(ids):
            raise ValueError("Missing original word IDs")
        forecasts = []
        for k in range(0, len(ids), 32):
            p = np.exp(logs[k : k + 32].astype(float))
            p /= p.sum(axis=1)[:, None]
            forecasts.append(predict(statistics(p), fit))
        path = OUT / f"{study}-forecasts.npz"
        np.savez_compressed(
            path, ids=ids, forecast=np.concatenate(forecasts), methods=np.array(METHODS)
        )
        sealed[study] = {
            "previous": digest(DISPERSION / receipt),
            "forecast": digest(path),
            "locations": len(ids),
        }
    write_json(
        OUT / "fit-lock.json",
        {
            "protocol": digest(PLAN),
            "training_native": digest(native / "receipt.json"),
            "source_counts": digest(ROOT / "source-counts.json"),
            "weights": fit,
            "studies": sealed,
            "new_external_response_data_opened": False,
        },
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("native", type=Path)
    lock(parser.parse_args().native)
