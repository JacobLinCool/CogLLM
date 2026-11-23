"""Integrate a persistent reporting propensity over each reader's six original stories."""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit, logsumexp

from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.first_cue import OUT as PREVIOUS
from research.reading_validation.intervention_moment import measurements
from research.reading_validation.memory_data import BASE, load_rows
from research.reading_validation.memory_score import (
    reveal,
    summarize,
    trial_observation,
)
from research.reading_validation.referent_features import setup
from research.reading_validation.taxonomic_monitor import Route
from research.reading_validation.taxonomic_pairs import (
    Ontology,
    form,
    native_words,
    opportunities,
    source_links,
)

OUT = Path("research/results/reading-validation-reporting-propensity")
PLAN = Path(__file__).with_name("reporting-propensity-protocol.md")


def fit(x, observations, vary):
    keys = list(x)
    base = 0
    parts = []
    two = {}
    for k in keys:
        parts.append(x[k][1:])
        two[k] = (base, len(parts[-1]))
        base += len(parts[-1])
    raw = np.vstack(parts)
    mu = raw.mean(0)
    sd = raw.std(0)
    sd[sd < 1e-6] = 1.0

    def design(X):
        return np.column_stack((np.ones(len(X)), (X - mu) / sd))

    X = design(raw)
    people = sorted({r["reader"] for r in observations})
    idx = {r: i for i, r in enumerate(people)}
    risk = np.zeros((len(people), len(X)))
    event = np.zeros_like(risk)
    for r in observations:
        i = idx[r["reader"]]
        o, length = two[r["id"]]
        at = trial_observation(r) - 2
        if len(r["keys"]) - 1 != length:
            raise ValueError("Source risk positions differ from original article")
        risk[i, o : o + min(at + 1, length)] += 1
        if at < length:
            event[i, o + at] += 1
    risk = csr_matrix(risk)
    event = csr_matrix(event)
    nodes, w = np.polynomial.hermite.hermgauss(15)
    nodes *= np.sqrt(2)
    w = w / np.sqrt(np.pi)
    if not vary:
        nodes = np.zeros(1)
        w = np.ones(1)

    def objective(v):
        b = v[: X.shape[1]]
        scale = v[-1] if vary else 0
        eta = (X @ b)[:, None] + scale * nodes
        probability = expit(eta)
        lp = event @ eta - risk @ np.logaddexp(0, eta) + np.log(w)
        den = logsumexp(lp, axis=1)
        post = np.exp(lp - den[:, None])
        residue = (risk.T @ post) * probability - event.T @ post
        gradb = X.T @ residue.sum(1)
        loss = -den.sum() + 0.5 * (b[1:] @ b[1:]) + 0.125 * (scale**2)
        gradb[1:] += b[1:]
        gradient = (
            np.r_[gradb, np.sum(residue * nodes) + 0.25 * scale] if vary else gradb
        )
        return float(loss / len(people)), gradient / len(people)

    initial = np.zeros(X.shape[1] + int(vary))
    initial[0] = -4
    if vary:
        initial[-1] = 0.65
    probe = np.random.default_rng(57291).normal(size=len(initial))
    eps = 1e-6
    check = abs(
        (objective(initial + eps * probe)[0] - objective(initial - eps * probe)[0])
        / (2 * eps)
        - objective(initial)[1] @ probe
    )
    if check > 1e-6:
        raise ArithmeticError("Reader-path gradient mismatch")
    limits = [(None, None)] * X.shape[1] + ([(0, None)] if vary else [])
    sol = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        bounds=limits,
        options={"maxiter": 900, "maxcor": 24, "ftol": 1e-15, "gtol": 3e-8},
    )
    val, grad = objective(sol.x)
    if vary and sol.x[-1] == 0:
        grad[-1] = min(grad[-1], 0)
    station = float(np.abs(grad).max())
    if station > 4e-6:
        raise ArithmeticError(f"Reader-hazard fit failed convergence: {station}")
    sd_reader = float(sol.x[-1]) if vary else 0.0

    def predict(raw):
        phi = design(raw[1:])
        p = expit((phi @ sol.x[: X.shape[1]])[:, None] + sd_reader * nodes)
        stay = np.cumprod(1 - p, axis=0)
        before = np.vstack((np.ones(len(w)), stay[:-1]))
        mass = (before * p) @ w
        living = before @ w
        if abs(mass.sum() + stay[-1] @ w - 1) > 1e-9:
            raise ArithmeticError("Lost a response path")
        return np.r_[0, mass / np.maximum(living, 1e-15)]

    return predict, {
        "scale": sd_reader,
        "coefficients": sol.x[: X.shape[1]].tolist(),
        "state_mean": mu.tolist(),
        "state_sd": sd.tolist(),
        "gradient_inf": station,
        "gradient_check": float(check),
        "iterations": int(sol.nit),
        "converged": bool(sol.success),
        "readers": len(people),
        "trials": len(observations),
        "source_loss": val,
    }


def train(archive):
    setup()
    OUT.mkdir(exist_ok=True)
    rows = json.loads((BASE / "memory/study-map.json").read_text())
    raw = {r["id"]: r["sentences"] for r in load_rows(BASE / "memory/stories.jsonl")}
    docs = {k: list(map(form, v)) for k, v in raw.items()}
    native = native_words(archive, raw)
    ontology = Ontology()
    bags = {k: opportunities(v, native[k], ontology) for k, v in raw.items()}
    source = [r for r in rows if r["study"] in (1, 2)]
    e2 = [r for r in source if r["study"] == 2]
    observations = reveal(2, rows)
    links = source_links(source, raw)

    @lru_cache(None)
    def router(exclusions):
        return Route(
            [r for r in source if r["story"] not in exclusions], docs, bags, links
        )

    z = {}
    history = {}
    for family in sorted({r["story"] for r in e2}):
        held = [r for r in rows if r["story"] == family and r["study"] == 3]
        if not held:
            continue
        seen = [r for r in e2 if r["story"] != family]
        input0 = {
            r["id"]: measurements(
                router(tuple(sorted((family, r["story"])))),
                docs[r["id"]],
                bags[r["id"]],
            )
            for r in seen
        }
        events = [r for r in observations if r["story"] != family]
        history[family] = {}
        xh = {
            r["id"]: measurements(router((family,)), docs[r["id"]], bags[r["id"]])
            for r in held
        }
        for mode in ("fixed_hazard", "reader_frailty"):
            model, diagnostic = fit(input0, events, mode == "reader_frailty")
            history[family][mode] = diagnostic
            for key, X in xh.items():
                z["strict." + mode + "." + key] = model(X)
    baseline = PREVIOUS / "forecast/hazards.npz"
    if (
        digest(baseline)
        != json.loads((PREVIOUS / "forecast/seal.json").read_text())["files"][
            "hazards.npz"
        ]
    ):
        raise ValueError("Original risk-set predictors changed")
    with np.load(baseline) as old:
        for k in old.files:
            if k.startswith(
                ("strict.directional.", "strict.oracle.", "strict.first_cue_moment.")
            ):
                z[k] = old[k]
    path = OUT / "hazards.npz"
    np.savez_compressed(path, **z)
    write_json(
        OUT / "fit-lock.json",
        {
            "protocol": digest(PLAN),
            "native": digest(archive / "receipt.json"),
            "baseline": digest(baseline),
            "forecast": digest(path),
            "source_folds": history,
            "third_cohort_responses_in_fit": False,
        },
    )


def grade():
    h = OUT / "hazards.npz"
    l = json.loads((OUT / "fit-lock.json").read_text())
    if l["forecast"] != digest(h) or l["protocol"] != digest(PLAN):
        raise ValueError("Unregistered first-report forecasts")
    rows = json.loads((BASE / "memory/study-map.json").read_text())
    observed = reveal(3, rows)
    models = (
        "fixed_hazard",
        "reader_frailty",
        "directional",
        "first_cue_moment",
        "oracle",
    )
    with np.load(h) as forecasts:
        scores = summarize(
            forecasts,
            observed,
            190164,
            cohorts=("strict",),
            methods=models,
            focus="reader_frailty",
            controls=("fixed_hazard", "directional", "first_cue_moment", "oracle"),
        )
    write_json(OUT / "scores.json", dict(lock=digest(OUT / "fit-lock.json"), **scores))
    print(
        {
            k: (
                scores["models"]["strict"][k]["path_nll"]["estimate"],
                scores["models"]["strict"][k]["condition_effect"]["estimate"],
            )
            for k in models
        }
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    s = sub.add_parser("fit")
    s.add_argument("native", type=Path)
    sub.add_parser("score")
    args = p.parse_args()
    if args.action == "fit":
        train(args.native)
    else:
        grade()
