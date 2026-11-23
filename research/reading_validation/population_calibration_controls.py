"""Grade a target median and equally calibrated native-Qwen control on the locked people and stems."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.cloze_population_model import material
from research.reading_validation.dispersion_transfer import HISTORY
from research.reading_validation.population_inference import DEFAULT
from research.reading_validation.reading_population_calibration import (
    OUT,
    Panel,
    adjust,
    inputs,
)


def median_each(y, w):
    sorted_indices = np.argsort(y, axis=1)
    v = np.take_along_axis(y, sorted_indices, axis=1)
    k = np.take_along_axis(w * np.isfinite(y), sorted_indices, axis=1)
    total = k.sum(1)
    ranks = k.cumsum(1)
    a = np.argmax(ranks > ((total - 1) // 2)[:, None], axis=1)
    b = np.argmax(ranks > (total // 2)[:, None], axis=1)
    return (v[np.arange(len(v)), a] + v[np.arange(len(v)), b]) / 2


def native_disagreement(path):
    receipt = read_json(path / "receipt.json")
    if (
        digest(path / "receipt.json")
        != read_json(HISTORY / "external-lock.json")["native"]
        or digest(path / "logs.npz") != receipt["output"]
    ):
        raise ValueError("Ungraded native stem capture differs from the source archive")
    with np.load(path / "logs.npz") as f, np.load(DEFAULT / "weights.npz") as weights:
        lp = f["native"]
        blocks = []
        for i in range(0, len(lp), 32):
            p = np.exp(
                material(
                    np.arange(i, min(i + 32, len(lp))),
                    lp,
                    weights["route"],
                    weights["share"],
                ).astype(float)
            )
            p /= p.sum(1)[:, None]
            blocks.append(1 - (p * p).sum(1))
        return np.concatenate(blocks)


def evaluate(path):
    p, events, roles = inputs()
    seal = read_json(OUT / "forecast-lock.json")
    if seal["roles"] != list(roles) or digest(OUT / "forecast.npz") != seal["forecast"]:
        raise ValueError(
            "Previously graded held readers, stems or comparison model changed"
        )
    source, target = Panel(events, roles[0], 5), Panel(events, roles[1], 2)
    raw = native_disagreement(path)
    rng = np.random.default_rng(83032)

    def get(n, b):
        return (
            np.ones((1, n), dtype=int)
            if b == 1
            else rng.multinomial(n, np.full(n, 1 / n), b)
        )

    measures = []
    medians = []
    for i in range(81):
        b = 1 if i == 0 else 10
        ys = source.disagreement(get(source.people, b))
        yt = target.disagreement(get(target.people, b))
        ws = get(source.items, b)
        wt = get(target.items, b) * np.isfinite(yt)
        m = median_each(ys, ws)
        models = []
        for base in (p, raw):
            coefficient, mean = adjust(base[source.stems], ys, ws)
            models.append(expit(logit(base[target.stems]) + coefficient[:, None]))
        models += [m[:, None], mean[:, None]]
        if i == 0:
            if not np.isclose(m[0], np.nanmedian(ys[0])):
                raise ArithmeticError("Weighted and literal sample medians diverge")
            medians = [float(m[0]), float(mean[0])]
        err = np.stack([np.nan_to_num(abs(v - yt)) for v in models], axis=1)
        measures.append(
            np.stack(
                [
                    (err * wt[:, None]).sum(2) / wt.sum(1)[:, None],
                    (err**2 * wt[:, None]).sum(2) / wt.sum(1)[:, None],
                ],
                axis=2,
            )
        )
    v = np.concatenate(measures)
    names = (
        "ucl_calibrated",
        "native_calibrated",
        "reference_median",
        "reference_mean",
    )
    diff = v[:, 1:3, 0] - v[:, [0], 0]
    se = diff[1:].std(0)
    critical = np.quantile(np.max(abs((diff[1:] - diff[0]) / se), axis=1), 0.95)
    result = {
        "protocol": digest(
            Path(__file__).with_name("population-calibration-controls.md")
        ),
        "fixed_forecast": digest(OUT / "forecast-lock.json"),
        "native": digest(path / "receipt.json"),
        "reference_median": medians[0],
        "reference_mean": medians[1],
        "scores": {
            k: {"mae": float(v[0, j, 0]), "mean_squared_error": float(v[0, j, 1])}
            for j, k in enumerate(names)
        },
        "mae_reduction_over": {
            k: {
                "gain": float(diff[0, j]),
                "simultaneous": (
                    diff[0, j] + np.array([-1, 1]) * critical * se[j]
                ).tolist(),
            }
            for j, k in enumerate(names[1:3])
        },
    }
    write_json(OUT / "controls.json", result)
    print(result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("native", type=Path)
    evaluate(parser.parse_args().native)
