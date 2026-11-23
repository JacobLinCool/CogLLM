"""Grade the locked distribution at each independently observed Provo word position."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.cloze_population_data import ROOT
from research.reading_validation.cloze_population_fit import data
from research.reading_validation.cloze_population_score import opened
from research.reading_validation.word_population_dispersion import (
    OUT,
    PLAN,
    adjacent,
    components,
    measurement,
)


def grade(native):
    lock = read_json(OUT / "forecast-lock.json")
    logfile = ROOT / "dispersion-external-logp.npy"
    gatesfile = OUT / "held-gates.npz"
    if (
        lock["protocol"] != digest(PLAN)
        or lock["held_logp"] != digest(logfile)
        or lock["held_gates"] != digest(gatesfile)
        or lock["native"] != digest(native / "receipt.json")
    ):
        raise ValueError("Frozen probabilities differ from registration")
    raw, _, _, _, orig, meta = data(native)
    obs, _, bad, _ = opened(meta, orig)
    all_ext = np.flatnonzero(~meta["source"])
    ok = np.array([str(meta["ids"][i]) not in bad for i in all_ext])
    inds = all_ext[ok]
    row, col, hits, human = measurement(obs, inds)
    num = np.bincount(row, weights=hits)
    starts = np.searchsorted(row, np.arange(len(inds) + 1))
    ext_rows = np.flatnonzero(ok)
    with np.load(gatesfile) as bundle:
        if not np.array_equal(meta["ids"][all_ext], bundle["ids"]):
            raise ValueError("Word positions do not match the forecast IDs")
        gs = {k: bundle[k][ok] for k in ("choices", "joint")}
    logp = np.load(logfile, mmap_mode="r")
    d = {k: np.empty(len(inds)) for k in ("raw", "original", "choices", "joint")}
    costs = {k: np.empty(len(inds)) for k in d}
    for a in range(0, len(inds), 24):
        ix = np.arange(a, min(len(inds), a + 24))
        pp = components(logp[ext_rows[ix]])
        pred = {"original": pp[:, :, 1], "raw": np.exp(raw[inds[ix]].astype(float))}
        pred.update(
            {k: np.einsum("ijk,ik->ij", pp, gate[ix]) for k, gate in gs.items()}
        )
        for k, p in pred.items():
            if np.max(abs(p.sum(1) - 1)) > 2e-6:
                raise ValueError("Invalid probability distribution")
            d[k][ix] = 1 - np.sum(p * p, axis=1)
            for j, q in enumerate(ix):
                sl = slice(starts[q], starts[q + 1])
                costs[k][q] = -(hits[sl] @ np.log(p[j, col[sl]]))
    stories = meta["articles"][inds]
    _, grp = np.unique(stories, return_inverse=True)
    G = 1 + grp.max()
    a, b = adjacent(stories, meta["pos"][inds])
    delta = human[b] - human[a]
    cmp = np.abs(delta) > 1e-10
    ref = {
        r["id"]: r
        for r in map(
            json.loads,
            (ROOT / "external-scored-word-rows.jsonl").read_text().splitlines(),
        )
    }
    old = [
        ref[str(meta["ids"][i])]["model_category_disagreement"]["context"] for i in inds
    ]
    parity = float(np.max(np.abs(d["original"] - old)))
    if parity > np.finfo(np.float32).eps:
        raise ValueError(
            "Source observer did not reproduce published human-choice forecasts"
        )
    rng = np.random.default_rng(420191)
    B = np.vstack((np.ones(G), rng.multinomial(G, np.full(G, 1 / G), size=10000)))

    def means(values, groups, counts):
        total = np.bincount(groups, weights=values, minlength=G)
        n = np.bincount(groups, weights=counts, minlength=G)
        return (B @ total) / (B @ n)

    draws = {}
    est = {}
    for k, disagreement in d.items():
        update = disagreement[b] - disagreement[a]
        guess = np.sign(update[cmp])
        v = {
            "choice_nll": means(costs[k], grp, num),
            "update_mae": means(abs(delta - update), grp[a], np.ones(len(a))),
            "direction": means(
                (1 + guess * np.sign(delta[cmp])) / 2, grp[a][cmp], np.ones(cmp.sum())
            ),
            "level_mae": means(abs(human - disagreement), grp, np.ones(len(grp))),
        }
        draws[k] = v
        est[k] = {
            label: {
                "mean": float(u[0]), "interval": np.quantile(u[1:], [0.025, 0.975]).tolist()
            }
            for label, u in v.items()
        }
    contrasts = {
        k + "." + name: draws[k][name] - draws["joint"][name]
        for k in ("original", "choices")
        for name in ("choice_nll", "update_mae")
    }
    Z = np.stack(list(contrasts.values()))
    sd = Z[:, 1:].std(1)
    if np.min(sd) <= 0:
        raise ArithmeticError("Unidentifiable bootstrap contrast")
    critical = np.quantile(
        np.max(abs((Z[:, 1:] - Z[:, :1]) / sd[:, None]), axis=0), 0.95
    )
    C = {
        k: {
            "joint_reduction": float(v[0]),
            "pointwise": np.quantile(v[1:], [0.025, 0.975]).tolist(),
            "simultaneous": [float(v[0] - critical * s), float(v[0] + critical * s)],
        }
        for (k, v), s in zip(contrasts.items(), sd, strict=True)
    }
    file = OUT / "provo-word-forecasts.npz"
    np.savez_compressed(
        file,
        id=meta["ids"][inds],
        article=stories,
        pos=meta["pos"][inds],
        human=human,
        opportunities=num,
        **{"D_" + k: v for k, v in d.items()},
        **{"NLL_" + k: v for k, v in costs.items()},
    )
    write_json(
        OUT / "scores.json",
        {
            "lock": digest(OUT / "forecast-lock.json"),
            "forecast": digest(file),
            "paragraphs": int(G),
            "word_ends": len(inds),
            "choices": int(num.sum()),
            "adjacent_changes": len(delta),
            "nonzero": int(cmp.sum()),
            "unchanged_base_max_abs_discrepancy": parity,
            "estimates": est,
            "primary_contrasts": C,
        },
    )
    print(json.dumps({"estimates": est["joint"], "contrasts": C}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("native", type=Path)
    grade(parser.parse_args().native)
