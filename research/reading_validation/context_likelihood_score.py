"""Score jointly held same-word comparisons and frozen cross-language clue effects."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from scipy.stats import nchypergeom_fisher, spearmanr
from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.context_likelihood import (
    conditional_counts,
    count_slope,
)
from research.reading_validation.context_likelihood_train import (
    OUT,
    native,
    ARMS,
    SPANISH,
)
from research.reading_validation.spanish_eeg_score import buffer, rho


def verify():
    n1 = np.array([4, 8, 15, 9])
    n2 = np.array([7, 4, 8, 8])
    k1 = np.array([2, 3, 10, 1])
    k2 = np.array([3, 1, 3, 0])
    b = np.array([-1.3, 0.5, 2.1, -0.6])
    u, z = conditional_counts(n1, n2, k1, k2)
    lp = np.logaddexp.reduce(z + b[:, None] * u, axis=1)
    reference = -nchypergeom_fisher.logpmf(k1, n1 + n2, n1, k1 + k2, np.exp(b))
    err = float(np.max(abs(lp - reference)))
    rng = np.random.default_rng(586)
    X = rng.normal(size=(450, 1))
    nk = np.full(450, 60)
    total = rng.integers(1, 105, len(X))
    y = nchypergeom_fisher.rvs(120, 60, total, np.exp(0.78 * X[:, 0]), random_state=rng)
    a, fit = count_slope(X, np.arange(len(X)), nk, nk, y, total - y)
    a2, _ = count_slope(-X, np.arange(len(X)), nk, nk, total - y, y)
    mir = float(abs(a2[0] - a[0]))
    if err > 1e-11 or abs(a[0] - 0.78) > 0.12 or mir > 1e-8:
        raise ArithmeticError(
            "Conditional likelihood or orientation failed independent reference"
        )
    return dict(
        reference_distribution_max_error=err,
        generated_slope=0.78,
        estimated_slope=float(a[0]),
        swapped_context_deviation=mir,
        stationarity=fit["gradient"],
    )


def quantile(a):
    return np.quantile(a, [0.025, 0.975], axis=0).T.tolist()


def english(z, Y, n, sents, words, position):
    y = Y.copy()
    y[:, 1] /= n
    i = z["i"]
    j = z["j"]
    un, t = np.unique(words, return_inverse=True)
    sn, s = np.unique(sents, return_inverse=True)
    rng = np.random.default_rng(19219)
    sw = rng.exponential(size=(2000, len(sn)))
    tw = rng.exponential(size=(2000, len(un)))
    out = {}
    comps = [(2, 3), (2, 0), (2, 1), (3, 1), (0, 1)]
    names = ("N400_negativity", "cloze_miss", "rated_unpredictability")
    for variant, choose in [
        ("all", np.ones(len(i), bool)),
        ("same_index", position[i] == position[j]),
    ]:
        group = {}
        across = []
        cov = []
        for k, name in enumerate(names):
            sign = np.sign(y[i, k] - y[j, k])
            obs = choose & (sign != 0)
            a, b = i[obs], j[obs]
            typ = t[a]
            selected = np.unique(typ)
            w1 = s[a]
            w2 = s[b]
            V = np.column_stack(
                [0.5 * (1 + sign[obs] * np.sign(z[arm][obs, k])) for arm in ARMS]
            )

            def macro(ws, wt):
                pair = ws[w1] * ws[w2]
                den = np.bincount(typ, weights=pair, minlength=len(un))[typ]
                return ((wt[typ] / sum(wt[selected])) * (pair / den)) @ V

            point = macro(np.ones(len(sn)), np.ones(len(un)))
            repl = np.array([macro(w, t0) for w, t0 in zip(sw, tw)])
            d = np.stack([repl[:, u] - repl[:, v] for u, v in comps], axis=1)
            ds = np.array([point[u] - point[v] for u, v in comps])
            across.append(d - ds)
            cov.append(ds)
            group[name] = {
                "words": len(np.union1d(a, b)),
                "types": len(selected),
                "pairs": len(a),
                "sentences": len(np.unique(np.r_[w1, w2])),
                "tied_human_pairs_excluded": int((choose & (sign == 0)).sum()),
                "arms": {
                    key: {
                        "concordance": float(point[t0]),
                        "crossed_interval": quantile(repl[:, t0]),
                    }
                    for t0, key in enumerate(ARMS)
                },
                "comparisons": {
                    ARMS[u]
                    + "_over_"
                    + ARMS[v]: {
                        "gain": float(ds[t0]),
                        "crossed_interval": quantile(d[:, t0]),
                    }
                    for t0, (u, v) in enumerate(comps)
                },
            }
        radii = np.quantile(
            np.max(abs(np.stack(across, axis=-1)), axis=-1), 0.95, axis=0
        )
        for k, name in enumerate(names):
            for j0, (u, v) in enumerate(comps):
                group[name]["comparisons"][ARMS[u] + "_over_" + ARMS[v]][
                    "simultaneous_endpoint_interval"
                ] = [float(cov[k][j0] - radii[j0]), float(cov[k][j0] + radii[j0])]
        out[variant] = group
    return out


def transfer(z):
    f = read_json(SPANISH / "cue-scores.json")["families"]
    Y = buffer(f, "valid")[:, :, :, [6, 17], 0]
    gold = np.nanmean(Y, axis=0)
    female = gold[:, 1] - gold[:, 0]
    ok = np.isfinite(female[:, 1])
    rng = np.random.default_rng(826)
    fam = rng.integers(sum(ok), size=(2000, sum(ok)))
    ppl = rng.multinomial(len(Y), np.full(len(Y), 1 / len(Y)), size=2000)
    original = Y[:, :, :, 1]
    sums = ppl @ np.nan_to_num(original).reshape(len(Y), -1)
    n = ppl @ np.isfinite(original).reshape(len(Y), -1)
    avg = np.divide(sums, n, out=np.full_like(sums, np.nan), where=n > 0).reshape(
        len(ppl), len(f), 2
    )
    dr = avg[:, ok, 1] - avg[:, ok, 0]
    one = female[ok, 1]
    result = {}
    sr = {}
    for method in (*ARMS, "surprisal"):
        x = z["spanish_" + method]
        x = x[:, :, 0] if x.ndim == 3 else x
        x = x[ok]
        if np.max(abs(x[:, 0])) > 0.001:
            raise ArithmeticError(
                "An unchanged earlier Spanish word acquired a future-context effect"
            )
        target = x[:, 1]
        point = float(rho(target, one))
        rank = np.array([rho(target[j], one[j]) for j in fam])
        rp = np.array([rho(target[v], d[v]) for d in dr for v in [np.isfinite(d)]])
        sr[method] = (point, rank, rp)
        result[method] = {
            "frozen_effect": target.mean(),
            "family_interval": quantile(target[fam].mean(1)),
            "families_positive": int((target > 0).sum()),
            "same_early_word_max_difference": float(np.max(abs(x[:, 0]))),
            "N400_item_rho": point,
            "rho_family_interval": quantile(rank),
            "rho_reader_interval": quantile(rp),
        }
    comp = {}
    for m in (
        "surface_conditional",
        "context_marginal",
        "surface_marginal",
        "surprisal",
    ):
        c = sr["context_conditional"]
        h = sr[m]
        comp[m] = {
            "rho_gain": float(c[0] - h[0]),
            "family_interval": quantile(c[1] - h[1]),
            "reader_interval": quantile(c[2] - h[2]),
        }
    return dict(
        families=int(ok.sum()),
        readers=len(Y),
        item_signed_EEG_microvolts=np.mean(one),
        arms=result,
        conditional_comparisons=comp,
    )


def main(path):
    test = verify()
    seal = read_json(OUT / "forecast-lock.json")
    p = OUT / "predictions.npz"
    if seal["forecast"] != digest(p):
        raise ValueError("Unsealed cross-study predictions")
    items, X, Y, o, s, words, position, fold = native(path)
    with np.load(p) as z:
        if not np.array_equal(items, z["items"]):
            raise ValueError("Different held words or ordering")
        e = english(z, Y, o[:, 0], s, words, position)
        es = transfer(z)
    write_json(
        OUT / "scores.json",
        dict(
            provenance=digest(OUT / "forecast-lock.json"),
            verification=test,
            english=e,
            spanish=es,
        ),
    )
    print(
        "Completed same-word English/Spanish conditional study",
        {
            k: v["arms"]["context_conditional"]["concordance"]
            for k, v in e["all"].items()
        },
    )


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("native", type=Path)
    main(p.parse_args().native)
