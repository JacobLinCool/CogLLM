"""Paired comparisons across three unrelated encoders at identical human word observations."""
from pathlib import Path
from itertools import chain
import numpy as np
from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.multilab_score import unpack, human, summary
from research.reading_validation.physiology_portability_data import (
    OUT,
    PLAN,
    CANDIDATES,
)


def fast_ranks(values):
    _, code = np.unique(values, return_inverse=True)

    def apply(weights):
        count = np.bincount(code, weights=weights)
        return ((np.cumsum(count) - count * 0.5)[code] / weights.sum()) - 0.5

    return apply


def R(p, z, w):
    denom = np.sqrt((w @ (p * p)) * (w @ (z * z)))
    return (w @ (p * z)) / denom if denom > 1e-12 else float("nan")


def paired(predictions, y, groups, draws=1600):
    units, gi = np.unique(groups, return_inverse=True)
    gold = fast_ranks(y)
    ranks = {k: fast_ranks(p) for k, p in predictions.items()}
    rng = np.random.default_rng(97531)
    res = rng.multinomial(len(units), np.ones(len(units)) / len(units), size=draws)
    out = {k: [] for k in ranks}
    for w in chain(
        [np.ones(len(groups))], (weights[gi].astype(float) for weights in res)
    ):
        a = gold(w)
        for k, rf in ranks.items():
            out[k].append(R(rf(w), a, w))
    a = {k: np.asarray(v) for k, v in out.items()}
    pairs = [(k + "_context", k + "_surface") for k in CANDIDATES] + [
        (k + "_context", k + "_raw") for k in CANDIDATES
    ]
    archs = [
        (c + "_context", d + "_context")
        for i, c in enumerate(CANDIDATES)
        for d in CANDIDATES[i + 1 :]
    ]
    pairs += archs
    reports = {}
    between = []
    for k, j in pairs:
        delta = a[k] - a[j]
        name = k + "__over__" + j
        reports[name] = dict(
            rho_difference=float(delta[0]), interval=summary(delta[1:])
        )
        if (k, j) in archs:
            between.append((name, delta))
    mixed = np.column_stack([q for _, q in between])
    sd = mixed[1:].std(0)
    sim = np.quantile(
        np.max(abs((mixed[1:] - mixed[0]) / np.maximum(sd, 1e-12)), axis=1), 0.95
    )
    for i, (name, delta) in enumerate(between):
        reports[name]["simultaneous_architecture_ci"] = [
            float(delta[0] - sim * sd[i]),
            float(delta[0] + sim * sd[i]),
        ]
    return dict(
        words=len(y),
        independent_groups=len(units),
        group_bootstraps=draws,
        rho={k: dict(point=float(v[0]), interval=summary(v[1:])) for k, v in a.items()},
        paired=reports,
    )


def main():
    archives = {}
    lock = {}
    ref = None
    for c in CANDIDATES:
        folder = OUT / c
        f = read_json(folder / "forecast-lock.json")
        path = folder / "forecasts.npz"
        if (
            digest(path) != f["forecast"]
            or f["protocol"] != digest(PLAN)
            or any(
                digest(folder / f"fold-{i}.json") != v
                for i, v in enumerate(f["models"])
            )
        ):
            raise ValueError("Uncertified prospective predictions")
        archives[c] = np.load(path)
        lock[c] = digest(folder / "forecast-lock.json")
        if ref is not None:
            for key in ("eeg_items", "sentence", "gold", "n", "es_family", "es_y"):
                if not np.array_equal(archives[c][key], ref[key]):
                    raise ValueError("Mismatched human item support")
        ref = archives[c]
    if ref is None:
        raise ValueError("No encoders")
    head = {}
    for i, target in enumerate(["N400", "cloze_miss", "unpredictability"]):
        y = ref["gold"][:, i].copy()
        if i == 1:
            y /= ref["n"]
        preds = {}
        for c, r in archives.items():
            preds.update(
                {
                    c + "_context": r["context_P"][:, i],
                    c + "_surface": r["surface_P"][:, i],
                    c + "_raw": r["raw"],
                }
            )
        head[target] = paired(preds, y, ref["sentence"])
    spanish = paired(
        {
            c + "_" + method: r["es_" + method]
            for c, r in archives.items()
            for method in ["context", "surface", "raw"]
        },
        ref["es_y"],
        ref["es_family"],
    )
    _, _, Y, _, sl, items, coverage, hd = unpack()
    src = read_json(OUT / CANDIDATES[0] / "forecast-lock.json")["nine_lab_order"]
    from research.reading_validation.multilab_materials import ROOT as NINE
    from research.backbone_transfer.extract import stream_rows

    material = list(stream_rows(NINE / "conditions.jsonl"))
    if [r["id"] for r in material] != src:
        raise ValueError("Original nine-laboratory candidate ordering differs")
    mean = human(Y, np.ones((1, len(sl))))[0]
    human_d = mean[:, 1] - mean[:, 0]
    nl = len(items)
    groups = np.arange(nl)

    def by_item(f):
        out = np.zeros((len(items), 2, 2))
        for i, row in enumerate(material):
            out[items.index(row["item"]), int(row["condition"] == "unexpected")] = f[i]
        return out[:, 1] - out[:, 0]

    d = {}
    for c, arc in archives.items():
        for method in ["context", "surface", "raw"]:
            zz = arc["multi_" + method]
            scores = zz[:, :, 0] if zz.ndim == 3 else zz
            d[c + "_" + method] = by_item(scores)
    rng = np.random.default_rng(26941)
    w = rng.multinomial(nl, np.ones(nl) / nl, size=10000)
    obs = {}
    for k, value in d.items():
        vmean = value.mean(0)
        diff = value[:, 1] - value[:, 0]
        u = w @ diff / nl
        obs[k] = dict(
            contrast_article=float(vmean[0]),
            contrast_noun=float(vmean[1]),
            noun_minus_article=float(diff.mean()),
            difference_interval=summary(u),
            article_over_noun=float(vmean[0] / vmean[1]),
            item_effect_rank=[
                float(
                    R(
                        fast_ranks(value[:, j])(np.ones(nl)),
                        fast_ranks(human_d[:, j])(np.ones(nl)),
                        np.ones(nl),
                    )
                )
                for j in range(2)
            ],
        )
    done = dict(
        protocol=digest(PLAN),
        forecast_checkpoints=lock,
        english_held_sentences=head,
        spanish_external=spanish,
        nine_lab_external=dict(
            families=nl,
            verified_trials=coverage,
            readers=hd["subjects"],
            human_contrast=human_d.mean(0).tolist(),
            model=obs,
        ),
    )
    write_json(OUT / "scores.json", done)
    print(
        "English, Spanish EEG",
        [
            (
                c,
                head["N400"]["rho"][c + "_context"]["point"],
                spanish["rho"][c + "_context"]["point"],
            )
            for c in CANDIDATES
        ],
    )
    for r in archives.values():
        r.close()


if __name__ == "__main__":
    main()
