"""Compare dimension-matched word heads with and without training-time alternative encoders."""
import numpy as np
from scipy.stats import spearmanr
from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.multilab_score import human, unpack
from research.reading_validation.privileged_word_fit import RESULT, PLAN
from research.reading_validation.physiology_portability_data import OUT
from research.reading_validation.physiology_portability_score import fast_ranks, R
from research.reading_validation.multilab_materials import ROOT as NINE
from research.backbone_transfer.extract import stream_rows


def compare(a, b, groups):
    names, g = np.unique(groups, return_inverse=True)
    cnt = np.bincount(g)
    rng = np.random.default_rng(81426)
    w = rng.multinomial(len(names), np.ones(len(names)) / len(names), size=10000)
    d = b - a
    d = np.column_stack((d, d.mean(1)))
    arr = np.vstack([d[g == i].sum(0) for i in range(len(names))])
    z = (w @ arr) / (w @ cnt)[:, None]
    m = d.mean(0)
    sd = z.std(0)
    sim = np.quantile(
        np.max(abs((z[:, :3] - m[:3]) / np.maximum(sd[:3], 1e-12)), axis=1), 0.95
    )
    return dict(
        gain=m.tolist(),
        ci=np.quantile(z, [0.025, 0.975], axis=0).T.tolist(),
        simultaneous_endpoint_ci=np.column_stack(
            (m[:3] - sim * sd[:3], m[:3] + sim * sd[:3])
        ).tolist(),
    )


def rank_pair(v, u, y, family):
    group, g = np.unique(family, return_inverse=True)
    rng = np.random.default_rng(1881)
    ranks = [fast_ranks(a) for a in (v, u, y)]
    rep = []
    for rw in rng.multinomial(len(group), np.ones(len(group)) / len(group), size=1400):
        w = rw[g].astype(float)
        a, b, c = [r(w) for r in ranks]
        rep.append([R(a, c, w), R(b, c, w)])
    r = np.array(rep)
    x = float(spearmanr(v, y).statistic)
    z = float(spearmanr(u, y).statistic)
    return dict(
        shared=x,
        qwen_pca=z,
        gain=x - z,
        ci=np.quantile(r[:, 0] - r[:, 1], [0.025, 0.975]).tolist(),
    )


def main():
    lock = read_json(RESULT / "forecast-lock.json")
    file = RESULT / "forecasts.npz"
    if digest(file) != lock["archive"] or lock["protocol"] != digest(PLAN):
        raise ValueError("Study forecasts are not sealed")
    with np.load(file) as z, np.load(OUT / "qwen/forecasts.npz") as base:
        if not np.array_equal(z["items"], base["eeg_items"]):
            raise ValueError("Original held measurements differ")
        y = z["Y"].copy()
        y[:, 1] /= z["n"]
        sent = z["sentence"]
        stats = {}
        for name in [
            "residual_pca",
            "shared_directions",
            "residual_pca_sparse",
            "shared_directions_sparse",
        ]:
            p = z["P_" + name]
            stats[name] = dict(
                correlation=[
                    float(spearmanr(p[:, j], y[:, j]).statistic) for j in range(3)
                ],
                nll=z["L_" + name].mean(0).tolist(),
            )
        cmp = {
            "primary": compare(z["L_shared_directions"], z["L_residual_pca"], sent),
            "over_surface": compare(z["L_shared_directions"], base["surface_L"], sent),
            "sparse": compare(
                z["L_shared_directions_sparse"], z["L_residual_pca_sparse"], sent
            ),
        }
        external = rank_pair(
            z["es_shared_directions"], z["es_residual_pca"], z["es_y"], z["es_family"]
        )
        _, _, YY, _, sl, items, coverage, d = unpack()
        m = human(YY, np.ones((1, len(sl))))[0]
        gd = m[:, 1] - m[:, 0]
        docs = list(stream_rows(NINE / "conditions.jsonl"))
        if [r["id"] for r in docs] != lock["labs_order"]:
            raise ValueError("Original lab presentation order changed")
        nine = {}
        for key in ["shared_directions", "residual_pca"]:
            vals = np.zeros((len(items), 2, 2))
            for row, pr in zip(docs, z["multi_" + key]):
                vals[
                    items.index(row["item"]), int(row["condition"] == "unexpected")
                ] = pr[:, 0]
            delta = vals[:, 1] - vals[:, 0]
            nine[key] = dict(
                article=float(delta[:, 0].mean()),
                noun=float(delta[:, 1].mean()),
                within_item_contrast_rank=[
                    float(spearmanr(delta[:, j], gd[:, j]).statistic) for j in range(2)
                ],
            )
    write_json(
        RESULT / "scores.json",
        dict(
            protocol=digest(PLAN),
            lock=digest(RESULT / "forecast-lock.json"),
            held_sentences=len(set(sent)),
            held_words=len(y),
            crossfit=stats,
            paired=cmp,
            independent_spanish=external,
            independent_nine_labs=nine,
        ),
    )
    print("Cross-view gain:", cmp["primary"])


if __name__ == "__main__":
    main()
