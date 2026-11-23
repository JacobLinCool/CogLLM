"""Measure the registered effect of swap distance in physically identified held articles."""

import json
from pathlib import Path

import numpy as np

from research.reading_validation.coherence_order_data import OUT as IT, read
from research.reading_validation.coherence_order_fit import source_targets
from research.reading_validation.coherence_order_genre_score import group_sample
from research.reading_validation.coherence_order_score import point
from research.reading_validation.italian_contrast_data import OUT as AUG
from research.reading_validation.italian_order_data import OUT
from research.reading_validation.italian_order_score import average
from research.reading_validation.local_forecast import digest


def main():
    seq = {
        k: r
        for k, r in read(OUT / "provenance.json").items()
        if r["split"] == "held" and r["type"] == "transposed"
    }
    if not (IT / "scores.json").is_file() or not (AUG / "scores.json").is_file():
        raise ValueError("Grade original and frozen augmented models first")
    with (
        np.load(IT / "forecast/predictions.npz", allow_pickle=False) as o,
        np.load(AUG / "predictions.npz", allow_pickle=False) as aug,
    ):
        ids = sorted(set(o["italian.keys"]) & set(seq))
        ix = {k: i for i, k in enumerate(o["italian.keys"])}
        which = [ix[k] for k in ids]
        if not np.array_equal(o["italian.keys"], aug["italian.keys"]):
            raise ValueError("Different target conclusions or observations")
        m = {k: o["italian." + k][which] for k in ("adaptive", "surprisal", "form")}
        m["augmented"] = aug["italian.augmented_adaptive"][which]
    y = source_targets(held=True)["italian"]
    m["human"] = np.array([y[k] for k in ids])
    g = []
    groups = []
    for k in ids:
        r = seq[k]
        pos = [a[1] for a in r["coordinates"]]
        delta = [i for i, v in enumerate(pos) if i != v - min(pos)]
        if len(delta) != 2:
            raise ValueError(
                "A certified transposition must exchange exactly two positions"
            )
        g.append(delta[-1] - delta[0])
        groups.append(r["source"])
    gap = np.array(g)
    c = group_sample(groups)
    res = {}
    for key, v in m.items():
        near = gap == 1
        d = average(v[near], c[:, near]) - average(v[~near], c[:, ~near])
        res[key] = {
            "means_by_ordinal_distance": {
                str(j): float(v[gap == j].mean()) for j in (1, 2, 3)
            },
            "adjacent_minus_more_distant": point(d[0], d[1:]),
            "finite_draws": int(np.isfinite(d[1:]).sum()),
        }
    result = {
        "observed_materials": len(ids),
        "source_groups": len(set(groups)),
        "counts_by_ordinal_distance": {str(j): int(sum(gap == j)) for j in (1, 2, 3)},
        "results": res,
        "protocol": digest(Path(__file__).with_name("italian-swap-distance.md")),
    }
    (OUT / "distance-scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(res["human"], res["augmented"])


if __name__ == "__main__":
    main()
