"""Report study-wise held-family preference performance for real ASSET votes."""

from collections import defaultdict
import json

import numpy as np

from research.reading_validation.joint_data import digest
from research.reading_validation.revision_data import ASPECTS, build
from research.reading_validation.revision_prepare import OUT, records
from research.reading_validation.revision_score import item


def score(folder):
    seal = json.loads((folder / "forecast-seal.json").read_bytes())
    path = folder / "development.npz"
    if digest(path) != seal["files_sha256"][path.name]:
        raise ValueError("Held-family forecasts have changed")
    data = build()
    where = {t: i for i, t in enumerate(data.passages)}
    with np.load(path, allow_pickle=False) as f:
        ix = {k: i for i, k in enumerate(f["case"])}
        methods = [k for k in f if k not in ("case", "fold")]
        predictions = {k: f[k] for k in methods}
    point = defaultdict(lambda: defaultdict(list))
    for r in records("asset_pair"):
        aspect = r["aspect"]
        if r["winner"] == "similar":
            continue
        a, b = (where[r["source"], r[k]] for k in ("asset_simplification", "other_simplification"))
        observed = 1 if r["winner"] == "asset" else -1
        j = ASPECTS.index(aspect)
        signed = [np.sign(predictions[m][ix[a], j] - predictions[m][ix[b], j]) for m in methods]
        short = np.sign(len(r["other_simplification"].split()) - len(r["asset_simplification"].split()))
        scores = 0.5 + 0.5 * np.r_[signed, short] * observed
        point[aspect][data.groups[a]].append(scores)
    result = {}
    names = methods + ["length"]
    for asp, groups in point.items():
        cells = np.stack([np.mean(v, axis=0) for v in groups.values()])
        n = len(cells)
        b = np.random.default_rng(440761).multinomial(n, np.ones(n) / n, 10000) @ cells / n
        v = cells.mean(axis=0)
        output = {name: item(v[i], b[:, i]) for i, name in enumerate(names)}
        for m in methods:
            for baseline in ("edit", "length"):
                if m == baseline:
                    continue
                i, j = names.index(m), names.index(baseline)
                output[m + " minus " + baseline] = item(v[i] - v[j], b[:, i] - b[:, j])
        result[asp] = dict(
            independent_source_groups=n, directional_votes=sum(map(len, groups.values())), estimates=output
        )
    (OUT / "development-scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: {m: round(v["estimates"][m]["estimate"], 3) for m in names} for k, v in result.items()})


if __name__ == "__main__":
    score(OUT / "forecast")
