"""Grade Italian coherence within source genres and bootstrap partially recovered document clusters."""

import json
from pathlib import Path

import numpy as np

from research.reading_validation.coherence_order_data import OUT, read
from research.reading_validation.coherence_order_fit import source_targets
from research.reading_validation.coherence_order_score import DRAWS, SEED, summary
from research.reading_validation.local_forecast import digest

ARMS = (
    "adaptive",
    "representation",
    "shared",
    "surprisal",
    "form",
    "constant",
    "genre",
    "genre_plus_form",
)


def ranks(values, counts):
    """Return midranks as if each printed passage appeared exactly its sampled count."""
    levels, inv = np.unique(values, return_inverse=True)
    order = np.argsort(inv)
    first = np.r_[0, 1 + np.flatnonzero(np.diff(inv[order]))]
    mass = np.add.reduceat(counts[:, order], first, axis=1)
    if len(levels) != mass.shape[1]:
        raise ValueError("Ranked group count differs")
    return (np.cumsum(mass, axis=1) - 0.5 * mass + 0.5)[:, inv]


def weighted_correlation(x, y, w):
    ax = x - (w * x).sum(1, keepdims=True)
    ay = y - (w * y).sum(1, keepdims=True)
    denom = np.sqrt((w * ax**2).sum(1) * (w * ay**2).sum(1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 1e-12, (w * ax * ay).sum(1) / denom, np.nan)


def metrics(y, p, counts):
    arrays = []
    for off in range(0, len(counts), 160):
        batch = counts[off : off + 160]
        w = batch / batch.sum(1, keepdims=True)
        arrays.append(
            np.stack(
                [
                    w @ (y - p) ** 2,
                    w @ abs(y - p),
                    weighted_correlation(y, p, w),
                    weighted_correlation(ranks(y, batch), ranks(p, batch), w),
                ]
            )
        )
    all_values = np.concatenate(arrays, axis=1)
    return {
        key: (values[0], values[1:])
        for key, values in zip(
            ("mse", "mae", "pearson", "spearman"), all_values, strict=True
        )
    }


def group_sample(groups):
    material, inv = np.unique(groups, return_inverse=True)
    size = len(material)
    rng = np.random.default_rng(SEED)
    mult = rng.multinomial(size, np.full(size, 1 / size), size=DRAWS)
    return np.vstack([np.ones(len(groups), int), mult[:, inv]])


def panel(models, y, groups, q):
    counts = group_sample(groups[q])
    values = {k: metrics(y[q], v[q], counts) for k, v in models.items()}
    result = summary(values, [k for k in models if k != "adaptive"])
    result["passages"] = int(q.sum())
    result["groups"] = len(np.unique(groups[q]))
    result["mean_observed"] = float(y[q].mean())
    return result


def run():
    path = OUT / "forecast/predictions.npz"
    genre_path = OUT / "genre-controls.npz"
    primary = read(OUT / "scores.json")
    if read(OUT / "forecast/seal.json")["predictions"] != digest(path) or primary[
        "forecast_seal"
    ] != digest(OUT / "forecast/seal.json"):
        raise ValueError("Primary forecasts must be sealed and graded first")
    gs = read(OUT / "genre-controls-seal.json")
    protocol = Path(__file__).with_name("coherence_order_genre_protocol.md")
    if gs["forecast"] != digest(genre_path) or gs[
        "registered_genre_question"
    ] != digest(protocol):
        raise ValueError("Unknown genre estimates or protocol")
    if read(OUT / "genre-seal.json")["source"] != digest(OUT / "genre-forecasts.json"):
        raise ValueError("Genre provenance changed")
    with (
        np.load(path, allow_pickle=False) as f,
        np.load(genre_path, allow_pickle=False) as g,
    ):
        ids = f["italian.keys"]
        lookup = {k: j for j, k in enumerate(g["keys"])}
        if set(lookup) != set(ids):
            raise ValueError(
                "Comparisons must contain the same declared test population"
            )
        order = np.array([lookup[k] for k in ids])
        models = {k: g[k][order] if k in g else f["italian." + k] for k in ARMS}
        predicted = g["predicted_genre"][order]
        observed = g["observed_genre"][order]
    provenance = read(OUT / "genre-forecasts.json")
    sources = np.array([provenance[k]["matched_source"] or "unknown/" + k for k in ids])
    reference = source_targets(held=True)["italian"]
    y = np.array([reference[k] for k in ids])
    all_rows = np.ones(len(y), bool)
    output = {
        "source_uncertainty": "A recovered document is sampled as a whole. Give an unknown source ID its own sampling unit. This underestimates dependence if currently unmatched passages share a source. Point estimates weight printed passages equally.",
        "known_source_documents": len(
            set(sources) - {s for s in sources if s.startswith("unknown/")}
        ),
        "unidentified_sources": int(sum(s.startswith("unknown/") for s in sources)),
        "with_recovered_source_clusters": panel(models, y, sources, all_rows),
        "using_individual_passages": panel(models, y, ids, all_rows),
        "predicted_genre": {
            v: panel(models, y, sources, predicted == v) for v in ("ted", "wiki")
        },
        "directly_identified_genre": {
            v: panel(models, y, sources, observed == v) for v in ("ted", "wiki")
        },
        "forecast_seal": digest(OUT / "forecast/seal.json"),
        "genre_seal": digest(OUT / "genre-controls-seal.json"),
        "draws": DRAWS,
        "seed": SEED,
    }
    for k in ("adaptive", "representation", "surprisal", "form"):
        for metric in ("mse", "spearman", "mae", "pearson"):
            a = primary["italian_all"]["scores"][k][metric]["estimate"]
            b = output["with_recovered_source_clusters"]["scores"][k][metric][
                "estimate"
            ]
            if abs(a - b) > 1e-10:
                raise ValueError(
                    "Material-weighted point estimate differs from primary estimator"
                )
    (OUT / "genre-scores.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    print(
        {
            s: {k: v["estimate"] for k, v in output[s]["scores"]["adaptive"].items()}
            for s in ("with_recovered_source_clusters", "using_individual_passages")
        }
    )


if __name__ == "__main__":
    run()
