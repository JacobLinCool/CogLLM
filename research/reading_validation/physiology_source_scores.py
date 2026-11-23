"""Release English source and Spanish transfer estimates while the nine-site endpoint stays sealed."""
import numpy as np
from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.physiology_portability_data import OUT, CANDIDATES
from research.reading_validation.physiology_portability_score import paired


def main():
    data = {}
    locks = {}
    for key in CANDIDATES:
        d = OUT / key
        lock = read_json(d / "forecast-lock.json")
        f = d / "forecasts.npz"
        if digest(f) != lock["forecast"]:
            raise ValueError("Corrupt source predictions")
        with np.load(f) as z:
            data[key] = {
                n: z[n]
                for n in (
                    "eeg_items",
                    "gold",
                    "n",
                    "sentence",
                    "raw",
                    "surface_P",
                    "context_P",
                    "es_y",
                    "es_family",
                    "es_surface",
                    "es_raw",
                    "es_context",
                )
            }
        locks[key] = digest(d / "forecast-lock.json")
    ref = data["qwen"]
    results = {}
    for c, d in data.items():
        if not all(
            np.array_equal(ref[k], d[k])
            for k in ("eeg_items", "gold", "n", "sentence", "es_y", "es_family")
        ):
            raise ValueError("Mismatched observations")
    for j, name in enumerate(("N400", "cloze_miss", "unpredictability")):
        y = ref["gold"][:, j].copy()
        p = {}
        if j == 1:
            y /= ref["n"]
        for c, x in data.items():
            p.update(
                {
                    c + "_context": x["context_P"][:, j],
                    c + "_surface": x["surface_P"][:, j],
                    c + "_raw": x["raw"],
                }
            )
        results[name] = paired(p, y, ref["sentence"])
    es = paired(
        {
            c + "_" + key: r["es_" + key]
            for c, r in data.items()
            for key in ("surface", "raw", "context")
        },
        ref["es_y"],
        ref["es_family"],
    )
    write_json(
        OUT / "english-spanish-scores.json",
        dict(locks=locks, source=results, spanish=es, nine_lab_outcomes_opened=False),
    )
    print("Three-backbone source and transfer comparisons sealed")


if __name__ == "__main__":
    main()
