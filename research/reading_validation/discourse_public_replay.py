"""Recompute independent and shared coherence losses from untouched public grades and saved forecasts.

Requires Python 3.10+ and NumPy. The optional --fetch flag downloads only the authors' three
public test files to the supplied directory. No personal access token or training cache is used.
"""

import argparse
import csv
import hashlib
import json
from io import StringIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np

FILES = {
    "held.tsv": (
        "https://raw.githubusercontent.com/davidecolla/DisCoTex/e113774f1b5f9c1eb4ea5581767429241ebb8cc9/data/test/gold/discotex_2_test_set_release.tsv",
        "88095f67200099e0a87e54ce5925413f359e5c01df3ee1102f4d73eab06f9a57",
    ),
    "TestData.json": (
        "https://raw.githubusercontent.com/AviyaMn/CoheSentia/891891af4d38518df2f305fa20a12bc9e63fd2e3/data/TestData.json",
        "691b3b519c82cbe53fd8cae85cd67736535e31309a3afa2634b44c09a73a2331",
    ),
    "test_1355_clean.csv": (
        "https://raw.githubusercontent.com/BillyZhang24kobe/writing2coherence/382b4fa086351202f5ac514fcd2b86a5479882e1/data/test/test_1355_clean.csv",
        "971c82a28d6efcdc4fd526e13290f840f95cdb32a2ac07aad6229dd7cc4ff62e",
    ),
}
SEED = 378283
N = 10000


def verified_bytes(path, name, fetch):
    url, expected = FILES[name]
    p = path / name
    if p.is_file():
        raw = p.read_bytes()
    elif fetch:
        with urlopen(url, timeout=45) as s:
            raw = s.read()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(
                "Downloaded public data differs from the authors' pinned file"
            )
        path.mkdir(exist_ok=True, parents=True)
        p.write_bytes(raw)
    else:
        raise FileNotFoundError(
            f"Missing {name}. Supply the downloaded file or choose --fetch."
        )
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Unexpected input checksum for {name}")
    return raw


def summarize(q, bootstrap):
    return {
        "estimate": float(q),
        "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def votes(stories, essays, ref):
    if ref[0] == "TestData.json":
        _, sid, index = ref
        data = stories[sid]["IncrementalData"]
        n = data["num_sentences"]
        labels = []
        for worker, marks in data["reasons"].items():
            if worker not in data["scores"]:
                raise ValueError("Missing original reader")
            if all(str(k).isdigit() and 0 <= int(k) < n for k in marks):
                labels.append(bool(marks.get(str(index), [])))
        return sum(labels), len(labels)
    if ref[0] == "test_1355_clean.csv":
        raw = essays[ref[1]]["label"]
        if raw not in ("1.0", "0.0"):
            raise ValueError("Unexpected DECOR outcome")
        return int(raw == "0.0"), 1
    raise ValueError("Unexpected public dataset")


def run(package, cache, fetch):
    pred = package / "forecast" / "predictions.npz"
    stamp = json.loads((package / "forecast" / "seal.json").read_text())
    if hashlib.sha256(pred.read_bytes()).hexdigest() != stamp["predictions"]:
        raise ValueError("Mismatched fixed forecasts")
    metadata = json.loads((package / "public-observation-index.json").read_text())
    yit = {
        r["ID"]: float(r["SCORE"])
        for r in csv.DictReader(
            StringIO(verified_bytes(cache, "held.tsv", fetch).decode("utf-8-sig")),
            delimiter="\t",
        )
    }
    stories = json.loads(verified_bytes(cache, "TestData.json", fetch))
    essays = list(
        csv.DictReader(
            StringIO(
                verified_bytes(cache, "test_1355_clean.csv", fetch).decode("utf-8-sig")
            )
        )
    )
    result = {}
    arms = ("form", "surprisal", "representation", "shared", "adaptive")
    with np.load(pred, allow_pickle=False) as bank:
        ids = bank["italian.keys"]
        y = np.array([yit[k] for k in ids])
        ix = np.random.default_rng(SEED).integers(len(y), size=(N, len(y)))
        square = {arm: (y - bank["italian." + arm]) ** 2 for arm in arms}
        result["italian_all"] = {
            k: summarize(v.mean(), v[ix].mean(1)) for k, v in square.items()
        }
        for base in ("form", "surprisal", "representation"):
            d = square[base] - square["adaptive"]
            result["italian_all"]["adaptive_over_" + base] = summarize(
                d.mean(), d[ix].mean(1)
            )
        ids = bank["english.keys"]
        yes, n = np.array([votes(stories, essays, metadata[k]) for k in ids]).T
        for study in ("cohesentia", "decor"):
            sel = (bank["english.study"] == study) & (n > 0)
            groups = bank["english.group"][sel]
            levels = np.unique(groups)
            w = np.random.default_rng(SEED).multinomial(
                len(levels), np.ones(len(levels)) / len(levels), size=N
            ) / len(levels)
            loss = {}
            for arm in arms:
                p = np.clip(bank["english." + arm][sel], 1e-12, 1 - 1e-12)
                negative = -(yes[sel] * np.log(p) + (n[sel] - yes[sel]) * np.log1p(-p))
                loss[arm] = np.array(
                    [
                        negative[groups == g].sum() / n[sel][groups == g].sum()
                        for g in levels
                    ]
                )
            result[study] = {k: summarize(v.mean(), w @ v) for k, v in loss.items()}
            for base in ("form", "surprisal", "representation"):
                d = loss[base] - loss["adaptive"]
                result[study]["adaptive_over_" + base] = summarize(d.mean(), w @ d)
    prior = json.loads((package / "scores.json").read_text())
    maximum = 0.0
    for role, panel in result.items():
        metric = "mse" if role == "italian_all" else "logloss"
        for key, v in panel.items():
            branch = "paired_gain" if key.startswith("adaptive_over_") else "scores"
            ref = prior[role][branch][key][metric]
            maximum = max(
                maximum,
                abs(v["estimate"] - ref["estimate"]),
                *abs(np.array(v["ci95"]) - ref["ci95"]),
            )
    if maximum > 1e-10:
        raise ArithmeticError(f"Independent public reconstruction differs by {maximum}")
    print(
        json.dumps(
            {
                "reproduced_estimates_and_95_intervals": 24,
                "max_absolute_discrepancy": maximum,
                "results": result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--public-data", type=Path, required=True)
    p.add_argument("--fetch", action="store_true")
    a = p.parse_args()
    run(a.package, a.public_data, a.fetch)
