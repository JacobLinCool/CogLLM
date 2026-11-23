"""Freeze predecessor-sensitive continuation scores across every pair before reading observed ordering choices."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from research.reading_validation.coherence_order_data import OUT as FIT, read
from research.reading_validation.coherence_order_estimator import predict
from research.reading_validation.coherence_order_features import (
    BASE_COLUMNS,
    PRED_COLUMNS,
    SEM_COLUMNS,
)
from research.reading_validation.columbia_order_data import OUT, PLAN, WORK, units
from research.reading_validation.italian_contrast_data import OUT as AUG
from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.transport_word_archive import article_boundaries

COLUMNS = (*BASE_COLUMNS, *PRED_COLUMNS, *SEM_COLUMNS)


def predictions(x, p, aug):
    z = (x - p["english.center"]) / p["english.scale"]
    vals = {}
    for arm, size in (
        ("form", len(BASE_COLUMNS)),
        ("surprisal", len(BASE_COLUMNS) + len(PRED_COLUMNS)),
        ("independent", z.shape[1]),
    ):
        weights = p[("representation" if arm == "independent" else arm) + ".english"]
        vals[arm] = 1 - predict(weights, {"language": "english", "x": z[:, :size]})
    for arm, t in (("adaptive", p), ("augmented", aug)):
        native = 1 - predict(
            t["representation.shared"], {"language": "english", "x": z}
        )
        a = t["english.mixture"].item()
        vals[arm] = (1 - a) * vals["independent"] + a * native
    vals["mean_surprisal"] = -x[:, COLUMNS.index("word_surprisal")]
    vals["native_cosine"] = x[:, COLUMNS.index("cosine_previous")]
    return vals


def parity(p, aug):
    with (
        np.load(FIT / "native-features.npz", allow_pickle=False) as src,
        np.load(FIT / "forecast/predictions.npz", allow_pickle=False) as historical,
        np.load(AUG / "predictions.npz", allow_pickle=False) as new,
    ):
        keep = (src["ids"][:, 4] == "held") & (src["ids"][:, 3] != "italian")
        ix = src["ids"][keep, 0]
        v = predictions(src["x"][keep], p, aug)
        if not np.array_equal(ix, historical["english.keys"]) or not np.array_equal(
            ix, new["english.keys"]
        ):
            raise ValueError("Mismatched English source adapter validation")
        ref = {
            "form": historical["english.form"],
            "surprisal": historical["english.surprisal"],
            "independent": historical["english.representation"],
            "adaptive": historical["english.adaptive"],
            "augmented": new["english.augmented_adaptive"],
        }
        err = max(float(np.max(abs(v[key] - (1 - pred)))) for key, pred in ref.items())
        if err > 1e-9:
            raise ArithmeticError(
                "Model cannot reproduce historical English predictions"
            )
        return {
            "max_absolute_discrepancy": err,
            "original_boundaries": len(ix),
            "replayed_arms": len(ref),
        }


def vectorize(archive):
    d = read(archive / "receipt.json")
    index = read(OUT / "pair-geometry.json")
    source = {r["id"]: r["text"] for r in rows(WORK / "local-stimuli.jsonl")}
    stamp = read(OUT / "inputs.json")
    if (
        d["source"] != stamp["packet"]
        or stamp["packet"] != digest(WORK / "local-stimuli.jsonl")
        or d["file"] != digest(archive / "features.npz")
        or stamp["geometry"] != digest(OUT / "pair-geometry.json")
        or stamp["protocol"] != digest(PLAN)
    ):
        raise ValueError("Unknown native alternative forecasts")
    if set(source) != set(d["articles"]) | set(d["skipped"]):
        raise ValueError("Receipt does not cover the entire candidate set")
    x = []
    pairs = []
    removed = {}
    with np.load(archive / "features.npz", allow_pickle=False) as z:
        for k, r in sorted(index.items()):
            art = r["article"]
            if art in d["skipped"]:
                removed[k] = d["skipped"][art]
                continue
            observed, _ = article_boundaries(z, art, d["articles"][art], source[art])
            target = tuple(r["target"])
            if target not in observed:
                removed[k] = "no_complete_word_in_first_target_sentence"
                continue
            x.append(observed[target])
            pairs.append(k)
    return np.asarray(x), np.asarray(pairs), removed, d


def lexical(all_units):
    docs = {
        (g, k): Counter(w.casefold() for w in re.findall(r"[^\W\d_]+", t))
        for g, items in all_units.items()
        for k, t in items.items()
    }
    df = Counter(w for d in docs.values() for w in d)
    idf = {w: np.log((len(docs) + 1) / (n + 1)) + 1 for w, n in df.items()}
    features = {k: {w: n * idf[w] for w, n in doc.items()} for k, doc in docs.items()}
    mag = {k: np.sqrt(sum(q * q for q in v.values())) for k, v in features.items()}

    def score(topic, a, b):
        i, j = (topic, a), (topic, b)
        return sum(v * features[j].get(w, 0.0) for w, v in features[i].items()) / (
            mag[i] * mag[j]
        )

    return score


def center(v, keys):
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        event, _, target = k.split("/")
        groups[event, target].append(i)
    out = np.empty_like(v)
    for idx in groups.values():
        if len(idx) < 2:
            raise ValueError(
                "Target-specific contextual adjustment needs multiple possible predecessors"
            )
        out[idx] = v[idx] - (v[idx].sum() - v[idx]) / (len(idx) - 1)
    return out


def run(archive):
    ws = FIT / "forecast/weights.npz"
    wa = AUG / "weights.npz"
    if (
        digest(ws) != read(FIT / "forecast/seal.json")["parameters"]
        or digest(wa) != read(AUG / "prediction-seal.json")["parameters"]
    ):
        raise ValueError("Unregistered source model")
    x, k, removed, rec = vectorize(archive)
    with np.load(ws, allow_pickle=False) as w, np.load(wa, allow_pickle=False) as aug:
        values = predictions(x, w, aug)
        check = parity(w, aug)
    lex = lexical(units())
    values["tfidf"] = np.array([lex(*s.split("/")) for s in k])
    centr = {key + ".centered": center(v, k) for key, v in values.items()}
    if not all(np.isfinite(v).all() for v in [*values.values(), *centr.values()]):
        raise ValueError("Nonfinite choice forecasts")
    dest = OUT / "forecasts.npz"
    np.savez_compressed(dest, keys=k, **values, **centr)
    feature_path = OUT / "native-features.npz"
    np.savez_compressed(feature_path, keys=k, x=x, columns=np.array(COLUMNS))
    code = {
        name: digest(Path(__file__).with_name(name))
        for name in (
            "columbia_order_data.py",
            "columbia_order_forecast.py",
            "columbia_order_score.py",
            "transport_word_archive.py",
        )
    }
    info = {
        "forecasted": len(k),
        "omitted": removed,
        "native_file": rec["file"],
        "source_input": rec["source"],
        "encoder_batches": rec["encoder_batches"],
        "prefix_controls": rec["prefix_control_forwards"],
        "source_parity": check,
        "old_weights": digest(ws),
        "augmented_weights": digest(wa),
        "forecasts": digest(dest),
        "native_features": digest(feature_path),
        "protocol": digest(PLAN),
        "code": code,
        "individual_order_tables_opened": False,
    }
    (OUT / "forecast-seal.json").write_text(json.dumps(info, indent=2) + "\n")
    print(info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    run(parser.parse_args().archive)
