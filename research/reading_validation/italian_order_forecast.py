"""Score physically certified sentence permutations in identical lexical material."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research.reading_validation.coherence_order_data import ITALIAN, read, texts
from research.reading_validation.coherence_order_data import OUT as FIT
from research.reading_validation.coherence_order_estimator import predict, prepare
from research.reading_validation.coherence_order_features import (
    BASE_COLUMNS,
    PRED_COLUMNS,
)
from research.reading_validation.danish_coherence_forecast import transported
from research.reading_validation.italian_order_data import OUT, PLAN
from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.memory_data import h
from research.reading_validation.transport_word_archive import article_boundaries


def read_banks(archives, requests, params):
    passage_predictions = {}
    local_predictions = {}
    absent = {}
    sources = []
    for archive, expected, original in archives:
        header = read(archive / "receipt.json")
        if header["source"] != expected or header["file"] != digest(
            archive / "features.npz"
        ):
            raise ValueError("One of the original or permutation archives changed")
        sources.append(header["file"])
        with np.load(archive / "features.npz", allow_pickle=False) as z:
            for key, source in requests.items():
                if key not in original:
                    continue
                if key not in header["articles"]:
                    absent[key] = header["skipped"][key]
                    continue
                text = original[key]
                native = header["articles"][key]
                bs, _ = article_boundaries(z, key, native, text, include_initial=True)
                atoms = [tuple(v) for v in native["sentences"]]
                use = [a for a in (atoms[1:] if len(atoms) > 1 else atoms) if a in bs]
                if not use:
                    absent[key] = "no_eligible_boundary"
                    continue
                meta = lambda a, key=key, source=source: (
                    h(f"{key}/{a[0]}/{a[1]}"),
                    key,
                    source,
                    "italian",
                    "held",
                    key,
                )
                ids = np.array([meta(a) for a in use])
                x = np.array([bs[a] for a in use])
                measures = transported(ids, x, params)
                passage_predictions[key] = {
                    k: float(v[0]) for k, v in measures.items() if k != "keys"
                }
                for q, value in bs.items():
                    record = {}
                    v = (value - params["english.center"]) / params["english.scale"]
                    for a, n in (
                        ("form", len(BASE_COLUMNS)),
                        ("surprisal", len(BASE_COLUMNS) + len(PRED_COLUMNS)),
                        ("representation", len(v)),
                        ("shared", len(v)),
                    ):
                        wt = (
                            "representation.shared" if a == "shared" else a + ".english"
                        )
                        block = prepare(np.array([meta(q)]), v[None, :n], "english")
                        record[a] = float(predict(params[wt], block)[0])
                    mix = params["english.mixture"].item()
                    record["adaptive"] = (1 - mix) * record[
                        "representation"
                    ] + mix * record["shared"]
                    local_predictions[(key, *q)] = record
    if set(requests) != (set(passage_predictions) | set(absent)):
        raise ValueError("Missing requested authentic or generated variant")
    return passage_predictions, local_predictions, absent, sources


def run(base, other):
    design = read(OUT / "order-pairs.json")
    lock = read(OUT / "inputs.json")
    params = FIT / "forecast/weights.npz"
    if (
        digest(OUT / "order-pairs.json") != lock["geometry"]
        or digest(PLAN) != lock["registered"]
        or read(FIT / "forecast/seal.json")["parameters"] != digest(params)
    ):
        raise ValueError("Sealed intervention/source provenance changed")
    gold = read(OUT / "jaccard.json")
    if digest(OUT / "jaccard.json") != read(OUT / "jaccard-seal.json")["forecasts"]:
        raise ValueError("Unknown official lexical reference")
    official = texts()
    synthetic = {r["id"]: r["text"] for r in rows(ITALIAN / "counterfactuals.jsonl")}
    specs = {
        k: p["source"] for p in design.values() for k in (p["canonical"], p["permuted"])
    }
    banks = [
        (
            base,
            read(
                Path("research/results/reading-validation-state-transport/inputs.json")
            )["source"],
            official,
        ),
        (other, lock["source"], synthetic),
    ]
    if set(specs) & set(official) & set(synthetic):
        raise ValueError("Alternative sentences occurred in the source fitting bank")
    with np.load(params, allow_pickle=False) as f:
        pp, br, absent, archive_files = read_banks(banks, specs, f)
    per_pair = []
    replayed = []
    with np.load(FIT / "forecast/predictions.npz", allow_pickle=False) as original:
        lookup = {k: i for i, k in enumerate(original["italian.keys"])}
        for key, p in design.items():
            c, s = p["canonical"], p["permuted"]
            if c not in pp or s not in pp:
                continue
            for arm, val in pp[p["observed"]].items():
                replayed.append(abs(val - original["italian." + arm][lookup[key]]))
            i, j = (
                br.get((c, *p["canonical_interval"])),
                br.get((s, *p["permuted_interval"])),
            )
            d = {k: pp[c][k] - pp[s][k] for k in pp[c]}
            if c in gold and s in gold:
                d["official_jaccard"] = gold[c] - gold[s]
            per_pair.append(
                {
                    "id": key,
                    "group": p["source"],
                    "observed_type": p["observed_type"],
                    "canonical_has_predecessor": p["canonical_has_predecessor"],
                    "first_disrupted_index": p["first_disrupted"],
                    "canonical_minus_permuted_passage": d,
                    "permuted_minus_canonical_break_probability": None
                    if i is None or j is None
                    else {k: j[k] - i[k] for k in i},
                }
            )
    error = max(replayed)
    if error > 1e-9:
        raise ValueError(
            "Observed-passage outcome-independent order diagnostic does not reproduce official Italian predictions"
        )
    f = OUT / "order-differences.json"
    f.write_text(json.dumps(per_pair, indent=2) + "\n")
    seal = {
        "estimates": digest(f),
        "pareto_registered": digest(PLAN),
        "source_readout": digest(params),
        "native_archive_files": archive_files,
        "absent": absent,
        "retained": dict(Counter(r["observed_type"] for r in per_pair)),
        "native_words_replay_max_abs": error,
    }
    (OUT / "forecast-seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    print(seal)


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("original", type=Path)
    a.add_argument("variants", type=Path)
    args = a.parse_args()
    run(args.original, args.variants)
