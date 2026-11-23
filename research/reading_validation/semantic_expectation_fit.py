"""Evaluate the incoming word against the meanings of independently modeled earlier alternatives."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import (
    digest,
    read_json,
    write_json,
    stream_rows,
)
from research.reading_validation.cloze_population_data import OUT, ROOT
from research.reading_validation.cloze_population_fit import data, assign
from research.reading_validation.cloze_population_model import (
    material,
    project,
    distribution,
)
from research.reading_validation.convergent_word_score import assemble
from research.reading_validation.multilab_materials import ROOT as LABS

RESULT = OUT / "semantics"
PLAN = Path(__file__).with_name("semantic-expectation-protocol.md")
NAMES = ("lexical_and_print", "with_native_semantics", "with_population_semantics")


def normalized(w):
    return w.strip('.,?!:;()[]"').strip().casefold().replace("’", "'")


def row_norm(z):
    z = np.asarray(z, dtype=float)
    norm = np.linalg.norm(z, axis=1)
    if np.any(norm < 1e-12):
        raise ValueError("A printed word has no identifiable semantic direction")
    return z / norm[:, None]


def features(raw, ctx, models, U, V, form, candidates, target):
    unit = row_norm(candidates)
    n = len(raw)
    M = np.zeros((n, 5))
    j = np.arange(n)
    inside = target >= 0
    keep = np.where(inside, target, V.shape[0] - 1)
    for lo in range(0, n, 64):
        sl = slice(lo, min(n, lo + 64))
        word = form[sl]
        idx = keep[sl]
        isword = inside[sl]
        a = np.arange(len(idx))
        for k, model in enumerate(["raw", "context"]):
            lp = distribution(raw[sl], ctx[sl], models[model], U, V)
            prob = np.exp(lp)
            p = prob[:, :-1]
            pv = prob[a, idx] * isword
            remaining = p.sum(1) - pv
            if (remaining <= 1e-9).any():
                raise ValueError("The score cannot identify comparison alternatives")
            centroid = (p @ unit - pv[:, None] * word) / remaining[:, None]
            M[sl, 3 + k] = 1 - np.einsum("ij,ij->i", word, centroid)
            if k:
                M[sl, 0] = 1 - (prob * prob).sum(1)
                M[sl, 1] = -lp[a, idx]
                M[sl, 2] = isword
    return M


def ridge(x, y):
    mu = x.mean(0)
    s = np.maximum(x.std(0), 0.02)
    X = (x - mu) / s
    ym = y.mean()
    A = X.T @ X + np.eye(x.shape[1]) / 0.35**2
    rhs = X.T @ (y - ym)
    b = np.linalg.solve(A, rhs)
    err = float(np.max(abs(A @ b - rhs)))
    if err > 1e-7:
        raise ValueError("Physiological conditional mean fit failed")
    return dict(
        b=(b / s).tolist(), intercept=float(ym - mu @ (b / s)), stationarity=err
    )


def prep(folder, old, nine):
    seal = read_json(OUT / "semantic-input-seal.json")
    if (
        seal["protocol"] != digest(PLAN)
        or digest(LABS / "stimuli.jsonl") != seal["supplement"]
    ):
        raise ValueError("Supplementary experimental materials changed")
    raw, X, U, V, _, meta = data(folder)
    (_, _, _, E, H, coords, _, _), _ = assemble(old)
    eegid = coords[:, 1].astype(int)
    look = {v: i for i, v in enumerate(meta["ids"]) if meta["source"][i]}
    idx = np.array([look[str(k)] for k in eegid])
    full = read_json(ROOT / "geometry.json")
    forms = {
        q["source_id"]: q["target"]
        for v in full.values()
        if v["source"] == "UCL"
        for q in v["items"]
    }
    strings = [forms[str(i)] for i in eegid]
    eeg_groups = coords[:, 0]
    masks = assign(meta["articles"][meta["source"]])
    fold = np.array([masks[k] for k in eeg_groups])
    docs = {r["id"]: r for r in stream_rows(LABS / "stimuli.jsonl")}
    rows = list(stream_rows(LABS / "conditions.jsonl"))
    S = []
    L = []
    surface = []
    words = []
    order = []
    previous = read_json(nine / "receipt.json")
    if previous["word_states"] != digest(nine / "word-states.npz"):
        raise ValueError("Strong whole-word baseline differs")
    with np.load(folder / "lexicon.npz") as lex, np.load(
        folder / "prefix-states.npz"
    ) as zs, np.load(folder / "log-probabilities.npz") as ps, np.load(
        nine / "word-states.npz"
    ) as sf:
        actual = {w: v for w, v in zip(lex["observed_forms"], lex["observed_vectors"])}
        cand = lex["coordinates"]
        route = lex["route"]
        share = lex["share"]
        for row in rows:
            key = row["id"]
            d = docs[key]
            ii = [row["article"], row["noun"]]
            ids = np.asarray(ii) - 1
            S.extend(zs[key][ids])
            L.extend(ps[key][ids])
            surface.extend(sf[key][ii, :6])
            order.append(key)
            for i in ii:
                a, b, _, _ = d["words"][i]
                words.append(normalized(d["text"][a:b]))
    words = strings + words
    target = np.array([meta["words"].get(w, -1) for w in words])
    F = np.vstack([actual[w] for w in words])
    V0 = row_norm(F)
    actual_raw = np.vstack(
        [raw[idx], material(np.arange(len(L)), np.asarray(L), route, share)]
    )
    z = np.vstack([X[idx], S])
    surf = np.vstack([E[:, :6], surface])
    if (
        len(order) != 160
        or len(eegid) != 1546
        or abs(np.linalg.norm(V0, axis=1) - 1).max() > 1e-7
    ):
        raise ValueError("Lost an original word or lexical vector")
    return (
        actual_raw,
        z,
        U,
        V,
        cand,
        V0,
        target,
        surf,
        eegid,
        H[:, 0],
        fold,
        eeg_groups,
        order,
    )


def run(folder, old, nine):
    RESULT.mkdir(exist_ok=True)
    source_lock = read_json(OUT / "forecast-lock.json")
    if any(
        digest(OUT / f"fold-{i}.json") != v
        for i, v in enumerate(source_lock["source_fits"])
    ):
        raise ValueError("Source cloze fits are not frozen")
    raw, z, U, V, cand, words, target, surf, items, y, fold, sent, order = prep(
        folder, old, nine
    )
    P = np.zeros((len(y), 3))
    test = {}
    history = []
    for k in range(6):
        l = read_json(OUT / f"fold-{k}.json")
        ctx = project(z, l["geometry"])
        M = features(raw, ctx, l["models"], U, V, words, cand, target)
        tr = np.ones(len(y), bool) if k == 5 else fold != k
        records = {}
        for j, name in enumerate(NAMES):
            design = np.column_stack((surf, words, M[:, : 3 + j]))
            f = ridge(design[: len(y)][tr], y[tr])
            pred = design @ np.array(f["b"]) + f["intercept"]
            records[name] = f
            if k == 5:
                test[name] = pred[len(y) :].reshape(-1, 2).tolist()
            else:
                P[~tr, j] = pred[: len(y)][~tr]
        f = RESULT / f"fold-{k}.json"
        write_json(
            f,
            dict(
                cloze_source=digest(OUT / f"fold-{k}.json"),
                source_items=items[tr].tolist(),
                measure="English mean N400 negativity",
                models=records,
            ),
        )
        history.append(digest(f))
    buf = RESULT / "forecasts.npz"
    np.savez_compressed(buf, english=P, items=items, human=y, sent=sent, **test)
    write_json(
        RESULT / "forecast-lock.json",
        dict(
            protocol=digest(PLAN),
            population_lock=digest(OUT / "forecast-lock.json"),
            folds=history,
            measures=NAMES,
            labs_order=order,
            files={buf.name: digest(buf)},
            new_lab_outcomes_seen=False,
            inside_response_dictionary={
                "english": int((target[: len(y)] >= 0).sum()),
                "nine_labs": int((target[len(y) :] >= 0).sum()),
            },
        ),
    )
    print(
        "Sealed every semantic error reading before independent voltage scoring",
        len(items),
        len(order) * 2,
    )


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("native", type=Path)
    p.add_argument("english_word_native", type=Path)
    p.add_argument("nine_lab_native", type=Path)
    a = p.parse_args()
    run(a.native, a.english_word_native, a.nine_lab_native)
