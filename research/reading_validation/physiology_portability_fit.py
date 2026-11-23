"""Fit matched reading-response likelihoods to frozen states under each candidate."""
from argparse import ArgumentParser
from pathlib import Path
import csv
import numpy as np
from scipy.linalg import eigh
from scipy.special import expit
from research.backbone_transfer.extract import (
    digest,
    read_json,
    stream_rows,
    write_json,
)
from research.reading_validation.convergent_adaptive import (
    assemble,
    fit,
    forecast,
    labels,
)
from research.reading_validation.convergent_word_prepare import ROOT as WORDS
from research.reading_validation.meaning_score import folds
from research.reading_validation.observed_cloze import observe
from research.reading_validation.multilab_materials import ROOT as NINE
from research.reading_validation.physiology_portability_data import OUT, PLAN, ROOT
from research.reading_validation.spanish_clue_data import ROOT as SPANISH


def index_artifact(folder):
    r = read_json(folder / "receipt.json")
    candidate = r["candidate"]
    seat = read_json(OUT / "input-seal.json")
    cache = ROOT / candidate
    cache.mkdir(exist_ok=True)
    if (
        r["stimuli"] != seat["stimuli"]
        or r["texts"] != seat["complete_texts"]
        or seat["protocol"] != digest(PLAN)
    ):
        raise ValueError("Scientific input differs")
    for name, sha in r["files"].items():
        if digest(folder / name) != sha:
            raise ValueError("Forecast checkpoint failed verification")
    sentinel = cache / "geometry.json"
    if not sentinel.exists():
        mapped = np.lib.format.open_memmap(
            cache / "native.npy",
            mode="w+",
            dtype="float16",
            shape=(r["stored_words"], r["dimensions"]),
        )
        surface = np.lib.format.open_memmap(
            cache / "surface.npy",
            mode="w+",
            dtype="float32",
            shape=(r["stored_words"], 6),
        )
        offsets = {}
        start = 0
        with np.load(folder / "native-states.npz") as z, np.load(
            folder / "word-surprisal.npz"
        ) as s:
            for doc in stream_rows(ROOT / "stimuli.jsonl"):
                key = doc["id"]
                a = z[key]
                b = s[key]
                coords = doc["native"][candidate]
                if len(a) != len(doc["words"]) or len(a) != len(b):
                    raise ValueError("Mismatched original word count")
                stop = start + len(a)
                mapped[start:stop] = a
                offsets[key] = [start, stop]
                for i, ((l, h), (u, v)) in enumerate(zip(doc["words"], coords)):
                    w = doc["text"][l:h]
                    surface[start + i] = [
                        np.log1p(len(w)),
                        np.log1p(v - u + 1),
                        np.log1p(i),
                        w[0].isupper(),
                        w[-1] in ",.;:!?",
                        b[i],
                    ]
                start = stop
        if start != r["stored_words"]:
            raise ValueError("Incomplete native archive")
        mapped.flush()
        surface.flush()
        del mapped, surface
        write_json(
            sentinel,
            dict(
                receipt=digest(folder / "receipt.json"),
                native=digest(cache / "native.npy"),
                surface=digest(cache / "surface.npy"),
                offsets=offsets,
            ),
        )
    geo = read_json(sentinel)
    if geo["receipt"] != digest(folder / "receipt.json"):
        raise ValueError("Conflicting scientific archive")
    return (
        np.load(cache / "native.npy", mmap_mode="r"),
        np.load(cache / "surface.npy", mmap_mode="r"),
        geo["offsets"],
        candidate,
    )


def project(native, surface, train):
    train = np.asarray(train)
    a = native[train].astype(np.float32)
    mu = a.mean(0, dtype=np.float64).astype("float32")
    s = np.maximum(a.std(0, dtype=np.float64), 0.025).astype("float32")
    a -= mu
    a /= s
    n = len(a)
    # The training word Gram matrix is much smaller than native width.
    gram = np.asarray(a @ a.T / n, dtype=np.float64)
    eig, U = eigh(gram, subset_by_index=[n - 24, n - 1])
    eig = eig[::-1]
    U = U[:, ::-1]
    if np.min(eig) <= 0:
        raise ValueError("Training data cannot identify 24 distinct directions")
    basis = ((a.T @ U) / np.sqrt(n) / eig).astype("float32")
    z = np.empty((len(native), 24), dtype="float32")
    for start in range(0, len(native), 256):
        stop = min(start + 256, len(native))
        z[start:stop] = ((native[start:stop].astype("float32") - mu) / s) @ basis
    check = float(np.max(abs(z[train].T @ z[train] / n - np.eye(24))))
    if check > 0.002:
        raise ArithmeticError(
            f"Whitened eigenvectors do not reproduce training Gram basis: {check}"
        )
    surface_mu = surface[train].mean(0, dtype=float)
    surface_sd = np.maximum(surface[train].std(0, dtype=float), 0.02)
    x = np.column_stack((np.ones(len(native)), (surface - surface_mu) / surface_sd, z))
    return (
        x,
        dict(
            mean=mu,
            scale=s,
            basis=basis,
            surface_mean=surface_mu,
            surface_sd=surface_sd,
        ),
        check,
    )


def get_rows(old, offsets):
    (_, spanish, weights, _, en, c, _, _), _ = assemble(old)
    en_idx = {}
    es_idx = []
    families = []
    sources = list(stream_rows(WORDS / "map.jsonl"))
    span = {r["id"]: r for r in stream_rows(SPANISH / "mapping.jsonl")}
    for r in sources:
        start, end = offsets[r["id"]]
        if r["language"] == "Spanish":
            js = range(start + 1, end)
            es_idx.extend(js)
            families.extend([span[r["id"]]["target_word"]] * len(js))
            continue
        for i, item in enumerate(r["items"]):
            if item is not None:
                en_idx[item] = start + i
    idx = np.array([en_idx[int(k)] for k in c[:, 1]])
    ids = c[:, 0]
    items = c[:, 1].astype(int)
    table = observe(labels())
    o = np.asarray([table[i] for i in items])
    Y = en.copy()
    Y[:, 1] = o[:, 1]
    if len(es_idx) != len(spanish) or len(set(idx)) != 1546:
        raise ValueError("Common human response universe incomplete")
    nine = list(stream_rows(NINE / "conditions.jsonl"))
    nine_idx = np.array(
        [[offsets[r["id"]][0] + r[k] for k in ("article", "noun")] for r in nine]
    )
    return (
        idx,
        ids,
        items,
        Y,
        o,
        np.array(es_idx),
        spanish[:, 0],
        np.asarray(families),
        nine,
        nine_idx,
    )


def evaluate_logits(x, m):
    z = x @ np.array(m["B"])
    z[:, 1] = expit(z[:, 1])
    return z * np.array(m["scale"]) + m["mu"]


def main(folder, old):
    native, surface, off, candidate = index_artifact(folder)
    idx, ids, items, Y, o, es_idx, esy, family, nine, nine_idx = get_rows(old, off)
    partition = folds(ids, "UCL 2026 reading sentence:")
    fold = np.array([partition[s] for s in ids])
    dest = OUT / candidate
    dest.mkdir(exist_ok=True)
    arms = ["surface", "context"]
    P = {k: np.full_like(Y, np.nan) for k in arms}
    L = {k: np.full_like(Y, np.nan) for k in arms}
    saved = []
    external = {}
    multi = {}
    for f in range(6):
        prior = dest / f"fold-{f}.json"
        tr = (fold != f) if f < 5 else np.ones(len(ids), bool)
        te = ~tr
        b = dest / f"basis-{f}.npz"
        rec = None
        if prior.exists():
            rec = read_json(prior)
            if (
                rec.get("protocol") != digest(PLAN)
                or rec.get("acquisition") != digest(folder / "receipt.json")
                or rec["basis"] != digest(b)
                or rec["held_items"] != items[te].tolist()
            ):
                raise ValueError("Incompatible source checkpoint")
            with np.load(b) as part:
                basis = {k: part[k] for k in part.files}
            Z = np.empty((len(native), 24), dtype="float32")
            for i in range(0, len(native), 256):
                Z[i : i + 256] = (
                    (native[i : i + 256].astype("float32") - basis["mean"])
                    / basis["scale"]
                ) @ basis["basis"]
            X = np.column_stack(
                (
                    np.ones(len(native)),
                    (surface - basis["surface_mean"]) / basis["surface_sd"],
                    Z,
                )
            )
            error = rec["manual_whitening_max_error"]
        else:
            X, basis, error = project(native, surface, idx[tr])
        models = {}
        for name in arms:
            x = X[:, :7] if name == "surface" else X
            m = (
                rec["models"][name]
                if rec
                else fit(x[idx[tr]], Y[tr], o[tr, 0], o[tr, 2], arm="shared")
            )
            models[name] = m
            if f < 5:
                P[name][te], L[name][te] = forecast(
                    x[idx[te]], Y[te], o[te, 0], o[te, 2], m
                )
            else:
                external[name] = evaluate_logits(x[es_idx], m)[:, 0]
                multi[name] = evaluate_logits(x[nine_idx.ravel()], m).reshape(-1, 2, 3)
        if rec is None:
            np.savez_compressed(b, **basis)
            write_json(
                prior,
                dict(
                    protocol=digest(PLAN),
                    acquisition=digest(folder / "receipt.json"),
                    train_sentences=len(set(ids[tr])),
                    held_items=items[te].tolist(),
                    basis=digest(b),
                    manual_whitening_max_error=error,
                    models=models,
                ),
            )
        saved.append(digest(prior))
        print(candidate, "converged fold", f, flush=True)
    external["raw"] = np.asarray(surface[es_idx, 5], float)
    multi["raw"] = np.asarray(surface[nine_idx, 5], float)
    compact = dest / "forecasts.npz"
    np.savez_compressed(
        compact,
        eeg_items=items,
        sentence=ids,
        gold=Y,
        n=o[:, 0],
        surface_P=P["surface"],
        context_P=P["context"],
        surface_L=L["surface"],
        context_L=L["context"],
        raw=np.asarray(surface[idx, 5], float),
        es_family=family,
        es_y=esy,
        **{"es_" + k: v for k, v in external.items()},
        **{"multi_" + k: v for k, v in multi.items()},
    )
    write_json(
        dest / "forecast-lock.json",
        dict(
            protocol=digest(PLAN),
            source_inputs=digest(OUT / "input-seal.json"),
            word_sha=digest(folder / "receipt.json"),
            forecast=digest(compact),
            models=saved,
            english_words=len(idx),
            spanish_words=len(es_idx),
            nine_lab_order=[r["id"] for r in nine],
            program=digest(Path(__file__)),
        ),
    )
    print("Locked", candidate, len(idx), len(es_idx), len(nine))


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("historical", type=Path)
    a = p.parse_args()
    main(a.folder, a.historical)
