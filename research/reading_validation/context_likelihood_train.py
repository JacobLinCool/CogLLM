"""Jointly hold out both original articles when estimating a same-word context effect."""
from argparse import ArgumentParser
from itertools import combinations_with_replacement
from pathlib import Path
import csv
import numpy as np
from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.convergent_adaptive import (
    BASE,
    assemble,
    labels,
    transform,
    fit as marginal_fit,
)
from research.reading_validation.observed_cloze import observe
from research.reading_validation.context_likelihood import fit
from research.reading_validation.context_within_word import pairs
from research.reading_validation.meaning_score import folds
from research.reading_validation.convergent_word_prepare import ROOT
from research.reading_validation.spanish_clue_data import OUT as SPANISH

OUT = BASE / "conditional"
PLAN = Path(__file__).with_name("within-context-training-protocol.md")
ARMS = (
    "surface_conditional",
    "surface_marginal",
    "context_conditional",
    "context_marginal",
)


def native(path):
    (_, _, _, X, H, coords, _, _), _ = assemble(path)
    truth = observe(labels())
    items = coords[:, 1].astype(int)
    o = np.array([truth[k] for k in items])
    Y = H.copy()
    Y[:, 1] = o[:, 1]
    s = coords[:, 0]
    gold = {
        int(z["item_id"]): z for z in csv.DictReader((ROOT / "outcomes.csv").open())
    }
    words = np.array([gold[k]["word"] for k in items])
    position = np.array([int(gold[k]["context_length"]) for k in items])
    index = folds(s, "UCL 2026 reading sentence:")
    fold = np.array([index[k] for k in s])
    return items, X, Y, o, s, words, position, fold


def scores(x, head, cond):
    if cond:
        return x[:, 1:] @ np.asarray(head["slope"])
    z = x @ np.asarray(head["B"])
    return z * np.array(head["scale"]) + head["mu"]


def main(path):
    reg = read_json(OUT / "registration.json")
    src = digest(path / "receipt.json")
    if reg["protocol"] != digest(PLAN):
        raise ValueError("Changed within-word training protocol")
    items, X, Y, o, s, words, pos, fold = native(path)
    all_i, all_j = pairs(words, s, pos, fold, "all")
    pr = {arm: np.full((len(all_i), 3), np.nan) for arm in ARMS}
    hist = []
    family = read_json(SPANISH / "cue-scores.json")["families"]
    spanish = []
    with np.load(path / "word-states.npz") as z:
        for fam in family:
            spanish.append(
                np.stack([z[fam[k]][[6, 17]].astype(float) for k in ("high", "low")])
            )
    E = np.array(spanish)
    allX = np.concatenate([X, E.reshape(-1, X.shape[1])])
    extern = {}
    for f, g in [*combinations_with_replacement(range(5), 2), (-1, -1)]:
        selected = np.ones(len(fold), bool) if f < 0 else (fold != f) & (fold != g)
        tag = f"fold-{f}-{g}" if f >= 0 else "all-source"
        file = OUT / (tag + ".json")
        rec = read_json(file) if file.exists() else None
        if rec and (
            rec["source"] != src
            or rec["registration"] != digest(OUT / "registration.json")
            or rec["training_items"] != items[selected].tolist()
        ):
            raise ValueError("Incompatible conditional checkpoint")
        train = np.r_[selected, np.zeros(allX.shape[0] - len(X), bool)]
        a = all_i[selected[all_i] & selected[all_j]]
        b = all_j[selected[all_i] & selected[all_j]]
        test = (np.minimum(fold[all_i], fold[all_j]) == f) & (
            np.maximum(fold[all_i], fold[all_j]) == g
        )
        scale = np.maximum(Y[selected].std(axis=0), 0.01)
        checkpoint = {}
        for name in ("surface", "context"):
            x, projection = transform(allX, train, name == "context")
            for kind in ("marginal", "conditional"):
                k = name + "_" + kind
                if rec:
                    head = rec["fits"][k]["model"]
                elif kind == "conditional":
                    head = fit(x, Y, o[:, 0], o[:, 2], (a, b), words, scale)
                else:
                    head = marginal_fit(
                        x[: len(X)][selected],
                        Y[selected],
                        o[selected, 0],
                        o[selected, 2],
                        "shared",
                    )
                z = scores(x, head, kind == "conditional")
                checkpoint[k] = {"model": head, "projection": projection}
                if f >= 0:
                    pr[k][test] = z[all_i[test]] - z[all_j[test]]
                else:
                    native_pred = z[len(X) :].reshape(len(E), 2, 2, 3)
                    extern[k] = native_pred[:, 1] - native_pred[:, 0]
        if not rec:
            write_json(
                file,
                dict(
                    source=src,
                    registration=digest(OUT / "registration.json"),
                    training_items=items[selected].tolist(),
                    train_sentences=len(set(s[selected])),
                    usable_pairs=len(a),
                    fit_fold=[f, g],
                    held_comparison_pairs=int(test.sum()),
                    fits=checkpoint,
                ),
            )
        hist.append(digest(file))
        print(
            tag,
            "train",
            int(selected.sum()),
            "word pairs",
            len(a),
            "test",
            int(test.sum()),
            flush=True,
        )
    if any(not np.isfinite(p).all() for p in pr.values()) or len(family) != 112:
        raise ValueError("Unscored registered context pair")
    extern["surprisal"] = E[:, 1, :, 5] - E[:, 0, :, 5]
    preds = OUT / "predictions.npz"
    np.savez_compressed(
        preds,
        items=items,
        i=all_i,
        j=all_j,
        **pr,
        **{"spanish_" + k: v for k, v in extern.items()},
    )
    write_json(
        OUT / "forecast-lock.json",
        dict(
            source=src,
            protocol=digest(PLAN),
            program=digest(Path(__file__)),
            likelihood=digest(Path(__file__).with_name("context_likelihood.py")),
            folds=hist,
            forecast=digest(preds),
            english_words=len(items),
            english_comparison_pairs=len(all_i),
            spanish_families=len(family),
        ),
    )


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("native", type=Path)
    main(p.parse_args().native)
