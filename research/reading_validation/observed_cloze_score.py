"""Repeat primary likelihood comparisons after dropping verified blank completions."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.convergent_adaptive import (
    BASE,
    assemble,
    labels,
    transform,
    fit,
    forecast,
)
from research.reading_validation.convergent_sparse import fit as sparse_fit
from research.reading_validation.meaning_score import folds
from research.reading_validation.observed_cloze import observe
from research.reading_validation.privileged_word_score import compare

PLAN = Path(__file__).with_name("cloze-observation-audit.md")
RESULT = BASE / "observed"


def main(folder):
    if read_json(RESULT / "registration.json")["protocol"] != digest(PLAN):
        raise ValueError("Registered missingness audit changed")
    (_, _, _, X, H, c, _, _), _ = assemble(folder)
    items = c[:, 1].astype(int)
    ids = c[:, 0]
    observations = observe(labels())
    o = np.array([observations[j] for j in items])
    Y = H.copy()
    Y[:, 1] = o[:, 1]
    partition = folds(ids, "UCL 2026 reading sentence:")
    F = np.array([partition[k] for k in ids])
    hold = {}
    for f in range(5):
        tr = F != f
        te = ~tr
        selected = read_json(BASE / f"sparse/context-fold-{f}.json")[
            "selected_sentences"
        ]
        obs = np.isin(ids[tr], selected)
        masked = Y[tr].copy()
        masked[~obs, 0] = np.nan
        report = {}
        for name in ("surface", "context"):
            x, _ = transform(X, tr, name == "context")
            for arm in ("diagonal", "shared"):
                for budget in ("dense", "sparse"):
                    key = "_".join((name, arm, budget))
                    m = (
                        fit(x[tr], Y[tr], o[tr, 0], o[tr, 2], arm)
                        if budget == "dense"
                        else sparse_fit(x[tr], masked, o[tr, 0], o[tr, 2], obs, arm)
                    )
                    if key not in hold:
                        hold[key] = np.full_like(Y, np.nan)
                    _, hold[key][te] = forecast(x[te], Y[te], o[te, 0], o[te, 2], m)
                    report[key] = dict(coefficients=m["B"], iterations=m["iterations"])
        write_json(
            RESULT / f"fold-{f}.json",
            dict(held_items=items[te].tolist(), source_models=report),
        )
        print("Source-count verification fold", f, flush=True)
    comps = {}
    for budget in ["dense", "sparse"]:
        for baseline in ["context_diagonal", "surface_shared"]:
            comps[budget + "__over__" + baseline] = compare(
                hold["context_shared_" + budget], hold[baseline + "_" + budget], ids
            )
    write_json(
        RESULT / "scores.json",
        dict(
            registration=digest(RESULT / "registration.json"),
            trained_on_nonempty_responses=int(o[:, 0].sum()),
            word_count=len(Y),
            grouped_effects=comps,
        ),
    )
    print(
        comps["dense__over__surface_shared"]["gain"],
        comps["sparse__over__context_diagonal"]["gain"],
    )


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("folder", type=Path)
    main(p.parse_args().folder)
