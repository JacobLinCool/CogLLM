"""Exercise joint optimization with genuine training observations and lexical-identical source pairs."""

import json

import numpy as np

from research.reading_validation.argument_data import sentences
from research.reading_validation.coherence_order_data import (
    ITALIAN,
    WORD,
    csv_rows,
    read,
)
from research.reading_validation.coherence_order_data import OUT as COH
from research.reading_validation.coherence_order_estimator import prepare
from research.reading_validation.coherence_order_features import form
from research.reading_validation.coherence_order_fit import source_targets
from research.reading_validation.italian_contrast_data import OUT
from research.reading_validation.italian_contrast_estimator import fit, layout
from research.reading_validation.local_forecast import rows
from research.reading_validation.memory_data import h


def boundaries(t, center, sd):
    atoms = sentences(t)
    matches = list(WORD.finditer(t))
    words = [r.group().casefold() for r in matches]
    coord = np.array([r.span() for r in matches])
    by = {}
    for a, b in atoms[1:] if len(atoms) > 1 else atoms:
        r = form(t, words, coord, a, b, atoms)
        if r is not None:
            by[a, b] = (r[0] - center) / sd
    return by


def main():
    path = COH / "form-preflight"
    with np.load(path / "weights.npz", allow_pickle=False) as p:
        params = dict(p)
    with np.load(COH / "form-features.npz", allow_pickle=False) as p:
        ids = p["ids"]
        x = p["x"]
    observed = source_targets()
    blocks = []
    for lang in ("english", "italian"):
        mask = ids[:, 3] == "italian" if lang == "italian" else ids[:, 3] != "italian"
        q = mask & (ids[:, 4] == "fit")
        z = (x[q] - params[lang + ".center"]) / params[lang + ".scale"]
        blocks.append(prepare(ids[q], z, lang, observed[lang]))
    articles = {h(r["TEXT"]): r["TEXT"] for r in csv_rows(ITALIAN / "train.tsv")}
    articles.update(
        {r["id"]: r["text"] for r in rows(ITALIAN / "source-counterfactuals.jsonl")}
    )
    pairs = []
    for rec in read(OUT / "source-geometry.json").values():
        if not rec["eligible"] or rec["role"] != "fit":
            continue
        params_form = (params["italian.center"], params["italian.scale"])
        c = boundaries(articles[rec["canonical"]], *params_form)
        s = boundaries(articles[rec["permuted"]], *params_form)
        ai = tuple(rec["canonical_interval"])
        bi = tuple(rec["permuted_interval"])
        d = (
            (s[bi] - c[ai])
            if rec["canonical_has_predecessor"] and ai in c and bi in s
            else None
        )
        if c and s:
            pairs.append(
                {
                    "group": rec["source"],
                    "canonical": np.array(list(c.values())),
                    "permuted": np.array(list(s.values())),
                    "local": d,
                }
            )
    structure = layout(pairs)
    _, note = fit(
        blocks,
        structure,
        params["form.shared"],
        read(path / "seal.json")["fit"]["form.shared"],
    )
    result = {
        "training_material_only": True,
        "same_control_human_loss_verified": True,
        "source_pairs": structure["pairs"],
        "same_sentence_constraints": structure["localized"],
        "convergence": note,
    }
    (OUT / "form-training-preflight.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(result)


if __name__ == "__main__":
    main()
