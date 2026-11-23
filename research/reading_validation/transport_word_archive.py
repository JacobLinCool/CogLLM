"""Recover printed boundary states from a sealed native Qwen article archive."""

import numpy as np

from research.reading_validation.argument_data import sentences
from research.reading_validation.coherence_order_data import WORD
from research.reading_validation.coherence_order_features import (
    BASE_COLUMNS,
    PRED_COLUMNS,
    SEM_COLUMNS,
    form,
    neural,
)


def article_boundaries(z, key, doc, t, *, include_initial=False):
    atoms = sentences(t)
    raw = z[key + ".char"].astype(int)
    coords = raw[:, :2]
    matches = list(WORD.finditer(t))
    ws = [w.group().casefold() for w in matches]
    if atoms != [tuple(q) for q in doc["sentences"]] or not np.array_equal(
        coords, [q.span() for q in matches]
    ):
        raise ValueError("Native sentence or word locations changed")
    u = (
        coords,
        raw[:, 4].astype(bool),
        z[key + ".surprisal"].astype(float),
        z[key + ".words"].astype(float),
        z[key + ".rel"].astype(float),
    )
    choices = atoms if include_initial or len(atoms) == 1 else atoms[1:]
    rows = {}
    for a, b in choices:
        f = form(t, ws, coords, a, b, atoms)
        if f is None:
            continue
        v = neural(u, *f[1:], atoms, a)
        if v is None:
            continue
        row = np.r_[f[0], v]
        if row.size != len(BASE_COLUMNS) + len(PRED_COLUMNS) + len(
            SEM_COLUMNS
        ) or not np.all(np.isfinite(row)):
            raise ValueError("Unexpected native boundary features")
        rows[a, b] = row
    return rows, len(choices)
