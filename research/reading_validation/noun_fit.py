"""Fit the fixed two-endpoint noun-phrase estimators with family-held-out forecasts."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.reading_validation.joint_data import surface
from research.reading_validation.joint_fit import normalized_hidden
from research.reading_validation.local_forecast import digest
from research.reading_validation.noun_data import DATA, OUT, design
from research.reading_validation.noun_material import MODEL
from research.reading_validation.revision_forecast import WORD
from research.reading_validation.rewrite_forecast import extracted, predictions

RIDGE = 0.05
ARMS = (
    "mean",
    "condition",
    "surface",
    "extended",
    "surface_condition",
    "extended_condition",
)
FIXED = ("readme", "readme_surface", "mean_complex")
ENDPOINTS = ("difficulty", "error")
ORDER = [(a, b) for a in (False, True) for b in (False, True)]


def designs(extraction, records):
    seal = json.loads((OUT / "input-seal.json").read_bytes())
    if (
        digest(DATA / "local-stimuli.jsonl") != seal["packet"]
        or digest(MODEL)
        != json.loads((MODEL.parent / "seal.json").read_bytes())["files"]["models.npz"]
    ):
        raise ValueError("Forecast material or earlier source coefficients changed")
    unique = sorted({r["id"] for r in records})
    ids, states, strings, meta = extracted(
        extraction, DATA / "local-stimuli.jsonl", unique
    )
    if (
        meta["lexical_sha256"] != seal["lexical"]
        or not meta["one_forward_for_each_passage"]
    ):
        raise ValueError(
            "Feature extraction used different lexical weights or duplicate passages"
        )
    absolute = surface_matrix = np.stack([surface(t, "en") for t in strings])
    with np.load(MODEL, allow_pickle=False) as fit:
        grade = predictions(fit, states, strings)
        grade = {
            arm: 1 + grade["adaptive." + arm].sum(axis=1) for arm in ("qwen", "surface")
        }
        basis = (normalized_hidden(states) - fit["fit.basis.hidden_mean"]) @ fit[
            "fit.basis.projection"
        ][:, -24:]
    original = json.loads((extraction / "ids.json").read_bytes())
    idx = {k: i for i, k in enumerate(original)}
    word_measures = []
    total = aligned = 0
    with np.load(extraction / "words.npz", allow_pickle=False) as a:
        offs, masks, spans, values = (
            a[c] for c in ("range_offsets", "observed", "word_spans", "lexical")
        )
        for k, t in zip(ids, strings, strict=True):
            i = idx[k]
            sel = slice(offs[i], offs[i + 1])
            if not np.array_equal(
                spans[sel], np.asarray([m.span() for m in WORD.finditer(t)])
            ):
                raise ValueError("Word precision changed in native source")
            lex = values[sel][masks[sel]]
            if not len(lex):
                raise ValueError("No observed native word end in source sentence")
            total += int(offs[i + 1] - offs[i])
            aligned += len(lex)
            word_measures.append([float(lex.mean()), float(np.quantile(lex, 0.9))])
    lex = np.asarray(word_measures)
    numeric = np.column_stack([absolute, grade["qwen"], lex, basis])
    look = {k: i for i, k in enumerate(ids)}
    choice = [look[r["id"]] for r in records]
    cell = np.asarray(
        [4 * (r["gram"] - 2) + ORDER.index((r["split"], r["pseudo"])) for r in records],
        int,
    )
    condition = np.eye(12)[cell]
    surface_matrix, extended = surface_matrix[choice], numeric[choice]
    data = {
        "mean": np.zeros((len(records), 0)),
        "condition": condition,
        "surface": surface_matrix,
        "extended": extended,
        "surface_condition": np.column_stack([surface_matrix, condition]),
        "extended_condition": np.column_stack([extended, condition]),
    }
    static = {
        k: np.asarray(v)[choice]
        for k, v in zip(
            FIXED, [grade["qwen"], grade["surface"], lex[:, 0]], strict=True
        )
    }
    return (
        data,
        static,
        {
            "observed_word_ends": aligned,
            "total_words": total,
            "remote_extraction": meta,
        },
    )


def ridge(x, y, train, n_families):
    """Mean of the paired level and within-family losses, jointly solved for both endpoints."""
    if x.shape[0] != n_families * 4 or y.shape != (len(x), 2):
        raise ValueError("Expected four variants and two responses per family")
    if not np.all(train.reshape(-1, 4) == train.reshape(-1, 4)[:, :1]):
        raise ValueError("A text variant crosses the training split")
    xx, yy = x[train], y[train]
    center, scale, base = xx.mean(axis=0), xx.std(axis=0), yy.mean(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    z = (xx - center) / scale
    zfam = (
        z.reshape((-1, 4, z.shape[1])) if z.shape[1] else np.empty((len(z) // 4, 4, 0))
    )
    yfam = yy.reshape((-1, 4, 2))
    stacked = np.vstack(
        (z, (zfam - zfam.mean(axis=1, keepdims=True)).reshape(len(z), z.shape[1]))
    )
    target = np.vstack(
        (yy - base, (yfam - yfam.mean(axis=1, keepdims=True)).reshape(len(yy), 2))
    )
    width = z.shape[1]
    gram = stacked.T @ stacked / len(target) + RIDGE * np.eye(width)
    rhs = stacked.T @ target / len(target)
    if width:
        coef = np.linalg.solve(gram, rhs)
        err = float(np.max(abs(gram @ coef - rhs)))
        if err > 1e-9:
            raise ArithmeticError("Ridge optimum failed normal-equation check")
    else:
        coef = np.zeros((0, 2))
        err = 0.0
    return (
        (x - center) / scale @ coef + base,
        {"center": center, "scale": scale, "coef": coef, "intercept": base},
        err,
    )


def fit(extraction, *, independently_varying=False):
    output = OUT / "varying-blocks" if independently_varying else OUT
    output.mkdir(exist_ok=True)
    lengths = {2, 3} if independently_varying else {2, 3, 4}
    audit = json.loads((OUT / "material-audit.json").read_bytes())
    for p, v in json.loads((OUT / "estimator-lock.json").read_bytes())["files"].items():
        if digest(Path(p)) != v:
            raise ValueError(f"Preregistered stimulus or estimator changed: {p}")
    records = sorted(
        [r for r in design() if r["family"] in audit["folds"] and r["gram"] in lengths],
        key=lambda r: (r["family"], r["split"], r["pseudo"]),
    )
    families = sorted({r["family"] for r in records})
    if any(
        [(r["split"], r["pseudo"]) for r in records[i : i + 4]] != ORDER
        for i in range(0, len(records), 4)
    ):
        raise ValueError("Within-family contrast ordering changed")
    y = np.asarray([[r[k] for k in ENDPOINTS] for r in records], float)
    if (
        not np.isfinite(y).all()
        or (y[:, 0] < 1).any()
        or (y[:, 0] > 5).any()
        or (y[:, 1] < 0).any()
        or (y[:, 1] > 1).any()
    ):
        raise ValueError("Unexpected observed outcome scale")
    f = np.asarray([audit["folds"][r["family"]] for r in records])
    matrices, fixed, extraction_receipt = designs(extraction, records)
    predictions_to_save = {k: fixed[k] for k in FIXED}
    parameters = {}
    diagnostics = {}
    for arm, x in matrices.items():
        held = np.full(y.shape, np.nan)
        errors = []
        for fold in sorted(set(f)):
            result, _, err = ridge(x, y, f != fold, len(families))
            held[f == fold] = result[f == fold]
            errors.append(err)
        if not np.isfinite(held).all():
            raise ValueError("Omitted or invalid family forecast")
        _, full, error = ridge(x, y, np.ones(len(y), bool), len(families))
        parameters.update({arm + "." + name: v for name, v in full.items()})
        predictions_to_save[arm] = held
        diagnostics[arm] = {
            "normal_equations_max_abs_residual": max(errors + [error]),
            "features": x.shape[1],
            "heldout_range": [[float(v.min()), float(v.max())] for v in held.T],
        }
    np.savez_compressed(
        output / "predictions.npz",
        id=[r["id"] for r in records],
        family=[r["family"] for r in records],
        fold=f,
        **predictions_to_save,
    )
    np.savez_compressed(output / "weights.npz", **parameters)
    lock = {
        "native": extraction_receipt,
        "arms": ARMS,
        "fixed_controls": FIXED,
        "families": len(families),
        "rows": len(records),
        "endpoint_order": ENDPOINTS,
        "fold_diagnostics": diagnostics,
        "ridge": RIDGE,
        "material_audit_sha256": digest(OUT / "material-audit.json"),
        "score_script_sha256": digest(Path(__file__).with_name("noun_score.py")),
        "fit_script_sha256": digest(Path(__file__)),
        "predictions_sha256": digest(output / "predictions.npz"),
        "weights_sha256": digest(output / "weights.npz"),
    }
    (output / "forecast-seal.json").write_text(
        json.dumps(lock, indent=2, allow_nan=False) + "\n"
    )
    print({k: lock[k] for k in ("families", "rows", "predictions_sha256")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", required=True, type=Path)
    parser.add_argument("--independently-varying", action="store_true")
    args = parser.parse_args()
    fit(args.extraction, independently_varying=args.independently_varying)
