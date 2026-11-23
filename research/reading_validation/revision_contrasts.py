"""Matched-length simplicity and same-source editorial trade-offs from fixed predictions."""

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from research.reading_validation.joint_data import digest
from research.reading_validation.revision_prepare import OUT
from research.reading_validation.revision_score import aggregate_panel, item, DRAWS, SEED


def analysis(human, pred, methods):
    n, _, readers, _ = human.shape
    left, right = np.array(list(combinations(range(6), 2))).T
    valid = np.isfinite(human)
    ys = np.nan_to_num(human)
    diff = pred[:, left] - pred[:, right]
    lens = -pred[:, :, 0, methods.index("length")]
    dlen = abs(lens[:, left] - lens[:, right])
    allowed = {
        "equal_length": dlen == 0,
        "within_10_percent_length": 2 * dlen <= 0.1 * (lens[:, left] + lens[:, right]),
    }
    simple = ("length", "readme", "edit", "aligned", "full", "adaptive", "anchored")
    both = ("length", "readme", "edit", "adaptive", "anchored", "same_simplicity_score")
    sim = np.sign(diff[:, :, 0, [methods.index(m) for m in simple]])
    i = [methods.index(m if m != "same_simplicity_score" else "adaptive") for m in both]
    ss = np.sign(diff[:, :, 0, i])
    ps = np.stack(
        [np.sign(diff[:, :, int(m != "same_simplicity_score"), j]) for m, j in zip(both, i, strict=True)],
        axis=-1,
    )

    def summarize(rw):
        nums = np.einsum("br,fvra->bfva", rw, ys, optimize=True)
        nobs = np.einsum("br,fvra->bfva", rw, valid, optimize=True)
        means = np.divide(nums, nobs, out=np.full_like(nums, np.nan), where=nobs > 0)
        delta = means[:, :, left] - means[:, :, right]
        directions = np.where(abs(delta) > 1e-10, np.sign(delta), 0)
        ds, dm = directions[:, :, :, 0], directions[:, :, :, 1]
        simple_credit = 0.5 + 0.5 * ds[..., None] * sim
        both_credit = (0.5 + 0.5 * ds[..., None] * ss) * (0.5 + 0.5 * dm[..., None] * ps)
        out = {}
        for label, geometry in allowed.items():
            mask = (ds != 0) & geometry
            cnt = mask.sum(axis=2)
            avg = np.einsum("bfp,bfpm->bfm", mask, simple_credit) / np.maximum(1, cnt[..., None])
            out[label] = avg, cnt
        for label, sign in [("aligned_preferences", 1.0), ("opposed_preferences", -1.0)]:
            mask = ds * dm == sign
            cnt = mask.sum(axis=2)
            avg = np.einsum("bfp,bfpm->bfm", mask, both_credit) / np.maximum(1, cnt[..., None])
            out[label] = avg, cnt
        return out

    original = summarize(np.ones((1, readers)))
    readers_rng, src_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(2)]
    result = {}
    b = {k: [] for k in original}
    f = {k: [] for k in original}

    def macro(aw, state):
        val, cnt = state
        w = aw * (cnt > 0)
        total = w.sum(axis=1)
        return np.divide(
            np.einsum("bf,bfm->bm", w, val),
            total[:, None],
            out=np.full((len(w), val.shape[-1]), np.nan),
            where=total[:, None] > 0,
        )

    for first in range(0, DRAWS, 60):
        k = min(DRAWS - first, 60)
        rw = readers_rng.multinomial(readers, np.ones(readers) / readers, k)
        aw = src_rng.multinomial(n, np.ones(n) / n, k)
        res = summarize(rw)
        for key, (val, cnt) in original.items():
            b[key].append(macro(aw, res[key]))
            fac = aw * (cnt > 0)
            f[key].append(fac @ val[0] / np.maximum(1, fac.sum(axis=1, keepdims=True)))
    for key, (val, cnt) in original.items():
        point = macro(np.ones((1, n)), (val, cnt))[0]
        names = simple if key in allowed else both

        def estimates(boot):
            observed = np.isfinite(boot).all(axis=1)
            out = {m: item(point[j], boot[observed, j]) for j, m in enumerate(names)}
            for m in ("length", "readme", "edit", "same_simplicity_score"):
                if m in names:
                    a, c = names.index("adaptive"), names.index(m)
                    out["adaptive minus " + m] = item(
                        point[a] - point[c], boot[observed, a] - boot[observed, c]
                    )
            return out, DRAWS - int(observed.sum())

        cross, m = estimates(np.vstack(b[key]))
        fixed, _ = estimates(np.vstack(f[key]))
        result[key] = dict(
            pairs=int(cnt.sum()),
            sources=int((cnt > 0).sum()),
            crossed_resamples_empty=m,
            crossed_95=cross,
            source_95=fixed,
        )
    return result


def run():
    folder = OUT / "forecast"
    human, preds, methods = aggregate_panel(folder)
    result = analysis(human, preds, methods)
    result.update(
        protocol_sha256=digest(Path(__file__).with_name("revision-contrast-protocol.md")),
        forecast_sha256=digest(folder / "forecast-seal.json"),
        draws=DRAWS,
        seed=SEED,
    )
    (OUT / "contrasts.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        {
            s: (r["pairs"], r["crossed_95"]["adaptive"], r["crossed_95"]["edit"])
            for s, r in result.items()
            if isinstance(r, dict)
        }
    )


if __name__ == "__main__":
    run()
