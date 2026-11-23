"""Compare locked revision predictions to the 2022 direct-assessment panel."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from research.reading_validation.external_input import digest_string
from research.reading_validation.joint_data import digest, lines
from research.reading_validation.revision_data import ASPECTS
from research.reading_validation.revision_prepare import OUT, SOURCE, records


DRAWS = 10000
SEED = 440761


def item(point, draws):
    return dict(estimate=float(point), interval95=np.quantile(draws, [0.025, 0.975]).tolist())


def aggregate_panel(folder):
    seal = json.loads((folder / "forecast-seal.json").read_bytes())
    if any(digest(folder / k) != v for k, v in seal["files_sha256"].items()) or seal["extraction"] is None:
        raise ValueError(
            "Complete frozen revision estimates are required before opening the transfer ratings"
        )
    data = json.loads((OUT / "inputs.json").read_bytes())
    if digest(SOURCE / "simpda2022.csv") != data["material_files"]["simpda2022.csv"]:
        raise ValueError("Direct-assessment transfer file changed")
    predicted = list(lines(folder / "forecasts.jsonl"))
    estimators = list(predicted[0]["scores"])
    table = {(r["group"], r["draft_id"], r["source_id"]): r["scores"] for r in predicted}
    family = sorted({r["group"] for r in predicted})
    variants = {g: sorted({d for ff, d, s in table if ff == g}) for g in family}
    if set(map(len, variants.values())) != {6}:
        raise ValueError("Expected six revisions per source ID")
    source_rows = [r for r in records("simpda2022") if r["Input.id"] in family]
    raters = sorted({r["WorkerId"] for r in source_rows})
    human = np.full((len(family), 6, len(raters), 3), np.nan)
    preds = np.zeros((len(family), 6, 3, len(estimators)))
    count = np.zeros((len(family), 6), dtype=int)
    for r in source_rows:
        g = r["Input.id"]
        d = digest_string(r["Input.simplified"])
        s = digest_string(r["Input.original"])
        i = family.index(g)
        j = variants[g].index(d)
        k = raters.index(r["WorkerId"])
        if np.isfinite(human[i, j, k]).any():
            raise ValueError("Repeated direct-assessment vote by a worker for a variant")
        values = [float(r["Answer." + ext]) for ext in ("simplicity", "adequacy", "fluency")]
        if not np.isfinite(values).all() or not all(0 <= y <= 100 for y in values):
            raise ValueError("Unexpected direct-assessment response scale")
        human[i, j, k] = values
        for m, name in enumerate(estimators):
            val = table[g, d, s][name]
            preds[i, j, :, m] += val[:3] if isinstance(val, list) else val
        count[i, j] += 1
    if not (count == 3).all() or set(table) != {
        (r["Input.id"], digest_string(r["Input.simplified"]), digest_string(r["Input.original"]))
        for r in source_rows
    }:
        raise ValueError("Scored and presented candidate assignments differ")
    return human, preds / count[:, :, None, None], estimators


def summarize(human, pred, methods):
    n, _, nr, na = human.shape
    left, right = np.array(list(combinations(range(6), 2))).T
    model_sign = np.sign(pred[:, left] - pred[:, right])
    valid = np.isfinite(human)
    ys = np.nan_to_num(human)

    def sample(rw):
        counts = np.einsum("br,fvra->bfva", rw, valid, optimize=True)
        sums = np.einsum("br,fvra->bfva", rw, ys, optimize=True)
        means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        delta = means[:, :, left] - means[:, :, right]
        sign = np.where(np.abs(delta) > 1e-10, np.sign(delta), 0)
        ok = np.isfinite(delta) & (sign != 0)
        tot = ok.sum(axis=2)
        numer = np.einsum("bfpa,fpam->bfam", sign, model_sign)
        return 0.5 + np.divide(
            0.5 * numer, tot[..., None], out=np.zeros_like(numer), where=tot[..., None] > 0
        ), tot

    orig, counts = sample(np.ones((1, nr)))
    ok = counts[0] > 0
    point = np.sum(orig[0] * ok[:, :, None], axis=0) / ok.sum(axis=0)[:, None]
    rg, sg = (np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(2))
    boot, conditional = [], []
    missing = np.zeros(na, dtype=int)
    for start in range(0, DRAWS, 60):
        k = min(60, DRAWS - start)
        rw = rg.multinomial(nr, np.full(nr, 1 / nr), k)
        sw = sg.multinomial(n, np.full(n, 1 / n), k)
        values, obs = sample(rw)
        missing += ((obs == 0) & (sw[:, :, None] > 0)).any(axis=1).sum(axis=0)
        family_weights = sw[:, :, None] * (obs > 0)
        boot.append(
            np.einsum("bfa,bfam->bam", family_weights, values) / family_weights.sum(axis=1)[:, :, None]
        )
        fw = sw[:, :, None] * ok
        conditional.append(np.einsum("bfa,fam->bam", fw, orig[0]) / fw.sum(axis=1)[:, :, None])
    boot = np.concatenate(boot)
    conditional = np.concatenate(conditional)
    output = {}

    def report(a, vals):
        out = {m: item(point[a, j], vals[:, a, j]) for j, m in enumerate(methods)}
        for m in ("edit", "aligned", "full", "adaptive", "anchored"):
            for b in ("length", "word_coverage", "readme", "edit", "aligned", "adaptive"):
                if m == b:
                    continue
                i, j = methods.index(m), methods.index(b)
                out[m + " minus " + b] = item(point[a, i] - point[a, j], vals[:, a, i] - vals[:, a, j])
        if "independent_full" in methods:
            i, j = methods.index("full"), methods.index("independent_full")
            out["full minus independent_full"] = item(
                point[a, i] - point[a, j], vals[:, a, i] - vals[:, a, j]
            )
        return out

    human_mean = np.nanmean(human, axis=2)
    ratios = {}
    for a, aspect in enumerate(ASPECTS[:3]):
        per = {}
        for j, m in enumerate(methods):
            vals = []
            for f in range(n):
                x, y = pred[f, :, a, j], human_mean[f, :, a]
                if len(set(x)) > 1 and len(set(y)) > 1:
                    vals.append(float(spearmanr(x, y).statistic))
            full = float(spearmanr(pred[:, :, a, j].ravel(), human_mean[:, :, a].ravel()).statistic)
            per[m] = dict(
                within_source_mean=float(np.mean(vals)) if vals else None,
                eligible_sources=len(vals),
                pooled=full,
            )
        output[aspect] = dict(
            non_tied_comparisons=int(counts[:, :, a].sum()),
            eligible_sources=int(ok[:, a].sum()),
            source_interval=report(a, conditional),
            crossed_source_reader_interval=report(a, boot),
            resamples_omitting_at_least_one_selected_family=int(missing[a]),
            spearman=per,
        )
        ratios[aspect] = output[aspect]["source_interval"]["adaptive minus edit"]
    return output


def individual(human, pred, methods):
    n, _, nr, na = human.shape
    out = {}
    for a, aspect in enumerate(ASPECTS[:3]):
        by = [[] for _ in range(n)]
        h = human[:, :, :, a]
        for f in range(n):
            for i, j in combinations(range(6), 2):
                mz = np.sign(pred[f, i, a] - pred[f, j, a])
                for person in np.flatnonzero(np.isfinite(h[f, i]) & np.isfinite(h[f, j])):
                    diff = h[f, i, person] - h[f, j, person]
                    if not diff:
                        continue
                    other = np.arange(nr) != person
                    other_delta = np.nanmean(h[f, i, other]) - np.nanmean(h[f, j, other])
                    by[f].append(0.5 + 0.5 * np.r_[mz, np.sign(other_delta)] * np.sign(diff))
        names = methods + ["other_panel_members"]
        means = np.array([np.mean(rs, axis=0) for rs in by if rs])
        val = means.mean(axis=0)
        rng = np.random.default_rng(SEED)
        source = rng.multinomial(len(means), np.full(len(means), 1 / len(means)), DRAWS)
        boot = source @ means / len(means)
        out[aspect] = dict(
            paired_comparisons=sum(map(len, by)),
            sources=len(means),
            mean_per_source={m: item(val[k], boot[:, k]) for k, m in enumerate(names)},
        )
    return out


def run(folder):
    human, pred, methods = aggregate_panel(folder)
    a = summarize(human, pred, methods)
    b = individual(human, pred, methods)
    result = dict(
        families=len(human),
        variants=6 * len(human),
        readers=human.shape[2],
        votes_per_aspect=int(np.isfinite(human[:, :, :, 0]).sum()),
        forecast_seal_sha256=digest(folder / "forecast-seal.json"),
        score_sha256=digest(Path(__file__)),
        bootstrap_seed=SEED,
        draws=DRAWS,
        group_mean_preferences=a,
        individual_preferences=b,
    )
    (OUT / "scores.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for aspect in ASPECTS[:3]:
        q = a[aspect]["crossed_source_reader_interval"]
        print(
            aspect,
            {
                m: dict(q[m])
                for m in (
                    "adaptive",
                    "length",
                    "readme",
                    "edit",
                    "adaptive minus edit",
                    "adaptive minus length",
                )
            },
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast", type=Path, default=OUT / "forecast")
    run(parser.parse_args().forecast)
