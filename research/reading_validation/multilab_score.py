"""Test expected/unexpected response differences at two adjacent genuine word locations."""
from collections import defaultdict, Counter
import csv
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from research.backbone_transfer.extract import (
    digest,
    read_json,
    stream_rows,
    write_json,
)
from research.reading_validation.convergent_word_score import corr
from research.reading_validation.multilab_materials import OUT, ROOT

TARGETS = ("article", "noun")
BOOT = 3000


def summary(x):
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return np.quantile(a, [0.025, 0.975]).tolist() if len(a) else [None, None]


def unpack():
    native = read_json(OUT / "forecast-lock.json")
    path = OUT / "blinded-forecasts.json"
    if native["forecasts"] != digest(path) or native["source_inputs"] != digest(
        OUT / "input-seal.json"
    ):
        raise ValueError("Independent word forecasts must be sealed first")
    reg = read_json(OUT / "input-seal.json")
    maps = list(stream_rows(ROOT / "conditions.jsonl"))
    score = read_json(path)
    index = {r["id"]: r for r in score}
    items = sorted(set(r["item"] for r in maps))
    ii = {k: j for j, k in enumerate(items)}
    ids = defaultdict(list)
    M = {}
    cloze = np.full((len(items), 2, 2), np.nan)
    for r in maps:
        it = r["item"]
        j = ii[it]
        s = int(r["condition"] == "unexpected")
        ids[it].append(r)
        for k, v in index[r["id"]]["forecasts"].items():
            if k not in M:
                M[k] = np.full_like(cloze, np.nan)
            M[k][j, s] = v
        cloze[j, s] = [r[k + "_cloze"] for k in TARGETS]
    if any(np.isnan(v).any() for v in [cloze, *M.values()]):
        raise ValueError("Missing an original printed alternative")
    records = {}
    assigned = defaultdict(set)
    duplicate = {}
    labs = set()
    for target in TARGETS:
        source = ROOT / (target + ".txt")
        if digest(source) != reg["source_n400"][target]:
            raise ValueError("Original human outcome table changed")
        with source.open() as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        records[target] = rows
        keys = []
        for r in rows:
            lab = r["lab"].lower()
            sub = (lab, r["subject"])
            it = int(r["item"])
            norm = float(r["cloze"])
            labs.add(lab)
            slots = [k for k, m in enumerate(ids[it]) if m[target + "_cloze"] == norm]
            if len(slots) == 1:
                assigned[sub].add(slots[0])
            keys.append((sub, it))
        duplicate[target] = sum(n - 1 for n in Counter(keys).values() if n > 1)
    if any(duplicate.values()) or any(len(v) != 1 for v in assigned.values()):
        raise ValueError(
            "Participant does not have unique randomized list or item trial"
        )
    subjects = sorted(assigned)
    si = {s: i for i, s in enumerate(subjects)}
    labs = sorted(labs)
    sl = np.array([labs.index(s[0]) for s in subjects])
    shape = (len(subjects), len(items), 2, 2)
    Y = np.full(shape, np.nan)
    sensitivity = Y.copy()
    bad = defaultdict(int)
    coverage = {}
    for target, rows in records.items():
        t = TARGETS.index(target)
        used = 0
        for r in rows:
            sub = (r["lab"].lower(), r["subject"])
            it = int(r["item"])
            norm = float(r["cloze"])
            slot = next(iter(assigned[sub]))
            cond = ids[it][slot]
            s = int(cond["condition"] == "unexpected")
            voltage = float(r["base100" if t == 0 else "n400"])
            if not np.isfinite(voltage):
                raise ValueError("Unmodelled released EEG missingness")
            sensitivity[si[sub], ii[it], s, t] = -voltage
            if cond[target + "_cloze"] != norm:
                bad[(target, it, sub[0])] += 1
                continue
            Y[si[sub], ii[it], s, t] = -voltage
            used += 1
        coverage[target] = dict(
            released=len(rows), verified=used, excluded=len(rows) - used
        )
    return (
        M,
        cloze,
        Y,
        sensitivity,
        sl,
        items,
        coverage,
        {
            "labs": labs,
            "mismatch_records": {str(k): v for k, v in bad.items()},
            "subjects": len(subjects),
            "count_per_lab": np.bincount(sl).tolist(),
        },
    )


def human(Y, subject_weights):
    flat = Y.reshape(len(Y), -1)
    mask = np.isfinite(flat)
    counts = subject_weights @ mask.astype(float)
    sums = subject_weights @ np.where(mask, flat, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.divide(sums, counts)
    return a.reshape((-1, *Y.shape[1:]))


def ratio(arr):
    numerator = arr[..., 0]
    denom = arr[..., 1]
    return np.divide(
        numerator, denom, out=np.full_like(numerator, np.nan), where=abs(denom) > 1e-12
    )


def partial(x, y, c):
    z = np.column_stack([np.ones(len(c)), rankdata(c)])
    r = rankdata(x)
    u = rankdata(y)
    r -= z @ np.linalg.lstsq(z, r, rcond=None)[0]
    u -= z @ np.linalg.lstsq(z, u, rcond=None)[0]
    return float(np.corrcoef(r, u)[0, 1])


def score():
    models, cloze, Y, usable, sl, items, coverage, details = unpack()
    ni = len(items)
    ns = len(sl)
    rng = np.random.default_rng(91402)
    labs = np.unique(sl)
    W = np.zeros((BOOT, ns))
    I = rng.multinomial(ni, np.full(ni, 1 / ni), size=BOOT)
    site = rng.multinomial(len(labs), np.full(len(labs), 1 / len(labs)), size=BOOT)
    for level in labs:
        sub = np.where(sl == level)[0]
        W[:, sub] = (
            rng.multinomial(len(sub), np.full(len(sub), 1 / len(sub)), size=BOOT)
            * site[:, level, None]
        )
    mean = human(Y, np.ones((1, ns)))[0]
    cells = human(Y, W)
    d = mean[:, 1] - mean[:, 0]
    est = d.mean(0)
    D = cells[:, :, 1] - cells[:, :, 0]
    samples = (D * I[:, :, None]).sum(1) / ni
    # Resampling only laboratories excluded for a particular material can remove a measurement cell.
    # Eligibility uses the original observation mask alone, without conditioning on an EEG value.
    cell_complete=np.isfinite(D).all(axis=(1,2));removed=int((~cell_complete).sum())
    if removed>len(cell_complete)*.02:
        raise ValueError("Too many resampled designs have no observed participant for a verified stimulus")
    I=I[cell_complete];D=D[cell_complete];samples=samples[cell_complete]
    num = est[1] - est[0]
    r = ratio(est)
    lead = {
        "article": float(est[0]),
        "noun": float(est[1]),
        "ci": [summary(x) for x in samples.T],
        "noun_minus_article": float(num),
        "difference_ci": summary(samples[:, 1] - samples[:, 0]),
        "article_noun_ratio": float(r),
        "ratio_ci": summary(ratio(samples)),
    }
    leave = {}
    for lab in labs:
        s = human(Y, (sl != lab)[None, :].astype(float))[0]
        de = (s[:, 1] - s[:, 0]).mean(0)
        leave[details["labs"][lab]] = {
            "article": float(de[0]),
            "noun": float(de[1]),
            "difference": float(de[1] - de[0]),
        }
    alt = human(usable, np.ones((1, ns)))[0]
    without = np.array([i not in (177, 178) for i in items])
    sensitivity = {
        "all_model_list_reconstructions": (alt[:, 1] - alt[:, 0]).mean(0).tolist(),
        "exclude_both_source_discrepant_items": d[without].mean(0).tolist(),
        "both_discrepant_items": [177, 178],
    }
    goldgap = cloze[:, 0] - cloze[:, 1]
    model_results = {}
    predict = {k: v[:, 1] - v[:, 0] for k, v in models.items()}
    for name, contrasts in predict.items():
        avg = contrasts.mean(0)
        human_r = []
        paired = []
        partials = []
        continuous = []
        for j in range(2):
            human_r.append(float(spearmanr(contrasts[:, j], d[:, j]).statistic))
            # Model rank associations reflect uncertainty in both the actual human contrast and the choice of texts.
            paired.append(
                summary(
                    [
                        corr(contrasts[:, j], row[:, j], iw)
                        for row, iw in zip(D[::3], I[::3])
                    ]
                )
            )
            partials.append(partial(contrasts[:, j], d[:, j], goldgap[:, j]))
            continuous.append(
                float(
                    spearmanr(
                        models[name][:, :, j].ravel(), mean[:, :, j].ravel()
                    ).statistic
                )
            )
        pred = I @ contrasts / ni
        model_results[name] = dict(
            mean=avg.tolist(),
            article_noun_ratio=float(ratio(avg)),
            ratio_ci=summary(ratio(pred)),
            noun_minus_article=float(avg[1] - avg[0]),
            signed_direction_ci=summary(pred[:, 1] - pred[:, 0]),
            held_item_contrast_rho=human_r,
            contrast_rho_ci=paired,
            aggregate_trial_cell_rho=continuous,
            partial_cloze_gap_contrast_rho=partials,
        )
    half = []
    split_rng = np.random.default_rng(937)
    for step in range(200):
        chosen = np.zeros(ns)
        for lab in labs:
            group = np.flatnonzero(sl == lab)
            chosen[split_rng.permutation(group)[: len(group) // 2]] = 1
        a, b = human(Y, np.stack((chosen, 1 - chosen)))
        a = a[:, 1] - a[:, 0]
        b = b[:, 1] - b[:, 0]
        half.append([spearmanr(a[:, j], b[:, j]).statistic for j in range(2)])
    cloze_stats = {}
    for j, k in enumerate(TARGETS):
        g = goldgap[:, j]
        cloze_stats[k] = dict(
            equal_norm_items=int((g == 0).sum()),
            reverse_to_design_items=int((g < 0).sum()),
            measured_contrast_vs_cloze_rho=float(spearmanr(g, d[:, j]).statistic),
            all_cells_normalized_cloze_vs_voltage_rho=float(
                spearmanr(-cloze[:, :, j].ravel(), mean[:, :, j].ravel()).statistic
            ),
        )
    out = dict(
        forecast_lock=digest(OUT / "forecast-lock.json"),
        readers=ns,
        labs=len(labs),
        item_families=ni,
        source_coverage=coverage,
        coding=details,
        bootstrap="Original items, labs and readers nested in labs",
        bootstrap_design_coverage={"generated":BOOT,"retained":len(I),"sampled_missing_verified_source_cell":removed},
        location_only_baseline={
            "article_contrast": 0,
            "noun_contrast": 1,
            "variation_between_items": 0,
            "contrast_rank_correlation": None,
        },
        human_signed_microvolts=lead,
        human_leave_one_lab_out=leave,
        human_source_coding_sensitivity=sensitivity,
        cloze_summary=cloze_stats,
        split_half_item_contrast_rho=np.median(half, axis=0).tolist(),
        model_associations=model_results,
    )
    write_json(OUT / "scores.json", out)
    print("Nine-laboratory measured effect", lead)
    print(
        "Frozen models",
        [(k, v["mean"], v["held_item_contrast_rho"]) for k, v in model_results.items()],
    )


if __name__ == "__main__":
    score()
