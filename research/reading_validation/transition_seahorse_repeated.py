"""Fit source-only exchangeable Q1 counts and transport them to held repeat-rating groups."""

from collections import defaultdict
from math import comb

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from research.reading_validation.memory_data import h
from research.reading_validation.transition_seahorse_data import DEST, WORK, sha, source_rows
from research.reading_validation.transition_seahorse_features import load, save, signatures
from research.reading_validation.transition_seahorse_fit import LABEL, key, linear, solve, weights
from research.reading_validation.transition_seahorse_score import summary


def get_source():
    archive = {}
    for role in load(DEST / "exposure-seal.json")["packets"]:
        filename = WORK / f"seahorse-{role}-document-features.npz"
        if sha(filename) != load(DEST / f"{role}-feature-seal.json")["file"]:
            raise ValueError("Registered Q1 feature shard changed")
        with np.load(filename, allow_pickle=False) as arrays:
            archive.update(zip(arrays["keys"], arrays["x"], strict=True))
    population, panels, audit = {}, {}, {}
    overlap = load(DEST / "word-exposure-audit.json")
    for role in ("train", "validation", "test"):
        rejected = set(overlap[role]["exclude_overlap_with_held_roles"])
        original_rows = []
        for row in source_rows(role, grades=role != "test"):
            if row["gem_id"] in rejected or h(row["summary"]) not in archive:
                continue
            if role != "test" and row["question1"] not in LABEL:
                continue
            original_rows.append(row)
        official = {key(row): row for row in original_rows}
        duplicate = defaultdict(list)
        for row in source_rows(role, duplicates=True, grades=True):
            if row["question1"] in LABEL:
                duplicate[key(row)].append(int(LABEL[row["question1"]]))
        panel = {k: v for k, v in duplicate.items() if k in official and len(v) >= 2}
        panels[role] = {k: (official[k], votes) for k, votes in panel.items()}
        group_ids = {official[k]["gem_id"] for k in panel}
        reject_repeats = group_ids if role == "train" else set()
        keep = [r for r in original_rows if r["gem_id"] not in reject_repeats]
        population[role] = keep
        audit[role] = {
            "eligible_canonical": len(original_rows), "distinct_canonical_keys":len(official),
            "usable_panel_rows": sum(map(len, panel.values())),
            "usable_panels": len(panel), "panel_articles": len(group_ids),
            "mean_fit_eligible": len(keep), "canonical_rows_removed_for_panel_independence": len(original_rows)-len(keep),
        }
    source_panels = [r for r, _ in panels["train"].values()]
    audit["disjointness"] = {}
    for role in ("validation", "test"):
        originals = list(source_rows(role))
        target_articles = {r["gem_id"] for r in originals}
        target_strings = set().union(*(signatures(r["summary"]) for r in originals))
        seen_ids = {r["gem_id"] for r in source_panels} & target_articles
        hits = {key(r) for r in source_panels if signatures(r["summary"]) & target_strings}
        audit["disjointness"][role] = {"shared_articles":len(seen_ids), "source_panels_with_held_word_match":len(hits)}
        if seen_ids or hits:
            raise ValueError("A source repeat panel appeared in a later canonical cohort")
    return archive, population, panels, audit


def source_means(archive, population, panels, name):
    width = {"form": 15, "native": 187}[name]
    def design(rows):
        return np.stack([archive[h(r["summary"])][:width] for r in rows])
    def supervised(role):
        rows = population[role]
        return design(rows), np.array([LABEL[r["question1"]] for r in rows]), weights([r["gem_id"] for r in rows])
    x, y, w = supervised("train")
    v, target, vw = supervised("validation")
    center = w @ x
    scale = np.sqrt(w @ (x-center)**2)
    scale[scale < 1e-7] = 1.
    theta, convergence = solve((x-center)/scale, y, w, .002)
    cal, cal_diag = solve(linear((v-center)/scale, theta)[:, None], target, vw, 1e-8, positive=True)
    out = {}
    for role in ("train", "test"):
        x = design([r for r, _ in panels[role].values()])
        out[role] = linear(linear((x-center)/scale, theta)[:, None], cal)
    return out, {"mean": convergence, "calibration": cal_diag}


def vote_count_loss(k, n, p, rho):
    """Negative log probability of the unordered count, with a continuous binomial limit."""
    w = 1 - rho
    out = np.array([-np.log(comb(int(y), int(x))) for x, y in zip(k, n, strict=True)])
    for j in range(int(n.max())):
        out += (n > j) * np.log(w + j*rho)
        out -= (k > j) * np.log(w*p + j*rho)
        out -= (n-k > j) * np.log(w*(1-p) + j*rho)
    return out


def best_scalar(objective, lo, hi):
    fit = minimize_scalar(objective, method="bounded", bounds=(lo, hi), options={"xatol": 2e-10})
    if not fit.success:
        raise ArithmeticError("A source scalar fit did not converge")
    endpoint = min((lo, float(fit.x), hi), key=objective)
    return endpoint


def fit_counts(logodds, k, n, w, *, adjust_mean, dependence):
    def objective(offset, rho):
        p = np.clip(expit(logodds + offset), 1e-10, 1-1e-10)
        return float(w @ vote_count_loss(k, n, p, rho))
    def intercept(rho):
        return best_scalar(lambda t: objective(t, rho), -12., 12.) if adjust_mean else 0.
    rho = best_scalar(lambda r: objective(intercept(r), r), 0., 1-1e-7) if dependence else 0.
    offset = intercept(rho)
    if abs(offset) >= 12:
        raise ArithmeticError("The repeated-ballot intercept reached the numerical search limit")
    return {"logit_shift": offset, "rho": rho, "source_count_nll": objective(offset, rho)}


def outcomes(items):
    n = np.array([len(v) for _, v in items.values()])
    k = np.array([sum(v) for _, v in items.values()])
    g = np.array([row["gem_id"] for row, _ in items.values()])
    return k, n, g


def score_calibrator(base, parameters, k, n):
    p = np.clip(expit(base + parameters["logit_shift"]), 1e-10, 1-1e-10)
    rho = parameters["rho"]
    q = 2*(1-rho)*p*(1-p)
    observed = 2*k*(n-k)/(n*(n-1))
    return {
        "count_nll": vote_count_loss(k, n, p, rho),
        "pair_brier": observed*(1-q)**2 + (1-observed)*q**2,
        "estimated_pair_disagreement": q,
        "mean_predicted_no": p,
        "observed_pair_disagreement": observed,
        "observed_fraction_no": k/n,
    }


def resample_dispersion(source_odds, source_counts, source_sizes, source_articles,
                        held_odds, held_counts, held_sizes, held_boot, exact_rho):
    """Include source uncertainty by re-estimating the scalar on a dense probability grid."""
    grid = np.linspace(0, .999, 1000)
    p_train, p_test = expit(source_odds), expit(held_odds)
    _, ids, sizes = np.unique(source_articles, return_inverse=True, return_counts=True)
    source = np.stack([np.bincount(ids, weights=vote_count_loss(source_counts,source_sizes,p_train,r))/sizes
                       for r in grid], axis=1)
    if abs(grid[source.mean(axis=0).argmin()] - exact_rho) > .001:
        raise ArithmeticError("Bootstrap grid differs from the exact full-source scalar optimum")
    held = np.stack([vote_count_loss(held_counts,held_sizes,p_test,r) for r in grid])
    rng = np.random.default_rng(741363)
    g = len(sizes)
    counts = rng.multinomial(g,np.full(g,1/g),size=len(held_boot)-1)
    selected = (counts @ source).argmin(axis=1)
    d = np.sum(held_boot[1:]*(held[selected]-held[0]),axis=1)
    return {
        "source_articles": g, "independent_source_and_held_resamples":len(selected),
        "grid_resolving_step":float(grid[1]), "estimated_rho":exact_rho,
        "source_rho_interval":np.quantile(grid[selected],[.025,.975]).tolist(),
        "fraction_source_rho_at_zero":float(np.mean(selected==0)),
        "interval_held_count_nll_improvement_with_source_refits":np.quantile(d,[.025,.975]).tolist(),
        "scope":"Native single-reader mean is fixed from its nonoverlapping training articles. Both repeated-rating article samples are resampled."
    }


def main():
    table, population, panel, audit = get_source()
    kt, nt, gt = outcomes(panel["train"])
    k, n, group = outcomes(panel["test"])
    w = weights(gt)
    odds = {}
    conv = {}
    for method in ("native", "form"):
        odds[method], conv[method] = source_means(table, population, panel, method)
    odds["constant"] = {"train": np.zeros(len(kt)), "test": np.zeros(len(k))}
    previous = DEST / "forecasts-before-held-responses.npz"
    if sha(previous) != load(DEST / "fit-seal.json")["forecast_sha256"]:
        raise ValueError("Archived production mean forecasts changed")
    with np.load(previous, allow_pickle=False) as a:
        forecasts = {}
        for method in ("native", "form"):
            original = dict(zip(a["keys"], a[method], strict=True))
            forecasts[method] = np.array([logit(original[x]) for x in panel["test"]])
    parameters, outcomes_test, outcomes_shipped = {}, {}, {}
    for method, bases in odds.items():
        for name, shift, dep in (("fixed",False,False), ("dispersion",False,True), ("shift",True,False), ("shift_dispersion",True,True)):
            if method == "constant" and not shift:
                continue
            identity = method + "_" + name
            param = fit_counts(bases["train"], kt, nt, w, adjust_mean=shift, dependence=dep)
            parameters[identity] = param
            outcomes_test[identity] = score_calibrator(bases["test"], param, k, n)
            if method in forecasts:
                outcomes_shipped[identity] = score_calibrator(forecasts[method], param, k, n)
    # Equal source article weight and equal pair weight within its summary, identical to the target item estimand.
    q = float(w @ (2*kt*(nt-kt)/(nt*(nt-1))))
    obs = 2*k*(n-k)/(n*(n-1))
    outcomes_test["constant_disagreement"] = {
        "pair_brier": obs*(1-q)**2 + (1-obs)*q*q,
        "estimated_pair_disagreement": np.full(len(k),q),
    }
    articles, ix = np.unique(group, return_inverse=True)
    draws = np.random.default_rng(88216).multinomial(len(articles),np.full(len(articles),1/len(articles)),size=4000)
    boot = np.vstack([np.ones(len(articles)), draws])[:,ix]
    boot /= boot.sum(axis=1)[:,None]
    two_sample = resample_dispersion(
        odds["native"]["train"], kt, nt, gt, odds["native"]["test"], k, n, boot,
        parameters["native_dispersion"]["rho"],
    )
    def aggregate(results):
        packed = {}
        for method, metrics in results.items():
            packed[method] = {}
            for metric, values in metrics.items():
                means = boot @ values
                packed[method][metric] = summary(means[0], means[1:])
        return packed
    contrasts = {}
    for a,b in (("native_dispersion","native_fixed"),("native_shift_dispersion","native_shift"),
                ("native_shift_dispersion","constant_shift_dispersion"),("native_shift_dispersion","form_shift_dispersion"),
                ("native_dispersion","constant_disagreement")):
        for name in ("count_nll", "pair_brier"):
            if name not in outcomes_test[b]:
                continue
            z=boot@(outcomes_test[a][name]-outcomes_test[b][name])
            contrasts[f"{a} minus {b}, {name}"]=summary(z[0],z[1:])
    save(DEST / "repeated-calibration.json", {
        "study":"Post-primary source-only repeated-ballot likelihood", "repeated_source":audit,
        "held_article_clusters":len(articles), "probability_models":conv,
        "source_dispersion_and_mean":parameters, "constant_source_pair_disagreement":q,
        "source_groups_with_competing_votes":int(((kt>0)&(kt<nt)).sum()),
        "held_groups_with_competing_votes":int(((k>0)&(k<n)).sum()),
        "scores":aggregate(outcomes_test), "paired":contrasts,
        "same_parameters_on_original_held_production_probabilities":aggregate(outcomes_shipped),
        "pre_existing_forecast_sha256":sha(previous), "two_sample_dependence_sensitivity":two_sample,
        "interpretation":"Count and pair scoring use equally weighted repeated-rating summaries. Bootstrap resamples articles, conditional on source calibration. Rater identities are unavailable."
    })
    print(audit,parameters["native_dispersion"],contrasts["native_dispersion minus native_fixed, count_nll"])


if __name__ == "__main__":
    main()
