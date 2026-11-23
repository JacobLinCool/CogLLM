"""Evaluate normalized source-locked lexical populations on independent reader productions."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
import csv
import numpy as np
from scipy.stats import spearmanr
from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.cloze_population_fit import data
from research.reading_validation.cloze_population_data import (
    PROVO,
    OUT,
    ROOT,
    normalized,
)
from research.reading_validation.cloze_population_model import Choices, predictions


def opened(meta, training):
    alphabet = meta["words"]
    end = len(alphabet)
    allc = defaultdict(Counter)
    denoms = defaultdict(set)
    no_response = Counter()
    if digest(PROVO) != read_json(OUT / "registration.json")["external_file_sha256"]:
        raise ValueError("Held response table changed")
    for r in csv.DictReader(PROVO.open(encoding="mac_roman")):
        uid = r["Word_Unique_ID"]
        n = int(r["Response_Count"])
        denoms[uid].add(int(r["Total_Response_Count"]))
        name = normalized(r["Response"])
        if not name:
            no_response[uid] += n
        else:
            allc[uid][name] += n
    if any(len(v) != 1 for v in denoms.values()):
        raise ValueError("Conflicting reader denominators")
    details = {}
    mistyped = {}
    for uid, c in allc.items():
        n = next(iter(denoms[uid]))
        counted = sum(c.values()) + no_response[uid]
        if n != counted:
            mistyped[uid] = {"stated": n, "counted": counted}
    ii = []
    cc = []
    remaining = {}
    bytype = Counter()
    outside = Counter()
    unknown = []
    for k, (source, id0) in enumerate(zip(meta["source"], meta["ids"])):
        if source:
            ii.append(training.indices[k])
            cc.append(training.counts[k])
            continue
        count = Counter()
        bytype.update(allc[id0])
        for w, n in allc[id0].items():
            ix = alphabet.get(w, end)
            count[ix] += n
            if ix == end:
                outside[w] += n
        if not count:
            raise ValueError("No reader choices at verified stimulus " + id0)
        exact = allc[id0]
        total = sum(exact.values())
        if total < 2:
            raise ValueError("At least two nonempty observations are required for empirical disagreement")
        maximum = max(exact.values())
        details[k] = (
            {alphabet[w] for w, n in exact.items() if n == maximum and w in alphabet},
            1.0 - sum(n * (n - 1) for n in exact.values()) / (total * (total - 1)),
        )
        keys = sorted(count)
        ii.append(np.array(keys))
        cc.append(np.array([count[k] for k in keys]))
        remaining[id0] = sum(count.values())
        unknown.append(no_response[id0])
    source_words = set(
        w for c in read_json(ROOT / "source-counts.json").values() for w in c
    )
    return Choices(ii, cc), details, mistyped, dict(
        released_contexts=len(denoms),
        verified_contexts=len(remaining),
        participants_reported_by_authors=470,
        eligible_nonempty_choices=sum(remaining.values()),
        eligible_blank_choices=sum(unknown),
        verified_human_response_types=len(bytype),
        externally_observed_types_absent_from_training=len(set(bytype) - source_words),
        externally_observed_choices_absent_from_training=sum(
            n for k, n in bytype.items() if k not in source_words
        ),
        outside_fixed_alphabet=sum(outside.values()),
        represented_choices=sum(remaining.values()) - sum(outside.values()),
        outside_spellings=len(outside),
        all_released_denominator_mismatches=len(mistyped),
        aligned_denominator_mismatches=sum(k in mistyped for k in remaining),
        aligned_consistent_nonempty_choices=sum(v for k,v in remaining.items() if k not in mistyped),

    )


def metric(rows):
    r = np.asarray(rows, float)
    n = r[:, 1]
    targets = r[:, 8] >= 0
    disagreement = 1 - r[:, 3]
    human = 1 - r[:, 4]
    rated = (
        float(spearmanr(disagreement, human).statistic)
        if np.ptp(disagreement) > 1e-10
        else None
    )
    return dict(
        prefixes=len(r),
        choices=int(n.sum()),
        NLL_per_choice=float(r[:, 2].sum() / n.sum()),
        top1_matches_any_mode=float(r[:, 5].mean()),
        human_disagreement_MAE=float(np.mean(abs(disagreement - human))),
        human_disagreement_rho=rated,
        actual_target_probability_MAE=float(abs(r[targets, 8] - r[targets, 9]).mean())
        if targets.any()
        else None,
        target_words_represented=int(targets.sum()),
        mean_predicted_disagreement=float(disagreement.mean()),
        mean_observed_disagreement=float(human.mean()),
    )


def grouped(measured, meta):
    r = {k: np.asarray(v) for k, v in measured.items()}
    indices = r["context"][:, 0].astype(int)
    output = {}
    for label, mask in [
        ("all", np.ones(len(indices), bool)),
        ("complete_articles", meta["complete"][indices]),
    ]:
        ids = indices[mask]
        g, group = np.unique(meta["articles"][ids], return_inverse=True)
        hits = r["context"][mask, 1]
        n = np.bincount(group, weights=hits)
        rng = np.random.default_rng(781)
        w = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=4000)
        den = w @ n
        comp = {}
        for a, b in [
            ("context", "calibrated"),
            ("context", "raw"),
            ("calibrated", "raw"),
            ("context", "frequency"),
        ]:
            delta = r[b][mask, 2] - r[a][mask, 2]
            q = np.bincount(group, weights=delta)
            diff = (w @ q) / den
            comp[a + "_over_" + b] = dict(
                gain_in_nats_per_choice=float(q.sum() / n.sum()),
                passage_interval=np.quantile(diff, [0.025, 0.975]).tolist(),
                equal_passage_gain=float(np.mean(q / n)),
            )
        output[label] = dict(
            unique_articles=len(g),
            metrics={k: metric(v[mask]) for k, v in r.items()},
            paired_nll=comp,
        )
    return output


def one_token(external, raw, X, geometry, models, U, V, obs, meta):
    from scipy.special import logsumexp
    from research.reading_validation.cloze_population_model import batches
    lex = read_json(ROOT / "dictionary.json")
    aliases = Counter(r["tokens"][0] for r in lex)
    ok = np.asarray([len(r["tokens"]) == 1 and aliases[r["tokens"][0]] == 1 for r in lex] + [False])
    if ok.sum() != 7275:
        raise ValueError("Changed native single-token geometry")
    g, group = np.unique(meta["articles"][external],return_inverse=True)
    pos = {i:a for i,a in zip(external,group,strict=True)}
    losses = {k: np.zeros(len(g)) for k in models}
    n = np.zeros(len(g))
    for ids, p in batches(external,raw,X,geometry,models,U,V):
        for j,i in enumerate(ids):
            words = obs.indices[i]
            counts = obs.counts[i][ok[words]]
            words = words[ok[words]]
            n[pos[i]] += counts.sum()
            for name,logp in p.items():
                z = logp[j]
                losses[name][pos[i]] += (logsumexp(z[ok]) * counts.sum() - z[words] @ counts)
    w = np.random.default_rng(39049).multinomial(len(g),np.ones(len(g))/len(g),size=3000)
    comparisons = {}
    for name in ("calibrated","raw"):
        change = losses[name] - losses["context"]
        comparisons["context_over_"+name] = {"gain":float(change.sum()/n.sum()),
            "paired_passage_interval":np.quantile((w@change)/(w@n), [.025,.975]).tolist()}
    return {"identified_words":int(ok.sum()),"included_reader_choices":int(n.sum()),"article_groups":len(g),
            "per_choice_NLL":{name:float(value.sum()/n.sum()) for name,value in losses.items()},"paired":comparisons}


def main(folder):
    lock = read_json(OUT / "forecast-lock.json")
    raw, X, U, V, training, meta = data(folder)
    source = np.where(meta["source"])[0]
    external = np.where(~meta["source"])[0]
    f = read_json(OUT / "fold-5.json")
    models = f["models"]
    geometry = f["geometry"]
    if (
        digest(folder / "receipt.json") != lock["native"]
        or [digest(OUT / f"fold-{t}.json") for t in range(6)] != lock["source_fits"]
    ):
        raise ValueError("Source fit or acquisition changed after sealing")
    compare, _ = predictions(external, raw, X, geometry, models, U, V)
    if compare != lock["external_forecast_sha256"]:
        raise ValueError("External forecasts changed before opening grades")
    obs, exact, mistyped, cov = opened(meta, training)
    probed, held = predictions(
        external, raw, X, geometry, models, U, V, obs, meta["targets"]
    )
    if probed != compare:
        raise ValueError("Observations altered independent predictions")
    source_out = {k: [] for k in models}
    for t in range(5):
        fold = read_json(OUT / f"source-fold-{t}-scores.json")["scores"]
        for k, v in fold.items():
            source_out[k].extend(v)
    valid = np.array([i for i in external if meta["ids"][i] not in mistyped])
    valid_set = set(valid)
    clean = {k:[row for row in v if int(row[0]) in valid_set] for k,v in held.items()}
    exact_scores = {}
    for name, values in clean.items():
        rows = np.asarray(values, float)
        units = rows[:,0].astype(int)
        lower = 1 - rows[:,3]
        upper = lower + rows[:,10] ** 2
        observed = np.array([exact[i][1] for i in units])
        category = rows[:,6].astype(int)
        named = category != len(meta["words"])
        agrees = np.array([k in exact[i][0] for k, i in zip(category, units, strict=True)])
        exact_scores[name] = dict(
            exact_modal_spelling_accuracy=float(agrees.mean()),
            named_forecasts=int(named.sum()),
            unnamed_other_forecasts=int((~named).sum()),
            accuracy_given_named_forecast=float(agrees[named].mean()) if named.any() else None,
            observed_exact_spelling_disagreement=float(observed.mean()),
            predicted_exact_spelling_disagreement_mean_bounds=[float(lower.mean()),float(upper.mean())],
            mean_distance_to_identified_disagreement_range=float(np.maximum.reduce(
                [lower-observed, observed-upper, np.zeros_like(observed)]).mean()),
        )
    from research.reading_validation.cloze_change_score import evaluate

    evaluate(source_out, clean, held, meta, exact, mistyped)
    rr = grouped(clean, meta)
    ss = grouped(source_out, meta)
    ix = np.array([i[0] for i in held["context"]], int)
    if len(set(ix)) != len(external) or len(
        set(r[0] for r in source_out["context"])
    ) != len(source):
        raise ValueError("Missing original passage positions")
    write_json(
        OUT / "scores.json",
        dict(
            forecasts=digest(OUT / "forecast-lock.json"),
            program=digest(Path(__file__)),
            coverage=cov,
            exact_spelling_support=exact_scores,
            unambiguous_single_token_choices=one_token(valid, raw, X, geometry, models, U, V, obs, meta),
            sensitivity_all_observed_row_mass=grouped(held,meta),
            response_denominator_audit=digest(Path(__file__).with_name("cloze-denominator-audit.md")),
            vocabulary_identification=digest(Path(__file__).with_name("cloze-vocabulary-identification.md")),
            independent_Provo=rr,
            source_UCL=ss,
        ),
    )
    print(
        "Full human alternative choices",
        cov["eligible_nonempty_choices"],
        "primary context-over-calibrated",
        rr["all"]["paired_nll"]["context_over_calibrated"],
    )


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("native", type=Path)
    main(p.parse_args().native)
