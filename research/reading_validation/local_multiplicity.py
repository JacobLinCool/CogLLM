"""Recalculate simultaneous intervals for the fixed difficulty and revision inventory."""

import argparse
import csv
import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.reading_validation.local_claims import RESULTS, inventory
from research.reading_validation.local_forecast import digest

COUNT = 32
TAIL = 0.025 / COUNT
OUT = RESULTS / "reading-validation-local" / "multiplicity"
INTERVAL_KEYS = (
    "ci95",
    "interval",
    "crossed_95",
    "article_bootstrap_95",
    "reader_article_95",
    "interval95",
)


def read(path):
    return json.loads(path.read_bytes())


def interval(point, values):
    values = np.asarray(values)
    return {
        "estimate": float(point),
        "ci95": np.quantile(values, [0.025, 0.975, TAIL, 1 - TAIL]).tolist(),
    }


class _ReportingNumpy:
    """Preserve original bootstrap calculations and request two additional report quantiles."""

    def __getattr__(self, name):
        return getattr(np, name)

    def quantile(self, values, q, *args, **kwargs):
        if np.array_equal(q, [0.025, 0.975]):
            q = [0.025, 0.975, TAIL, 1 - TAIL]
        return np.quantile(values, q, *args, **kwargs)


@contextmanager
def report_quantiles(module):
    # The committed preceding analyses are hash-locked. Override only their reporting
    # quantile requests in memory and never write to a source result directory.
    with patch.object(module, "np", _ReportingNumpy()):
        yield


def plaba():
    from research.reading_validation import local_rank, local_ratings

    folder = RESULTS / "reading-validation-local"
    rows = read(folder / "localization/predictions.json")
    with report_quantiles(local_rank):
        out = local_rank.grouped_scores(rows, ["local", "form"])["paired"]["sentence"]
    answer = {1: out["hit5"]["local minus form"], 2: out["hit1"]["local minus form"]}
    rows = read(folder / "article-localization/predictions.json")
    delta = np.array(
        [[r["local_joint"]["hit5"] - r["form_joint"]["hit5"]] for r in rows]
    )
    pt, boot = local_ratings.intervals(delta, [r["group"] for r in rows])
    answer[3] = interval(pt[0], boot[:, 0])
    return answer


def headings():
    from research.reading_validation import local_heading as ht

    directory = RESULTS / "reading-validation-local" / "headings"
    rows = read(directory / "predictions.json")
    families = defaultdict(dict)
    values = {}
    for r in rows:
        families[r["doi"]][r["arm"]] = r["id"]
        values[r["id"]] = np.mean([q["forecasts"]["local"] for q in r["sentences"]])
    cases = [[v[k] for k in ht.ARMS] for _, v in sorted(families.items())]
    delta = np.array([(values[b] - values[a]) * 7 / 4 for a, b in cases])
    raw = ht.CACHE / "kerwer-2021/holdout/scoring/public-study-ratings.jsonl"
    if digest(raw) != read(directory / "scores.json")["receipts"]["source_ratings"]:
        raise ValueError("Changed source heading responses")
    pt, boot, _ = ht.kerwer_effects(
        ht.kerwer_readers(raw, cases), np.column_stack((delta, delta))
    )
    return {6: interval(pt[:, 1].mean(), boot[:, 1])}


def guo():
    from research.reading_validation import local_help_score as help
    from research.reading_validation import local_jargon as jargon

    folder = RESULTS / "reading-validation-local" / "jargon"
    pred = {
        (r["paper"], r["term"]): r["scores"]
        for r in jargon.rows(folder / "predictions.jsonl")
    }
    rec, _ = jargon.outcomes(jargon.JARGON / "source.csv", pred)
    with report_quantiles(jargon):
        assess = jargon.rank_stats(rec, "after", list(next(iter(pred.values()))))
    other, _ = help.data()
    with np.load(help.OUT / "forecasts.npz", allow_pickle=False) as bank:
        if bank["key"].tolist() != [
            help.code("/".join((r["paper"], r["term"], r["reader"]))) for r in other
        ]:
            raise ValueError("Inconsistent order in Guo assistance responses")
        with report_quantiles(help):
            need = help.evaluate(
                other,
                bank,
                help.OUTCOMES.index("definition"),
                "definition",
                matched=False,
            )
    return {
        7: assess["estimates"]["local minus form"],
        8: assess["estimates"]["local minus external_lexical"],
        10: need["concordance"]["lexical_prior minus lexical"],
    }


def revisions():
    from research.reading_validation import revision_contrasts, revision_score

    humans, pred, names = revision_score.aggregate_panel(
        revision_score.OUT / "forecast"
    )
    with patch.object(revision_score, "item", interval):
        scores = revision_score.summarize(humans, pred, names)["simplicity"][
            "crossed_source_reader_interval"
        ]
    with patch.object(revision_contrasts, "item", interval):
        opposed = revision_contrasts.analysis(humans, pred, names)[
            "opposed_preferences"
        ]["crossed_95"]
    return {
        14: scores["full minus length"],
        15: scores["full minus edit"],
        16: opposed["adaptive minus edit"],
    }


def correctness():
    from research.reading_validation import revision_comprehension_score as rs

    path = rs.RESULT / "forecasts.jsonl"
    if digest(path) != read(rs.RESULT / "forecast-seal.json")["forecasts_sha256"]:
        raise ValueError("Unsealed OneStopQA revision forecasts")
    records = {
        (r["article"], r["paragraph"], r["condition"]): r["scores"]
        for r in rs.lines(path)
    }
    clean = {k[0] for k in records} - set(
        read(rs.RESULT / "input-seal.json")["excluded_from_primary"]
    )
    with report_quantiles(rs):
        r = rs.estimate(records, rs.read_responses(), clean, False)["correctness"][
            "paired"
        ]
    return {
        i: r["meaning minus " + k]
        for i, k in enumerate((*rs.CONTROLS[::-1], "simplicity"), 17)
    }


def learners():
    from research.reading_validation import nns_rewrite_score as rw
    from research.reading_validation import nns_score as un

    dense = rw.pack_model_input("dense")
    flags, totals, _ = un.selected_counts("dense", dense)
    n = totals.sum(1)
    use = n > 0
    doc, fam = dense["doc"][use], dense["family"][use]
    with np.load(un.OUT / "word-forecasts.npz", allow_pickle=False) as b:
        assert np.array_equal(b["doc"], dense["doc"]) and np.array_equal(
            b["coords"], dense["coords"]
        )
        loss = []
        for k in ("text.adaptive", "cefr.adaptive"):
            p = b[k][use]
            loss.append(
                (
                    -(
                        flags[use] * np.log(p)
                        + (totals[use] - flags[use]) * np.log1p(-p)
                    )
                ).sum(1)
                / n[use]
            )
    m = rw.family_matrix(np.column_stack(loss), doc, fam)
    w = np.random.default_rng(un.SEED).multinomial(
        len(m), np.ones(len(m)) / len(m), size=un.B
    )
    avg, boot = un.aggregate(m, w)
    size = {25: interval(avg[0] - avg[1], boot[:, 0] - boot[:, 1])}
    with (
        np.load(rw.OUT / "forecasts.npz", allow_pickle=False) as f,
        np.load(rw.DATA / "word-features.npz", allow_pickle=False) as pos,
    ):
        pairs = rw.comparisons(pos, "held", outcomes=True)
        if not np.array_equal(pairs["keys"], f["r_keys"]) or not np.array_equal(
            f["u_coords"], dense["coords"]
        ):
            raise ValueError("Learner joint model geometry changed")
        true = pairs["counts"] / pairs["counts"].sum(1, keepdims=True)
        y = flags[use].sum(1) / n[use]
        label = ["contextual.adaptive", "contextual.independent"]
        us = np.column_stack([f["u." + k][use] for k in label])
        ub = -(y[:, None] * np.log(us) + (1 - y[:, None]) * np.log1p(-us))
        r = np.stack([f["r." + k] for k in label], axis=1)
        rb = -np.sum(rw.xlogy(true[:, None, :], np.clip(r, 1e-12, 1)), axis=2)
        for i, z, d, g in ((26, ub, doc, fam), (27, rb, pairs["doc"], pairs["family"])):
            with patch.object(rw, "ci", interval):
                out = rw.reported(z, d, g, label, [tuple(label)], lower=True)
            size[i] = out["oriented_gains"][
                "contextual.adaptive_gain_vs_contextual.independent"
            ]
    return size


def semlex():
    from research.reading_validation import semlex_score as s

    with np.load(s.OUT / "forecasts.npz", allow_pickle=False) as f:
        keys, pred = f["keys"], f["signed_a_preference"]
    sets = {
        k: tuple(v["candidates"]) for k, v in read(s.DATA / "geometry.json").items()
    }
    panel = s.read_ranks(s.DATA / "outcomes-reserve.txt")
    keep = np.array([a in panel[k] and b in panel[k] for k, _, a, b in keys])
    h = np.array([np.sign(panel[k][b] - panel[k][a]) for k, _, a, b in keys[keep]])
    with patch.object(s, "ci", interval):
        r = s.test(keys[keep], pred[keep], h, sets)["all_shared_candidate_pairs"][
            "relative_confidence"
        ]["contrasts"]
    return {
        i: r["mix_minus_" + k]
        for i, k in enumerate(
            ("context_invariant", "lexical_plus_surprisal", "paper_adaptive_ngram"), 30
        )
    }


def values(record, direction):
    key = next(k for k in INTERVAL_KEYS if k in record)
    x = record["estimate"] if "estimate" in record else record["mean"]
    raw = record[key]
    groups = [
        sorted(direction * v for v in raw[j : j + 2]) for j in range(0, len(raw), 2)
    ]
    return float(x) * direction, groups


def run(early=False):
    claims = inventory()
    if not early and not claims[-1].file.exists():
        raise ValueError("The prespecified SemEval external evaluation is still sealed")
    fixed = {}
    for source in (plaba, headings, guo, revisions, correctness, learners):
        fixed.update(source())
    if not early:
        fixed.update(semlex())
    result = []
    for i, c in enumerate(claims, 1):
        if early and i > 29:
            continue
        record = read(c.file)
        for k in c.path:
            record = record[k]
        point, (old,) = values(record, c.direction)
        wide = None
        if i in fixed:
            check, (rep, wide) = values(fixed[i], c.direction)
            if not np.allclose([point, *old], [check, *rep], rtol=0, atol=2e-10):
                raise ValueError(
                    f"Failed to reproduce original point and interval for {i}: {point, old} versus {check, rep}"
                )
        elif old[0] > 0:
            raise ValueError(f"Missing required wider replay for claim {i}")
        result.append(
            {
                "index": i,
                "study": c.study,
                "comparison": c.label,
                "artifact": c.artifact,
                "outcome_path": list(c.path),
                "source_sha256": digest(c.file),
                "estimate": point,
                "ordinary_95": old,
                "familywise_interval": wide,
                "positive_after_correction": bool(wide is not None and wide[0] > 0),
            }
        )
    target = (
        Path(".cache/cogllm/reading-validation/local-simultaneous-preflight")
        if early
        else OUT
    )
    target.mkdir(exist_ok=True, parents=True)
    summary = {
        "scope": "Local word, sentence and revision results; selected retrospective headline inventory",
        "comparisons": COUNT,
        "two_sided_marginal_coverage": 1 - 2 * TAIL,
        "per_tail": TAIL,
        "positive_original_95": sum(x["ordinary_95"][0] > 0 for x in result),
        "positive_after_correction": sum(
            x["positive_after_correction"] for x in result
        ),
        "evaluated": len(result),
        "protocol_sha256": digest(
            Path(__file__).with_name("local-multiplicity-protocol.md")
        ),
        "code_sha256": digest(Path(__file__)),
        "inventory_sha256": digest(Path(__file__).with_name("local_claims.py")),
        "results": result,
    }
    (target / "scores.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    with (target / "source-data.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "study",
                "comparison",
                "estimate",
                "ordinary_low",
                "ordinary_high",
                "simultaneous_low",
                "simultaneous_high",
                "positive_after_correction",
            ]
        )
        for v in result:
            writer.writerow(
                [
                    v["index"],
                    v["study"],
                    v["comparison"],
                    v["estimate"],
                    *v["ordinary_95"],
                    *(v["familywise_interval"] or ("", "")),
                    v["positive_after_correction"],
                ]
            )
    print(
        target,
        summary["positive_original_95"],
        summary["positive_after_correction"],
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    run(parser.parse_args().preflight)
