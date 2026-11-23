"""Estimate where a named expression will occur in a full biomedical abstract."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from research.reading_validation.local_features import OUT, build
from research.reading_validation.local_forecast import digest
from research.reading_validation.local_rank import targets_and_groups, fit, ranking, by_target
from research.reading_validation.local_ratings import intervals


def fit_sentence_selection(docs, groups, targets, nodes, group, columns):
    from scipy.optimize import minimize

    training = [
        rows for rows in docs.values() if any(s.key in groups and groups[s.key] != group for s in rows)
    ]

    def extract(s, k):
        if k == "surface":
            return s.sentence[k][[0, 1, 2, 4, 5, 6]]
        return s.sentence[k]

    xraw = lambda rows: np.vstack([np.concatenate([extract(s, k) for k in columns]) for s in rows])
    raw = [xraw(rows) for rows in training]
    flat = np.vstack(raw)
    mean = np.mean(flat, axis=0)
    scale = np.maximum(flat.std(axis=0), 1e-6)
    offset = 0
    for col in columns:
        size = len(extract(training[0][0], col))
        if col == "representation":
            scale[offset : offset + size] *= np.sqrt(size / 4)
        offset += size
    x = [(a - mean) / scale for a in raw]
    labels = [np.array([len(targets.get(s.key, ())) for s in rows], dtype=float) for rows in training]
    total = sum(a.sum() for a in labels)

    def objective(coef):
        loss = 0.06 * np.dot(coef, coef)
        grad = 0.12 * coef
        for xs, ys in zip(x, labels):
            z = xs @ coef
            logp = z - logsumexp(z)
            loss -= ys @ logp / total
            grad += ((np.exp(logp) * ys.sum() - ys) @ xs) / total
        return loss, grad

    fit = minimize(
        objective,
        np.zeros(flat.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"gtol": 1e-7, "ftol": 1e-12},
    )
    if not fit.success or np.max(np.abs(fit.jac)) > 2e-5:
        raise ArithmeticError("Sentence-selection model failed convergence")
    return lambda sentences: ((xraw(sentences) - mean) / scale) @ fit.x, dict(
        coef=fit.x.tolist(),
        center=mean.tolist(),
        scale=scale.tolist(),
        features=columns,
        gradient=float(abs(fit.jac).max()),
    )


def run(gpu, out):
    nodes, _ = build(gpu)
    targets, groups = targets_and_groups()
    docs = defaultdict(list)
    for node in nodes.values():
        if node.document.startswith("plaba/"):
            docs[node.document].append(node)
    arms = {"form": ("surface",), "local": ("surface", "medical", "lexical", "representation")}
    primary, params = [], {}
    for group in [*sorted(set(groups.values())), None]:
        train = [nodes[key] for key in sorted(targets) if groups[key] != group]
        test = [nodes[key] for key in sorted(targets) if groups[key] == group]
        relevant = {node.document for node in test}
        forecasts = defaultdict(dict)
        for name, columns in arms.items():
            span_head, span_state = fit(train, targets, columns)
            sentence_head, sentence_state = fit_sentence_selection(
                docs, groups, targets, nodes, group, columns
            )
            if group is None:
                params[name] = dict(span=span_state, sentence=sentence_state)
                continue
            for doc in relevant:
                ns = docs[doc]
                sent_logits = sentence_head(ns)
                grids = [s.coords for s in ns]
                raw = [span_head(s) for s in ns]
                cond = [v - logsumexp(v) for v in raw]
                options = {
                    "joint": np.concatenate([q + z for q, z in zip(cond, sent_logits)]),
                    "uniform_sentence": np.concatenate(cond),
                    "raw": np.concatenate(raw),
                }
                for variant, scores in options.items():
                    forecasts[doc][name + "_" + variant] = ranking(np.vstack(grids), scores)
        for n in test:
            for a, b in sorted(targets[n.key]):
                record = dict(group=group, sentence=n.key, start=a, end=b)
                for name, coords in forecasts[n.document].items():
                    record[name] = dict(top5=coords[:5].tolist(), **by_target(coords, (a, b)))
                primary.append(record)
    methods = [k for k in primary[0] if k not in ("group", "sentence", "start", "end")]
    summary = {}
    for measure in ("hit1", "hit5", "rr"):
        vals = np.array([[r[k][measure] for k in methods] for r in primary])
        point, boot = intervals(vals, [r["group"] for r in primary])
        desc = lambda p, b: dict(mean=float(p), interval=np.quantile(b, [0.025, 0.975]).tolist())
        stat = {k: desc(point[j], boot[:, j]) for j, k in enumerate(methods)}
        for a, b in [
            ("local_joint", "form_joint"),
            ("local_joint", "local_raw"),
            ("local_joint", "local_uniform_sentence"),
            ("form_joint", "form_raw"),
        ]:
            i, j = methods.index(a), methods.index(b)
            stat[a + " minus " + b] = desc(point[i] - point[j], boot[:, i] - boot[:, j])
        summary[measure] = stat
    out.mkdir(exist_ok=True, parents=True)
    pred = out / "predictions.json"
    pred.write_text(json.dumps(primary, separators=(",", ":")) + "\n")
    (out / "deployment-weights.json").write_text(json.dumps(params, separators=(",", ":")) + "\n")
    (out / "scores.json").write_text(
        json.dumps(
            dict(
                metrics=summary,
                protocol_sha256=digest(Path(__file__).with_name("local-article-protocol.md")),
                predictions_sha256=digest(pred),
                spans=len(primary),
                groups=len(set(groups.values())),
            ),
            indent=2,
        )
        + "\n"
    )
    print(summary["hit5"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=Path, required=True)
    p.add_argument("--out", type=Path, default=OUT / "article-localization")
    a = p.parse_args()
    run(a.gpu, a.out)
