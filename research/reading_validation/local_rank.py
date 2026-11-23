"""Cross-fit a conditional model of the passage expressions readers singled out."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from research.reading_validation.local_features import RESULTS, build, disjoint_ranking
from research.reading_validation.local_forecast import digest


MODELS = {
    "form": ("surface",),
    "medical_form": ("surface", "medical"),
    "local_lexical": ("surface", "medical", "lexical"),
    "local": ("surface", "medical", "lexical", "representation"),
    "local_shared": ("surface", "medical", "lexical", "representation", "shared"),
    "local_and_persistent": ("surface", "medical", "lexical", "representation", "shared", "persistent"),
}
RIDGE = 0.12
SEED = 60821


def overlap(coords, target):
    a, b = target
    inter = (np.minimum(coords[:, 1], b) - np.maximum(coords[:, 0], a)).clip(0)
    return inter / (b - a + coords[:, 1] - coords[:, 0] - inter)


def grouped_scores(records, methods):
    groups = sorted({r["group"] for r in records})
    index = np.array([groups.index(r["group"]) for r in records])
    draws = np.random.default_rng(SEED).multinomial(
        len(groups), np.ones(len(groups)) / len(groups), size=10000
    )
    weights = draws[:, index]
    point, paired = {}, {}
    domains = ("sentence", "article")
    measure_names = ("hit1", "hit5", "exact1", "exact5", "rr")
    for mode in domains:
        point[mode], paired[mode] = {}, {}
        for k in measure_names:
            observed = {m: np.array([r[m][mode][k] for r in records], dtype=float) for m in methods}
            intervals = {m: (weights @ vals) / weights.sum(axis=1) for m, vals in observed.items()}
            summary = lambda v, boot: dict(
                mean=float(v.mean()), interval=np.quantile(boot, [0.025, 0.975]).tolist()
            )
            point[mode][k] = {m: summary(observed[m], intervals[m]) for m in methods}
            paired[mode][k] = {}
            for m, b in [
                ("local", "form"),
                ("medical_form", "form"),
                ("local_lexical", "medical_form"),
                ("local", "local_lexical"),
                ("local", "displayed_mass"),
                ("local", "displayed_increase"),
                ("local_and_persistent", "local"),
                ("local_and_persistent", "local_shared"),
            ]:
                if m in methods and b in methods:
                    paired[mode][k][m + " minus " + b] = summary(
                        observed[m] - observed[b], intervals[m] - intervals[b]
                    )
    return dict(
        estimates=point,
        paired=paired,
        bootstrap_groups=len(groups),
        bootstrap_draws=10000,
        bootstrap_seed=SEED,
    )


def ranking(coords, scores, limit=None):
    return coords[disjoint_ranking(coords, scores, limit)]


def by_target(proposals, target):
    ious = overlap(proposals, target)
    exact = (proposals == target).all(axis=1)
    hits = np.flatnonzero(ious >= 0.5)
    return dict(
        hit1=int(ious[0] >= 0.5),
        hit5=int((ious[:5] >= 0.5).any()),
        exact1=int(exact[0]),
        exact5=int(exact[:5].any()),
        rr=float(1 / (1 + hits[0])) if len(hits) else 0.0,
    )


def scaler(nodes, parts):
    center, scales = {}, {}
    for name in parts:
        stack = np.vstack([s.features[name] for s in nodes])
        center[name] = stack.mean(axis=0)
        scales[name] = np.maximum(stack.std(axis=0), 1e-6)
        if name in ("representation", "shared"):
            scales[name] *= np.sqrt(stack.shape[1] / 4)
    return center, scales


def fit(nodes, targets, parts):
    center, scales = scaler(nodes, parts)
    design = lambda node: np.column_stack([(node.features[k] - center[k]) / scales[k] for k in parts])
    matrices = [design(node) for node in nodes]
    # Repeatedly named expressions contribute once per sentence and unique exact coordinates.
    distribution, weights = [], []
    for node in nodes:
        grade = np.zeros(len(node.coords))
        for target in targets[node.key]:
            ious = overlap(node.coords, target)
            peaks = ious == ious.max()
            grade += peaks / peaks.sum()
        distribution.append(grade / len(targets[node.key]))
        weights.append(len(targets[node.key]))
    weights = np.array(weights) / sum(weights)

    def objective(w):
        grad, total = RIDGE * w.copy(), 0.5 * RIDGE * (w @ w)
        for x, q, a in zip(matrices, distribution, weights, strict=True):
            z = x @ w
            total += a * (logsumexp(z) - q @ z)
            grad += a * ((softmax(z) - q) @ x)
        return float(total), grad

    result = minimize(
        objective,
        np.zeros(matrices[0].shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-7, "maxiter": 250, "maxls": 40},
    )
    grad_norm = np.max(abs(result.jac))
    if not result.success or grad_norm > 2e-5:
        raise ValueError(f"Conditional span model failed convergence: {result.message}, {grad_norm}")
    params = dict(
        coef=result.x.tolist(),
        center={k: v.tolist() for k, v in center.items()},
        scale={k: v.tolist() for k, v in scales.items()},
        features=parts,
        updates=int(result.nit),
        maximum_absolute_gradient=float(grad_norm),
        ridge=RIDGE,
    )
    return lambda node: design(node) @ result.x, params


def targets_and_groups():
    source = RESULTS / "plaba-explicit-span-audit.json"
    doc = json.loads(source.read_bytes())
    if digest(source) != "0275ef4214df8c794862f0d0454ac6a72a03e84ecb4366d64f8bb796a3ba1874":
        raise ValueError("Unexpected named-span inventory")
    targets, groups = defaultdict(set), {}
    for row in doc["positives"]:
        key, group = row["sentence_id"], row["round1_reader_connected_group"]
        if groups.setdefault(key, group) != group:
            raise ValueError("Inconsistent reader/article grouping")
        for span in row["spans"]:
            targets[key].add((span["start"], span["end"]))
    if sum(map(len, targets.values())) != 94 or len(set(groups.values())) != 12:
        raise ValueError("Positive inventory differs from the audited endpoint")
    return targets, groups


def assay(out, gpu=None):
    nodes, source = build(gpu)
    targets, groups = targets_and_groups()
    docs = defaultdict(list)
    for node in nodes.values():
        docs[node.document].append(node)
    names = dict(
        (name, parts)
        for name, parts in MODELS.items()
        if all(k in nodes[next(iter(targets))].features for k in parts)
    )
    methods = list(names) + list(nodes[next(iter(targets))].raw)
    records, fits = [], {}
    for group in sorted(set(groups.values())):
        train = [nodes[k] for k in sorted(targets) if groups[k] != group]
        test = [nodes[k] for k in sorted(targets) if groups[k] == group]
        test_docs = sorted({node.document for node in test})
        if any(node.document in test_docs for node in train):
            raise ValueError("Article overlap across folds")
        predictions, fits[group] = {}, {}
        for method in methods:
            if method in names:
                predictor, data = fit(train, targets, names[method])
                fits[group][method] = data
            else:
                predictor = lambda node, label=method: node.raw[label]
            within, full = {}, {}
            for doc in test_docs:
                signals = [predictor(node) for node in docs[doc]]
                for node, score in zip(docs[doc], signals, strict=True):
                    within[node.key] = ranking(node.coords, score)
                full[doc] = ranking(np.vstack([node.coords for node in docs[doc]]), np.concatenate(signals))
            predictions[method] = within, full
        for node in test:
            for target in sorted(targets[node.key]):
                item = dict(sentence=node.key, document=node.document, group=group, target=target)
                for method, (within, full) in predictions.items():
                    s, d = within[node.key], full[node.document]
                    item[method] = dict(
                        sentence=by_target(s, target),
                        article=by_target(d, target),
                        top5_sentence=s[:5].tolist(),
                        top5_article=d[:5].tolist(),
                    )
                records.append(item)
    full_fit = {
        method: fit([nodes[k] for k in sorted(targets)], targets, parts)[1]
        for method, parts in names.items()
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [("predictions", records), ("fold-weights", fits), ("deployment-weights", full_fit)]:
        (out / (name + ".json")).write_text(json.dumps(value, separators=(",", ":")) + "\n")
    report = grouped_scores(records, methods)
    report.update(
        source=source,
        methods=methods,
        positive_spans=len(records),
        predictions_sha256=digest(out / "predictions.json"),
        gpu_sha256=digest(gpu / "seal.json") if gpu else None,
        historical_exposure=True,
    )
    (out / "scores.json").write_text(json.dumps(report, indent=2) + "\n")
    for m in methods:
        print(m, {level: report["estimates"][level]["hit5"][m] for level in ("sentence", "article")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assay(args.out, args.gpu)
