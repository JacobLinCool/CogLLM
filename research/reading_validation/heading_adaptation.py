"""Fit and cross-validate an article-level cue under crossed reader and article exclusion."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np

from research.expository_lora.score import ARMS, interval, kerwer_readers
from research.reading_validation.local_features import CACHE, DATA, OUT, verified_rows
from research.reading_validation.local_forecast import digest, rows


OUT_DIR = OUT / "native-headings"
HEADINGS = ("background", "why was this study done?", "what did the researchers do and find?",
            "what do these findings mean?")
METHODS = ("frozen", "heading_only", "native_and_heading")
SEED = 784562
DRAWS = 10000
BLOCKS = 5


def template_score(text):
    lines = [re.sub(r"^\d+\.\s*", "", l.strip()).casefold() for l in text.splitlines()]
    return int(tuple(l for l in lines if l in HEADINGS) == HEADINGS)


def data():
    ratings = CACHE / "kerwer-2021/holdout/scoring/public-study-ratings.jsonl"
    certified = json.loads((OUT_DIR / "scores.json").read_bytes())
    if digest(ratings) != certified["ratings_sha256"]:
        raise ValueError("Human endpoint has changed")
    yhat = OUT_DIR / "native-headings.jsonl"
    receipt = json.loads((OUT_DIR / "receipt.json").read_bytes())
    source = DATA / "native-heading-stimuli.jsonl"
    if digest(yhat) != receipt["predictions_sha256"] or digest(source) != receipt["stimuli_sha256"]:
        raise ValueError("The original native word or stimulus archive changed")
    text = {r["id"]: r["text"] for r in rows(source)}
    inputs = {r["id"]: r for r in rows(yhat)}
    sorted_cases = [r["presentations"] for r in certified["families"]]
    marker = np.array([[template_score(text[key]) for key in ids] for ids in sorted_cases])
    if not np.all(marker == np.broadcast_to([0, 1], marker.shape)) or marker.shape != (12, 2):
        raise ValueError("Visible organization cue differs from the published manipulation")
    # Original assignment tags verify grade-free parsing but never enter the fitted predictor.
    design = {r["id"]:r["arm"] for r in verified_rows("kerwer", "layout.jsonl")}
    if any([design[key] for key in ids] != list(ARMS) for ids in sorted_cases):
        raise ValueError("Source conditions are misaligned")
    states = np.asarray([[1 + 1.75 * (np.mean([s["rating"]["understanding"] for s in inputs[key]["sentences"]]) - 1)
                          for key in ids] for ids in sorted_cases])
    readers = np.array(sorted({r["reader"] for r in rows(ratings)}))
    order = sorted(range(len(readers)),
                   key=lambda i: hashlib.sha256(("cogllm-kerwer-article-fold/" + readers[i]).encode()).digest())
    memberships = np.empty(len(order), dtype=int)
    memberships[order] = np.arange(len(order)) % BLOCKS
    return kerwer_readers(ratings, sorted_cases), memberships, states, marker, [r["doi"] for r in certified["families"]]


def moments(y, who, reader_weights):
    """Sufficient statistics of individual 1–8 responses for each reader block and article arm."""
    all_stats = []
    for values in (y > 0, y, y ** 2):
        by_block = []
        for k in range(BLOCKS):
            use = who == k
            matrix = values[use].reshape(use.sum(), -1)
            by_block.append((reader_weights[:, use] @ matrix).reshape(-1, *y.shape[1:]))
        all_stats.append(np.stack(by_block, axis=1))
    return all_stats


def other_sources(target, article_weights):
    """Sum across training articles, independently omitting each currently held DOI."""
    w = article_weights[:, None, :]
    return (target * w).sum(axis=2, keepdims=True) - target * w


def evaluate(y, who, raw, visible, reader_weights, article_weights, detailed=False):
    counts, sums, squares = moments(y, who, reader_weights)
    eligible = ((counts.sum(axis=1)[:, None, :, :] - counts) > 0).all(axis=(1, 2, 3))
    if not eligible.all():
        if detailed:
            raise ValueError("A point estimate lacks training support")
        counts, sums, squares = (v[eligible] for v in (counts, sums, squares))
        article_weights = article_weights[eligible]
    total_n, total_y = counts.sum(axis=1), sums.sum(axis=1)
    train_n = total_n[:, None, :, :] - counts
    if np.any(train_n <= 0) or np.any(total_n == 0):
        raise ValueError("A reader-bootstrap draw omitted a training article condition")
    train_means = (total_y[:, None, :, :] - sums) / train_n
    test_means = total_y / total_n
    human = test_means[:, :, 1] - test_means[:, :, 0]
    # Design B × K × F × A: resample, held participant block, held source, assigned arm.
    family_weight = article_weights / article_weights.sum(axis=1, keepdims=True)
    remaining = (article_weights.sum(axis=1, keepdims=True) - article_weights)[:, None, :]
    by_block = counts.sum(axis=3)
    counterfactual_weight = by_block / by_block.sum(axis=1, keepdims=True)

    def combine(v):
        return np.sum(family_weight * v, axis=1)

    result = dict(human=combine(human))
    detail = {}
    human_center = result["human"][:, None]
    for method in METHODS:
        base = np.zeros_like(raw) if method == "heading_only" else raw
        if method == "frozen":
            forecasts = np.broadcast_to(base, train_means.shape)
            slope = np.zeros_like(train_means[:, :, :, 0])
            intercept = np.zeros_like(slope)
        else:
            residual = train_means - base
            difference = residual[:, :, :, 1] - residual[:, :, :, 0]
            slope = other_sources(difference, article_weights) / (remaining + 1)
            midpoint = other_sources(residual.mean(axis=3), article_weights) / remaining
            intercept = midpoint - slope / 2
            forecasts = base + intercept[..., None] + slope[..., None] * visible
        pred = np.sum(counterfactual_weight * (forecasts[..., 1] - forecasts[..., 0]), axis=1)
        err = pred - human
        expected = combine(pred)
        centered = (pred - expected[:, None]) - (human - human_center)
        sse = (squares - 2 * forecasts * sums + forecasts**2 * counts).sum(axis=(1, 3))
        n = total_n.sum(axis=2)
        individual_rmse = np.sqrt((sse * article_weights).sum(axis=1) / (n * article_weights).sum(axis=1))
        metrics = dict(mean=expected, minus_human=combine(err), absolute_article_error=combine(abs(err)),
                       centered_absolute_article_error=combine(abs(centered)), individual_rmse=individual_rmse)
        result.update({method + "." + k: v for k,v in metrics.items()})
        if detailed:
            detail[method] = dict(article_predictions=pred[0].tolist(), article_residuals=err[0].tolist(),
                                  slopes=slope[0].tolist(), intercepts=intercept[0].tolist())
    for k in ("absolute_article_error", "centered_absolute_article_error", "individual_rmse"):
        for comparator in ("frozen", "heading_only"):
            result["native_and_heading_minus_" + comparator + "." + k] = (
                result["native_and_heading." + k] - result[comparator + "." + k])
    return result, detail


def check_folds(y, who, raw, detail):
    """Replay every retained coefficient using literal row and article exclusion."""
    worst = 0.0
    retained = []
    for k in range(BLOCKS):
        people = who != k
        for f in range(y.shape[1]):
            articles = np.arange(y.shape[1]) != f
            train = y[people][:, articles, :]
            n = (train > 0).sum(axis=0)
            if not np.all(n > 0):
                raise ValueError("A source-only reference fit lacks a comparison arm")
            for name, scores in (("heading_only", np.zeros_like(raw)), ("native_and_heading", raw)):
                residual = train.sum(axis=0) / n - scores[articles]
                slope = (residual[:, 1] - residual[:, 0]).sum() / (len(residual) + 1)
                start = residual.mean() - slope / 2
                estimate = detail[name]
                worst = max(worst, abs(slope - estimate["slopes"][k][f]),
                            abs(start - estimate["intercepts"][k][f]))
            retained.append(int((train > 0).sum()))
    if worst > 1e-11:
        raise ValueError("The fast nested fitting equations differ from literal exclusion")
    return dict(distinct_test_folds=len(retained), estimated_methods=2,
                max_coefficient_difference=worst, min_training_ratings=min(retained),
                max_training_ratings=max(retained))


def main():
    y, who, raw, cues, dois = data()
    nr, nf, _ = y.shape
    base, detail = evaluate(y, who, raw, cues, np.ones((1,nr)), np.ones((1,nf)), detailed=True)
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(SEED).spawn(2)]
    draws = {k:[] for k in base}
    valid_draws = attempted = 0
    while valid_draws < DRAWS:
        size = min(DRAWS - valid_draws, 100)
        attempted += size
        rw = streams[0].multinomial(nr, np.full(nr,1/nr), size)
        fw = streams[1].multinomial(nf, np.full(nf,1/nf), size)
        part, _ = evaluate(y, who, raw, cues, rw, fw)
        valid_draws += len(part["human"])
        for k,v in part.items():
            draws[k].extend(v)
    estimates = {k:interval(v[0], draws[k]) for k,v in base.items()}
    frozen = json.loads((OUT_DIR/"scores.json").read_bytes())["native_vs_body"]
    if abs(estimates["frozen.mean"]["estimate"] - frozen["native"]["estimate"]) > 1e-10:
        raise ValueError("Raw native mean differs from signed archived result")
    report = dict(estimates=estimates, source_dois=dois, folds=detail, visible_template=cues.tolist(),
                  independent_coefficient_replay=check_folds(y, who, raw, detail),
                  per_block_readers=np.bincount(who).tolist(), participants=nr, ratings=int((y>0).sum()),
                  seed=SEED, resamples=DRAWS, undefined_resamples=attempted-DRAWS, protocol_sha256=digest(Path(__file__).with_name("heading-adaptation-protocol.md")),
                  code_sha256=digest(Path(__file__)), original_scores_sha256=digest(OUT_DIR/"scores.json"))
    (OUT_DIR/"crossed-adaptation.json").write_text(json.dumps(report,indent=2)+"\n")
    for k,v in estimates.items():
        print(f"{k:70s}{v['estimate']: .4f}  {v['ci95']}")


if __name__ == "__main__":
    main()
