"""One disjoint, nonrotating cohort test of an article-specific organization readout."""

import hashlib
import json
from pathlib import Path
import numpy as np

from research.expository_lora.score import interval
from research.reading_validation.heading_adaptation import OUT_DIR, METHODS, data
from research.reading_validation.local_features import CACHE
from research.reading_validation.local_forecast import digest, rows


def divide(keys, unit):
    prefix = f"cogllm-kerwer-single-split/{unit}/"
    order = sorted(range(len(keys)), key=lambda i: hashlib.sha256((prefix + keys[i]).encode()).digest())
    middle = len(order) // 2
    return np.array(order[:middle]), np.array(order[middle:])


def summaries(y, mult):
    matrices = [(mult @ a.reshape(len(y), -1)).reshape(len(mult), *y.shape[1:]) for a in (y>0, y, y**2)]
    return matrices


def estimate(y_train, y_test, raw_train, raw_test, h_test, reader_train, reader_test, article_train, article_test):
    n_train, s_train, _ = summaries(y_train, reader_train)
    n_test, s_test, q_test = summaries(y_test, reader_test)
    good = ((n_train > 0) | (article_train[...,None] == 0)).all(axis=(1,2))
    good &= ((n_test > 0) | (article_test[...,None] == 0)).all(axis=(1,2))
    n_train, n_test, s_train, s_test, q_test, article_train, article_test = (
        a[good] for a in (n_train,n_test,s_train,s_test,q_test,article_train,article_test))
    y_fit = s_train / np.maximum(n_train, 1)
    y_held = s_test / np.maximum(n_test, 1)
    human = y_held[...,1] - y_held[...,0]
    nf = article_train.sum(axis=1)
    def test_mean(a):
        return (a * article_test).sum(axis=1) / article_test.sum(axis=1)
    result = dict(human=test_mean(human))
    for method in METHODS:
        if method == "frozen":
            pred = np.broadcast_to(raw_test, y_held.shape)
        else:
            base_train, base_test = (raw_train, raw_test) if method == "native_and_heading" else (0.0, 0.0)
            residual = y_fit - base_train
            d = residual[...,1] - residual[...,0]
            slope = (article_train * d).sum(axis=1) / (nf + 1)
            intercept = (article_train * residual.mean(axis=2)).sum(axis=1) / nf - slope / 2
            pred = base_test + intercept[:,None,None] + slope[:,None,None] * h_test
        effect = pred[...,1] - pred[...,0]
        err = effect - human
        sse = q_test - 2 * pred * s_test + pred**2 * n_test
        rmse = np.sqrt((article_test * sse.sum(axis=2)).sum(axis=1) / (article_test * n_test.sum(axis=2)).sum(axis=1))
        for k, v in dict(mean=test_mean(effect), minus_human=test_mean(err), absolute_article_error=test_mean(abs(err)),
                         individual_rmse=rmse).items():
            result[f"{method}.{k}"] = v
    for k in ("individual_rmse","absolute_article_error"):
        for baseline in ("frozen","heading_only"):
            result[f"native_and_heading_minus_{baseline}.{k}"] = (
                result[f"native_and_heading.{k}"] - result[f"{baseline}.{k}"])
    return result


def main():
    y, _, states, markers, dois = data()
    readers = sorted({r["reader"] for r in rows(CACHE / "kerwer-2021/holdout/scoring/public-study-ratings.jsonl")})
    fit_people, held_people = divide(readers, "reader")
    fit_articles, held_articles = divide(dois, "doi")
    y_fit, y_test = y[fit_people][:,fit_articles], y[held_people][:,held_articles]
    fixed = y_fit, y_test, states[fit_articles], states[held_articles], markers[held_articles]
    sizes = len(fit_people),len(held_people),len(fit_articles),len(held_articles)
    point = estimate(*fixed, *(np.ones((1,n)) for n in sizes))
    if len(point["human"]) != 1:
        raise ValueError("The prescribed original fixed split has an unobserved comparison arm")
    seed, count = 14817, 10000
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    samples = {k:[] for k in point}
    attempts = 0
    while len(samples["human"]) < count:
        b = min(100, count - len(samples["human"]))
        weights = [stream.multinomial(n, np.full(n,1/n), b) for n,stream in zip(sizes,streams, strict=True)]
        batch = estimate(*fixed, *weights)
        attempts += b
        for k,v in batch.items():
            samples[k].extend(v)
    estimates = {k:interval(v[0],samples[k]) for k,v in point.items()}
    detail = dict(train_dois=[dois[i] for i in fit_articles], test_dois=[dois[i] for i in held_articles],
                  train_n=len(fit_people), test_n=len(held_people), train_ratings=int((y_fit>0).sum()),
                  test_ratings=int((y_test>0).sum()), excluded_cross_ratings=int((y>0).sum()-(y_fit>0).sum()-(y_test>0).sum()),
                  source_arm_n=(y_fit>0).sum(axis=0).tolist(), held_arm_n=(y_test>0).sum(axis=0).tolist())
    obj = dict(split=detail, estimates=estimates, seed=seed, resamples=count, undefined_resamples=attempts-count,
               protocol_sha256=digest(Path(__file__).with_name("heading-fixed-split-protocol.md")),
               code_sha256=digest(Path(__file__)))
    (OUT_DIR / "independent-rectangle.json").write_text(json.dumps(obj,indent=2)+"\n")
    print(detail)
    for k,v in estimates.items():
        print(k, f"{v['estimate']:+.4f}", v["ci95"])


if __name__ == "__main__":
    main()
