"""Map raw-scale ratings and original target-word times to blinded material identifiers."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.reading_validation.transition_source import CACHE, check, SOURCE


def load():
    manifest = json.loads(SOURCE.read_text())["files"]
    for name in ("160_coh_ratings.csv", "exp1_prereg_data.csv"):
        check(CACHE / name, manifest[name])
    return pd.read_csv(CACHE / "160_coh_ratings.csv"), pd.read_csv(CACHE / "exp1_prereg_data.csv", low_memory=False)


def prepare(metadata, packets):
    ratings, times = load()
    ratings = ratings[ratings.cond.isin(list("abcd"))].rename(columns={"trial_num": "item", "cond": "condition"})
    if ratings.duplicated(["item", "condition"]).any() or not ratings.trial_mean_rating.between(1, 5).all():
        raise ValueError("Ratings require one observed original-scale mean per stimulus")
    rows = metadata.merge(ratings[["item", "condition", "trial_mean_rating"]], validate="1:1")
    if len(rows) != len(metadata):
        raise ValueError("Missing rating target")
    times = times[times["type"].eq("target")][["participant", "trial_num", "cond", "rt", "pre_crit", "stimulus"]].copy()
    times.rename(columns={"trial_num": "item", "cond": "condition", "rt": "rt_ms"}, inplace=True)
    # Original trial order within a participant is unchanged in the released CSV.
    times["word_index"] = times.groupby(["participant", "item", "condition"]).cumcount()
    times = times.merge(rows[["item", "condition", "id", "split", "critical_index"]], validate="m:1")
    if times.empty or not times.rt_ms.gt(0).all():
        raise ValueError("Online outcomes require positive word latencies")
    expected = {}
    for p in packets:
        for k, (s, e) in enumerate(p["target_words"]):
            expected[p["id"], k] = p["text"][s:e]
    normalized = lambda w: " ".join(str(w).casefold().replace("’", "'").split())
    actual = times[["id", "word_index", "stimulus"]].drop_duplicates()
    misaligned = [(r.id, r.word_index, r.stimulus, expected.get((r.id, r.word_index)))
                  for r in actual.itertuples() if normalized(r.stimulus) != normalized(expected.get((r.id, r.word_index)))]
    if misaligned:
        raise ValueError(f"Original presentation alignment failed for {len(misaligned)} positions. First: {misaligned[:2]}")
    if not (times.pre_crit.eq(1) == times.word_index.lt(times.critical_index)).all():
        raise ValueError("Public precritical flags disagree with the experimental minimal pairs")
    pre = times[times.pre_crit.eq(1)].copy()
    clip = pre[pre.split.eq("calibration")].groupby(["participant","condition"]).rt_ms.agg(["mean","std"])
    pre = pre.join(clip, on=["participant","condition"], validate="m:1")
    if pre[["mean","std"]].isna().any().any():
        raise ValueError("Missing participant-condition winsorisation limits")
    clipped = pre.rt_ms.clip(pre["mean"] - 2 * pre["std"], pre["mean"] + 2 * pre["std"])
    pre["log_ms"] = np.log(clipped)
    trials = pre.groupby(["id","participant"], as_index=False).agg(log_ms=("log_ms","mean"), words=("log_ms","size"))
    expected_counts = rows.set_index("id").critical_index
    if not trials.words.eq(trials.id.map(expected_counts)).all():
        raise ValueError("Partially observed precritical trials cannot be merged without censoring")
    summary = trials.groupby("id", as_index=False).agg(log_ms=("log_ms","mean"), readers=("participant","size"))
    rows = rows.merge(summary, how="left", validate="1:1")
    missing = rows.loc[rows.log_ms.isna(), ["item", "condition", "split"]].to_dict("records")
    summary = {"rater_scale":[1,5], "timed_participants": int(pre.participant.nunique()),
               "time_missing_cells": missing, "timed_trials": len(trials), "timed_precritical_words": len(pre),
               "holdout_timed_trials": int(rows[rows.split.eq("holdout")].readers.sum()),
               "holdout_timed_precritical_words": int(pre.split.eq("holdout").sum()),
               "rt_units":"mean natural log milliseconds per pre-critical word, equal weight per reader per item",
               "original_exclusions":"released Experiment 1 pre-registration preprocessing",
               "word_time_clipping":"reader and condition mean ±2 sample standard deviations from training families"}
    return rows, pre, summary
