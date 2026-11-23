"""Replay all non-reserve Guo feature rows and forecasts from archived native components."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.reading_validation.definition_features import Corpus
from research.reading_validation.joint_data import digest, lines
from research.reading_validation.local_features import read_gpu
from research.reading_validation.local_help_article import ArticleHelpFeatures
from research.reading_validation.local_help_inference import (
    COLUMNS,
    DESIGNS,
    HelpReadout,
)
from research.reading_validation.local_help_prepare import CACHE, DEFT, OLD, OUT, code


def run(deft, native):
    converter = ArticleHelpFeatures()
    model = HelpReadout()
    papers = {r["id"]: r["text"] for r in lines(CACHE / "jargon/local-stimuli.jsonl")}
    by_paper = {r["id"]: r["terms"] for r in lines(CACHE / "jargon/geometry.jsonl")}
    old = read_gpu(
        native, CACHE / "jargon/local-stimuli.jsonl", OLD / "source-seal.json"
    )
    codes = json.loads((DEFT / "external-keys.json").read_bytes())["jargon"]
    bank = Corpus(deft)
    rows = {
        (r["paper"], r["term_key"]): r["features"]
        for r in lines(OUT / "features.jsonl")
    }
    papers_used = sorted({p for p, _ in rows})
    deltas = np.zeros(len(COLUMNS))
    locked, rebuilt = [], []
    for paper in papers_used:
        qs = {t["term"]: (t["term"], t["start"], t["end"]) for t in by_paper[paper]}
        values, excluded = converter.from_components(
            papers[paper], qs, old[paper], bank.page(codes[paper])
        )
        if excluded or len(values) != len(qs):
            raise ValueError("Known eligible terms were lost")
        for term, x in values.items():
            ref = rows[paper, code(term)]
            locked.append(ref)
            rebuilt.append(x)
            deltas = np.maximum(deltas, np.abs(x - ref))
    ref, x = np.asarray(locked), np.asarray(rebuilt)
    if len(x) != len(rows) or deltas.max() > 1e-5:
        raise ArithmeticError(
            f"New feature generator differs from the native reference: {deltas}"
        )
    # All five known prior ratings, all response types and both known- and unknown-reader settings.
    largest = 0.0
    for prior in range(5):
        known = np.full(len(x), prior)
        for arm in DESIGNS:
            for endpoint in model.weights:
                oldp = model.predict(
                    ref, arm=arm, endpoint=endpoint, familiarity_before=known
                )["probability"]
                newp = model.predict(
                    x, arm=arm, endpoint=endpoint, familiarity_before=known
                )["probability"]
                largest = max(largest, float(np.max(np.abs(oldp - newp))))
    result = {
        "terms": len(rows),
        "articles": len(papers_used),
        "outcome_file_opened": False,
        "max_abs_feature_difference": dict(zip(COLUMNS, deltas, strict=True)),
        "max_abs_response_probability_difference": largest,
        "weights_sha256": digest(OUT / "weights.json"),
        "converter_sha256": digest(Path(__file__).with_name("local_help_article.py")),
    }
    (OUT / "native-feature-replay.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deft", type=Path, required=True)
    p.add_argument("--local", type=Path, required=True)
    args = p.parse_args()
    run(args.deft, args.local)
