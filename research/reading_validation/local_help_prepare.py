"""Prepare outcome-free explanatory and lexical scores for designated science terms."""

import hashlib
import json
from pathlib import Path

import numpy as np

from research.reading_validation.joint_data import digest, lines
from research.reading_validation.local_forecast import WORD
from research.reading_validation.local_help_inference import COLUMNS

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CACHE = PROJECT / ".cache/cogllm/reading-validation"
RESULTS = HERE.parent / "results"
OLD = RESULTS / "reading-validation-local/jargon"
DEFT = CACHE / "definitions"
OUT = RESULTS / "reading-validation-local/help-choice"


def fold(tag, paper):
    return int.from_bytes(hashlib.sha256((tag + paper).encode()).digest(), "big") % 5


def code(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    frozen = json.loads((OLD / "prediction-lock.json").read_bytes())
    if digest(OLD / "predictions.jsonl") != frozen["files_sha256"]["predictions.jsonl"]:
        raise ValueError("Local phrase forecasts changed")
    inputs = {r["id"]: r["text"] for r in lines(CACHE / "jargon/local-stimuli.jsonl")}
    base = {(r["paper"], r["term"]): r for r in lines(OLD / "predictions.jsonl")}
    mapping = json.loads((DEFT / "external-keys.json").read_bytes())["jargon"]
    queries = {
        r["id"]: r["query_terms"]
        for r in lines(DEFT / "probe-layout-v2.jsonl")
        if r["kind"] == "jargon"
    }
    checks = json.loads(
        (RESULTS / "reading-validation-definitions/forecast-seal.json").read_bytes()
    )
    rows = []
    reserved = sorted({p for p, _ in base if fold("guo-help-reserve-v1/", p) == 0})
    for paper, h in sorted(mapping.items()):
        if paper in reserved:
            continue
        path = DEFT / "forecasts" / (h + ".npz")
        if digest(path) != checks["predictions_sha256"][h]:
            raise ValueError("DEFT predictions changed")
        text = inputs[paper]
        with np.load(path, allow_pickle=False) as z:
            start = z["right"]
            rows_in = sorted(queries[h].items())
            for idx, (term, span) in enumerate(rows_in):
                if (paper, term) not in base:
                    continue
                row = base[paper, term]
                left, right = span
                if (left, right) != (row["start"], row["end"]):
                    raise ValueError(
                        "Prior score and DEFT target character span differ"
                    )
                j = z["qidx"][idx]
                if j < 0:
                    continue
                later = int(start[j]) + 1
                old = row["scores"]
                scores = [
                    old["external_lexical"],
                    old["form"],
                    np.log1p(len(WORD.findall(term))),
                    np.log1p(len(term)),
                    left / len(text),
                    float(max(z["surface_qprob"][idx, later:], default=0)),
                    old["local"],
                    float(z["semantic_term"][j]),
                    float(max(z["semantic_qprob"][idx, later:], default=0)),
                ]
                if not np.isfinite(scores).all():
                    raise ArithmeticError("Nonfinite features")
                rows.append(
                    {
                        "paper": paper,
                        "term_key": code(term),
                        "article_fold": fold("guo-help-fit-v1/", paper),
                        "start": left,
                        "end": right,
                        "features": [float(x) for x in scores],
                    }
                )
    expect = {(p, code(t)) for p, t in base if p not in reserved}
    if {(r["paper"], r["term_key"]) for r in rows} != expect:
        raise ValueError("Incomplete joined term inventory")
    path = OUT / "features.jsonl"
    path.write_text(
        "".join(
            json.dumps(r) + "\n"
            for r in sorted(rows, key=lambda v: (v["paper"], v["start"]))
        )
    )
    result = {
        "protocol_sha256": digest(HERE / "local-help-protocol.md"),
        "source_sha256": digest(CACHE / "jargon/source.csv"),
        "prior_guo_seal_sha256": digest(OLD / "prediction-lock.json"),
        "prior_deft_seal_sha256": digest(
            RESULTS / "reading-validation-definitions/forecast-seal.json"
        ),
        "columns": COLUMNS,
        "features_sha256": digest(path),
        "terms": len(rows),
        "available_articles": len({p for p, _ in base}),
        "reserved": reserved,
        "used_articles": len({r["paper"] for r in rows}),
        "script_sha256": digest(Path(__file__)),
    }
    (OUT / "geometry.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["terms"], result["used_articles"], len(reserved))


if __name__ == "__main__":
    main()
