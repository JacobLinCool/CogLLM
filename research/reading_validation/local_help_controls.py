"""Seal and verify a public, response-free native-L4 replay of the Guo feature contract."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.local_help_inference import (
    COLUMNS,
    DESIGNS,
    HelpReadout,
)
from research.reading_validation.local_help_prepare import CACHE, OUT, PROJECT, code

FIXTURE = CACHE / "guo-help-controls.jsonl"
REFERENCE = OUT / "l4-control-reference.json"
PACKAGE = OUT / "l4-worker-package.json"
FEATURE_TOLERANCES = (0.0002, 1e-9, 1e-9, 1e-9, 1e-9, 0.005, 0.005, 0.015, 0.005)
PROBABILITY_TOLERANCE = 0.003
RESOURCES = [
    "research/results/reading-validation-local/serving/manifest.json",
    "research/results/reading-validation-local/serving/lexicon.json.gz",
    "research/results/reading-validation-local/serving/weights.npz",
    "research/results/reading-validation-local/localization/deployment-weights.json",
    "research/results/reading-validation-local/jargon/prediction-lock.json",
    "research/results/reading-validation-local/help-choice/geometry.json",
    "research/results/reading-validation-definitions/weights.json",
    "research/results/reading-validation-definitions/forecast-seal.json",
]


def prepare():
    with (OUT / "features.jsonl").open() as f:
        reference = defaultdict(dict)
        for line in f:
            r = json.loads(line)
            reference[r["paper"]][r["term_key"]] = r["features"]
    texts = {r["id"]: r["text"] for r in rows(CACHE / "jargon/local-stimuli.jsonl")}
    queries = {r["id"]: r["terms"] for r in rows(CACHE / "jargon/geometry.jsonl")}
    by_length = sorted((len(texts[p]), p) for p in reference if len(reference[p]) >= 3)
    papers = [by_length[round(f * (len(by_length) - 1))][1] for f in (0.1, 0.5, 0.9)]
    controls, expected = [], []
    for p in papers:
        terms = [
            {
                "id": code(t["term"]),
                "term": t["term"],
                "start": t["start"],
                "end": t["end"],
            }
            for t in queries[p]
            if code(t["term"]) in reference[p]
        ]
        if set(t["id"] for t in terms) != set(reference[p]):
            raise ValueError("Unmapped Guo geometry")
        controls.append({"id": p, "text": texts[p], "terms": terms})
        expected.append({"id": p, "terms": reference[p]})
    FIXTURE.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in controls)
    )
    REFERENCE.write_text(
        json.dumps(
            {
                "fixture_sha256": digest(FIXTURE),
                "source_features_sha256": digest(OUT / "features.jsonl"),
                "columns": COLUMNS,
                "feature_absolute_tolerances": FEATURE_TOLERANCES,
                "probability_absolute_tolerance": PROBABILITY_TOLERANCE,
                "selection": "10th, 50th, and 90th percentiles of character length among non-reserve Guo papers with at least three eligible terms, ties broken by article ID",
                "articles": expected,
            },
            indent=2,
        )
        + "\n"
    )
    worker = Path(__file__).with_name("local_help_l4.py")
    solver = Path(__file__).with_name("local_help_article.py")
    PACKAGE.write_text(
        json.dumps(
            {
                "entry": "research.reading_validation.local_help_l4.run(runtime, reader, checkpoint, input_path, output_dir)",
                "runtime": "pinned Qwen3.5-2B Backbone(native_attention=True)",
                "worker_sha256": digest(worker),
                "converter_sha256": digest(solver),
                "worker_source": str(worker.relative_to(PROJECT))
                if worker.is_absolute()
                else str(worker),
                "converter_source": str(solver.relative_to(PROJECT))
                if solver.is_absolute()
                else str(solver),
                "resource_sha256": {s: digest(PROJECT / s) for s in RESOURCES},
                "controls": str(FIXTURE.relative_to(PROJECT)),
                "controls_sha256": digest(FIXTURE),
                "reference_sha256": digest(REFERENCE),
                "external_input": ".cache/cogllm/reading-validation/relay/local-stimuli.jsonl",
                "controls_papers": len(expected),
                "controls_terms": sum(len(p["terms"]) for p in expected),
            },
            indent=2,
        )
        + "\n"
    )
    print(PACKAGE, digest(PACKAGE))


def verify(features):
    ref = json.loads(REFERENCE.read_bytes())
    extraction = json.loads((features.parent / "extraction.json").read_bytes())
    if (
        digest(features) != extraction["output_sha256"]
        or extraction["data_sha256"] != ref["fixture_sha256"]
        or extraction["backbone_forwards"] != len(ref["articles"])
    ):
        raise ValueError(
            "Returned L4 controls differ from the prespecified isolated fixture"
        )
    observed = {r["id"]: r for r in rows(features)}
    if set(observed) != {r["id"] for r in ref["articles"]}:
        raise ValueError("Supply isolated Guo control output only")
    old, new = [], []
    for paper in ref["articles"]:
        real = observed[paper["id"]]
        if real["excluded_terms"] or real["length_domain"] != "source":
            raise ValueError("Eligible Guo target unexpectedly lost")
        made = {t["id"]: t["features"] for t in real["terms"]}
        if set(made) != set(paper["terms"]):
            raise ValueError("Target IDs differ from gold fit matrix")
        for t, vec in paper["terms"].items():
            new.append(made[t])
            old.append(vec)
    old, new = np.asarray(old), np.asarray(new)
    delta = np.abs(old - new).max(axis=0)
    model = HelpReadout()
    worst = 0.0
    for prior in range(5):
        known = np.full(len(old), prior)
        for endpoint in model.weights:
            for arm in DESIGNS:
                kwargs = {"endpoint": endpoint, "arm": arm, "familiarity_before": known}
                worst = max(
                    worst,
                    float(
                        abs(
                            model.predict(old, **kwargs)["probability"]
                            - model.predict(new, **kwargs)["probability"]
                        ).max()
                    ),
                )
    passed = bool(
        np.all(delta <= ref["feature_absolute_tolerances"])
        and worst <= ref["probability_absolute_tolerance"]
    )
    out = {
        "passed": passed,
        "documents": len(observed),
        "terms": len(old),
        "passes": extraction["backbone_forwards"],
        "max_abs_probability_difference": worst,
        "max_abs_feature_differences": dict(zip(COLUMNS, delta, strict=True)),
        "remote_extraction": extraction,
        "reference_sha256": digest(REFERENCE),
    }
    (OUT / "l4-control-replay.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    if not passed:
        raise ArithmeticError("Numerical parity exceeded prespecified tolerances")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-controls", type=Path)
    args = parser.parse_args()
    if args.verify_controls:
        verify(args.verify_controls)
    else:
        prepare()
