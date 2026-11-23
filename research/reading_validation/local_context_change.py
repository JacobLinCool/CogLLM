"""Evaluate sealed term forecasts within exact reader, paper and prereading grade matches."""

from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from research.reading_validation.local_features import OUT
from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.local_jargon import JARGON, outcomes, two_way


def run():
    root = OUT / "jargon"
    lock = json.loads((root / "prediction-lock.json").read_bytes())
    path = root / "predictions.jsonl"
    if digest(path) != lock["files_sha256"][path.name]:
        raise ValueError("The primary transfer predictions changed")
    predictions = {(r["paper"], r["term"]): r["scores"] for r in rows(path)}
    records, _ = outcomes(JARGON / "source.csv", predictions)
    groups = defaultdict(list)
    papers = sorted({r["paper"] for r in records})
    readers = sorted({r["reader"] for r in records})
    for r in records:
        if r["before"] is not None and r["after"] is not None:
            groups[r["paper"], r["reader"]].append(r)
    methods = [
        "form",
        "medical_form",
        "local",
        "external_lexical",
        "displayed_mass",
        "local_and_persistent",
    ]
    cube = np.zeros((len(papers), len(readers), len(methods)))
    eligible = np.zeros((len(papers), len(readers)), dtype=bool)
    pairs = aligned = 0
    for (paper, reader), records in groups.items():
        observed = []
        for a, b in combinations(records, 2):
            if a["before"] != b["before"]:
                continue
            aligned += 1
            delta = a["after"] - b["after"]
            if delta == 0:
                continue
            scores = np.array([a["scores"][m] - b["scores"][m] for m in methods])
            observed.append(0.5 + 0.5 * np.sign(delta) * np.sign(scores))
        if observed:
            i, j = papers.index(paper), readers.index(reader)
            eligible[i, j] = True
            cube[i, j] = np.mean(observed, axis=0)
            pairs += len(observed)
    point, boot, fixed = two_way(cube, eligible)

    def summary(p, b, t):
        return dict(
            estimate=float(p),
            crossed_95=np.quantile(b, [0.025, 0.975]).tolist(),
            fixed_reader_95=np.quantile(t, [0.025, 0.975]).tolist(),
        )

    table = {m: summary(point[i], boot[:, i], fixed[:, i]) for i, m in enumerate(methods)}
    for a, b in [("local", "form"), ("external_lexical", "form"), ("local", "external_lexical")]:
        i, j = methods.index(a), methods.index(b)
        table[a + " minus " + b] = summary(
            point[i] - point[j], boot[:, i] - boot[:, j], fixed[:, i] - fixed[:, j]
        )
    report = dict(
        scope="Secondary analysis specified after inspecting the Guo primary transfer.",
        paired_same_before=aligned,
        unequal_after_pairs=pairs,
        reader_article_units=int(eligible.sum()),
        papers=int(eligible.any(1).sum()),
        readers=int(eligible.any(0).sum()),
        forecast_sha256=digest(root / "prediction-lock.json"),
        protocol_sha256=digest(Path(__file__).with_name("local-context-change.md")),
        estimates=table,
    )
    (root / "matched-start.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        {k: report[k] for k in ("unequal_after_pairs", "reader_article_units", "papers", "readers")},
        table["external_lexical"],
        table["local"],
    )


if __name__ == "__main__":
    run()
