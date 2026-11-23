"""Resample distinct released works while keeping pronoun comparisons inside each excerpt."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_direct_score import concordance
from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_game_prepare import OUT, ROOT
from research.reading_validation.transition_score import dump, sha, with_interval


DIRECT = Path("research/results/reading-validation-transition-direct")
PROTOCOL = Path("research/reading_validation/transition-gutenberg-clusters.md")


def combine(rows, methods, grouping, same_form):
    result, per_method = {}, {}
    for method in methods:
        by_text, counts = concordance(rows, method, "antecedent_disagreement", same_form)
        by_work = defaultdict(list)
        for text, point in by_text.items():
            by_work[grouping[text]].append(point)
        works = sorted(by_work)
        _, _, draws = grouped_distribution(works)
        values = np.array([np.mean(by_work[work]) for work in works])
        estimates = draws @ values / len(works)
        point = values.mean()
        result[method] = with_interval(point, estimates)
        per_method[method] = point, estimates
    differences = {}
    for model in ("direct", "shared"):
        point, boot = per_method[model]
        for other in ("surface", "direct"):
            if other == model:
                continue
            reference, ref_boot = per_method[other]
            differences[f"{model}_minus_{other}"] = with_interval(point - reference, boot - ref_boot)
    return {"texts": len(by_text), "works": len(works), "within_text_eligible_pairs": sum(counts.values()),
            "concordance": result, "paired": differences}


def run():
    packets = {r["id"]: r for r in map(json.loads, (ROOT / "game-native-stimuli.jsonl").read_text().splitlines())
               if r["domain"] == "game" and r["fold"] == "test"}
    grouping = {article: row["group"] for article, row in packets.items()}
    annotations = json.loads((DIRECT / "external-reader-outcomes.json").read_text())["rows"]
    primary = [r for r in annotations if r["antecedent_disagreement"] is not None]
    models = {}
    for candidate in ("gemma4", "qwen", "lfm3"):
        folder = DIRECT / candidate
        meta = json.loads((folder / "receipt.json").read_text())
        file = folder / "word-probabilities.npz"
        if sha(file) != meta["predictions_sha256"]:
            raise ValueError("A native forecast changed")
        methods = [k for k in meta["arms"] if k != "ner"]
        with np.load(file) as probabilities:
            cached = {d: probabilities[d] for d in {r["doc"] for r in primary}}
        rows = [{**r, **{m: float(cached[r["doc"]][r["word"], meta["arms"].index(m), meta["tasks"].index("CorefE")])
                         for m in methods}} for r in primary]
        models[candidate] = {"forecast_sha256": sha(file), "within_text": combine(rows, methods, grouping, False),
                             "same_form_within_text": combine(rows, methods, grouping, True)}
    dump(DIRECT / "distinct-work-pronoun-sensitivity.json", {"protocol_sha256": sha(PROTOCOL), "candidates": models})
    if not (OUT / "scores.json").is_file():
        return
    meta = json.loads((OUT / "receipt.json").read_text())
    file = OUT / "game-forecasts.npz"
    if sha(file) != meta["forecasts_sha256"]:
        raise ValueError("The supervised game forecast changed")
    with np.load(file) as probabilities:
        cached = {d: dict(zip(row["save"], probabilities[d], strict=True)) for d, row in packets.items()}
    methods = meta["arms"]
    rows = [{**r, **{m: float(cached[r["doc"]][r["word"]][methods.index(m)]) for m in methods}} for r in primary]
    dump(OUT / "distinct-work-pronoun-sensitivity.json", {"protocol_sha256": sha(PROTOCOL), "forecast_sha256": sha(file),
         "within_text": combine(rows, methods, grouping, False), "same_form_within_text": combine(rows, methods, grouping, True)})


if __name__ == "__main__":
    run()
