"""Apply externally fitted PLABA readouts to Kerwer's same-body heading intervention."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.expository_lora.score import ARMS, kerwer_effects, kerwer_readers, interval
from research.reading_validation.local_features import CACHE, OUT, PACKET, RESULTS, build, verified_rows
from research.reading_validation.local_forecast import digest, rows


def evaluate(features):
    nodes, _ = build(features)
    source = PACKET / "source-audit.json"
    endpoint = CACHE / "kerwer-2021/holdout/scoring/public-study-ratings.jsonl"
    if digest(endpoint) != json.loads(source.read_bytes())["sources"]["kerwer"]["source_ratings_sha256"]:
        raise ValueError("External Kerwer responses differ from the accepted original source")
    fitted = OUT / "ratings/deployment-weights.json"
    archive = json.loads(fitted.read_bytes())
    historical = RESULTS / "expository-lora-l4/predictions.jsonl"
    old = {
        (r["id"], s): np.asarray(p[1]) @ np.arange(1, 6)
        for r in rows(historical)
        if r["dataset"] == "kerwer" and r["model"] == "frozen"
        for s, p in zip(r["stops"], r["probabilities"], strict=True)
    }
    models = ("text_form", "medical", "local", "local_and_persistent")
    families = defaultdict(dict)
    predictions = []
    for row in verified_rows("kerwer", "layout.jsonl"):
        key = row["id"]
        families[row["doi"]][row["arm"]] = key
        sentences = []
        for stop in row["matched_stops"]:
            s = dict(nodes[f"kerwer/{key}@{stop}"].sentence)
            s["surface"] = s["surface"][[0, 1, 2, 4, 5, 6]]
            results = {"historical_frozen": float(old[key, stop])}
            for m in models:
                params = archive[m]
                norm = params["normalization"]
                x = np.concatenate(
                    [(s[f] - norm[f]["mean"]) / norm[f]["scale"] for f in params["features"]]
                )
                coef = np.asarray(params["coef"])[:, 1]
                logits = (x @ coef) * (np.arange(1, 6) - 3) + params["offset"][1]
                p = np.exp(logits - logits.max())
                results[m] = float(p @ np.arange(1, 6) / p.sum())
            sentences.append(dict(stop=stop, forecasts=results))
        predictions.append(dict(id=key, doi=row["doi"], arm=row["arm"], sentences=sentences))
    if len(families) != 12 or sum(len(d["sentences"]) for d in predictions) != 362:
        raise ValueError("Matched Kerwer assignment geometry changed")
    cases = [[arms[name] for name in ARMS] for _, arms in sorted(families.items())]
    target = OUT / "headings"
    target.mkdir(parents=True, exist_ok=True)
    path = target / "predictions.json"
    path.write_text(json.dumps(predictions, separators=(",", ":")) + "\n")
    receipts = dict(
        source_file=digest(source),
        source_ratings=digest(endpoint),
        projection_input=digest(features / "seal.json"),
        plaba_coefficients=digest(fitted),
        historical_forecasts=digest(historical),
        predictions_sha256=digest(path),
        protocol_sha256=digest(Path(__file__).with_name("local-heading-protocol.md")),
    )
    observed = kerwer_readers(endpoint, cases)
    by_document = {d["id"]: d["sentences"] for d in predictions}
    estimates = {}
    for m in (*models, "historical_frozen"):
        contrast = []
        for left, right in cases:
            if len(by_document[left]) != len(by_document[right]):
                raise ValueError("Source assignments have different matched body-sentence counts")
            contrast.append(
                (
                    np.mean([r["forecasts"][m] for r in by_document[right]])
                    - np.mean([r["forecasts"][m] for r in by_document[left]])
                )
                * 7
                / 4
            )
        points, boot, _ = kerwer_effects(observed, np.column_stack((contrast, contrast)))
        means = points.mean(axis=0)
        estimates[m] = {
            k: interval(means[i], boot[:, i])
            for k, i in (
                ("human_effect", 0),
                ("model_effect", 1),
                ("model_minus_human", 4),
                ("mean_absolute_item_error", 6),
            )
        }
    result = dict(
        receipts=receipts,
        scale=[1, 8],
        families=len(cases),
        readers=len(observed),
        valid_article_responses=int((observed > 0).sum()),
        matched_sentence_pairs=181,
        estimates=estimates,
        full_state_local="Full native state archive unavailable for Kerwer; reduced local model only.",
    )
    (target / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print({m: v["model_effect"]["estimate"] for m, v in estimates.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=Path, required=True)
    evaluate(parser.parse_args().gpu)
