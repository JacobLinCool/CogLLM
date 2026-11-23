"""Compare Guo transfer forecasts to the other readers who rated the same displayed term."""

from collections import defaultdict
import json
import numpy as np
from research.reading_validation.local_features import OUT
from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.local_jargon import JARGON, outcomes, rank_stats


def run():
    root = OUT / "jargon"
    pred = root / "predictions.jsonl"
    lock = json.loads((root / "prediction-lock.json").read_bytes())
    if digest(pred) != lock["files_sha256"][pred.name]:
        raise ValueError("Unexpected independent transfer forecast file")
    group = defaultdict(dict)
    forecasts = {(r["paper"], r["term"]): r["scores"] for r in rows(pred)}
    records, _ = outcomes(JARGON / "source.csv", forecasts)
    for row in records:
        g = row["paper"], row["term"]
        if row["reader"] in group[g]:
            raise ValueError("Duplicated reader response")
        group[g][row["reader"]] = row
    result = {}
    for field in ("before", "after", "definition", "background"):
        matched = []
        for r in records:
            others = [
                j[field]
                for reader, j in group[r["paper"], r["term"]].items()
                if reader != r["reader"] and j[field] is not None
            ]
            if not others or r[field] is None:
                continue
            s = {k: r["scores"][k] for k in ("form", "local", "external_lexical")}
            s["other_readers"] = float(np.mean(others))
            matched.append(dict(r, scores=s))
        result[field] = rank_stats(matched, field, list(s))
    result["scope"] = (
        "After the primary scores were opened. Ten other raters supplied a retrospective consensus baseline conditional on this small panel."
    )
    result["locked_forecast_sha256"] = digest(root / "prediction-lock.json")
    (root / "human-reference.json").write_text(json.dumps(result, indent=2) + "\n")
    print({f: r["estimates"]["other_readers"]["mean"] for f, r in result.items() if isinstance(r, dict)})


if __name__ == "__main__":
    run()
