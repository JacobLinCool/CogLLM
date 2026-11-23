"""Verify the shipped source ensemble on every archived, independently held essay stop."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from research.reading_validation.essay_hazard_inference import EssayHazardInference, WEIGHTS
from research.reading_validation.transition_essay_reasons import DEST
from research.reading_validation.transition_external_fit import read_features
from research.reading_validation.transition_external_prepare import csv_rows
from research.reading_validation.transition_score import dump, sha


def verify(directory):
    records, _, features, arms = read_features(directory)
    chosen = [i for i, r in enumerate(records) if r["source"] == "decor"]
    data = csv_rows("test_1355_clean.csv")
    topics = [data[records[i]["source_ref"][1]]["topic"] for i in chosen]
    source = json.loads((DEST / "source-anchored-hazard-scores.json").read_text())
    topic_order = [r["topic"] for r in source["folds"]]
    select = np.array([topic_order.index(t) for t in topics])
    native, lexical = EssayHazardInference.load().probabilities(features[chosen][:, arms["integration"]], per_model=True)
    original = np.load(DEST / "source-anchored-hazards.npz")
    errs = {k: float(np.max(abs(p[np.arange(len(select)), select] - original[k])))
            for k, p in (("source_reliability", native), ("lexical_surprisal", lexical))}
    if max(errs.values()) > 1e-8:
        raise ArithmeticError("The released essay observer differs from the source observation forecasts")
    receipt = {"weights_sha256": sha(WEIGHTS), "source_result_sha256": sha(DEST / "source-anchored-hazard-scores.json"),
               "topic_disjoint_boundaries": len(select), "source_essays": 100, "source_topics": len(topic_order),
               "maximum_absolute_probability_error": errs}
    dump(DEST / "runtime-replay.json", receipt)
    print(receipt)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    verify(parser.parse_args().features)
