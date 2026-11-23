"""Certify both source-selected word settings against every archived movie prediction."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from research.reading_validation.narrative_inference import DECISIONS_SHA, MARGINAL_SHA, MarginalNarrativeInference
from research.reading_validation.transition_narrative_verify import run as native_replay


PACKET = 'b915ebd74440c810817ba7e9de90e247531fae73b04bc3270b08863a9f3e2d25'


def run(runtime, reader, checkpoint, input_path, output_dir):
    raw = Path(input_path).read_bytes()
    if sha256(raw).hexdigest() != PACKET:
        raise ValueError("Movie replay source changed")
    docs = [json.loads(line) for line in raw.splitlines()]
    balanced = MarginalNarrativeInference.for_writers()
    sensitive = MarginalNarrativeInference.for_writers("high_recall")
    counters = {"balanced": 0, "high_recall": 0, "words": 0}

    class Both:
        threshold = balanced.threshold

        def score(self, *args, **kwargs):
            return balanced.score(*args, **kwargs)

        def score_segments(self, *args, **kwargs):
            rows = balanced.score_segments(*args, **kwargs)
            doc = docs[self.index]
            self.index += 1
            scores = np.array([r["risk"] for r in rows])
            expected = np.array([p for _, _, p in doc["expected"]])
            for model, name in [(balanced, "balanced"), (sensitive, "high_recall")]:
                flags = model.flags(scores)
                if not np.array_equal(flags, model.flags(expected)):
                    raise ValueError("Native thresholding differs from archived scores")
                if name == "balanced" and not np.array_equal(flags, [r["flagged"] for r in rows]):
                    raise ValueError("Returned production marks differ from thresholded scores")
                counters[name] += int(flags.sum())
            counters["words"] += len(rows)
            return rows

    model = Both()
    model.index = 0
    native_replay(runtime, reader, checkpoint, input_path, output_dir, model=model)
    dest = Path(output_dir) / "receipt.json"
    receipt = json.loads(dest.read_text())
    if counters["words"] != 26417 or model.index != 39:
        raise ValueError("Incomplete comparison of balanced and high-recall settings")
    receipt.update(settings=counters, weights_sha256=MARGINAL_SHA, operating_policy_sha256=DECISIONS_SHA,
                   balanced=balanced.threshold, high_recall=sensitive.threshold)
    dest.write_text(json.dumps(receipt, indent=2) + "\n")
