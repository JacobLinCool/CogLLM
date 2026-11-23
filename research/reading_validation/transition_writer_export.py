"""Export two independently source-calibrated screening settings for the narrative reader."""
import json
from pathlib import Path

from research.reading_validation.narrative_inference import MARGINAL_BUNDLE, MARGINAL_SHA
from research.reading_validation.transition_score import dump, sha

SOURCE = Path("research/results/reading-validation-transition-t5/source-equal-f1-thresholds.json")
OUTPUT = MARGINAL_BUNDLE.with_name("character-operating-points.json")


def main():
    if sha(MARGINAL_BUNDLE) != MARGINAL_SHA:
        raise ValueError("Narrative weights changed")
    original = json.loads(MARGINAL_BUNDLE.read_text())
    source = json.loads(SOURCE.read_text())
    selected = source["model_decisions"]["words"]["served_marginal"]
    dump(OUTPUT, {
        "schema": "cogllm-narrative-character-en-decisions-v1",
        "weights_sha256": MARGINAL_SHA, "source_decisions_sha256": sha(SOURCE),
        "target": "An English narrative character has insufficient setup in this summary",
        "calibration_books": source["development_books"], "calibration_words": source["development_words"],
        "settings": {
            "balanced": {"cutoff": selected["cutoff"], "inclusive": True,
                         "criterion": "maximum F1 on measured book-development words",
                         "development_tp_fp_fn_tn": selected["tp_fp_fn_tn"]},
            "high_recall": {"cutoff": original["threshold"], "inclusive": False,
                            "criterion": "historical 95th percentile of book-development negative-word scores"}
        }
    })
    print(sha(OUTPUT))


if __name__ == "__main__":
    main()
