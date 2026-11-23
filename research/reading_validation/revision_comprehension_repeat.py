"""Check agreement between independent presentations of the same visible text."""

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from research.reading_validation.joint_data import digest, lines
from research.reading_validation.revision_comprehension_prepare import RESULT, versions
from research.reading_validation.revision_comprehension_score import read_responses


def main():
    seal = json.loads((RESULT / "forecast-seal.json").read_bytes())
    if digest(RESULT / "forecasts.jsonl") != seal["forecasts_sha256"] or not (RESULT / "scores.json").is_file():
        raise ValueError("Score the registered primary analysis first")
    forecasts = {
        (r["article"], r["paragraph"], r["condition"]): r["scores"]
        for r in lines(RESULT / "forecasts.jsonl")
    }
    observed = read_responses()
    differences = []
    articles = set()
    for (article, paragraph), conditions in versions().items():
        same = defaultdict(list)
        for condition, text in conditions.items():
            key = article, paragraph, condition
            if len(observed[key]) == 1:
                same[text].append(key)
        for cells in same.values():
            for a, b in combinations(cells, 2):
                if forecasts[a] != forecasts[b]:
                    raise ArithmeticError("Identical input produced unequal forecasts")
                differences.append(observed[a][0] != observed[b][0])
                articles.add(article)
    if not differences:
        raise ValueError("No duplicated presentation")
    unequal = np.stack(differences)
    result = {
        "protocol_sha256": digest(Path(__file__).with_name("revision-comprehension-repeat.md")),
        "pairs_of_identical_version_texts": len(unequal),
        "article_count": len(articles),
        "questions_in_pairs": int(np.prod(unequal.shape[:2])),
        "differing_observed_correctness": int(unequal[:, :, 0].sum()),
        "differing_observed_answer_availability": int(unequal[:, :, 1].sum()),
        "model_disagreements": 0,
    }
    (RESULT / "identical-texts.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
