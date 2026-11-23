"""Calculate the unmodified academic benchmark's Jaccard estimate on each new word-identical variant."""

import csv
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from research.reading_validation.coherence_order_reference import (
    DATA,
    ORIGINAL,
    SOURCE,
    mask_labels,
    sha,
)

OUT = Path("research/results/reading-validation-italian-interventions")


def run():
    import nltk

    if sha(ORIGINAL) != SOURCE or nltk.__version__ != "3.8.1":
        raise ValueError("Unexpected official baseline implementation")
    spec = importlib.util.spec_from_file_location("disc_ref", ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ms = [
        json.loads(l) for l in (DATA / "counterfactuals.jsonl").read_text().splitlines()
    ]
    originals = json.loads(
        Path(
            "research/results/reading-validation-coherence-order/official-italian.json"
        ).read_text()
    )
    source = {r["id"]: r["official"] for r in originals}
    pairs = json.loads((OUT / "order-pairs.json").read_text())
    result = {d["observed"]: source[k] for k, d in pairs.items() if k in source}
    with TemporaryDirectory() as folder:
        p = Path(folder)
        mask_labels(DATA / "train.tsv", p / "train.tsv")
        with (p / "test.tsv").open("w", newline="") as stream:
            fs = csv.DictWriter(
                stream, fieldnames=("ID", "TEXT", "SCORE"), delimiter="\t"
            )
            fs.writeheader()
            for i, r in enumerate(ms):
                fs.writerow({"ID": r["id"], "TEXT": r["text"], "SCORE": 1 + i % 5})
        rows, _ = module.load_dataset_t2(p / "test.tsv")
        undefined = [r["id"] for r in rows if len(r["sentences"]) < 2]
        retained = mask_labels(p / "test.tsv", p / "eligible.tsv", undefined)
        orig = module.pearsonr

        def collect(known, values):
            if list(known) != [1 + i % 5 for i in range(len(known))]:
                raise ValueError("Unexpected true ratings in shape-only Jaccard query")
            result.update(zip(retained, map(float, values), strict=True))
            return orig(known, values)

        with patch.object(module, "pearsonr", collect):
            module.main(p / "train.tsv", p / "eligible.tsv")
    f = OUT / "jaccard.json"
    f.write_text(json.dumps(result, indent=2) + "\n")
    (OUT / "jaccard-seal.json").write_text(
        json.dumps(
            {
                "original_code": SOURCE,
                "forecasts": sha(f),
                "variants": len(ms),
                "variant_undefined": undefined,
                "source_labels_masked": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    run()
