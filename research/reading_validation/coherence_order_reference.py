"""Extract outcome-blind predictions from the authors' unmodified Italian coherence baseline."""

import csv
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

PIN = "e113774f1b5f9c1eb4ea5581767429241ebb8cc9"
SOURCE = "575ec8953ef0b56abab98e5d962822b5435e177ff66d98127613606b8b6bdc15"
DATA = Path(".cache/cogllm/reading-validation/italian-order")
OUT = Path("research/results/reading-validation-coherence-order")
ORIGINAL = DATA / "run_baseline_t2.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_labels(raw, out, omit=()):
    ids = []
    with raw.open() as inp, out.open("w") as dst:
        entries = csv.DictReader(inp, delimiter="\t")
        writer = csv.DictWriter(dst, fieldnames=entries.fieldnames, delimiter="\t")
        writer.writeheader()
        for r in entries:
            if r["ID"] in omit:
                continue
            i = len(ids)
            ids.append(r["ID"])
            writer.writerow({"ID": r["ID"], "TEXT": r["TEXT"], "SCORE": 1 + i % 5})
    return ids


def run():
    import numpy as np

    if sha(ORIGINAL) != SOURCE:
        raise ValueError(
            "Official baseline implementation differs from the pinned commit"
        )
    spec = importlib.util.spec_from_file_location("official_discotex", ORIGINAL)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    raw = baseline.pearsonr
    frames = []

    def capture(rates, sequence):
        if list(rates) != [1 + i % 5 for i in range(len(rates))]:
            raise ValueError("The evaluator did not receive the synthetic score column")
        frames.extend(map(float, sequence))
        return raw(rates, sequence)

    all_rows = []
    undefined = {}
    with TemporaryDirectory() as folder:
        work = Path(folder)
        for f in ("train", "held"):
            mask_labels(DATA / (f + ".tsv"), work / (f + ".tsv"))
        with patch.object(baseline, "pearsonr", capture):
            for split in ("train", "held"):
                panel, _ = baseline.load_dataset_t2(work / (split + ".tsv"))
                undefined[split] = [r["id"] for r in panel if len(r["sentences"]) < 2]
                eligible = work / (split + "-scorable.tsv")
                ids = mask_labels(DATA / (split + ".tsv"), eligible, undefined[split])
                baseline.main(work / "train.tsv", eligible)
                if len(frames) != len(ids) or not np.isfinite(frames).all():
                    raise ValueError("Official Jaccard score missing")
                all_rows.extend(
                    {"id": k, "split": split, "official": p}
                    for k, p in zip(ids, frames)
                )
                frames.clear()
    OUT.mkdir(exist_ok=True, parents=True)
    forecasts = OUT / "official-italian.json"
    forecasts.write_text(json.dumps(all_rows, separators=(",", ":")) + "\n")
    rec = {
        "repository": "davidecolla/DisCoTex",
        "commit": PIN,
        "code": SOURCE,
        "source_labels_masked": True,
        "undefined_adjacent_sentence_pairs": undefined,
        "language": "Italian",
        "word_tokenize_default_language": "english",
        "forecast": sha(forecasts),
        "packages": {
            k: importlib.metadata.version(k)
            for k in (
                "nltk",
                "scikit-learn",
                "scipy",
                "numpy",
                "pandas",
                "prettytable",
                "tqdm",
            )
        },
    }
    (OUT / "official-italian-seal.json").write_text(json.dumps(rec, indent=2) + "\n")
    print("Sealed official adjacent Jaccard forecasts", len(all_rows))


if __name__ == "__main__":
    run()
