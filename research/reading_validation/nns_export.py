"""Package the fitted word-report readout and its attributed public vocabulary reference."""

import gzip
import json
from importlib.metadata import version
from pathlib import Path

from research.reading_validation.local_forecast import digest
from research.reading_validation.nns_features import CEFR, curriculum
from research.reading_validation.nns_material import OUT


def run():
    seal = json.loads((OUT / "forecast-seal.json").read_bytes())
    weights = OUT / "weights.npz"
    if digest(weights) != seal["weights_sha256"]:
        raise ValueError("The source-only fit changed")
    surface, by_stem = curriculum()
    profile = OUT / "vocabulary.json.gz"
    parts = {
        source: {
            axis: {w: n for w, n in ref.items() if (n <= 4) == (source == "cefrj")}
            for axis, ref in (("surface", surface), ("stem", by_stem))
        }
        for source in ("cefrj", "octanove")
    }
    content = json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()
    profile.write_bytes(gzip.compress(content, compresslevel=9, mtime=0))
    manifest = {
        "schema": "nns-self-report-en-v1",
        "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "weights_sha256": digest(weights),
        "vocabulary_sha256": digest(profile),
        "source_profile_sha256": CEFR,
        "local_source_sha256": digest(
            Path("research/results/reading-validation-local/serving/manifest.json")
        ),
        "extractor_sha256": digest(Path(__file__).with_name("nns_l4.py")),
        "proficiency": ["A1", "A2", "B1", "B2", "C1", "C2"],
        "dependencies": {p: version(p) for p in ("wordfreq", "nltk")},
        "target": "Probability that an English learner in the source population reports this word was hard to understand when shown an isolated English sentence.",
        "validation_unit": "One complete English sentence in a single causal Qwen forward.",
        "source_profile": "CEFR-J Wordlist v1.5, Yukio Tono, TUFS; Octanove Vocabulary Profile C1/C2 v1.0, Octanove Labs, CC BY-SA 4.0. Corpus and license details in vocabulary-sources.md.",
    }
    (OUT / "serving.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(len(surface), len(by_stem), len(profile.read_bytes()))


if __name__ == "__main__":
    run()
