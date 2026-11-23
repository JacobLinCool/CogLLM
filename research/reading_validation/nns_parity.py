"""Replay the new packed-word source logits against archived token-axis Qwen outputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.reading_validation.local_features import sentence_features
from research.reading_validation.local_forecast import WORD, digest, rows
from research.reading_validation.local_inference import LocalInference, _matrix_blocks
from research.reading_validation.nns_features import single_word_rank
from research.reading_validation.nns_material import BASE, OUT


def run(path):
    model = LocalInference.load()
    seal = json.loads((path / "seal.json").read_bytes())
    material = {r["id"]: r["text"] for r in rows(BASE / "local-stimuli.jsonl")}
    count = 0
    worst = 0.0
    texts = 0
    for ref in seal["documents"]:
        file = path / ref["file"]
        if digest(file) != ref["sha256"]:
            raise ValueError("Archived source changed")
        text = material[ref["id"]]
        with np.load(file, allow_pickle=False) as native:
            words = native["words"]
            positions = words[:, :2]
            take = words[:, 2]
            ok = take >= 0
            q = native["lexical"][np.maximum(0, take)]
            h = native["projection"][np.maximum(0, take)]
            packed = single_word_rank(model, text, ref["id"], positions, ok, q, h)
            node = sentence_features(
                ref["id"],
                "article",
                text,
                0,
                len(text),
                model.lexicon,
                {k: native[k] for k in ("words", "lexical", "projection")},
                0,
                0,
                sentence_endpoint=False,
            )
            expected = (
                _matrix_blocks(
                    model.rank["parts"],
                    model.rank["center"],
                    model.rank["scale"],
                    node.features,
                )
                @ model.rank["coef"]
            )
            lookup = {
                tuple(span): v for span, v in zip(node.coords, expected, strict=True)
            }
            for word, allowed in zip(WORD.finditer(text), ok, strict=True):
                if allowed:
                    worst = max(worst, abs(packed[word.span()] - lookup[word.span()]))
                    count += 1
            texts += 1
    if worst > 1e-6 or count < 1000:
        raise ArithmeticError("Source-native word logit parity failed")
    out = {
        "source_documents": texts,
        "certified_words": count,
        "max_score_difference": float(worst),
        "native_source_seal": digest(path / "seal.json"),
        "code_sha256": digest(Path(__file__)),
        "prospective_observations_used": False,
    }
    (OUT / "source-parity.json").write_text(json.dumps(out, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("--source", type=Path, required=True)
    run(args.parse_args().source)
