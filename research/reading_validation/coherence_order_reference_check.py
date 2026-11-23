"""Independently reconstruct original Jaccard forecasts as word sets with the original tokenizer."""

import csv
import json
from itertools import pairwise

import nltk
import numpy as np
from nltk import word_tokenize

from research.reading_validation.coherence_order_reference import DATA, OUT


def reference_check():
    def rows(path):
        with path.open() as stream:
            for r in csv.DictReader(stream, delimiter="\t"):
                text = r["TEXT"]
                phrases = [v + "." for v in text.split(".") if v != ""]
                if len(phrases) == 1:
                    phrases = [v + "?" for v in text.split("?") if v != ""]
                yield r["ID"], [set(word_tokenize(p.lower())) for p in phrases]

    official = {
        v["id"]: v["official"]
        for v in json.loads((OUT / "official-italian.json").read_text())
    }
    voc = set()
    cs = tuple(rows(DATA / "train.tsv"))
    for _, passage in cs:
        for phrase in passage:
            voc.update(phrase)
    absent = []
    errors = []
    for key, parts in (*cs, *rows(DATA / "held.tsv")):
        if len(parts) < 2:
            absent.append(key)
            continue
        scores = []
        for left, right in pairwise(parts):
            a, b = left & voc, right & voc
            scores.append(len(a & b) / len(a | b) if a | b else 0.0)
        errors.append(abs(np.mean(scores) - official[key]))
    if (
        max(errors) > 1e-12
        or sorted(absent) != ["task_2_test_108"]
        or len(errors) != 1063
    ):
        raise ValueError(
            "The sparse reference differs from the original official one-hot script"
        )
    return {
        "defined": len(errors),
        "undefined": absent,
        "max_abs_error": float(max(errors)),
        "vocabulary": len(voc),
    }


if __name__ == "__main__":
    official = json.loads((OUT / "official-italian-seal.json").read_text())
    if nltk.__version__ != official["packages"]["nltk"]:
        raise ValueError(
            "Use the original evaluation environment and English-default Punkt vocabulary"
        )
    result = reference_check()
    result["nltk"] = nltk.__version__
    (OUT / "reference-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(result)
