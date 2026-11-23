"""Evaluate indirect access to an unmentioned object through the rest of its original event."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.character_access_data import OUT as CHARACTER
from research.reading_validation.reinstatement_data import FILES, INPUT, OUT, PLAN
from research.reading_validation.reinstatement_l4 import FORM
from research.reading_validation.transition_goal_access_data import DEST as GOALS
from research.reading_validation.transition_goal_access_data import STOP
from research.reading_validation.transition_goal_access_score import describe

CONDITIONS = ("rementioned", "neutral", "new")


def normalize(term):
    return term.casefold().replace("’", "'")


def associates(intro, word):
    sentences = []
    sentence = []
    pos = 0
    for m in FORM.finditer(intro):
        if re.search(r"[.!?]", intro[pos : m.start()]):
            sentences.append(sentence)
            sentence = []
        pos = m.end()
        sentence.append(normalize(m.group()))
    sentences.append(sentence)
    total = Counter()
    n = 0
    for s in sentences:
        for i, w in enumerate(s):
            if w != normalize(word):
                continue
            local = Counter()
            for j, v in enumerate(s):
                if v != w and v not in STOP:
                    local[v] += 1 / (1 + abs(i - j))
            if local:
                z = sum(local.values())
                total.update({k: a / z for k, a in local.items()})
                n += 1
    return {k: v / n for k, v in total.items()} if n else {}


def score(r, native):
    text = r["text"]
    words = []
    backend = []
    distance = []
    for a, b, w, d in native["cues"]:
        token = normalize(text[a:b])
        if token not in STOP:
            words.append(token)
            backend.append(w)
            distance.append(d)
    if not words:
        raise ValueError("Missing intervening content words")
    bindings = [associates(text[: r["prior_end"]], t) for t in r["targets"]]
    affinity = np.array([[g.get(w, 0) for w in words] for g in bindings])

    def propagate(q):
        q = np.array(q)
        q = 0.5 * q / q.sum() + 0.5 / len(q)
        return affinity @ q

    return np.stack((propagate(backend), propagate(distance))), int(
        np.sum(affinity[0] > 0)
    )


def main(bank):
    spec = read_json(OUT / "input.json")
    receipt = read_json(bank / "receipt.json")
    file = bank / "cue-strengths.jsonl"
    if (
        digest(INPUT) != receipt["source_sha256"]
        or spec["presentations"] != receipt["forwards"]
        or spec["protocol"] != digest(PLAN)
        or receipt["forecast_sha256"] != digest(file)
        or receipt["provided_targets"]
    ):
        raise ValueError("Inputs, masking or causal inference failed verification")
    predicted = {r["id"]: r for r in map(json.loads, file.read_text().splitlines())}
    results = {}
    for num, (label, path) in enumerate(FILES.items()):
        if spec["files"][label] != digest(path):
            raise ValueError("Study input changed")
        src = read_json(
            (GOALS / "exposure-seal.json")
            if label == "goal"
            else (CHARACTER / (label + "-input.json"))
        )
        inputwords = {
            r["id"]: r for r in map(json.loads, path.read_text().splitlines())
        }
        values = []
        coverage = []
        ids = []
        for family, g in src["families"].items():
            ids.append(family)
            v = []
            c = []
            for k in CONDITIONS:
                uid = g["variants"][k]["id"]
                x, n = score(inputwords[uid], predicted[uid])
                v.append(x)
                c.append(n)
            values.append(v)
            coverage.append(c)
        scores = np.stack(values)
        coverage = np.stack(coverage)
        rng = np.random.default_rng(99951 + num)
        b = rng.integers(0, len(scores), size=(10000, len(scores)))
        estimates = {}
        for name, d in (
            ("native", scores[:, :, 0]),
            ("distance", scores[:, :, 1]),
            ("native_minus_distance", scores[:, :, 0] - scores[:, :, 1]),
        ):
            for scope, mat in (
                ("target", d[:, :, 0]),
                ("target_minus_controls", d[:, :, 0] - d[:, :, 1:].mean(2)),
            ):
                estimates[name + "." + scope] = {
                    k: describe(mat[:, a] - mat[:, c], b)
                    for k, (a, c) in {
                        "rementioned_minus_new": (0, 2),
                        "neutral_minus_new": (1, 2),
                        "rementioned_minus_neutral": (0, 1),
                    }.items()
                }
        outfile = OUT / (label + "-item-scores.npz")
        np.savez_compressed(
            outfile, family=np.array(ids), values=scores, cue_occurrences=coverage
        )
        results[label] = {
            "families": len(ids),
            "item_scores": digest(outfile),
            "target_with_no_reinstated_associate": dict(
                zip(CONDITIONS, (coverage == 0).sum(0).tolist(), strict=True)
            ),
            "estimates": estimates,
        }
    write_json(
        OUT / "scores.json",
        {"protocol": digest(PLAN), "receipt": receipt, "source": spec, "results": results},
    )
    for k, r in results.items():
        print(k, r["estimates"]["native.target_minus_controls"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("bank", type=Path)
    main(p.parse_args().bank)
