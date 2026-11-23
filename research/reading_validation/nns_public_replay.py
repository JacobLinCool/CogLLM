"""Recompute two published held reader endpoints from the author's ZIP and NumPy alone."""

import argparse
import json
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from research.reading_validation.local_forecast import ROOT, WORD, digest
from research.reading_validation.nns_material import LEVELS, SOURCE, identifier

RESULTS = ROOT / "research" / "results"
GAP = re.compile(r"_{3,}")
TIE = "The words are equally simple or equally complex."


def arrays(folder, filename, seal, field):
    path = folder / filename
    if digest(path) != json.loads((folder / seal).read_bytes())[field]:
        raise ValueError(f"Changed published forecasts in {filename}")
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def members(archive, filename, size):
    with archive.open(filename) as stream:
        for line in stream:
            fields = line.decode().rstrip("\r\n").split("\t")
            if len(fields) != size:
                raise ValueError("Unexpected study fields")
            yield fields[3:]


@lru_cache(maxsize=16384)
def source_id(text):
    return identifier(text)


def word_votes(archive, words):
    lookup = {
        (doc, *coords): i
        for i, (doc, coords) in enumerate(
            zip(words["doc"], words["coords"], strict=True)
        )
    }
    ids = set(words["doc"])
    denominators = np.zeros((len(lookup), len(LEVELS)), int)
    flags = np.zeros_like(denominators)
    geometry = {}
    entries = members(archive, "Complex Word Identification/CWI_Annotations.txt", 7)
    for level, text, term, mark in entries:
        if (doc := source_id(text)) not in ids:
            continue
        if doc not in geometry:
            matches = list(WORD.finditer(text))
            count = Counter(q.group().casefold() for q in matches)
            geometry[doc] = {
                q.group().casefold(): q.span()
                for q in matches
                if count[q.group().casefold()] == 1
            }
        span = geometry[doc].get(term.casefold())
        if span is None or (row := lookup.get((doc, *span))) is None:
            continue
        if mark not in ("Simple", "Complex"):
            raise ValueError("Invalid released word mark")
        j = LEVELS.index(level)
        denominators[row, j] += 1
        flags[row, j] += mark == "Complex"
    if geometry.keys() != ids or np.any(flags > denominators):
        raise ValueError("Missing published study material")
    return flags, denominators


def rewrite_votes(archive, bank):
    mapping = {tuple(k): i for i, k in enumerate(bank["r_keys"])}
    groups = set(bank["r_doc"])
    votes = np.zeros((len(mapping), 3), int)

    @lru_cache(maxsize=12000)
    def focus(text, word):
        m = GAP.search(text)
        if not m or GAP.search(text, m.end()):
            raise ValueError("A response has an ambiguous replacement position")
        rendered = text[: m.start()] + word + text[m.end() :]
        return identifier(f"{identifier(rendered)}/{m.start()}/{m.start() + len(word)}")

    for _, text, _, _, first, second, answer in members(
        archive, "Substitution Ranking/SR_Annotations.txt", 10
    ):
        if (src := source_id(text)) not in groups:
            continue
        a, b = sorted((first, second))
        pos = mapping.get((src, focus(text, a), focus(text, b)))
        if pos is None:
            continue
        if answer not in (a, b, TIE) or a == b:
            raise ValueError("Invalid released replacement choice")
        votes[pos, (a, b, TIE).index(answer)] += 1
    if (votes.sum(1) == 0).any():
        raise ValueError("A published pair has no votes")
    return votes


def grouped(values, docs, families, seed, population=None):
    by_doc = defaultdict(list)
    for i, k in enumerate(docs):
        by_doc[k].append(i)
    by_family = defaultdict(list)
    for indices in by_doc.values():
        k = families[indices[0]]
        if not np.all(families[indices] == k):
            raise ValueError("One sentence was split across families")
        by_family[k].append(np.mean(values[indices], axis=0))
    order = sorted(set(families if population is None else population))
    selected = [k in by_family for k in order]
    means = np.array([np.mean(by_family[k]) for k in order if k in by_family])
    n = len(order)
    draws = np.random.default_rng(seed).multinomial(n, np.ones(n) / n, size=10000)
    informative = draws[:, selected]
    samples = (informative @ means) / informative.sum(1)
    return {
        "estimate": float(means.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "families": len(means),
    }


def run(path):
    if digest(path) != SOURCE:
        raise ValueError("Expected Paetzold and Specia's original User_Studies_NNS.zip")
    word_dir = RESULTS / "reading-validation-nns"
    rewrite_dir = RESULTS / "reading-validation-nns-rewrite"
    u = arrays(
        word_dir,
        "word-forecasts.npz",
        "forecast-seal.json",
        "outcome_free_forecast_sha256",
    )
    r = arrays(rewrite_dir, "forecasts.npz", "forecast-seal.json", "forecasts_sha256")
    if not np.array_equal(u["doc"], r["u_doc"]) or not np.array_equal(
        u["coords"], r["u_coords"]
    ):
        raise ValueError("Inconsistent word locations between computations")
    with ZipFile(path) as archive:
        flags, total = word_votes(archive, u)
        choices = rewrite_votes(archive, r)
    hits, n = flags.sum(1), total.sum(1)
    good = n > 0
    pair_hits, ds, fs = [], [], []
    num = 0
    for d in sorted(set(u["doc"])):
        ix = np.flatnonzero((u["doc"] == d) & good)
        left, right = np.triu_indices(len(ix), 1)
        a, b = ix[left], ix[right]
        eligible = hits[a] * n[b] != hits[b] * n[a]
        a, b = a[eligible], b[eligible]
        if not len(a):
            continue
        rate_dir = np.sign(hits[a] * n[b] - hits[b] * n[a])
        prediction = u["text.adaptive"][:, 0]
        pair_hits.append(
            np.mean((1 + np.sign(prediction[a] - prediction[b]) * rate_dir) / 2)
        )
        ds.append(d)
        fs.append(u["family"][ix[0]])
        num += len(a)
    outcome = {
        "original_sha256": SOURCE,
        "word_responses": int(n.sum()),
        "word_marks": int(hits.sum()),
        "word_pairs": num,
        "replacement_votes": int(choices.sum()),
        "ties": int(choices[:, 2].sum()),
        "within_sentence_rank": grouped(
            np.array(pair_hits),
            np.asarray(ds),
            np.asarray(fs),
            631402,
            population=u["family"][good],
        ),
    }
    pairs = [r["r.contextual." + k] for k in ("independent", "adaptive")]
    pref = -(choices[:, None, :] * np.log(np.stack(pairs, axis=1))).sum(
        2
    ) / choices.sum(1, keepdims=True)
    outcome["revision_joint_gain"] = grouped(
        pref[:, 0] - pref[:, 1], r["r_doc"], r["r_family"], 231907
    )
    risks = [
        r["u.contextual.independent"][:, None],
        r["u.contextual.adaptive"][:, None],
        u["text.adaptive"],
        u["cefr.adaptive"],
    ]
    loss = [
        -(
            flags[good] * np.log(q[good])
            + (total[good] - flags[good]) * np.log1p(-q[good])
        ).sum(1)
        / n[good]
        for q in risks
    ]
    for name, delta, seed in (
        ("word_joint_gain", loss[0] - loss[1], 231907),
        ("cefr_gain", loss[2] - loss[3], 631402),
    ):
        outcome[name] = grouped(delta, u["doc"][good], u["family"][good], seed)
    reference = json.loads((word_dir / "scores.json").read_bytes())["reported"]
    joint = json.loads((rewrite_dir / "scores.json").read_bytes())
    comp = "contextual.adaptive_gain_vs_contextual.independent"
    expected = {
        "within_sentence_rank": reference["rank"]["scores"]["adaptive"],
        "cefr_gain": reference["logloss"]["paired_difference"][
            "text.adaptive_minus_cefr.adaptive"
        ],
        "word_joint_gain": joint["unknown_word"]["logloss"]["oriented_gains"][comp],
        "revision_joint_gain": joint["revision_preference"]["logloss"][
            "oriented_gains"
        ][comp],
    }
    for k, x in expected.items():
        observed = [outcome[k]["estimate"], *outcome[k]["ci95"]]
        target = [x["estimate"], *x["ci95"]]
        if not np.allclose(target, observed, rtol=0, atol=1e-12):
            raise ValueError(f"Failed to reproduce {k}")
    return outcome


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(run(parser.parse_args().archive), indent=2, allow_nan=False))
