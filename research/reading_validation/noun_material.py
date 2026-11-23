"""Verify intervention geometry and earlier supervised-material exposure without using scores."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from openpyxl import load_workbook

from research.reading_validation.external_input import material_key
from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.noun_data import BASE, DATA, OUT, SHA
from research.reading_validation.release_overlap import WINDOW, learning_material, words
from research.reading_validation.revision_prepare import (
    earlier_training_texts,
    matches,
    near_terms,
)

MODEL = Path("research/results/reading-validation-joint/forecast/models.npz")


def source_geometry():
    path = BASE / "leroy.xlsx"
    if digest(path) != SHA:
        raise ValueError("Original workbook hash changed")
    book = load_workbook(path, read_only=True, data_only=True)
    it = iter(book["Data"].values)
    cols = next(it)
    seen = Counter()
    geometry = {}
    stimuli = {r["id"]: r["text"] for r in rows(DATA / "local-stimuli.jsonl")}
    for vals in it:
        r = dict(zip(cols, vals, strict=True))
        gram = r["GRAM"]
        split = {"split": True, "unsplit": False}[r["compound"]]
        pseudo = {"YES": True, "NO": False}[r["pseudoword"]]
        cell = gram, split, pseudo
        family = f"{gram}-{seen[cell]:02d}"
        seen[cell] += 1
        text = r["Original Sentence"]
        key = hashlib.sha256(text.encode()).hexdigest()
        if stimuli[key] != text:
            raise ValueError("Single-forward forecast stimulus changed")
        geometry.setdefault(family, {})[split, pseudo] = (
            key,
            text,
            r["Noun Phrase"],
            r["Head Noun"],
        )
    book.close()
    if len(geometry) != 90 or len(stimuli) != 360:
        raise ValueError("Expected 90 complete four-condition families")
    return geometry


def surround(text, phrase):
    target = words(phrase)
    line = words(text)
    locations = [
        i
        for i in range(len(line) - len(target) + 1)
        if line[i : i + len(target)] == target
    ]
    if len(locations) != 1:
        return None
    i = locations[0]
    return line[:i], line[i + len(target) :]


def compare_material():
    geo = source_geometry()
    known, _, historical = learning_material()
    fit = {
        "educational_grade_and_projection": earlier_training_texts(),
        "CompLex": {
            r["text"]
            for r in rows(BASE.parent / "stage5/complex-l4-input/stimuli.jsonl")
            if r["split"] in ("train", "trial")
        },
    }
    strings = {
        k: (family, t)
        for family, variants in geo.items()
        for k, t, _, _ in variants.values()
    }
    exact = defaultdict(set)
    window = defaultdict(set)
    for k, (family, t) in strings.items():
        w = words(t)
        exact[w].add(family)
        for i in range(len(w) - WINDOW + 1):
            window[w[i : i + WINDOW]].add(family)
    prior = {}
    primary_excluded = set()
    for name, docs in {**fit, **{"historic_" + k: v for k, v in known.items()}}.items():
        full, partly = set(), set()
        for t in docs:
            w = words(t)
            full.update(exact.get(w, ()))
            for i in range(len(w) - WINDOW + 1):
                partly.update(window.get(w[i : i + WINDOW], ()))
        strong = set()
        if name in fit:
            known_keys = {material_key(x) for x in docs}
            grams, inv = near_terms(sorted(docs))
            for _, (family, sentence) in strings.items():
                if material_key(sentence) in known_keys or matches(
                    sentence, grams, inv
                ):
                    strong.add(family)
            primary_excluded.update(full | partly | strong)
        prior[name] = {
            "strings": len(docs),
            "identical": sorted(full),
            "shared_twelve": sorted(partly),
            "strong_fivegram": sorted(strong) if name in fit else None,
        }
    counts, issues = Counter(), []
    for f, versions in sorted(geo.items()):
        if len(versions) != 4 or len({v[3].casefold() for v in versions.values()}) != 1:
            raise ValueError("Family members differ on designated head noun")
        for p in (False, True):
            w, x = [versions[s, p] for s in (False, True)]
            a, b = surround(w[1], w[2]), surround(x[1], x[2])
            if a is None or b is None:
                counts["literal_phrase_not_uniquely_matched"] += 1
                kind = "phrase"
            elif a == b:
                counts["identical_surroundings"] += 1
                continue
            else:
                counts["literal_surroundings_differ"] += 1
                kind = "surroundings"
            issues.append({"family": f, "pseudo": p, "type": kind})
    eligible = [f for f in sorted(geo) if f not in primary_excluded]
    folds = {}
    for gram in (2, 3, 4):
        ordered = sorted(
            (f for f in eligible if int(f.split("-")[0]) == gram),
            key=lambda f: hashlib.sha256(("noun-family-v1/" + f).encode()).digest(),
        )
        folds.update({f: i % 5 for i, f in enumerate(ordered)})
    seen_norm = {}
    for _, (f, t) in strings.items():
        if f not in folds:
            continue
        key = words(t)
        if seen_norm.setdefault(key, folds[f]) != folds[f]:
            raise ValueError("Normalized sentence crosses family-held-out folds")
    return {
        "inputs_sha256": digest(DATA / "local-stimuli.jsonl"),
        "actual_source_model_sha256": digest(MODEL),
        "historical": historical,
        "exposure": prior,
        "eligible": eligible,
        "excluded": sorted(primary_excluded),
        "folds": folds,
        "geometry": dict(counts),
        "textual_geometry_flags": issues,
        "materials_program_sha256": digest(Path(__file__)),
    }


if __name__ == "__main__":
    result = compare_material()
    (OUT / "material-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print({k: result[k] for k in ("excluded", "geometry", "exposure")})
