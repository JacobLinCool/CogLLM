"""Represent independent reader grades and paired votes at their observed units."""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
import numpy as np

from research.reading_validation.external_input import digest_string, material_key
from research.reading_validation.revision_prepare import OUT, SOURCE, records


ASPECTS = ("simplicity", "meaning", "fluency", "overall")
STUDIES = ("asset_pair", "asset_rating", "wikida", "past")


@dataclass(frozen=True)
class Observation:
    study: str
    aspect: str
    group: str
    first: int
    second: int
    value: float
    weight: float


@dataclass
class Task:
    passages: list
    groups: list
    observations: list
    heldout: list


def build():
    groups = json.loads((SOURCE / "foldmap.json").read_bytes())
    exclude = set(json.loads((OUT / "inputs.json").read_bytes())["excluded_test_source_ids"])
    passages = []
    families = []
    index = {}
    observations = []
    heldout = set()

    def key(source, revision, group=None):
        case = source, revision
        family = "train/" + groups[material_key(source)] if group is None else "test/" + group
        if case not in index:
            index[case] = len(passages)
            passages.append(case)
            families.append(family)
        elif families[index[case]] != family:
            raise ValueError("Repeated passage maps to competing source families")
        return index[case]

    def add(study, aspect, i, j, target, weight=1.0):
        if not 0 <= target <= 1 or (j >= 0 and families[i] != families[j]):
            raise ValueError("Invalid observed comparison across source families")
        observations.append(Observation(study, aspect, families[i], i, j, target, weight))

    binary = defaultdict(list)
    for r in records("asset_pair"):
        a = key(r["source"], r["asset_simplification"])
        b = key(r["source"], r["other_simplification"])
        if a == b:
            continue
        win = r["winner"].lower()
        fav = {"asset": 1.0, r["other_dataset"].lower(): 0.0, "similar": 0.5}[win]
        if a > b:
            a, b, fav = b, a, 1 - fav
        binary[r["aspect"], a, b].append(fav)
    for (aspect, i, j), labels in binary.items():
        add("asset_pair", aspect, i, j, np.mean(labels), len(labels))
    grades = defaultdict(list)
    for r in records("asset_rating"):
        i = key(r["original"], r["simplification"])
        grades["asset_rating", r["aspect"], i].append(float(r["rating"]) / 100)
    for r in records("wikida"):
        i = key(r["orig_sent"], r["simp_sent"])
        for aspect in ASPECTS[:3]:
            grades["wikida", aspect, i].append(float(r[aspect]) / 100)
    for r in records("past"):
        i = key(r["original"], r["generation"])
        grades["past", "overall", i].extend(float(r["rating_" + str(k)]) / 100 for k in range(1, 6))
    grouped = defaultdict(list)
    for (study, aspect, i), grades_i in grades.items():
        value = float(np.mean(grades_i))
        add(study, aspect, i, -1, value)
        grouped[study, aspect, families[i]].append((i, value))
    for (study, aspect, family), grade_rows in grouped.items():
        for (i, y), (j, z) in combinations(grade_rows, 2):
            if y == z:
                continue
            # Paired term uses grade differences as ordered evidence with a bounded confidence weight.
            add(study, aspect, i, j, float(y > z), min(1.0, abs(y - z) / 0.2))
    for r in records("simpda2022"):
        g = r["Input.id"]
        if g in exclude:
            continue
        i = key(r["Input.original"], r["Input.simplified"], g)
        heldout.add((g, digest_string(r["Input.simplified"]), i))
    return Task(passages, families, observations, sorted(heldout))
