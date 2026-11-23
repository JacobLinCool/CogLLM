"""Freeze source-story splits and identical public summary material without consulting any expert or crowd ballot."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from research.reading_validation.coherence_order_data import OUT as TRAIN, read, texts
from research.reading_validation.danish_coherence_data import grams
from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.memory_data import BASE, h

WORK = BASE / "summeval"
OUT = Path("research/results/reading-validation-summeval")
RAW = WORK / "original-model-annotations.jsonl"
PLAN = Path(__file__).with_name("summeval-protocol.md")
SHA = "f0d4166e0cdeb439b387c4449634b067b7e9f721b9ef4c2b722f6b7550d3d6ab"


def material():
    if digest(RAW) != SHA:
        raise ValueError("Unknown published SummEval release")
    return [(d["id"], d["model_id"], d["decoded"], d["references"]) for d in rows(RAW)]


def make():
    raw = material()
    story = {}
    group = {a: a for a, *_ in raw}
    texts_by_key = {h(t): t for _, _, t, _ in raw}
    units = defaultdict(list)
    first = {}
    wording = {}

    def root(k):
        while k != group[k]:
            group[k] = group[group[k]]
            k = group[k]
        return k

    for k, m, t, _ in raw:
        doc = h(t)
        units[k].append((m, doc))
        rows_seen = grams(t) | {("exact", doc)}
        for g in rows_seen:
            other = first.setdefault(g, k)
            a, b = sorted([root(k), root(other)])
            group[b] = a
    components = defaultdict(list)
    for a in sorted(units):
        components[root(a)].append(a)
    source_txt = texts()
    trained = set()
    for k, d in read(TRAIN / "geometry.json").items():
        if d["role"] in ("fit", "dev"):
            trained.update(grams(source_txt[k]))
    excluded = set()
    for a, rs in units.items():
        if any(grams(texts_by_key[doc]) & trained for _, doc in rs):
            excluded.add(root(a))
    choices = sorted(
        set(components) - excluded, key=lambda k: h("cogllm-summeval-public-v1/" + k)
    )
    cut = int(0.4 * len(choices))
    fin = cut + int(0.5 * len(choices))
    role = {
        k: ("calibration" if i < cut else "confirm" if i < fin else "future")
        for i, k in enumerate(choices)
    }
    for a, vs in units.items():
        groupid = root(a)
        story[a] = {
            "group": groupid,
            "role": role.get(groupid, "exposed"),
            "systems": {m: d for m, d in sorted(vs)},
        }
        if len(story[a]["systems"]) != 16:
            raise ValueError("Changing released system inventory")
    fname = WORK / "local-stimuli.jsonl"
    fname.write_text(
        "".join(
            json.dumps({"id": k, "text": t}, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for k, t in sorted(texts_by_key.items())
        )
    )
    (OUT / "story-geometry.json").write_text(
        json.dumps(story, indent=2, sort_keys=True) + "\n"
    )
    info = {
        "raw": SHA,
        "protocol": digest(PLAN),
        "packet": digest(fname),
        "geometry": digest(OUT / "story-geometry.json"),
        "stories": len(units),
        "source_groups": len(components),
        "stories_per_role": dict(Counter(r["role"] for r in story.values())),
        "source_groups_per_role": dict(Counter(role.values())),
        "merged_source_groups": {k: v for k, v in components.items() if len(v) > 1},
        "prior_supervised_overlap_groups": sorted(excluded),
        "original_drafts": len(raw),
        "unique_drafts": len(texts_by_key),
        "expert_and_crowd_values_read": False,
    }
    (OUT / "inputs.json").write_text(json.dumps(info, indent=2) + "\n")
    print({k: v for k, v in info.items() if k != "merged_source_groups"})


if __name__ == "__main__":
    make()
