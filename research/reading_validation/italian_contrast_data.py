"""Create exact-content source-restoration stimuli using original training families only."""

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from research.reading_validation.coherence_order_data import ITALIAN, csv_rows, read
from research.reading_validation.coherence_order_data import OUT as COH
from research.reading_validation.danish_coherence_data import grams
from research.reading_validation.italian_order_data import OUT as HELD
from research.reading_validation.italian_order_data import sentence_bank, spans
from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.memory_data import h

OUT = Path("research/results/reading-validation-order-training")
PLAN = Path(__file__).with_name("italian-contrast-protocol.md")


def create():
    _, raw, _ = sentence_bank()
    provenance = read(HELD / "provenance.json")
    geo = read(COH / "geometry.json")
    observations = {r["ID"]: r["TEXT"] for r in csv_rows(ITALIAN / "train.tsv")}
    taken = {}
    samples = {}
    for name, rec in provenance.items():
        if (
            rec["split"] != "train"
            or rec["type"] not in ("original", "transposed")
            or not rec["verbatim"]
        ):
            continue
        pos = [tuple(v) for v in rec["coordinates"]]
        src = rec["source"]
        orig = observations[name]
        k = h(orig)
        pieces = [next(iter(raw[src][v])) for v in pos]
        if " ".join(pieces) != orig:
            raise ValueError("Actual registered wording differs")
        n = list(range(4))
        canonical = sorted(n, key=lambda j: pos[j])
        if rec["type"] == "original":
            a, b = list(combinations(n, 2))[
                int(h("cogllm-italian-order-swap-v1." + k)[:16], 16) % 6
            ]
            sw = n.copy()
            sw[a], sw[b] = sw[b], sw[a]
            new = sw
        else:
            sw = n
            new = canonical
        alt = " ".join(pieces[j] for j in new)
        other = h(alt)
        if other == k or other in taken:
            raise ValueError("Duplicate augmented training article")
        taken[other] = alt
        indices = [canonical.index(i) for i in sw]
        first = next(i for i in range(1, 4) if indices[i] != indices[i - 1] + 1)
        c = [pieces[j] for j in canonical]
        s = [pieces[j] for j in sw]
        samples[name] = {
            "observed": k,
            "alternative": other,
            "observed_type": rec["type"],
            "group": geo[k]["group"],
            "role": geo[k]["role"],
            "source": src,
            "canonical": h(" ".join(c)),
            "permuted": h(" ".join(s)),
            "canonical_interval": spans(c)[indices[first]],
            "permuted_interval": spans(s)[first],
            "canonical_has_predecessor": indices[first] > 0,
        }
    if len(taken) != 113:
        raise ValueError("Outcome-independent original inventory changed")
    test = {r["TEXT"] for r in csv_rows(ITALIAN / "held.tsv")} | {
        r["text"] for r in rows(ITALIAN / "counterfactuals.jsonl")
    }
    testgrams = set().union(*(grams(t) for t in test))
    held_sources = {v["source"] for v in provenance.values() if v["split"] == "held"}
    dev_phrases = set()
    forbidden = set()
    dev_sources = set()
    for name, s in samples.items():
        g = grams(observations[name]) | grams(taken[s["alternative"]])
        if s["source"] in held_sources or g & testgrams:
            forbidden.add(s["group"])
        if s["role"] == "dev":
            dev_sources.add(s["source"])
            dev_phrases.update(g)
    for name, s in samples.items():
        if s["role"] == "fit" and (
            s["source"] in dev_sources
            or (grams(observations[name]) | grams(taken[s["alternative"]]))
            & dev_phrases
        ):
            forbidden.add(s["group"])
    for s in samples.values():
        s["eligible"] = s["group"] not in forbidden
    file = ITALIAN / "source-counterfactuals.jsonl"
    file.write_text(
        "".join(
            json.dumps({"id": k, "text": v}, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for k, v in sorted(taken.items())
        )
    )
    OUT.mkdir(parents=True, exist_ok=True)
    key = OUT / "source-geometry.json"
    key.write_text(json.dumps(samples, indent=2) + "\n")
    seal = {
        "source": digest(file),
        "plan": digest(PLAN),
        "geometry": digest(key),
        "pairs": len(samples),
        "source_roles": dict(Counter(s["role"] for s in samples.values())),
        "excluded_groups": sorted(forbidden),
        "eligible": dict(Counter(s["role"] for s in samples.values() if s["eligible"])),
        "new_human_held_outcomes_read": False,
    }
    (OUT / "inputs.json").write_text(json.dumps(seal, indent=2) + "\n")
    print(seal)


if __name__ == "__main__":
    create()
