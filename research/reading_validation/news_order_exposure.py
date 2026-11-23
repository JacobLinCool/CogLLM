"""Print material intersections with the supervised checkpoints used by both news-order observers."""

import json
from collections import defaultdict

from research.reading_validation.columbia_order_data import units
from research.reading_validation.emtec_material import exposures
from research.reading_validation.local_forecast import rows
from research.reading_validation.danish_coherence_data import grams
from research.reading_validation.summeval_data import WORK, OUT
from research.reading_validation.coherence_order_data import read


def main():
    target = rows(WORK / "local-stimuli.jsonl")
    ids = [r["id"] for r in target]
    strings = [r["text"] for r in target]
    over, counts = exposures(strings)
    source = units()
    by_raw = defaultdict(set)
    for k, group in source.items():
        for text in group.values():
            for gram in grams(text):
                by_raw[gram].add(k)
    extra = {}
    for k, text in zip(ids, strings, strict=True):
        common = {v for window in grams(text) for v in by_raw.get(window, ())}
        if common:
            extra[k] = sorted(common)
    geo = read(OUT / "story-geometry.json")
    roleset = defaultdict(set)
    for group, rec in geo.items():
        for k in rec["systems"].values():
            roleset[k].add((rec["role"], group))
    aux = {k: sorted(roleset[k]) for k in sorted(ids[i] for i in over)}
    output = {
        "unique_drafts": len(ids),
        "sources": counts,
        "any_historical_supervised_match": aux,
        "columbia_human_order_material_shared_12": extra,
        "no_ballots_consulted": True,
    }
    (OUT / "historical-exposure.json").write_text(json.dumps(output, indent=2) + "\n")
    print(
        "unique news drafts with possible earlier supervised source",
        len(aux),
        "Columbia overlap",
        len(extra),
    )


if __name__ == "__main__":
    main()
