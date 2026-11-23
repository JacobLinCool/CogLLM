"""Seal paragraph geometry and test previous supervision before examining OneStop responses."""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from research.reading_validation.external_input import digest_string, material_key
from research.reading_validation.joint_data import digest, lines
from research.reading_validation.release_overlap import WINDOW, learning_material, words
from research.reading_validation.revision_prepare import (
    BASE,
    SOURCE,
    earlier_training_texts,
    geometry,
    matches,
    near_terms,
)
from research.reading_validation.revision_prepare import (
    OUT as REVISION,
)

DATA = BASE / "ats-comprehension"
RAW = SOURCE / "ats_eval.csv"
RESULT = Path("research/results/reading-validation-comprehension")
PIN = "c59328ab111b59e954763707b1051cf34feeead6"
RAW_HASH = "e4cf8464a22b543686f800c4705e126cd74ded2a3b386f865a7e7f7ff6c8acc6"
LEXICAL_HASH = "caef04fb8b90154e208e528fee39fc2ccba96cd34a6945fac836771427784254"
PACKET_HASH = "d2d13db6f3ac472d7fd6c92db136a0a6db9bd9e1d6c0f5517ae8e3d886c20de1"


def versions():
    if digest(RAW) != RAW_HASH:
        raise ValueError("Pinned human study has changed")
    families = defaultdict(dict)
    questions = defaultdict(set)
    with RAW.open(newline="", encoding="utf8") as stream:
        for r in csv.DictReader(stream):
            key = r["article_id"], r["para_id"]
            name = r["model_name"]
            if name in families[key] and families[key][name] != r["paragraph"]:
                raise ValueError(f"Conflicting version text for {name} in {key}")
            families[key][name] = r["paragraph"]
            questions[key].add(tuple(r[f"question_{i}"] for i in (1, 2, 3)))
    if len(families) != 60 or len({a for a, _ in families}) != 30:
        raise ValueError("Unexpected test study geometry")
    if (
        sum(map(len, families.values())) != 658
        or any("Adv" not in v for v in families.values())
        or any(len(q) != 1 for q in questions.values())
    ):
        raise ValueError("The paired passages do not share the same three questions")
    return dict(families)


def overlap(families):
    prior = defaultdict(set)
    prior["absolute_reading_heads"] = earlier_training_texts()
    for (split, _), texts in geometry().items():
        if split == "train":
            prior["revision_training"].update(texts)
    for row in lines(BASE.parent / "stage5/complex-l4-input/stimuli.jsonl"):
        if row["split"] in ("train", "trial"):
            prior["complex_lexical"].add(row["text"])
    direct = {}
    for study, texts in sorted(prior.items()):
        normalized = {material_key(text) for text in texts}
        grams, inverted = near_terms(sorted(texts))
        affected = set()
        for (article, _), group in families.items():
            if any(
                material_key(t) in normalized or matches(t, grams, inverted)
                for t in group.values()
            ):
                affected.add(article)
        direct[study] = sorted(affected)
    # This is a program-wide material check. The revision predictor takes no historically learned
    # portable-reader states. The OneStop questions never enter the revision model.
    target_index = defaultdict(set)
    for (article, _), group in families.items():
        for t in group.values():
            seq = words(t)
            for start in range(len(seq) - WINDOW + 1):
                target_index[seq[start : start + WINDOW]].add(article)
    old, _, provenance = learning_material()
    historical = {}
    for role, texts in old.items():
        affected = set()
        for t in texts:
            seq = words(t)
            for i in range(len(seq) - WINDOW + 1):
                affected.update(target_index.get(seq[i : i + WINDOW], ()))
        historical[role] = sorted(affected)
    return direct, historical, provenance


def main():
    families = versions()
    DATA.mkdir(parents=True, exist_ok=True)
    RESULT.mkdir(parents=True, exist_ok=True)
    strings = sorted(
        {t for info in families.values() for t in info.values()}, key=digest_string
    )
    packet = DATA / "stimuli.jsonl"
    packet.write_text(
        "".join(
            json.dumps({"id": digest_string(t), "text": t}, ensure_ascii=False) + "\n"
            for t in strings
        )
    )
    if digest(packet) != PACKET_HASH or digest(SOURCE / "lexical.npz") != LEXICAL_HASH:
        raise ValueError("Native extraction packet mismatch")
    shutil.copyfile(SOURCE / "lexical.npz", DATA / "lexical.npz")
    direct, historical, provenance = overlap(families)
    excluded = set().union(*direct.values())
    seal = {
        "author_release_revision": PIN,
        "responses_file_sha256": RAW_HASH,
        "packet_sha256": PACKET_HASH,
        "lexical_sha256": LEXICAL_HASH,
        "passages": len(strings),
        "families": len(families),
        "articles": len({a for a, _ in families}),
        "versions": sum(map(len, families.values())),
        "explicit_prior_study_articles": direct,
        "excluded_from_primary": sorted(excluded),
        "historical_portable_reader_source_articles": historical,
        "historical_reader_receipt": provenance,
        "revision_seal_sha256": digest(REVISION / "forecast/forecast-seal.json"),
        "protocol_sha256": digest(
            Path(__file__).with_name("revision-comprehension-protocol.md")
        ),
        "preparation_sha256": digest(Path(__file__)),
    }
    (RESULT / "input-seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: seal[k]
                for k in (
                    "passages",
                    "articles",
                    "excluded_from_primary",
                    "historical_portable_reader_source_articles",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
