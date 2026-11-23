"""Separate public summary geometry and source exposure from locally scored static visits."""

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.relay_data import CACHE, DATA, SOURCE_HASH
from research.reading_validation.release_overlap import WINDOW, learning_material, words
from research.reading_validation.revision_prepare import (
    earlier_training_texts,
    matches,
    near_terms,
)

OUT = Path("research/results/reading-validation-relay-summary")
PRIVATE = DATA / "summary-analysis"
STIM = DATA / "summary-stimuli.jsonl"
PACKET = "955fbebd97551c526d0149e2ccbdd8b6fd2a30dcf69d456c78fa92a74e5de681"
SCORES = ("simplicity", "coherence", "understanding", "explanation")
ATTENTION = {
    "select all of the following that are colors",
    "select all of the following that are fruits",
    "select the items that are animals",
    "select the items that are articles of clothing",
    "select the items that is an article of clothing",
}


def normalize(t):
    return " ".join(unicodedata.normalize("NFKC", t).split()).casefold()


def inventory():
    source = CACHE / "relay-participant.json"
    if digest(source) != SOURCE_HASH or digest(STIM) != PACKET:
        raise ValueError("Public release or stimulus packet changed")
    docs = {r["id"].removeprefix("relay-summary/"): r["text"] for r in rows(STIM)}
    title = {}
    abstract = {}
    visits = []
    answers = Counter()
    for subject, participant in enumerate(json.loads(source.read_bytes())):
        batch = participant["phases"]["static"]["batches"].values()
        complete = [
            (k, e)
            for b in batch
            for k, e in b.get("abstracts", {}).items()
            if e["completed"]
        ]
        if len(complete) not in (3, 6):
            raise ValueError("Unspecified number of completed visits")
        for aid, e in complete:
            if not (e["likert_submitted"] and e["sata_submitted"]):
                raise ValueError("Completed visit missing submitted instrument")
            if (
                docs[aid] != e["human_written_pls"]
                or title.setdefault(aid, normalize(e["abstract_title"]))
                != normalize(e["abstract_title"])
                or abstract.setdefault(aid, normalize(e["abstract"]))
                != normalize(e["abstract"])
            ):
                raise ValueError("Article text fields changed between reader visits")
            rate = {}
            for k in SCORES:
                value = e["likert"]["responses"][k]
                found = re.fullmatch(r"([1-5])(?:\s*[-–—]\s*(.+))?", str(value))
                if not found:
                    raise ValueError("Unrecognized ordinal response")
                v = int(found[1])
                if (
                    found[2]
                    and found[2].casefold()
                    != ("very poor", "poor", "fair", "good", "excellent")[v - 1]
                ):
                    raise ValueError("Ordinal answer code and name disagree")
                rate[k] = v
            score = []
            option = []
            attention = []
            for i in range(1, 6):
                q = normalize(e[f"question_{i}"])
                opts = {
                    normalize(s)
                    for s in e[f"question_{i}_answers_choices"].split(";")
                    if s.strip()
                }
                key = {
                    normalize(s)
                    for s in e[f"question_{i}_correct_answers"].split(";")
                    if s.strip()
                }
                selected = {normalize(s) for s in e["sata"]["sata_answers"][f"q{i}"]}
                if not key <= opts or not selected <= opts:
                    raise ValueError(
                        "A released answer differs from the displayed options"
                    )
                hit = int(selected == key)
                if q in ATTENTION:
                    attention.append(hit)
                    answers[i] += 1
                else:
                    if len(opts) != 3:
                        raise ValueError("Expected three options in a content item")
                    score.append(hit)
                    option.append(sum((o in selected) == (o in key) for o in opts))
            if len(attention) != 1 or len(score) != 4:
                raise ValueError("Invalid content/attention split")
            prior = [r["familiarity_score"] for r in e["term_familarity"]]
            if not prior or any(v not in range(1, 6) for v in prior):
                raise ValueError("Invalid pre-summary term ratings")
            visits.append(
                {
                    "reader": subject,
                    "article": aid,
                    "regular": len(complete) == 3,
                    **rate,
                    "correct": sum(score),
                    "option_decisions": sum(option),
                    "check": attention[0],
                    "prior": sum(prior) / len(prior),
                }
            )
    return docs, title, abstract, visits, dict(answers)


def clusters(docs, title, abstract):
    keys = sorted(docs)
    t = [docs[k] for k in keys]
    grams, inv = near_terms(t)
    parent = list(range(len(t)))

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i, k in enumerate(keys):
        near = matches(t[i], grams, inv)
        for j in range(i):
            h = keys[j]
            if (
                j in near
                or title[k] == title[h]
                or abstract[k] == abstract[h]
                or normalize(t[i]) == normalize(t[j])
            ):
                a, b = root(i), root(j)
                parent[max(a, b)] = min(a, b)
    return {k: keys[root(i)] for i, k in enumerate(keys)}


def exposure(docs, groups):
    teacher = {
        "PLABA_CLeAR_students_ReadMe": earlier_training_texts(),
        "CompLex": {
            r["text"]
            for r in rows(CACHE.parent / "stage5/complex-l4-input/stimuli.jsonl")
            if r["split"] in ("train", "trial")
        },
        "noun_intervention": {
            r["text"] for r in rows(CACHE / "noun/local-stimuli.jsonl")
        },
    }
    historical, _, receipt = learning_material()
    exact = defaultdict(set)
    windows = defaultdict(set)
    for k, s in docs.items():
        seq = words(s)
        exact[seq].add(k)
        for i in range(len(seq) - WINDOW + 1):
            windows[seq[i : i + WINDOW]].add(k)
    result = {}
    exclude = set()
    for source, materials in {
        **teacher,
        **{"historical_" + k: v for k, v in historical.items()},
    }.items():
        full = set()
        part = set()
        strong = set()
        for s in materials:
            seq = words(s)
            full.update(exact.get(seq, ()))
            for i in range(len(seq) - WINDOW + 1):
                part.update(windows.get(seq[i : i + WINDOW], ()))
        if source in teacher:
            grams, inv = near_terms(sorted(materials))
            strong = {k for k, s in docs.items() if matches(s, grams, inv)}
            exclude.update(groups[k] for k in (strong | full))
        result[source] = {
            "prior_strings": len(materials),
            "exact": sorted(full),
            "twelve": sorted(part),
            "strong": sorted(strong) if source in teacher else None,
        }
    return result, exclude, receipt


def main():
    OUT.mkdir(exist_ok=True)
    PRIVATE.mkdir(exist_ok=True)
    docs, title, abstract, visits, slots = inventory()
    group = clusters(docs, title, abstract)
    exp, excluded, history = exposure(docs, group)
    for v in visits:
        v["family"] = group[v["article"]]
        v["eligible"] = v["family"] not in excluded
    # Private group IDs, article keys and ratings are never written to a published row-level table.
    ph = PRIVATE / "static-visits.jsonl"
    ph.write_text("".join(json.dumps(v, sort_keys=True) + "\n" for v in visits))
    eligible = [v for v in visits if v["regular"] and v["eligible"]]
    result = {
        "study_commit": "860f6bc5c2a23e5ef8ed22bdf86ebc4f960ebec9",
        "source_sha256": SOURCE_HASH,
        "packet_sha256": PACKET,
        "private_response_sha256": digest(ph),
        "public_article_families": group,
        "total_visits": len(visits),
        "articles": len(docs),
        "distinct_families": len(set(group.values())),
        "primary_visits": len(eligible),
        "primary_subjects": len({v["reader"] for v in eligible}),
        "primary_articles": len({v["article"] for v in eligible}),
        "primary_families": len({v["family"] for v in eligible}),
        "attention_slots": slots,
        "exposure": exp,
        "excluded_families": sorted(excluded),
        "historical_receipt": history,
        "protocol_sha256": digest(
            Path(__file__).with_name("relay-summary-protocol.md")
        ),
        "code_sha256": digest(Path(__file__)),
    }
    (OUT / "inputs.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        {
            k: v
            for k, v in result.items()
            if k
            in [
                "primary_visits",
                "primary_subjects",
                "primary_articles",
                "primary_families",
                "attention_slots",
                "excluded_families",
                "distinct_families",
            ]
        }
    )


if __name__ == "__main__":
    main()
