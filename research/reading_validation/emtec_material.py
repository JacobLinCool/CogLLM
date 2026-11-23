"""Reconstruct the 42 Latin-square items using only text visible before human responses."""

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile

from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.nns_exposure import grams, sources
from research.reading_validation.nns_material import BASE, DATA as NNS
from research.reading_validation.revision_prepare import matches, near_terms
from research.reading_validation.release_overlap import words

DATA = BASE / "emtec"
OUT = Path("research/results/reading-validation-emtec")
RELEASE = "83c64a2020f5127598138143317fde993e41bf6f"


def identifier(t):
    return sha256(t.encode()).hexdigest()


def geometry():
    with (DATA / "public-material.csv").open(newline="") as stream:
        docs = {}
        for row in csv.DictReader(stream):
            header = row["introductory_screen_item"]
            body = row["gen_seq_trunc"]
            text = header + "\n\n" + body
            key = (row["item_id"], row["list"])
            piece = {
                "id": identifier(text),
                "text": text,
                "body_start": len(header) + 2,
            }
            if key in docs:
                raise ValueError("Duplicate released item/list")
            docs[key] = {
                "public": piece,
                "body": body,
                "header": header,
                "answer_file": row["filename"],
                "genre": row["task"],
                "generation": row["model"] + "/" + row["decoding_strategy"],
            }
    return docs


def exposures(texts):
    g, ix = near_terms(texts)
    norm = defaultdict(set)
    ng = defaultdict(set)
    for i, t in enumerate(texts):
        u = words(t)
        norm[u].add(i)
        for gram in grams(u, 12):
            ng[gram].add(i)
    prior, _ = sources()
    prior["both_nns_words"] = {
        r["text"]
        for group in ("dense", "sparse")
        for r in rows(NNS / (group + ".jsonl"))
    }
    prior["nns_revision_material"] = {
        r["text"] for r in rows(BASE / "nns-rewrite/stimuli.jsonl")
    }
    aux = [
        "guardian/stimuli.jsonl",
        "relay/summary-stimuli.jsonl",
        "relay/local-stimuli.jsonl",
        "local-stimuli.jsonl",
        "ats-comprehension/local-stimuli.jsonl",
    ]
    for filename in aux:
        p = BASE / filename
        if p.is_file():
            prior["additional_" + filename] = {r["text"] for r in rows(p)}
    out = {}
    bad = set()
    for label, docs in prior.items():
        found = defaultdict(set)
        for t in docs:
            u = words(t)
            found["exact"].update(norm.get(u, ()))
            for z in grams(u, 12):
                found["shared_12"].update(ng.get(z, ()))
            found["near"].update(matches(t, g, ix))
        hit = set().union(*found.values())
        bad.update(hit)
        out[label] = {k: len(v) for k, v in found.items()} | {
            "any": len(hit),
            "source_texts": len(docs),
        }
    return bad, out


def run():
    docs = geometry()
    with (DATA / "shown-stimuli.csv").open() as stream:
        for doc in csv.DictReader(stream, delimiter="\t"):
            item=docs[doc["item_id"],doc["list"]]
            if (doc["gen_seq_trunc"],doc["introductory_screen_item"])!=(item["body"],item["header"]):
                raise ValueError("Final OSF display and publication's GitHub stimulus disagree")
    material = sorted({p["body"] for p in docs.values()}, key=identifier)
    bad, audit = exposures(material)
    columns = set()
    joins = Counter()
    seen = defaultdict(set)
    levels = set()
    ap = DATA / "response-archive.zip"
    with ZipFile(ap) as z, z.open("participant_info/participant_results.csv") as f:
        import io

        reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t")
        columns = set(reader.fieldnames)
        for row in reader:
            key = (row["item_id"], row["list"])
            j = docs.get(key)
            if j is None:
                joins["outside_design"] += 1
                continue
            def lexemes(value):
                return re.findall(r"[^\W_]+",unicodedata.normalize("NFKC",value.replace(r"\n","\n")).casefold())
            if lexemes(row["gen_seq_trunc"])!=lexemes(j["body"]) or lexemes(row["introductory_screen_item"])!=lexemes(j["header"]):
                raise ValueError("The participant log contains words absent from the displayed release")
            if (row["gen_seq_trunc"],row["introductory_screen_item"])==(j["body"],j["header"]):
                joins["exact_including_log_formatting"]+=1
            joins["aligned"] += 1
            seen[row["subject_id"]].add(key)
            levels.add(key)
    original_texts = {p["public"]["id"]: p["public"] for p in docs.values()}
    omitted = {material[i] for i in bad}
    withheld = {k[0] for k,v in docs.items() if v["body"] in omitted}
    ids = {v["public"]["id"] for k,v in docs.items() if k[0] in withheld}
    eligible = {k: v for k, v in original_texts.items() if k not in ids}
    OUT.mkdir(parents=True, exist_ok=True)
    p = DATA / "stimuli.jsonl"
    p.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for k, row in sorted(eligible.items())
        )
    )
    index = {
        str(k[0]) + "/" + str(k[1]): {
            "id": v["public"]["id"],
            "genre": v["genre"],
            "generation": v["generation"],
        }
        for k, v in docs.items()
        if v["public"]["id"] in eligible
    }
    g = DATA / "geometry.json"
    g.write_text(json.dumps(index, sort_keys=True) + "\n")
    release = {
        "original_code_revision": RELEASE,
        "public_material_sha256": digest(DATA / "public-material.csv"),
        "final_presented_material_sha256": digest(DATA / "shown-stimuli.csv"),
        "response_archive_sha256": digest(ap),
        "expected_response_columns": sorted(
            k for k in columns if k.startswith(("RATING", "QUESTION"))
        ),
        "outcome_values_inspected": False,
        "public_packet_sha256": digest(p),
        "public_material_rows": len(docs),
        "unique_headers": len({v["header"] for v in docs.values()}),
        "unique_bodies": len(material),
        "unique_header_body": len(original_texts),
        "excluded_bodies": len(omitted),
        "excluded_complete_prompt_families":sorted(withheld),
        "eligible_prompt_families":len({k[0] for k in docs}-withheld),
        "eligible_stimuli": len(eligible),
        "participating_readers": len(seen),
        "released_item_list_pairs_with_a_visit": len(levels),
        "visits_by_geometry": dict(joins),
        "visits_per_reader": dict(Counter(map(len, seen.values()))),
        "historical_source_exposure": audit,
        "local_geometry_sha256": digest(g),
        "preparation_sha256": digest(Path(__file__)),
    }
    (OUT / "material.json").write_text(json.dumps(release, indent=2) + "\n")
    print({k: v for k, v in release.items() if k != "historical_source_exposure"})


if __name__ == "__main__":
    run()
