"""Assemble text through the focal sentence and keep reader votes off the accelerator."""

from collections import Counter
import csv
import json
from pathlib import Path

from research.reading_validation.external_input import digest_string, material_key
from research.reading_validation.joint_data import DEFAULT_CACHE, digest

BASE = DEFAULT_CACHE / "reading-validation"
RAW = BASE / "discogem"
OUT = Path("research/results/reading-validation-discourse")
PROTO = Path(__file__).with_name("discourse-protocol.md")


def read_table(name, delimiter):
    with (RAW / name).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream, delimiter=delimiter)


def run():
    public = {r["itemid"]: r["domain"] for r in read_table("15.csv", "\t")}
    rows = []
    texts = {}
    mismatch = Counter()
    for r in read_table("wide.csv", ","):
        uid = r["itemid"]
        if uid not in public:
            continue
        a, b = r["arg1_singlesentence"].strip(), r["arg2_singlesentence"].strip()
        left = r["arg1"].strip()
        if not (a and b and left):
            raise ValueError(f"Missing original argument: {uid}")
        doc = left + "\n\n" + b
        key = digest_string(doc)
        texts[key] = doc
        genre = r["genre"]
        if {"novel":"NV", "europarl":"EP", "wikipedia":"WK"}[genre] != public[uid]:
            mismatch[genre] += 1
        source = r["subgenre"] if genre == "novel" else r["datasource"]
        group = digest_string(genre + "\n" + source)
        rows.append(dict(id=key, item=uid, genre=genre, group=group, left_end=len(left),
                         first=material_key(a), second=material_key(b)))
    if {r["item"] for r in rows} != set(public):
        raise ValueError("The two author releases do not cover identical items")
    wiki_sentences = {r[k] for r in rows if r["genre"] == "wikipedia" for k in ("first", "second")}
    exclude = {r["group"] for r in rows if r["genre"] != "wikipedia"
               and (r["first"] in wiki_sentences or r["second"] in wiki_sentences)}
    retained = []
    seen = set()
    for r in rows:
        key = (r["group"], r["first"], r["second"])
        if r["group"] in exclude or key in seen:
            continue
        retained.append(r)
        seen.add(key)
    packet = BASE / "discourse-stimuli.jsonl"
    core = set(r["id"] for r in retained)
    # Add independently identified, already published DECOR sentence stops without human scores.
    meta = json.loads(Path("research/results/reading-validation-transition-external/stimuli.json").read_bytes())
    external = {r["id"] for r in meta["material"] if r["source"] == "decor" and r["split"] == "external"}
    ext_map = {}
    for row in (json.loads(line) for line in (BASE/"cohesion-stimuli.jsonl").read_text().splitlines()):
        for stop in row["stops"]:
            if stop["id"] in external:
                doc = row["text"][:stop["end"]]
                k = digest_string(doc)
                texts[k] = doc
                ext_map[stop["id"]] = dict(id=k, left_end=stop["start"])
    if set(ext_map) != external or not ext_map:
        raise ValueError("Missing essay material")
    ids = sorted(core | {r["id"] for r in ext_map.values()})
    packet.write_text("".join(json.dumps(dict(id=k,text=texts[k]), ensure_ascii=False)+"\n" for k in ids))
    OUT.mkdir(parents=True, exist_ok=True)
    for name,data in (("materials",retained),("external",ext_map)):
        (OUT/(name+".json")).write_text(json.dumps(data, separators=(",",":"))+"\n")
    receipt = dict(revision="635515c06259ae40696649cba1d9f097418c8bd4",protocol=digest(PROTO),
                   source={k:digest(RAW/k) for k in ("15.csv","wide.csv","item.csv")},
                   packet_sha256=digest(packet),file_count=len(ids),rows=len(retained),
                   counts=dict(Counter(r["genre"] for r in retained)),targets=len(ext_map),
                   inconsistent_genre_in_reannotation=dict(mismatch),excluded_groups=len(exclude),
                   excluded_rows=len(rows)-len(retained))
    (OUT/"inputs.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt))


if __name__=="__main__":
    run()
