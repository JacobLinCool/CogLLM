"""Register complete English SEAHORSE summary exposures while withholding reader outcomes."""
from collections import Counter
import csv
from hashlib import sha256
from io import TextIOWrapper
import json
from pathlib import Path
from zipfile import ZipFile

from research.reading_validation.memory_data import h

WORK = Path(".cache/cogllm/reading-validation")
ZIP = WORK / "seahorse-original.zip"
DEST = Path("research/results/reading-validation-transition-external/seahorse")
PLAN = Path(__file__).with_name("transition-seahorse-protocol.md")
EXPECTED = "0a39d585e36b5f7d9282da2605f8ff8fa35bef35327b2b46509ae285a1add60b"
ROLES = ("test", "validation", "train")


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def source_rows(role, *, duplicates=False, grades=False):
    if role not in ROLES:
        raise ValueError("Unknown official partition")
    name = ("duplicates/duplicates_" if duplicates else "") + role + ".tsv"
    with ZipFile(ZIP) as data, data.open("seahorse_data/" + name) as raw:
        records = csv.DictReader(TextIOWrapper(raw, encoding="utf8", newline=""), delimiter="\t")
        for r in records:
            if r["worker_lang"] == "en-US":
                yield {k: r[k] for k in ("gem_id", "model", "summary", *(("question1",) if grades else ()))}


def shard(story):
    return int(h("cogllm-seahorse-v1/" + story)[:8], 16) % 4


def main():
    if sha(ZIP) != EXPECTED:
        raise ValueError("Unrecognized SEAHORSE official source")
    DEST.mkdir(parents=True, exist_ok=True)
    by_key = {}
    stats = {}
    packets = {k: {} for k in ("test", "validation", *(f"train{i}" for i in range(4)))}
    for role in ROLES:
        entries = list(source_rows(role))
        stats[role] = {"rows": len(entries), "articles": len({r["gem_id"] for r in entries}),
                       "strings": len({r["summary"] for r in entries}), "models": dict(Counter(r["model"] for r in entries))}
        for r in entries:
            k = h(r["summary"])
            name = role + str(shard(r["gem_id"])) if role == "train" else role
            if k not in by_key:
                by_key[k] = name
                packets[name][k] = r["summary"]
    manifest = {}
    for name, items in packets.items():
        p = WORK / f"seahorse-{name}-stimuli.jsonl"
        with p.open("w", encoding="utf8") as stream:
            for k, t in sorted(items.items()):
                stream.write(json.dumps({"id": k, "text": t}, ensure_ascii=False) + "\n")
        manifest[name] = {"file": str(p), "sha256": sha(p), "unique": len(items)}
    document = {"source": EXPECTED, "protocol": sha(PLAN), "input_only_rows": stats,
                "total_distinct": len(by_key), "packets": manifest}
    (DEST/"exposure-seal.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
