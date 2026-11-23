"""Assemble text-only inputs and an independent inventory of observed task opportunities."""

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

from research.reading_validation.memory_data import BASE, h, load_rows, sha

WORK = BASE / "expectation"
OUT = Path("research/results/reading-validation-expectation")
PLAN = Path(__file__).with_name("expectation-protocol.md")
PUBLIC = {
    "michaelov": ("michaelov_2023.stims", "michaelov_2024.tsv", "ContextCode", "TargetWords"),
    "brothers": ("brothers_2021.stims", "brothers_2021.tsv", "ITEM", "critical_word"),
    "szewczyk": ("szewczyk_2022.stims", "szewczyk_2022.tsv", "Item", "TargetWords"),
}


def prepare():
    OUT.mkdir(exist_ok=True)
    articles, units, files = {}, {}, {}
    mapping = {}
    for study, (stimuli, data, family, target) in PUBLIC.items():
        files[stimuli] = sha(WORK / stimuli)
        files[data] = sha(WORK / data)
        original = {}
        for line in (WORK / stimuli).read_text().splitlines():
            marked = re.fullmatch(r"(.*)\*([^*]+)\*(.*)", line)
            if not marked:
                raise ValueError(f"Unrecognized source coordinates in {study}")
            left, word, right = marked.groups()
            plain = left + word + right
            original[plain] = (len(left), len(left) + len(word))
        def letters(text):
            return "".join(c for c in text.casefold() if c.isalnum())
        index = defaultdict(set)
        for wording in original:
            index[letters(wording)].add(wording)
        records = {}
        mapping_status = Counter()
        rows = 0
        families = defaultdict(set)
        with (WORK / data).open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                released = row["FullSentence"]
                text = released
                if released not in original:
                    matches = index[letters(released)]
                    if len(matches) != 1:
                        mapping_status["unresolved_source_wording"] += 1
                        continue
                    text = next(iter(matches))
                a,b = original[text]
                if letters(text[a:b]) != letters(row[target]):
                    mapping_status["different_word"] += 1
                    continue
                mapping_status["exact" if text == released else "unique_lexical_sequence"] += 1
                k = h(text)
                articles[k] = text
                lab = row.get("dataset", study)
                group = row[family]
                families[lab].add(group)
                ident = (lab, k, group)
                records[ident] = dict(study=lab, family=group, id=k, span=[a,b], source_id=h(released))
                rows += 1
        units[study] = dict(verified_rows=rows, alignment=mapping_status, studied_articles=len({r["id"] for r in records.values()}),
                            families={k: len(v) for k,v in families.items()})
        mapping[study] = list(records.values())
    # The sentence ending inventory alone is adequate for new experiments. Historical
    # passage stimuli require every word to locate competition before a critical sentence.
    mapping["frick"] = {}
    for row in load_rows(BASE / "memory/stories.jsonl"):
        text = "\n\n".join(row["sentences"])
        articles[h(text)] = text
        mapping["frick"][row["id"]] = h(text)
    textfile = WORK / "stimuli.jsonl"
    textfile.write_text("".join(json.dumps({"id":k,"text":v},ensure_ascii=False)+"\n" for k,v in sorted(articles.items())))
    keys = WORK / "opportunities.json"
    keys.write_text(json.dumps(mapping,ensure_ascii=False)+"\n")
    seal = dict(registration=sha(PLAN),files=files,text=sha(textfile),opportunities=sha(keys),
                original_revision="1c7152a2b10ed309130932b0d7f624ee6c9b85e3",
                inventories=units,unique_texts=len(articles))
    (OUT / "input-seal.json").write_text(json.dumps(seal,indent=2)+"\n")
    print(units, len(articles))


if __name__ == "__main__":
    prepare()
