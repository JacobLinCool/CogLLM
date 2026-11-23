"""Create complete adjacent-unit alternatives from the authors' randomly displayed lists only."""

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen

from research.reading_validation.argument_data import sentences
from research.reading_validation.coherence_order_data import OUT as COHERENCE
from research.reading_validation.coherence_order_data import read, texts
from research.reading_validation.danish_coherence_data import grams
from research.reading_validation.local_forecast import digest
from research.reading_validation.memory_data import BASE, h

TOPICS = (
    "congo",
    "crash",
    "disease",
    "ebola",
    "greenspan",
    "mozambique",
    "salvadorA",
    "salvadorB",
    "taliban",
    "tdt89",
)
URL = "https://www.cs.columbia.edu/~noemie/ordering/coll/multi/"
WORK = BASE / "columbia-order"
OUT = Path("research/results/reading-validation-columbia-order")
PLAN = Path(__file__).with_name("columbia-order-protocol.md")


class Cells(HTMLParser):
    """Keep text in actual HTML table cells in display order; tolerate omitted row end tags."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
            self.rows.append(self.row)
        if tag == "td" and self.row is not None:
            self.cell = []
            self.row.append(self.cell)
        if tag == "br" and self.cell is not None:
            self.cell.append(" ")

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "tr"):
            self.cell = None
        if tag == "tr":
            self.row = None

    def extracted(self):
        return [[" ".join("".join(c).split()) for c in row] for row in self.rows if row]


def fetch(topic):
    if topic not in TOPICS:
        raise ValueError("Unknown topic")
    target = WORK / "random-units" / f"{topic}.html"
    if not target.is_file():
        with urlopen(URL + topic + ".html", timeout=30) as reader:
            body = reader.read()
        if b"Sentences in Random Order" not in body:
            raise ValueError("This is not a randomized input list")
        target.write_bytes(body)


def units():
    docs = {}
    for topic in TOPICS:
        page = (
            (WORK / "random-units" / f"{topic}.html")
            .read_bytes()
            .decode("windows-1252")
        )
        p = Cells()
        p.feed(page)
        source = {}
        for r in p.extracted():
            if len(r) != 2 or not re.fullmatch(r"\[[A-Z]\]", r[0]) or not r[1]:
                raise ValueError(
                    f"Unexpected source unit structure in {topic}: {[s[:5] for s in r]}"
                )
            k = r[0][1]
            if k in source:
                raise ValueError("Duplicate human-readable random-list ID")
            source[k] = r[1]
        if len(source) < 3 or len(set(source.values())) != len(source):
            raise ValueError("Insufficient distinct units")
        docs[topic] = source
    return docs


def make():
    docs = units()
    source_text = texts()
    geo = read(COHERENCE / "geometry.json")
    trained = set()
    for key, meta in geo.items():
        if meta["role"] in ("train", "fit", "dev"):
            trained.update(grams(source_text[key]))
    if not trained:
        raise ValueError("Missing actual supervised source")
    index = {}
    data = {}
    exclusion = []
    for topic, parts in docs.items():
        for unit, t in parts.items():
            if trained & grams(t):
                exclusion.append((topic, unit))
        for i, left in parts.items():
            for j, right in parts.items():
                if i == j:
                    continue
                t = left + "\n" + right
                k = h(t)
                data[k] = t
                start = len(left) + 1
                target = next((s for s in sentences(t) if s[0] == start), None)
                if target is None:
                    raise ValueError("Joined unit does not start a visible sentence")
                index[f"{topic}/{i}/{j}"] = {
                    "article": k,
                    "target": target,
                    "left": h(left),
                    "right": h(right),
                }
    (WORK / "local-stimuli.jsonl").write_text(
        "".join(
            json.dumps({"id": k, "text": v}, ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for k, v in sorted(data.items())
        )
    )
    (OUT / "pair-geometry.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    counts = {k: len(v) for k, v in docs.items()}
    coverage = {
        "protocol": digest(PLAN),
        "geometry": digest(OUT / "pair-geometry.json"),
        "packet": digest(WORK / "local-stimuli.jsonl"),
        "topics": counts,
        "number_of_units": sum(counts.values()),
        "expected_ordered_pairs": sum(n * (n - 1) for n in counts.values()),
        "distinct_pair_texts": len(data),
        "unit_intersection_with_bilingual_fit": exclusion,
        "source_inputs": {
            k: digest(WORK / "random-units" / f"{k}.html") for k in TOPICS
        },
        "published_order_files_opened": False,
        "multi_sentence_units": dict(
            Counter(
                len(sentences(t)) for parts in docs.values() for t in parts.values()
            )
        ),
    }
    (OUT / "inputs.json").write_text(json.dumps(coverage, indent=2) + "\n")
    print(coverage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        (WORK / "random-units").mkdir(exist_ok=True, parents=True)
        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(fetch, TOPICS))
    OUT.mkdir(exist_ok=True, parents=True)
    make()
