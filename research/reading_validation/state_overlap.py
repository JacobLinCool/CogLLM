"""Audit held physical stories against the actual training and calibration versions."""

from collections import defaultdict
import json
from pathlib import Path

from research.reading_validation.memory_data import load_rows, sha
from research.reading_validation.release_overlap import WINDOW, words
from research.reading_validation.state_transport_data import OUT, TRIP

PLAN = Path(__file__).with_name("state-overlap-protocol.md")


def fingerprints(text):
    tokens = words(text)
    windows = {tokens[i : i + WINDOW] for i in range(len(tokens) - WINDOW + 1)}
    sentences = {part for line in text.splitlines() if len(part := words(line)) >= 4}
    return tokens, windows, sentences


def run():
    records = json.loads((OUT / "inputs.json").read_bytes())["groups"]["physical"]
    text = {row["id"]: row["text"] for row in load_rows(TRIP / "stimuli.jsonl")}
    exact, window, sentence = (defaultdict(set) for _ in range(3))
    for split in ("train", "dev"):
        for key, row in records[split].items():
            tokens, windows, sentences = fingerprints(text[row["key"]])
            label = (split, row["family"], key)
            exact[tokens].add(label)
            for part in windows:
                window[part].add(label)
            for part in sentences:
                sentence[part].add(label)
    matched = {}
    flagged = set()
    for key, row in records["test"].items():
        tokens, windows, sentences = fingerprints(text[row["key"]])
        hits = defaultdict(set)
        for label in exact.get(tokens, ()):
            hits[label].add("identical_article")
        for part in windows:
            for label in window.get(part, ()):
                hits[label].add("twelve_word_sequence")
        common = defaultdict(int)
        for part in sentences:
            for label in sentence.get(part, ()):
                common[label] += 1
        for label, n in common.items():
            if n >= 2:
                hits[label].add("two_whole_sentences")
        if hits:
            flagged.add(row["family"])
            matched[key] = [{"source": label, "matches": sorted(value)} for label, value in sorted(hits.items())]
    families = {r["family"] for r in records["test"].values()}
    result = dict(protocol=sha(PLAN), inputs=sha(OUT / "inputs.json"), held_articles=len(records["test"]),
                  held_families=len(families), excluded_families=sorted(flagged),
                  clean_families=sorted(families - flagged), matched_articles=matched)
    (OUT / "family-exposure.json").write_text(json.dumps(result, indent=2) + "\n")
    print(len(matched), "held articles matched training or development;", len(flagged), "of", len(families), "families affected.")


if __name__ == "__main__":
    run()
