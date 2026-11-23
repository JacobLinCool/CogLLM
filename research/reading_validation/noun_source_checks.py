"""Compare the released complete text to a literal restoration of the four-noun answer key."""

from collections import Counter
import json
from pathlib import Path
import re

from openpyxl import load_workbook

from research.reading_validation.local_forecast import digest
from research.reading_validation.noun_data import BASE, OUT, SHA
from research.reading_validation.release_overlap import words


def main():
    path = BASE / "leroy.xlsx"
    if digest(path) != SHA:
        raise ValueError("Unexpected source workbook")
    b = load_workbook(path, read_only=True, data_only=True)
    data = iter(b["Data"].values)
    keys = next(data)
    seen = Counter()
    counts = Counter()
    flag = []
    for v in data:
        r = dict(zip(keys, v, strict=True))
        gram, split, pseudo = r["GRAM"], r["compound"], r["pseudoword"]
        group = (gram, split, pseudo)
        family = f"{gram}-{seen[group]:02d}"
        seen[group] += 1
        raw = r["Sentence with blanks"]
        solutions = [t.strip() for t in r["Correct Solution"].split(",")]
        slots = re.findall(r"_{2,}", raw)
        if len(slots) != 4 or len(solutions) != 4:
            status = "not_four_slots_and_nouns"
        else:
            it = iter(solutions)
            restored = re.sub(r"_{2,}", lambda _: next(it), raw)
            status = (
                "same_word_sequence"
                if words(restored) == words(r["Original Sentence"])
                else "different_word_sequence"
            )
        counts[status] += 1
        if status != "same_word_sequence":
            flag.append(
                {
                    "family": family,
                    "compound": split,
                    "pseudoword": pseudo,
                    "status": status,
                }
            )
    b.close()
    payload = {
        "original_workbook_sha256": SHA,
        "program_sha256": digest(Path(__file__)),
        "comparison": "Restore the four released answers in their given order at the four blank sequences, then compare case-normalized alphanumeric words",
        "counts": dict(counts),
        "flags": flag,
    }
    (OUT / "source-restoration.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(counts)


if __name__ == "__main__":
    main()
