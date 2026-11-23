"""Extract grade-free public sentences for the direct word-understanding experiment."""

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from zipfile import ZipFile
import json

from research.reading_validation.local_forecast import digest


BASE = Path(".cache/cogllm/reading-validation")
DATA = BASE / "nns"
OUT = Path("research/results/reading-validation-nns")
ARCHIVE = BASE / "paetzold-nns.zip"
SOURCE = "261ab745f560f59e10b44d619e58aa3662ab7c75abd3f89a5770a4727bdf5cc2"
MEMBER = "Complex Word Identification/CWI_Annotations.txt"
LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")


def observations():
    if digest(ARCHIVE) != SOURCE:
        raise ValueError("Unexpected release from the original authors")
    with ZipFile(ARCHIVE) as archive, archive.open(MEMBER) as stream:
        for line in stream:
            parts = line.decode("utf-8").rstrip("\r\n").split("\t")
            if len(parts) != 7 or parts[3] not in LEVELS:
                raise ValueError("Unexpected author schema")
            # Identity and demographic fields are outside the empirical prediction target.
            yield tuple(parts[3:])


def material():
    counts = defaultdict(lambda: defaultdict(Counter))
    for level, sentence, word, _ in observations():
        counts[sentence][word][level] += 1
    return counts


def identifier(sentence):
    return sha256(sentence.encode()).hexdigest()


def run():
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True, parents=True)
    groups = defaultdict(list)
    counts = Counter()
    n = 0
    for text, words in material().items():
        multiplicities = {tuple(sorted(v.items())) for v in words.values()}
        if len(multiplicities) != 1:
            raise ValueError("Candidate words saw different proficiency strata")
        visits = sum(dict(next(iter(multiplicities))).values())
        if 5 < visits < 15 or visits > 26:
            raise ValueError("Density thresholds require re-audit")
        counts[visits] += 1
        n += sum(v.total() for v in words.values())
        groups["dense" if visits >= 15 else "sparse"].append(
            dict(id=identifier(text), text=text)
        )
    receipt = {}
    for role, records in sorted(groups.items()):
        target = DATA / (role + ".jsonl")
        with target.open("w") as file:
            for r in sorted(records, key=lambda x: x["id"]):
                file.write(json.dumps(r, ensure_ascii=False) + "\n")
        receipt[role] = {"sentences": len(records), "packet_sha256": digest(target)}
    receipt.update(
        archive=SOURCE, observed_rows=n, judgments_per_word=dict(sorted(counts.items()))
    )
    (OUT / "extraction-inputs.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(receipt)


if __name__ == "__main__":
    run()
