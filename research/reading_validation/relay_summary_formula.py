"""Evaluate the authors' four readability formulas on public English summaries."""

import json
import unicodedata
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import textstat

PUBLIC = Path(".cache/cogllm/reading-validation/relay/summary-stimuli.jsonl")
OUT = Path("research/results/reading-validation-relay-summary")


def main():
    if (
        version("textstat") != "0.7.10"
        or sha256(PUBLIC.read_bytes()).hexdigest()
        != "955fbebd97551c526d0149e2ccbdd8b6fd2a30dcf69d456c78fa92a74e5de681"
    ):
        raise ValueError("Unrecognized public input or readability implementation")
    items = {}
    for line in PUBLIC.read_text().splitlines():
        row = json.loads(line)
        text = (
            unicodedata.normalize("NFKD", row["text"])
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        grades = {
            k: float(fn(text))
            for k, fn in [
                ("flesch_kincaid", textstat.flesch_kincaid_grade),
                ("gunning_fog", textstat.gunning_fog),
                ("smog", textstat.smog_index),
                ("coleman_liau", textstat.coleman_liau_index),
            ]
        }
        grades["average"] = sum(grades.values()) / len(grades)
        items[row["id"].removeprefix("relay-summary/")] = grades
    OUT.mkdir(exist_ok=True)
    payload = {
        "source": "https://github.com/jchan58/ReLay/blob/860f6bc5c2a23e5ef8ed22bdf86ebc4f960ebec9/Metrics/Personalization/readability.py",
        "textstat": version("textstat"),
        "packet_sha256": sha256(PUBLIC.read_bytes()).hexdigest(),
        "grade": items,
    }
    (OUT / "formula.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    print(len(items), "public summaries")


if __name__ == "__main__":
    main()
