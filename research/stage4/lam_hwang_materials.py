"""Build an outcome-blind stimulus manifest for Lam and Hwang Experiment 1."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess


SOURCE_SHA256 = {
    "publisher-supplement.pdf": "ecafa119adba221af01b413cf7f4edb722c09b14737efd478d8d502f789ef89a",
    "author-material-keys.xlsx": "7bae109124c2ef182d2bce2406fb3e35b3548416fc387ee900652149b1becb8e",
    "experiment-1.csv": "d308975fcecdc4f8ae3b39ef9cc4febb9a3424d0031ca0d8cfe2987ee12f1557",
}
EXPERIMENT_KEY = "LamHwangLCPE1"
ITEM_IDS = tuple(f"I{index:02d}" for index in range(1, 25))
CONNECTIVE = "，因为"
VARIANTS = (("free", "", "free"), ("ta-m", "他", "overt_ta_m"), ("ta-f", "她", "overt_ta_f"))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _author_verbs(path):
    import openpyxl

    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        rows = workbook["simple_connectives"].values
        columns = {name: index for index, name in enumerate(next(rows))}
        experiment, item_id, verb = (columns[name] for name in ("Experiment", "Item_ID", "CN_Verb"))
        item_verbs = {
            (row[item_id].removesuffix("_" + EXPERIMENT_KEY), row[verb])
            for row in rows
            if row[experiment] == EXPERIMENT_KEY
        }
    finally:
        workbook.close()
    if len(item_verbs) != 24 or {item for item, _ in item_verbs} != set(ITEM_IDS):
        raise ValueError("Author material key must identify each implicit-causality item once")
    if len({verb for _, verb in item_verbs}) != 24:
        raise ValueError("Expected unique verbs in the author's implicit-causality material key")
    return {verb: item for item, verb in item_verbs}


def _trial_conditions(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        # The released human choice and referential form are never used here.
        conditions = {
            (row["ID"], row["Prompt_Type"], row["Bias"])
            for row in csv.DictReader(stream)
            if row["ID"] in ITEM_IDS
        }
    expected = {
        (item, prompt, "N1" if index % 2 else "N2")
        for index, item in enumerate(ITEM_IDS, start=1)
        for prompt in ("free", "pronoun")
    }
    if conditions != expected:
        raise ValueError("Published prompt conditions disagree with the author material key")


def prepare(source, output):
    paths = {name: source / name for name in SOURCE_SHA256}
    for name, path in paths.items():
        if _sha256(path) != SOURCE_SHA256[name]:
            raise ValueError(f"Published input hash differs for {name}")

    verbs = _author_verbs(paths["author-material-keys.xlsx"])
    _trial_conditions(paths["experiment-1.csv"])
    appendix = subprocess.run(
        ["pdftotext", "-f", "1", "-l", "4", "-layout", str(paths["publisher-supplement.pdf"]), "-"],
        check=True, capture_output=True, text=True, encoding="utf-8",
    ).stdout
    # This pattern projects out the exact Chinese prompt column only.
    texts = re.findall(r"[\u3400-\u9fff]+，因为", appendix)
    if len(texts) != 24 or len(set(texts)) != 24:
        raise ValueError("Expected the 24 implicit-causality prompts in Appendix A")

    families = {}
    for appendix_row, text in enumerate(texts, start=1):
        stem_end = len(text) - len(CONNECTIVE)
        verb_form = text[2:stem_end - 2]
        verb = verb_form.removesuffix("了")
        item_id = verbs[verb]
        if item_id in families:
            raise ValueError(f"Two Appendix A prompts map to {item_id}")
        expected_index = (2 * appendix_row - 1) if appendix_row <= 12 else (2 * (appendix_row - 12))
        if item_id != f"I{expected_index:02d}":
            raise ValueError(f"Unexpected Appendix A row order for {item_id}")
        families[item_id] = (text, stem_end, appendix_row)

    rows = []
    for item_id, (text, stem_end, appendix_row) in sorted(families.items()):
        for suffix, pronoun, prompt_type in VARIANTS:
            segments = [
                {"id": "np1", "start": 0, "end": 2},
                {"id": "np2", "start": stem_end - 2, "end": stem_end},
            ]
            if pronoun:
                segments.append({"id": "pronoun", "start": len(text), "end": len(text) + 1})
            rows.append({
                "id": f"{item_id}-{suffix}",
                "family_id": item_id,
                "language": "zh-Hans",
                "text": text + pronoun,
                "segments": segments,
                "metadata": {
                    "experiment": 1,
                    "appendix_row": appendix_row,
                    "prompt_type": prompt_type,
                    "verb_type": "IC-1" if appendix_row <= 12 else "IC-2",
                },
            })

    output.mkdir(parents=True, exist_ok=True)
    stimuli = output / "stimuli.jsonl"
    stimuli.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit = {
        "experiment": 1,
        "scope": "Appendix A implicit-causality materials and two overt-spelling sensitivity arms",
        "families": 24,
        "stimuli": len(rows),
        "exact_overt_spellings_known": False,
        "source_sha256": SOURCE_SHA256,
        "stimuli_sha256": _sha256(stimuli),
    }
    (output / "materials-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)
