"""Reproduce the Qiao–Lagnado source audit and prepare a local longitudinal panel."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean
from subprocess import check_output
from tempfile import TemporaryDirectory

from research.semantic_tracking.contractnli_prepare import digest
from research.stage2.records import write_jsonl


PROJECT = "https://osf.io/u72fr/"
MANUSCRIPT = "https://escholarship.org/uc/item/1zj2p06s"
SOURCES = {
    "study1.csv": ("whmzj", "f8c6ca10bda24d510e4f8f8f940faad054335dc0499b02100bc389ea5c3ac2e6"),
    "study1.Rmd": ("p9nfq", "7c851861a8a3d7a67aeb1bcc2b1f0ef318e942e37ffadb30309d3be6a0f174c7"),
    "study2.xlsx": ("2y4pr", "ff5aca229054500dbbe37a1e12a4d3294b6afc447cf99dce46f7c3c21cef6c6d"),
    "study2.Rmd": ("mn8xb", "8f4b83e27b8b71ee01bc301b7c89f63e4e3445d98b4a6f96a38c6d2c6b54dade"),
    "materials.pdf": ("znkef", "31c660cb7a79d573ecf558f7d8f4c21a8cc3ec4fe28f500502e2531d0ca41b21"),
}
PAGES_SHA256 = "c769714464950864daa31cc78a610e7b2ba2a7c22ceb4b3dd0ab2bbd5fb57ffd"
CASE = "qiao-case-1"
STAGES = {"SBS": tuple(range(1, 9)), "EOS": (4, 8), "MIXED": (1, 2, 3, 4, 5)}
CONDITIONS = {
    (1, "SBS", "DP"): 43, (1, "SBS", "PD"): 38,
    (1, "EOS", "DP"): 38, (1, "EOS", "PD"): 40,
    (2, "MIXED", "DP"): 45, (2, "MIXED", "PD"): 48,
}
SCALES = {"guilt": "guilt_rating", "confidence": "confidence",
          "not_guilty_lr": "evidence_likelihood_if_not_guilty", "guilty_lr": "evidence_likelihood_if_guilty"}


def read_source(raw):
    from openpyxl import load_workbook

    for name, (_, checksum) in SOURCES.items():
        if digest(raw / name) != checksum:
            raise ValueError(f"Published file differs from the audited OSF version: {name}")
    with (raw / "study1.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"id", "Prolific ID", "condition", "order", "evidence_type", "evidence_no", "stage", "verdict", *SCALES}
        if not required <= {key.strip() for key in reader.fieldnames}:
            raise ValueError("Study 1 data headers changed")
        first = [{key.strip(): value for key, value in row.items()} for row in reader]
    workbook = load_workbook(raw / "study2.xlsx", read_only=True, data_only=True)
    try:
        if workbook.sheetnames != ["Sheet1"]:
            raise ValueError("Study 2 worksheet changed")
        values = iter(workbook["Sheet1"].values)
        header = [value.strip() for value in next(values)]
        if not {"id", "Prolific ID", "order", "stage", "verdict", *SCALES} <= set(header):
            raise ValueError("Study 2 data headers changed")
        second = [dict(zip(header, row, strict=True)) for row in values]
    finally:
        workbook.close()
    if (len(first), len(second)) != (804, 465):
        raise ValueError("Published observation inventory changed")
    return {1: first, 2: second}


def material_blocks(pdf):
    cmd = ["pdftotext", "-layout", "-enc", "UTF-8", "-eol", "unix", str(pdf), "-"]
    content = check_output(cmd)
    if sha256(content).hexdigest() != PAGES_SHA256:
        raise ValueError("PDF extraction differed from the audited layout")
    overview, evidence, _, _, tail = content.decode("utf-8").split("\f")
    title, body = overview.strip().split("\n", 1)
    if title.strip() != "Case summary" or tail:
        raise ValueError("Unexpected stimulus pages")
    chunks = {"summary": "\n\n".join(" ".join(piece.split()) for piece in re.split(r"\n[ \t]*\n", body.strip()))}
    prosecution, defence = re.split(r"^[ \t]*Defence evidence[ \t]*$", evidence, flags=re.M)
    for label, section in (("P", prosecution), ("D", defence)):
        parts = re.split(r"^([1-4])\)[ \t]*", section, flags=re.M)
        if parts[1::2] != ["1", "2", "3", "4"]:
            raise ValueError("The four numbered passages in an evidence block could not be recovered")
        for ordinal, passage in zip(parts[1::2], parts[2::2], strict=True):
            chunks[f"{label}{ordinal}"] = " ".join(passage.split())
    if [len(chunks[key]) for key in ("summary", "P1", "P2", "P3", "P4", "D1", "D2", "D3", "D4")] != [
        1625, 247, 121, 134, 188, 97, 175, 167, 167,
    ]:
        raise ValueError("Recovered case and evidence lengths changed")
    return chunks


def build_variants(chunks):
    stimuli, design, lookup = [], [], {}
    for study, mode, order in CONDITIONS:
        identifier = f"qiao-s{study}-{mode.lower()}-{order.lower()}"
        item_order = [f"{side}{item}" for side in order for item in range(1, 5)]
        pieces = [chunks[k] for k in ("summary", *item_order)]
        offsets = []
        length = 0
        for piece in pieces:
            length += len(piece)
            offsets.append(length)
            length += 2
        totals = (1, 2, 3, 4, 8) if mode == "MIXED" else STAGES[mode]
        observations = {
            stage: {"stage": stage, "seen_evidence": item_order[:count], "stop": offsets[count]}
            for stage, count in zip(STAGES[mode], totals, strict=True)
        }
        text = "\n\n".join(pieces)
        if offsets[-1] != len(text):
            raise ValueError("Cumulative position disagrees with reconstructed text")
        stimuli.append({"id": identifier, "text": text, "stops": [item["stop"] for item in observations.values()]})
        design.append({"id": identifier, "case_id": CASE, "study": study, "mode": mode, "order": order,
                       "summary_stop": offsets[0], "observed_stages": list(observations.values())})
        lookup[study, mode, order] = identifier, observations
    return stimuli, design, lookup


def rating(value):
    score = int(value)
    if score != float(value) or not 0 <= score <= 100:
        raise ValueError("A published response does not match the recorded 0–100 scale")
    return score


def panels(rows, lookup):
    trials, platform_groups = defaultdict(list), defaultdict(list)
    platforms = {}
    for study, records in rows.items():
        for record in records:
            key = study, int(record["id"])
            platform = record["Prolific ID"]
            if not isinstance(platform, str) or not platform:
                raise ValueError("Published platform value is missing")
            if key in platforms and platforms[key] != platform:
                raise ValueError("An author-assigned trial maps to conflicting platform records")
            platforms[key] = platform
            trials[key].append(record)
    for key in sorted(trials):
        platform_groups[platforms[key]].append(key)
    repeated = sorted(tuple(keys) for keys in platform_groups.values() if len(keys) > 1)
    if len(repeated) != 2 or any(len(keys) != 2 or {study for study, _ in keys} != {1} for keys in repeated):
        raise ValueError("Platform overlap differs from the published-file audit")
    overlap_ids = {key: f"possible-repeat-{i:02d}" for i, keys in enumerate(repeated, 1) for key in keys}
    sequence_ids = Counter()
    allocations = Counter()
    outcomes, conflicts = [], []
    study_finals = defaultdict(list)
    def reader_order(key):
        study, author_id = key
        order = sha256(f"cogllm-qiao-reader-v1:{study}:{author_id}".encode("ascii")).hexdigest()
        return study, order

    for study, author_id in sorted(trials, key=reader_order):
        key = study, author_id
        sequence = sorted(trials[key], key=lambda row: int(row["stage"]))
        modes = {row["condition"] if study == 1 else "MIXED" for row in sequence}
        orders = {row["order"] for row in sequence}
        if len(modes) != 1 or len(orders) != 1:
            raise ValueError("An authored trial changes group during reading")
        mode, = modes
        order, = orders
        variant_id, observations = lookup[study, mode, order]
        if tuple(int(row["stage"]) for row in sequence) != STAGES[mode]:
            raise ValueError("Missing or repeated judgement stages")
        tags = {}
        if study == 1:
            tags = {int(row["stage"]): f"{row['evidence_type']}{int(row['evidence_no'])}" for row in sequence}
        conflict_stages = {stage for stage, tag in tags.items() if observations[stage]["seen_evidence"][-1] != tag}
        sequence_ids[study] += 1
        allocations[study, mode, order] += 1
        reader = f"qiao-s{study}-r{sequence_ids[study]:03d}"
        for row in sequence:
            stage = int(row["stage"])
            observation = observations[stage]
            final = stage == STAGES[mode][-1]
            verdict = row["verdict"]
            if verdict not in {"Guilty", "Not Guilty"}:
                raise ValueError("Unknown verdict response")
            expected = observation["seen_evidence"][-1]
            if stage in conflict_stages:
                conflicts.append({"study": study, "mode": mode, "order": order, "stage": stage,
                                  "expected": expected, "published_tag": tags[stage]})
                alignment = "conflicting_published_evidence_tag"
            elif conflict_stages:
                alignment = "uncertified_prefix_in_conflicting_trajectory"
            else:
                alignment = "verified_published_evidence_tag" if study == 1 else "inferred_from_published_design"
            values = {label: rating(row[field]) for field, label in SCALES.items()}
            if final:
                study_finals[study, mode, order].append((values["guilt_rating"], verdict == "Guilty"))
            outcomes.append({
                "id": variant_id,
                "reader_id": reader,
                "possible_repeat_group": overlap_ids.get(key),
                "stage": stage,
                "stop": None if conflict_stages else observation["stop"],
                "alignment": alignment,
                "final_stage": final,
                **values,
                "verdict": "guilty" if verdict == "Guilty" else "not_guilty",
            })
    if allocations != CONDITIONS or conflicts != [
        {"study": 1, "mode": "SBS", "order": "DP", "stage": 1, "expected": "D1", "published_tag": "P1"},
        {"study": 1, "mode": "SBS", "order": "DP", "stage": 5, "expected": "P1", "published_tag": "D1"},
    ]:
        raise ValueError("Design allocations or known source conflicts changed")
    summary = [
        {"study": study, "mode": mode, "order": order, "reader_records": len(values),
         "mean_guilt_rating": mean(rating for rating, _ in values),
         "guilty_verdicts": sum(verdict for _, verdict in values)}
        for (study, mode, order), values in sorted(study_finals.items())
    ]
    return outcomes, {
        "author_assigned_reader_records": dict(sequence_ids),
        "possible_repeated_platform_accounts_study_1": len(repeated),
        "platform_account_overlap_across_studies": 0,
        "source_alignment_conflicts": conflicts,
        "observations_per_alignment_status": Counter(row["alignment"] for row in outcomes),
        "final_observations": summary,
    }


def prepare(raw, output):
    if output.exists():
        raise FileExistsError(output)
    rows = read_source(raw)
    stimuli, design, lookup = build_variants(material_blocks(raw / "materials.pdf"))
    outcomes, inventory = panels(rows, lookup)
    tables = {"stimuli.jsonl": stimuli, "layout.jsonl": design, "outcomes.jsonl": outcomes}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".qiao-evidence-") as tmp:
        stage = Path(tmp) / "prepared"
        stage.mkdir(mode=0o700)
        for name, table in tables.items():
            write_jsonl(stage / name, table)
        receipt = {
            "schema": 1,
            "source_project": PROJECT,
            "manuscript": MANUSCRIPT,
            "source_code_sha256": digest(Path(__file__)),
            "sources": {name: {"url": f"https://osf.io/download/{key}/", "sha256": pin}
                        for name, (key, pin) in SOURCES.items()},
            "paper_license": "CC BY 4.0",
            "osf_project_license": None,
            "extracted_pdf_layout_sha256": PAGES_SHA256,
            "text_reconstruction": "Page 1 prose then page 2 numbered passages. Layout whitespace collapsed within "
                                   "paragraphs and structural headings and list markers omitted. Two newlines between pieces.",
            "case_families": 1,
            "distinct_evidence_orders": 2,
            "response_variants": len(stimuli),
            "withheld_observations": "All observations from a trajectory with a conflicting source tag retain their "
                                     "responses and have a null text alignment.",
            "endpoint": "A participant reports the same prompted guilt proposition at multiple scheduled stages. "
                        "Verdicts are responses and the experiment supplies no case truth.",
            "reader_ids": "Local sequential pseudonyms within each study's hashed trial order. "
                          "Possible platform-account repetitions use independent "
                          "local group labels. Source labels, raw platform identifiers, and free text do not enter outputs.",
            "counts": inventory,
            "output_sha256": {name: digest(stage / name) for name in tables},
        }
        (stage / "source-audit.json").write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n",
                                                 encoding="utf-8")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    prepare(**vars(parser.parse_args()))
