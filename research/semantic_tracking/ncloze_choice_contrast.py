"""Compare submitted nCloze choices at exact shared mechanical gaps."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import accumulate
import json
import math
from pathlib import Path
import random
from statistics import mean
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "research/results/ncloze-source-audit.json"
AUDIT_SHA256 = "df029ac1674d977222ab91f28b846a98664bd86e6df41b38e3b2f948d2c8e9aa"
PROTOCOL = Path(__file__).with_name("ncloze-choice-contrast.md")
PROTOCOL_SHA256 = "47ac91982a4ec36e6f69fa10c649362057a1ee7d9301781facc6aaa68a09d18f"
RELEASE_FILES = {"Data/exp-data.json", "Data/results.json", "Data/parseResults.py"}
LETTERS = "ABCD"
CHECK = 9
DRAWS = 10_000
SEED = 735972
MECHANICAL = ("ncloze-m", "cdgp-m", "t5multi-m")
TEMPLATES = {"cloth": "article", "ncloze": "article-ncloze",
             **dict.fromkeys(MECHANICAL, "article-cdgp-m")}
Gap = tuple[str, int]
Count = dict[Gap, dict[str, Counter]]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs(raw: Path) -> dict:
    prefix, marker, _ = PROTOCOL.read_bytes().partition(b"<!-- Results follow -->\n")
    if not marker or hashlib.sha256(prefix).hexdigest() != PROTOCOL_SHA256:
        raise ValueError("The prespecified comparison changed")
    if digest(AUDIT) != AUDIT_SHA256:
        raise ValueError("Expected the pinned nCloze source audit")
    files = json.loads(AUDIT.read_bytes())["source_files"]
    for name in RELEASE_FILES:
        if digest(raw / Path(name).name) != files[name]["sha256"]:
            raise ValueError(f"Released material differs from the source audit: {name}")
    return {name: files[name]["sha256"] for name in sorted(RELEASE_FILES)}


def restored_geometry(parts: list[str], keyed: list[str]) -> tuple[str, tuple[tuple[int, int], ...]]:
    ends = accumulate([len(parts[0]), *(len(word) + len(part) for word, part in zip(keyed, parts[1:], strict=True))])
    coordinates = tuple((start, start + len(word)) for start, word in zip(list(ends)[:-1], keyed, strict=True))
    restored = parts[0] + "".join(word + part for word, part in zip(keyed, parts[1:], strict=True))
    return restored, coordinates


def read_materials(path: Path) -> tuple[dict, set[Gap]]:
    materials = json.loads(path.read_bytes())
    answer_keys = {}
    eligible = set()
    for bundle, record in materials.items():
        geometry = {}
        options = {}
        for arm, template in TEMPLATES.items():
            suffix = "" if arm == "cloth" else f"-{arm}"
            parts = record[template].split("_")
            keys = record[f"answers{suffix}"]
            choices = record[f"options{suffix}"]
            if (len(parts) != len(keys) + 1 or len(keys) != len(choices) or len(keys) <= CHECK
                    or any(len(code) != 1 or code not in LETTERS for code in keys)
                    or any(len(variants) != len(LETTERS) for variants in choices)):
                raise ValueError("Unexpected released question layout or choice coding")
            options[arm] = choices
            answer_keys[bundle, arm] = keys
            keyed = [variants[LETTERS.index(code)] for code, variants in zip(keys, choices, strict=True)]
            geometry[arm] = restored_geometry(parts, keyed)
        primary = geometry[MECHANICAL[0]]
        if not all(geometry[arm] == primary for arm in MECHANICAL):
            raise ValueError("Mechanical conditions do not share every released gap and original text byte")
        for index in range(len(primary[1])):
            if index != CHECK and all(len(set(options[arm][index])) == 4 for arm in MECHANICAL):
                eligible.add((bundle, index))
    if len(materials) != 12 or len(eligible) != 92 or len({bundle for bundle, _ in eligible}) != 12:
        raise ValueError("Clean shared mechanical gaps differ from the frozen audit")
    return answer_keys, eligible


def response_blocks(answer: str, answer_keys: dict) -> dict[tuple[str, str], dict[int, str]]:
    xml = ET.fromstring(answer)
    free_text = [node.text for node in xml.iter() if node.tag.rsplit("}", 1)[-1] == "FreeText"]
    if len(free_text) != 1:
        raise ValueError("Expected one raw MTurk answer object")

    def unique_fields(pairs):
        fields = dict(pairs)
        if len(fields) != len(pairs):
            raise ValueError("Duplicate choice keys appeared in a submission")
        return fields

    parsed = json.loads(free_text[0], object_pairs_hook=unique_fields)
    if not isinstance(parsed, list) or len(parsed) != 1 or not isinstance(parsed[0], dict):
        raise ValueError("Expected one map of in-text gap responses")
    blocks = defaultdict(dict)
    for key, value in parsed[0].items():
        bundle, *condition_parts, ordinal = key.split("-")
        condition = "-".join(condition_parts)
        index = int(ordinal)
        if (type(value) is not str or len(value) != 1 or value not in LETTERS
                or (bundle, condition) not in answer_keys or index in blocks[bundle, condition]):
            raise ValueError("Unjoinable, duplicate or invalid submitted response")
        blocks[bundle, condition][index] = value
    for key, block in blocks.items():
        if block.keys() != set(range(len(answer_keys[key]))):
            raise ValueError("A submission does not contain every gap index from its released material")
    return blocks


def count_choices(path: Path, answer_keys: dict, eligible: set[Gap]) -> tuple[Count, Count, dict]:
    primary = defaultdict(lambda: defaultdict(Counter))
    sensitivity = defaultdict(lambda: defaultdict(Counter))
    assignments = set()
    workers = set()
    rows_by_condition = Counter()
    all_responses = 0
    sensitivity_counts = defaultdict(Counter)
    for submission in json.loads(path.read_bytes()):
        identity = submission["AssignmentId"]
        if identity in assignments or submission["AssignmentStatus"] != "Submitted":
            raise ValueError("Expected one occurrence per original submitted assignment")
        assignments.add(identity)
        workers.add(submission["WorkerId"])
        blocks = response_blocks(submission["Answer"], answer_keys)
        teachers = [key for key in blocks if key[1] == "cloth"]
        experiments = [key for key in blocks if key[1] != "cloth"]
        if len(teachers) != 1 or len(experiments) != 1 or teachers[0][0] == experiments[0][0]:
            raise ValueError("Expected one teacher and one experimental block on distinct article bundles")
        teacher, experiment = teachers[0], experiments[0]
        all_responses += sum(len(block) for block in blocks.values())
        bundle, arm = experiment
        if arm not in MECHANICAL:
            continue

        teacher_hits = sum(value == answer_keys[teacher][index] for index, value in blocks[teacher].items() if index != CHECK)
        passes_control = 4 * teacher_hits >= len(blocks[teacher]) - 1
        passes_check = blocks[experiment][CHECK] == answer_keys[experiment][CHECK]
        keep = passes_control and passes_check
        sensitivity_counts[arm].update({"all_submissions": 1, "kept_by_released_parser_rule": int(keep),
                                       "failed_experimental_slot_9": int(not passes_check),
                                       "teacher_below_quarter": int(not passes_control)})
        for index, value in blocks[experiment].items():
            key = (bundle, index)
            if key not in eligible:
                continue
            correct = int(value == answer_keys[experiment][index])
            rows_by_condition[arm] += 1
            primary[key][arm][correct] += 1
            if keep:
                sensitivity[key][arm][correct] += 1
    if (len(assignments), len(workers), all_responses) != (200, 199, 7707) or rows_by_condition != {
        "cdgp-m": 373, "ncloze-m": 387, "t5multi-m": 399
    }:
        raise ValueError("Submission inventory differs from the pinned unfiltered source audit")
    return primary, sensitivity, dict(sensitivity_counts)


def percentile(sorted_values: list[float], proportion: float) -> float:
    index = proportion * (len(sorted_values) - 1)
    lower = math.floor(index)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[math.ceil(index)] * weight


def evaluate(counts: Count, eligible: set[Gap]) -> dict:
    by_bundle = defaultdict(list)
    missing = dict.fromkeys(MECHANICAL, 0)
    bundles_missing_a_gap = set()
    kept_responses = Counter()
    available_responses = Counter()
    for gap in sorted(eligible):
        arms = counts.get(gap, {})
        totals = {arm: arms.get(arm, Counter()).total() for arm in MECHANICAL}
        available_responses.update(totals)
        absent = [arm for arm, n in totals.items() if n == 0]
        bundle, _ = gap
        if absent:
            for arm in absent:
                missing[arm] += 1
            bundles_missing_a_gap.add(bundle)
            continue
        kept_responses.update(totals)
        by_bundle[bundle].append(tuple(arms[arm][1] / totals[arm] for arm in MECHANICAL))

    if not by_bundle:
        raise ValueError("Every common gap is missing at least one operator")
    accuracy = [tuple(mean(column) for column in zip(*gaps, strict=True)) for gaps in by_bundle.values()]
    differences = [(our - cdgp, our - t5) for our, cdgp, t5 in accuracy]
    draws = [[], []]
    rng = random.Random(SEED)
    for _ in range(DRAWS):
        sampled = [differences[rng.randrange(len(differences))] for _ in differences]
        for dist, values in zip(draws, zip(*sampled, strict=True), strict=True):
            dist.append(mean(values))
    contrasts = {}
    for i, arm in enumerate(MECHANICAL[1:]):
        dist = sorted(draws[i])
        contrasts[arm] = {"ncloze_m_minus_control": mean(row[i] for row in differences),
                          "paired_bundle_95_ci": [percentile(dist, 0.025), percentile(dist, 0.975)]}
    return {
        "candidate_gaps": len(eligible), "complete_gaps": sum(map(len, by_bundle.values())),
        "complete_bundles": len(by_bundle), "bundles_missing_at_least_one_gap": len(bundles_missing_a_gap),
        "bundles_losing_all_gaps": len({bundle for bundle, _ in eligible}) - len(by_bundle),
        "gaps_missing_by_condition": missing, "responses_available_by_condition": dict(available_responses),
        "responses_at_complete_gaps_by_condition": dict(kept_responses),
        "bundle_macro_accuracy": {arm: mean(row[i] for row in accuracy) for i, arm in enumerate(MECHANICAL)},
        "contrasts": contrasts,
    }


def run(raw: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    input_sha256 = verify_inputs(raw)
    answer_keys, eligible = read_materials(raw / "exp-data.json")
    primary, filtered, selection = count_choices(raw / "results.json", answer_keys, eligible)
    document = {
        "schema_version": 1, "source_audit_sha256": AUDIT_SHA256, "protocol_prefix_sha256": PROTOCOL_SHA256,
        "script_sha256": digest(Path(__file__)), "input_sha256": input_sha256,
        "bootstrap": {"unit": "article bundle", "draws": DRAWS, "seed": SEED, "interval": 0.95,
                      "paired": True, "quantile_interpolation": "linear"},
        "unfiltered_submissions": evaluate(primary, eligible),
        "released_parser_sensitivity": {**evaluate(filtered, eligible), "selection_by_condition": selection},
    }
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as file:
        file.write(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.raw, args.output)


if __name__ == "__main__":
    main()
