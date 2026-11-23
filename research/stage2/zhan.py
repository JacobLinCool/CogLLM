"""Convert the released eligible trials from Zhan et al. (2020), Experiment 1."""

import csv
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .records import observed, write_jsonl


SOURCE_FILE = "experiment-1.csv"
TASK = "explanation_continuation"
VERB_TYPES = ("IC-1", "IC-2")
PROMPT_TYPES = {"free", "pronoun"}
RESPONSE_VALUES = {
    "reMention": {"np1", "np2"},
    "referenceForm": {"name", "overtPro", "nullPro"},
    "pronominalization": {"name", "pro"},
    "explanation": {"yes", "no"},
}
REQUIRED_COLUMNS = {
    "participantID", "itemNum", "formNum", "verbType", "prompt", "promptType",
    *RESPONSE_VALUES,
}

# The active sentences in S1 Appendix, pp. 1–3 (https://osf.io/download/q7bwn/),
# fix the candidate names and their order. Both slices were checked for all 36 items.
# Appendix PDF SHA-256: a418a45b8d3913488771bcc867ca22907af59cc92915417160d53305ab6326c9.
# Item 34 has an extra 一下 in the released trials. Keep the CSV prompt verbatim.
PROMPT_PATTERN = re.compile(
    r"(?P<np1>[\u4e00-\u9fff]{2})[\u4e00-\u9fff]{2}了"
    r"(?P<np2>[\u4e00-\u9fff]{2})(?:一[下番顿通])?。(?P<pronoun>[他她])?"
)


def _stimulus(item_number: int, row: dict) -> dict:
    prompt_type = row["promptType"]
    prompt = row["prompt"]
    match = PROMPT_PATTERN.fullmatch(prompt)
    if match is None or bool(match["pronoun"]) != (prompt_type == "pronoun"):
        raise ValueError(f"Unexpected prompt format for item {item_number}, {prompt_type}")

    family_id = f"zhan-e1-{item_number:02d}"
    segments = []
    for name in ("np1", "np2", "pronoun"):
        start, end = match.span(name)
        if start >= 0:
            segments.append({"id": name, "start": start, "end": end})

    return {
        "id": f"{family_id}-{prompt_type}",
        "family_id": family_id,
        "language": "zh-Hans",
        "text": prompt,
        "segments": segments,
        "metadata": {
            "experiment": 1,
            "item_number": item_number,
            "form_number": int(row["formNum"]),
            "prompt_type": prompt_type,
            "verb_type": row["verbType"],
        },
    }


def _fig1_summary(by_verb_and_reader: dict) -> dict:
    summary = {}
    for verb_type in VERB_TYPES:
        choices_by_reader = by_verb_and_reader[verb_type]
        rates = [statistics.mean(choices) for choices in choices_by_reader.values()]
        if len(rates) < 2:
            raise ValueError(f"Figure 1 requires at least two readers in free {verb_type}")
        summary[verb_type] = {
            "n_readers": len(rates),
            "n_trials": sum(map(len, choices_by_reader.values())),
            "np1_count": sum(map(sum, choices_by_reader.values())),
            "reader_equal_rate": statistics.fmean(rates),
            "reader_equal_se": statistics.stdev(rates) / math.sqrt(len(rates)),
        }
    return summary


def convert(source_dir: Path, output_dir: Path) -> dict:
    """Write eligible trials and exact stimuli, and recompute Fig. 1 free-prompt rates."""
    stimuli = {}
    trials = []
    reader_item_keys = set()
    counts_by_prompt = Counter()
    by_verb_and_reader = defaultdict(lambda: defaultdict(list))

    with (source_dir / SOURCE_FILE).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(f"{SOURCE_FILE} is missing columns: {', '.join(sorted(missing_columns))}")
        rows = list(reader)

    reader_ids = {
        source_id: f"zhan-e1-p{index:03d}"
        for index, source_id in enumerate(
            sorted({row["participantID"] for row in rows if row["participantID"]}), start=1
        )
    }

    for row_number, row in enumerate(rows, start=1):
        if None in row or None in row.values() or not row["participantID"]:
            raise ValueError(f"Malformed trial in {SOURCE_FILE} data row {row_number}")
        item_number = int(row["itemNum"])
        participant_id = reader_ids[row["participantID"]]
        prompt_type = row["promptType"]
        verb_type = row["verbType"]
        if prompt_type not in PROMPT_TYPES or verb_type not in VERB_TYPES:
            raise ValueError(f"Unexpected condition in {SOURCE_FILE} data row {row_number}")
        for name, allowed in RESPONSE_VALUES.items():
            if row[name] not in allowed:
                raise ValueError(f"Unexpected {name} in {SOURCE_FILE} data row {row_number}")
        if (participant_id, item_number) in reader_item_keys:
            raise ValueError(f"Repeated reader/item in {SOURCE_FILE} data row {row_number}")
        reader_item_keys.add((participant_id, item_number))

        stimulus = _stimulus(item_number, row)
        stimulus_id = stimulus["id"]
        if stimuli.setdefault(stimulus_id, stimulus) != stimulus:
            raise ValueError(f"Inconsistent stimulus {stimulus_id} in {SOURCE_FILE} data row {row_number}")

        observations = {
            name: observed(row[name])
            for name in RESPONSE_VALUES
        }
        task_context = {"prompt_type": prompt_type, "instruction": "write_natural_explanation"}
        if prompt_type == "free":
            by_verb_and_reader[verb_type][participant_id].append(row["reMention"] == "np1")
        else:
            if row["referenceForm"] != "overtPro" or row["pronominalization"] != "pro":
                raise ValueError(f"Unexpected supplied pronoun in {SOURCE_FILE} data row {row_number}")
            # The overt pronoun is supplied in this condition. Its form is a
            # property of the stimulus, so production remains unobserved.
            for name in ("referenceForm", "pronominalization"):
                observations[name] = {"status": "unobserved", "value": None}
            task_context["supplied_reference_form"] = row["referenceForm"]

        trials.append({
            "id": f"zhan-e1-trial-{row_number:04d}",
            "stimulus_id": stimulus_id,
            "participant_id": participant_id,
            "source_ref": {"file": SOURCE_FILE, "row": row_number},
            "exposure": {
                "mode": "full_text",
                "visible_end": len(stimulus["text"]),
                "task": TASK,
            },
            "observations": observations,
            "target_segment_id": "pronoun" if prompt_type == "pronoun" else None,
            "task_context": task_context,
            "metadata": {},
        })
        counts_by_prompt[prompt_type] += 1

    figure_summary = _fig1_summary(by_verb_and_reader)
    for stimulus in stimuli.values():
        family_id = stimulus["family_id"]
        free = stimuli.get(f"{family_id}-free")
        pronoun = stimuli.get(f"{family_id}-pronoun")
        if free is None or pronoun is None or pronoun["text"][:-1] != free["text"]:
            raise ValueError(f"Non-matching prompt variants in family {family_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "stimuli.jsonl", (stimuli[key] for key in sorted(stimuli)))
    write_jsonl(output_dir / "trials.jsonl", trials)

    return {
        "n_stimuli": len(stimuli),
        "n_families": len({stimulus["family_id"] for stimulus in stimuli.values()}),
        "n_trials": len(trials),
        "n_participants": len(reader_ids),
        "trials_by_prompt": dict(sorted(counts_by_prompt.items())),
        "fig1_free_prompt": figure_summary,
    }
