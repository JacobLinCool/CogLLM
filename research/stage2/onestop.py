"""Convert the author-released OneStopQA Prolific open-book QA trials."""

import csv
import re
from collections import Counter
from pathlib import Path

from .records import observed, write_jsonl


SOURCE_FILE = "qa.tsv"
REQUIRED_COLUMNS = {
    "answer_response", "difficulty", "item_id", "new_options", "options",
    "paragraph", "question", "source", "submission_ind", "trial_number", "trial_type",
}
ITEM_ID = re.compile(r"os([1-9][0-9]*)_([1-9][0-9]*)_([1-9][0-9]*)_(Adv|Ele)")
OPTION_LABELS = "ABCD"


def convert(source_dir: Path, output_dir: Path) -> dict:
    stimuli = {}
    trials = {}
    questions = {}
    items = set()
    participants = set()
    trials_by_difficulty = Counter()
    correct_by_difficulty = Counter()

    with (source_dir / SOURCE_FILE).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", strict=True)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not REQUIRED_COLUMNS.issubset(columns):
            raise ValueError(f"Invalid header in OneStopQA {SOURCE_FILE}")

        for row_number, row in enumerate(reader, start=1):
            location = f"{SOURCE_FILE} data row {row_number}"
            if None in row or any(row[column] is None for column in REQUIRED_COLUMNS):
                raise ValueError(f"Malformed OneStopQA record in {location}")
            if row["source"] != "Onestop" or row["trial_type"] != "reading_comprehension":
                continue

            item_id = row["item_id"]
            item = ITEM_ID.fullmatch(item_id)
            if item is None or item[4] != row["difficulty"]:
                raise ValueError(f"Invalid OneStopQA item identifier or difficulty in {location}")
            article, paragraph, question, difficulty = item.groups()
            family_id = f"onestop-article-{article}"
            stimulus_id = f"{family_id}-paragraph-{paragraph}-{difficulty.lower()}"
            question_id = f"os{article}_{paragraph}_{question}"
            text = row["paragraph"]
            question_text = row["question"]
            source_options = row["options"].split("|")
            if not text.strip() or not question_text.strip():
                raise ValueError(f"Incomplete OneStopQA stimulus or question in {location}")
            if len(source_options) != len(OPTION_LABELS) or any(not option.strip() for option in source_options):
                raise ValueError(f"Invalid OneStopQA answer options in {location}")
            question_data = (question_text, source_options)
            if questions.setdefault(question_id, question_data) != question_data:
                raise ValueError(f"Conflicting OneStopQA question or options in {location}")

            segment_id = f"{stimulus_id}-full"
            stimulus = {
                "id": stimulus_id,
                "family_id": family_id,
                "language": "en",
                "text": text,
                "segments": [{"id": segment_id, "start": 0, "end": len(text)}],
                "metadata": {
                    "source_article_number": int(article),
                    "source_paragraph_number": int(paragraph),
                    "difficulty": difficulty,
                },
            }
            if stimuli.setdefault(stimulus_id, stimulus)["text"] != text:
                raise ValueError(f"Conflicting OneStopQA paragraph text in {location}")

            submission = row["submission_ind"]
            trial_number = row["trial_number"]
            if not submission.isascii() or not submission.isdecimal():
                raise ValueError(f"Invalid OneStopQA submission in {location}")
            if trial_number not in {"1", "2", "3", "4"}:
                raise ValueError(f"Invalid OneStopQA trial number in {location}")
            participant_id = f"onestop-prolific-{submission}"
            trial_id = f"{participant_id}-trial-{trial_number}"
            if trial_id in trials:
                raise ValueError(f"Duplicate OneStopQA trial in {location}")

            # The source uses A/B/C/D indices 0–3 for the unshuffled options.
            # Interpret new_options as the source indices in presentation order.
            displayed_order = row["new_options"].split("|")
            if sorted(displayed_order) != ["0", "1", "2", "3"] or row["answer_response"] not in displayed_order:
                raise ValueError(f"Invalid OneStopQA option permutation or answer in {location}")
            selected_source_index = int(row["answer_response"])
            selected_position = displayed_order.index(row["answer_response"])
            correct_position = displayed_order.index("0")
            correct = selected_source_index == 0
            trials[trial_id] = {
                "id": trial_id,
                "stimulus_id": stimulus_id,
                "participant_id": participant_id,
                "source_ref": {"file": SOURCE_FILE, "row": row_number},
                "exposure": {"mode": "full_text", "visible_end": len(text), "task": "open_book_mcq"},
                "observations": {
                    "mcq_correct": observed(correct),
                    "mcq_correct_option": observed(OPTION_LABELS[correct_position]),
                    "mcq_selected_option": observed(OPTION_LABELS[selected_position]),
                    "mcq_selected_option_text": observed(source_options[selected_source_index]),
                    "mcq_selected_source_option_index": observed(selected_source_index),
                },
                "target_segment_id": None,
                "task_context": {
                    "question": question_text,
                    "options": [
                        {"id": label, "text": source_options[int(index)]}
                        for label, index in zip(OPTION_LABELS, displayed_order)
                    ],
                },
                "metadata": {
                    "item_id": item_id,
                    "question_id": question_id,
                    "trial_number": int(trial_number),
                    "options_order": "released_new_options",
                    "question_visibility": "with_passage",
                },
            }
            items.add(item_id)
            participants.add(participant_id)
            trials_by_difficulty[difficulty] += 1
            correct_by_difficulty[difficulty] += correct

    if not trials:
        raise ValueError(f"No OneStopQA comprehension trials in {SOURCE_FILE}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "stimuli.jsonl", (stimuli[key] for key in sorted(stimuli)))
    write_jsonl(output_dir / "trials.jsonl", trials.values())
    return {
        "n_stimuli": len(stimuli),
        "n_families": len({stimulus["family_id"] for stimulus in stimuli.values()}),
        "n_trials": len(trials),
        "n_participants": len(participants),
        "n_items": len(items),
        "n_questions": len(questions),
        "n_correct": sum(correct_by_difficulty.values()),
        "by_difficulty": {
            difficulty: {
                "n_trials": trials_by_difficulty[difficulty],
                "n_correct": correct_by_difficulty[difficulty],
            }
            for difficulty in sorted(trials_by_difficulty)
        },
    }
