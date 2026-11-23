"""Convert the author-released PLABA-EVAL CSVs to the Stage 2 schema."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .records import observed, write_jsonl


ANSWER_OPTIONS = ("A", "B", "C", "D")


def _rows(source_dir, filename):
    with (source_dir / filename).open(encoding="utf-8", newline="") as source:
        yield from enumerate(csv.DictReader(source), start=1)


def _missing():
    return {"status": "missing", "value": None}


def _unobserved():
    return {"status": "unobserved", "value": None}


def _load_stimuli(source_dir):
    documents = {}
    stimuli = []
    sentences = defaultdict(list)
    segment_by_index = {}

    for _, row in _rows(source_dir, "sentences.csv"):
        sentences[row["doc_id"]].append(row)

    for _, row in _rows(source_dir, "documents.csv"):
        doc_id = row["doc_id"]
        document_text = row["document_text"]
        if doc_id in documents:
            raise ValueError(f"Duplicate PLABA document {doc_id}")
        documents[doc_id] = row
        sentence_rows = sorted(sentences.pop(doc_id, []), key=lambda item: int(item["sentence_index"]))
        if len(sentence_rows) != int(row["n_sentences"]):
            raise ValueError(f"Unexpected sentence count for {doc_id}")

        segments = []
        cursor = 0
        for index, sentence in enumerate(sentence_rows, start=1):
            sentence_id = sentence["sentence_uid"]
            sentence_text = sentence["sentence_text"]
            start = document_text.find(sentence_text, cursor)
            if (
                start < 0
                or document_text[cursor:start].strip()
                or int(sentence["sentence_index"]) != index
            ):
                raise ValueError(f"Cannot align PLABA sentence {sentence_id} to {doc_id}")
            cursor = start + len(sentence_text)
            segments.append({"id": sentence_id, "start": start, "end": cursor})
            segment_by_index[(doc_id, index)] = sentence

        if document_text[cursor:].strip():
            raise ValueError(f"Unmapped text at end of PLABA document {doc_id}")
        stimuli.append({
            "id": doc_id,
            "family_id": row["base_text_id"],
            "language": "en",
            "text": document_text,
            "segments": segments,
            "metadata": {"variant": row["variant"]},
        })

    if sentences:
        raise ValueError(f"PLABA sentences reference unknown documents: {sorted(sentences)}")
    return stimuli, documents, segment_by_index


def _trial(row, filename, source_row, trial_id, task, document, observations):
    if any(row[key] != document[key] for key in ("base_text_id", "variant", "stimulus_id")):
        raise ValueError(f"Inconsistent PLABA document reference in {filename}, row {source_row}")
    return {
        "id": trial_id,
        "stimulus_id": row["doc_id"],
        "participant_id": row["worker_id"] or None,
        "source_ref": {"file": filename, "row": source_row},
        "exposure": {"mode": "full_text", "visible_end": len(document["document_text"]), "task": task},
        "observations": observations,
        "target_segment_id": None,
        "task_context": {},
        "metadata": {"round": row["round"]},
    }


def _sentence_trials(source_dir, documents, segment_by_index):
    filename = "annotations.csv"
    for source_row, row in _rows(source_dir, filename):
        if row["included_in_paper_sentence_analysis"] != "True":
            continue

        index = int(row["sentence_index"].removesuffix(".0"))
        anchor = int(row["anchor_id"].removesuffix(".0"))
        sentence = segment_by_index[(row["doc_id"], index)]
        if (
            row["sentence_uid"] != sentence["sentence_uid"]
            or row["annotation_text"] != sentence["sentence_text"]
            or anchor != int(sentence["mapped_orig_sentence_index"])
        ):
            raise ValueError(f"Unverified PLABA sentence reference in {filename}, row {source_row}")

        schema = row["feedback_schema"]
        other_text = _unobserved()
        if schema == "issue_tags":
            tags = json.loads(row["issue_tags_json"])
            no_issue = row["no_issue"] == "True"
            if row["no_issue"] not in {"True", "False"} or no_issue == bool(tags):
                raise ValueError(f"Inconsistent PLABA issue response in {filename}, row {source_row}")
            issues = {"issue_tags": observed(tags), "no_issue": observed(no_issue)}
            if "other" in tags:
                other_text = _missing()
        elif schema == "free_text":
            issues = {"issue_tags": _unobserved(), "no_issue": _unobserved()}
        else:
            raise ValueError(f"Unknown PLABA feedback schema: {schema}")

        response = {
            "processing_ease": observed(int(row["processing_ease"])),
            "perceived_understanding": observed(int(row["perceived_understanding"])),
            **issues,
            "feedback_text": observed(row["feedback_text"]) if row["feedback_text"] else _missing(),
            "other_text": observed(row["other_text"]) if row["other_text"] else other_text,
        }
        trial = _trial(
            row, filename, source_row, row["annotation_id"], "sentence_rating",
            documents[row["doc_id"]], response,
        )
        trial["target_segment_id"] = sentence["sentence_uid"]
        trial["metadata"].update({"feedback_schema": schema, "original_sentence_index": anchor})
        yield trial


def _document_trials(source_dir, documents):
    filename = "document_ratings.csv"
    for source_row, row in _rows(source_dir, filename):
        observations = {
            "overall_processing_ease": observed(int(row["overall_processing_ease"])),
            "overall_perceived_understanding": observed(int(row["overall_perceived_understanding"])),
            "post_reading_topic_familiarity": observed(int(row["topic_familiarity"])),
            "post_reading_academic_background": observed(json.loads(row["academic_reading_background_raw"])),
            "suggested_improvements": observed(row["improvements"]) if row["improvements"] else _unobserved(),
        }
        yield _trial(
            row, filename, source_row, row["reader_doc_id"], "document_rating",
            documents[row["doc_id"]], observations,
        )


def _mcq_trials(source_dir, documents, segment_by_index, items):
    filename = "mcq_responses.csv"
    for source_row, row in _rows(source_dir, filename):
        if row["included_in_final_paper_analysis"] != "True":
            continue

        item = items[row["mcq_id"]]
        correct_option = item["correct_answer"]
        selected_option = row["selected_option"]
        correct = row["is_correct"] == "True"
        if (
            item["base_text_id"] != row["base_text_id"]
            or row["question_text"] != item["question"]
            or row["correct_answer"] != correct_option
            or correct_option not in ANSWER_OPTIONS
            or selected_option not in ANSWER_OPTIONS
            or row["is_correct"] not in {"True", "False"}
            or correct != (selected_option == correct_option)
        ):
            raise ValueError(f"Inconsistent PLABA MCQ response in {filename}, row {source_row}")

        doc_id = row["doc_id"]
        if documents[doc_id]["variant"] == "orig":
            item_indexes = [int(item["critical_evidence_orig_sentence_index"])]
        else:
            item_indexes = json.loads(item["critical_evidence_simp_sentence_indices_json"])
        item_evidence_ids = [segment_by_index[(doc_id, index)]["sentence_uid"] for index in item_indexes]
        response_index = int(row["critical_evidence_sentence_index_this_variant"])
        response_sentence = segment_by_index.get((doc_id, response_index))

        observations = {
            "mcq_correct": observed(correct),
            "mcq_correct_option": observed(correct_option),
            "mcq_selected_option": observed(selected_option),
            "mcq_selected_option_text": observed(row["selected_option_text"]),
            "quiz_effort": observed(int(row["quiz_effort"])),
            "quiz_confidence": observed(int(row["quiz_confidence"])),
            "item_evidence_segment_ids": observed(item_evidence_ids) if item_evidence_ids else _missing(),
            "response_evidence_index": observed(response_index),
            "response_indexed_segment": (
                observed(response_sentence["sentence_uid"]) if response_sentence else _missing()
            ),
        }
        trial = _trial(
            row, filename, source_row, row["mcq_response_id"], "open_book_mcq",
            documents[doc_id], observations,
        )
        trial["task_context"] = {
            "question": item["question"],
            "options": [
                {"id": option, "text": item[f"option_{option.lower()}"]}
                for option in ANSWER_OPTIONS
            ],
        }
        trial["metadata"].update({
            "mcq_id": row["mcq_id"],
            "item_source_ref": item["source_ref"],
            "options_order": "released_item_order_display_order_unknown",
            "selected_text_matches_item_option": row["selected_option_text"] == item[f"option_{selected_option.lower()}"],
        })
        yield trial


def _alignment_contrast(sentence_trials, stimuli, segment_by_index):
    by_sentence = defaultdict(list)
    for trial in sentence_trials:
        key = (trial["stimulus_id"], trial["target_segment_id"])
        by_sentence[key].append(trial["observations"]["perceived_understanding"]["value"])

    by_anchor = defaultdict(lambda: defaultdict(list))
    for stimulus in stimuli:
        doc_id = stimulus["id"]
        for index, segment in enumerate(stimulus["segments"], start=1):
            anchor = int(segment_by_index[(doc_id, index)]["mapped_orig_sentence_index"])
            ratings = by_sentence.get((doc_id, segment["id"]))
            if ratings:
                by_anchor[(stimulus["family_id"], anchor)][stimulus["metadata"]["variant"]].append(
                    mean(ratings)
                )
    deltas = [
        mean(scores["simp"]) - mean(scores["orig"])
        for scores in by_anchor.values()
        if "orig" in scores and "simp" in scores
    ]
    return len(deltas), mean(deltas)


def convert(source_dir: Path, output_dir: Path) -> dict:
    """Normalize full-text trials and return independently computed release metrics."""
    stimuli, documents, segment_by_index = _load_stimuli(source_dir)
    items = {}
    for source_row, row in _rows(source_dir, "mcq_items.csv"):
        if row["included_in_final_paper_analysis"] == "True":
            items[row["mcq_id"]] = {**row, "source_ref": {"file": "mcq_items.csv", "row": source_row}}
    sentence_trials = list(_sentence_trials(source_dir, documents, segment_by_index))
    document_trials = list(_document_trials(source_dir, documents))
    mcq_trials = list(_mcq_trials(source_dir, documents, segment_by_index, items))
    ratings = [trial["observations"] for trial in sentence_trials]
    responses = [trial["observations"] for trial in mcq_trials]
    structured = [row for row in ratings if row["issue_tags"]["status"] == "observed"]
    canonical_evidence = [row for row in responses if row["item_evidence_segment_ids"]["status"] == "observed"]
    indexed_evidence = [row for row in responses if row["response_indexed_segment"]["status"] == "observed"]
    mismatched_evidence = sum(
        row["response_indexed_segment"]["value"] not in (row["item_evidence_segment_ids"]["value"] or [])
        for row in indexed_evidence
    )
    alignment_count, understanding_delta = _alignment_contrast(sentence_trials, stimuli, segment_by_index)
    metrics = {
        "article_pairs": len({stimulus["family_id"] for stimulus in stimuli}),
        "document_variants": len(stimuli),
        "sentences": len(segment_by_index),
        "sentence_ratings": len(ratings),
        "document_ratings": len(document_trials),
        "structured_feedback": len(structured),
        "structured_with_issue": sum(not row["no_issue"]["value"] for row in structured),
        "structured_no_issue": sum(row["no_issue"]["value"] for row in structured),
        "free_text_feedback": len(ratings) - len(structured),
        "mcq_items": len(items),
        "mcq_items_with_correct_option_a": sum(row["correct_answer"] == "A" for row in items.values()),
        "mcq_responses": len(responses),
        "mcq_correct_responses": sum(row["mcq_correct"]["value"] for row in responses),
        "mcq_with_canonical_evidence": len(canonical_evidence),
        "mcq_with_one_canonical_evidence_segment": sum(
            len(row["item_evidence_segment_ids"]["value"]) == 1 for row in canonical_evidence
        ),
        "mcq_with_response_indexed_sentence": len(indexed_evidence),
        "mcq_joinable_response_index_outside_canonical_evidence": mismatched_evidence,
        "mcq_response_index_outside_canonical_evidence": len(responses) - len(indexed_evidence) + mismatched_evidence,
        "mcq_selected_text_differs_from_item_option": sum(
            not trial["metadata"]["selected_text_matches_item_option"] for trial in mcq_trials
        ),
        "paired_sentence_alignments": alignment_count,
        "paired_mean_understanding_difference_simplified_minus_original": understanding_delta,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "stimuli.jsonl", stimuli)
    write_jsonl(output_dir / "trials.jsonl", sentence_trials + document_trials + mcq_trials)
    return metrics
