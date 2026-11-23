"""Recompute aggregate TED-Q elicitation results without exporting text or trials."""

import argparse
from ast import literal_eval
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from decimal import Decimal
import hashlib
from io import StringIO
import json
from pathlib import Path


REVISION = "10fb7e1a2b2d756f8ea190ee2ac223fecaec0f20"
CSV_SHA256 = "992a324ed452876c7364f7304cee24a58a1109fb3204a73adbc88e0937d18131"
REPOSITORY = f"https://github.com/amore-upf/ted-q/blob/{REVISION}"
FORMS = f"{REPOSITORY}/data_gathering/elicitation/chunk_includes"
THRESHOLD = 3


@dataclass(frozen=True)
class Observation:
    id: str
    excerpt: tuple[str, str, int]
    chunk: int
    chunk_span: tuple[int, int]
    highlight: tuple[int, int] | None
    kind: str
    score: int | None
    question_id: str | None
    best_answer_id: str


def integer(value):
    number = Decimal(value)
    if number != number.to_integral_value():
        raise ValueError(f"Expected an integer-valued field, got {value!r}")
    return int(number)


def observation(row):
    kind = row["type"]
    question_id = None
    if kind in {"answer", "non-answer"}:
        reference = literal_eval(row["prior_question"])
        if not isinstance(reference, list) or len(reference) != 1 or not isinstance(reference[0], str):
            raise ValueError("Expected one prior question ID")
        question_id = reference[0]
    elif kind not in {"question", "evaluation"} or row["prior_question"]:
        raise ValueError(f"Unexpected row type or question reference: {kind!r}")

    score = None if kind == "evaluation" else integer(row["answered"])
    if score is not None and not 1 <= score <= 5:
        raise ValueError("Rating lies outside the elicitation scale")
    if (kind == "answer" and score < THRESHOLD) or (kind == "non-answer" and score >= THRESHOLD):
        raise ValueError("Rating conflicts with the answer row type")

    highlight = None
    if kind in {"question", "answer"}:
        highlight = integer(row["highlight_start"]), integer(row["highlight_end"])
        if not row["highlight"] or len(row["highlight"]) != highlight[1] - highlight[0]:
            raise ValueError("Highlight length conflicts with its character offsets")
    elif row["highlight_start"] or row["highlight_end"]:
        raise ValueError("Unexpected highlight offsets")

    return Observation(
        id=row["id"],
        excerpt=(row["worker"], row["source"], integer(row["excerpt_number"])),
        chunk=integer(row["chunk_number"]),
        chunk_span=(integer(row["chunk_start"]), integer(row["chunk_end"])),
        highlight=highlight,
        kind=kind,
        score=score,
        question_id=question_id,
        best_answer_id=row["best_answer"],
    )


def validate_probes(probes):
    last = {}
    for excerpt, chunk in probes:
        last[excerpt] = max(chunk, last.get(excerpt, -1))
    if len(probes) != sum(chunk + 1 for chunk in last.values()):
        raise ValueError("Excerpt has a missing chunk")
    ends = set()
    valid_ends = set()
    question_ends = set()
    invalid_terminal = Counter()
    for (excerpt, chunk), rows in probes.items():
        terminal = chunk == last[excerpt]
        if sum(row.kind == "question" for row in rows) != int(not terminal):
            raise ValueError("A pre-terminal probe lacks its single required question")
        spans = {row.chunk_span for row in rows if row.kind != "evaluation"}
        if len(spans) != 1:
            raise ValueError("Probe rows do not agree on the current chunk")
        start, end = spans.pop()
        ends.add((excerpt[1], end))
        if 0 <= start < end:
            valid_ends.add((excerpt[1], end))
        elif terminal:
            invalid_terminal["probes"] += 1
            invalid_terminal.update(row.kind for row in rows if row.kind != "evaluation")
        else:
            raise ValueError("Invalid source span for a pre-terminal chunk")
        if not terminal:
            question_ends.add((excerpt[1], end))
    return last, {
        "source_probe_positions": len(ends),
        "source_probe_positions_with_valid_chunk_bounds": len(valid_ends),
        "source_probe_positions_with_questions": len(question_ends),
        "invalid_terminal_chunk_bounds": dict(sorted(invalid_terminal.items())),
    }


def audit(path):
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != CSV_SHA256:
        raise ValueError(f"Input must be the audited TED-Q elicitation file at {REVISION}")
    observations = [observation(row) for row in csv.DictReader(StringIO(contents.decode("utf-8-sig"), newline=""))]
    if len({row.id for row in observations}) != len(observations):
        raise ValueError("Duplicate annotation IDs")

    questions = {row.id: row for row in observations if row.kind == "question"}
    probes = defaultdict(list)
    by_question = defaultdict(dict)
    by_probe = defaultdict(dict)
    spans_outside_chunk = Counter()
    for row in observations:
        probe = row.excerpt, row.chunk
        probes[probe].append(row)
        if row.highlight is not None:
            start, end = row.highlight
            if not row.chunk_span[0] <= start < end <= row.chunk_span[1]:
                spans_outside_chunk[row.kind] += 1
        if row.question_id is None:
            continue
        question = questions[row.question_id]
        lag = row.chunk - question.chunk
        if row.excerpt != question.excerpt or lag not in {1, 2}:
            raise ValueError("Follow-up must refer to this reader's recent question in the same excerpt")
        if lag in by_question[question.id] or lag in by_probe[probe]:
            raise ValueError("Duplicate follow-up for this question or origin slot")
        by_question[question.id][lag] = row
        by_probe[probe][lag] = row

    last, probe_audit = validate_probes(probes)
    scores = {1: Counter(), 2: Counter()}
    two_scores = Counter()
    newer_followup = Counter()
    concurrent = Counter()
    unfinished_after_first = 0
    for question in questions.values():
        ratings = by_question[question.id]
        if 1 not in ratings or question.chunk_span[1] != ratings[1].chunk_span[0]:
            raise ValueError("Question lacks its contiguous first follow-up")
        first = ratings[1]
        reaches_at_first = first.score >= THRESHOLD
        expect_second = not reaches_at_first and first.chunk < last[first.excerpt]
        if expect_second != (2 in ratings):
            raise ValueError("Unexpected censoring or re-rating after threshold")
        if not reaches_at_first and not expect_second:
            unfinished_after_first += 1
        if 2 in ratings:
            second = ratings[2]
            if first.chunk_span[1] != second.chunk_span[0]:
                raise ValueError("Gap between follow-up chunks")
            two_scores[first.score, second.score] += 1
        best = {row.id: row.score for row in ratings.values()}
        if best.get(question.best_answer_id) != question.score or question.score != max(best.values()):
            raise ValueError("Best follow-up field disagrees with raw follow-ups")
        for lag, row in ratings.items():
            scores[lag][row.score] += 1

    for ratings in by_probe.values():
        if len(ratings) == 2:
            older, newer = ratings[2], ratings[1]
            older_reaches, newer_reaches = older.score >= THRESHOLD, newer.score >= THRESHOLD
            concurrent[older_reaches, newer_reaches] += 1
            if older_reaches and not newer_reaches:
                following = by_question[newer.question_id].get(2)
                outcome = None if following is None else following.score >= THRESHOLD
                newer_followup[outcome] += 1

    counts = Counter(row.kind for row in observations)
    first_positive = sum(count for score, count in scores[1].items() if score >= THRESHOLD)
    second_positive = sum(count for score, count in scores[2].items() if score >= THRESHOLD)
    return {
        "sources": {
            "paper_methods": "https://aclanthology.org/2020.lrec-1.141.pdf#page=3",
            "paper_results_table_1": "https://aclanthology.org/2020.lrec-1.141.pdf#page=5",
            "earlier_mandatory_elicitation_protocol": "https://mwestera.nl/files/westera_rohde_asking_between_the_lines_ac2019.pdf#page=4",
            "elicitation_csv": f"{REPOSITORY}/TED-Q_elicitation.csv",
            "raw_immutable_csv": f"https://raw.githubusercontent.com/amore-upf/ted-q/{REVISION}/TED-Q_elicitation.csv",
            "csv_sha256": CSV_SHA256,
            "question_form": f"{FORMS}/fragment_question.html#L163-L186",
            "followup_retirement_rule": f"{FORMS}/fragment_answer.html#L126-L139",
            "followup_scale": f"{FORMS}/fragment_answer.html#L209-L236",
            "ted_mdb_license_declaration": "https://github.com/MurathanKurfali/Ted-MDB-Annotations#license",
            "ted_transcript_usage_policy": "https://www.ted.com/about/our-organization/our-policies-terms/ted-talks-usage-policy",
            "ted_ml_terms": "https://www.ted.com/about/our-organization/our-policies-terms/ted-com-terms-of-use",
        },
        "scope": {
            "stimuli": "English TED-MDB transcript excerpts incrementally revealed a chunk at a time.",
            "model_use": "Methods reference and aggregate literature audit. Separate TED permission for transcript use in machine learning could not be verified. TED-Q annotations have no explicit data license in the author release.",
            "elicitation": "A question was required at every pre-terminal probe. Stop means the end of a newly shown chunk, ordinarily two sentence segments. A highlight designates the selected words in that chunk.",
            "followup": "The same reader rated answeredness in the new chunk from 1 to 5. Retesting stopped at 3 or above, at the excerpt end, or after two follow-ups. Only a visited follow-up below 3 is an observed below-threshold rating.",
        },
        "rows_by_kind": dict(sorted(counts.items())),
        "readers": len({excerpt[0] for excerpt in last}),
        "source_talks": len({excerpt[1] for excerpt in last}),
        "reader_excerpts": len(last),
        "reader_probes": len(probes),
        **probe_audit,
        "required_question_responses": len(questions),
        "terminal_probes_without_a_question_request": len(last),
        "explicit_no_question_responses": 0,
        "followup_scores": {lag: {score: counts[score] for score in range(1, 6)} for lag, counts in scores.items()},
        "answered_threshold": THRESHOLD,
        "first_followup_at_or_above_threshold": {"n": first_positive, "rated": scores[1].total()},
        "second_followup_at_or_above_threshold": {"n": second_positive, "rated": scores[2].total()},
        "first_followup_below_threshold_without_second_followup": unfinished_after_first,
        "below_threshold_after_second_followup": scores[2].total() - second_positive,
        "questions_with_observed_rating_at_or_above_threshold": {"n": first_positive + second_positive, "total_questions": len(questions)},
        "two_followup_rating_transitions": [
            {"first": first, "second": second, "n": n}
            for (first, second), n in sorted(two_scores.items())
        ],
        "probes_with_two_prior_questions_revisited": concurrent.total(),
        "two_question_threshold_outcomes": {
            "neither": concurrent[False, False],
            "newer_only": concurrent[False, True],
            "older_only": concurrent[True, False],
            "both": concurrent[True, True],
        },
        "newer_question_after_older_only_threshold": {
            "next_probe_at_or_above": newer_followup[True],
            "next_probe_below": newer_followup[False],
            "no_next_probe": newer_followup[None],
        },
        "out_of_chunk_highlight_offsets": {kind: spans_outside_chunk[kind] for kind in ("question", "answer")},
        "score_5_question_count": {
            "author_csv": scores[1][5] + scores[2][5],
            "reported_in_paper_section_4": 367,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elicitation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.elicitation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
