"""Audit public INQUISITIVE question records without constructing model inputs."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
from pathlib import Path
import re
import unicodedata


RELEASE = "9c364b689a3c134cc5a0bc886f0ce7ce190e6111"
REPO = "https://github.com/wjko2/INQUISITIVE"
SHA256 = "3d954e957d6df1dde297682b1c1f8c63ccc0c7cff1dcb9b995a7e054b3dd04ee"
DCQA_RELEASE = "011fd2cbb0bc49d2af41af4fb60da7a48aca5463"
DCQA_REPO = "https://github.com/wjko2/DCQA-Discourse-Comprehension-by-Question-Answering"
DCQA_SHA256 = "66156fd2949ae2c2fa204e527129f7da511c973415b53c26ec037f728670b144"
ARTICLE_COUNT = 1500
SENTENCES_PER_ARTICLE = 5
RELEASE_COUNTS = {"train": 15931, "validation": 1991, "test": 1894}
HEADER = [
    "Article_Id", "Sentence_Id", "Sentence", "Span", "Question",
    "Span_Start_Position", "Span_End_Position",
]
VALIDATION_ARTICLES = (range(1, 101), range(1051, 1101))
TEST_ARTICLES = (range(101, 151), range(501, 551), range(1101, 1151))


def split(article):
    if any(article in ids for ids in VALIDATION_ARTICLES):
        return "validation"
    if any(article in ids for ids in TEST_ARTICLES):
        return "test"
    return "train"


def focal_key(sentence):
    return " ".join(unicodedata.normalize("NFKC", sentence).casefold().split())


def span_status(sentence, span, start, end):
    # The author release counts literal spaces and retains consecutive white spaces.
    words = sentence.split(" ")
    if not span.strip():
        return "empty_span"
    if not 0 <= start < end <= len(words):
        return "invalid_offsets"
    if " ".join(words[start:end]) != span:
        return "span_does_not_match_offsets"
    return "exact"


def collect(path):
    focal_texts = {}
    seen_spans = set()
    row_counts = Counter()
    report_counts = Counter()
    question_counts = Counter()
    span_counts = Counter()
    failures = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        if next(reader, None) != HEADER:
            raise ValueError("Unexpected INQUISITIVE question file header")
        for row in reader:
            line_number = reader.line_num
            if len(row) != len(HEADER):
                raise ValueError(f"Unexpected number of columns on source line {line_number}")
            article_text, sentence_id_text, sentence, span, question, start_text, end_text = row
            article, sentence_id = int(article_text), int(sentence_id_text)
            start, end = int(start_text), int(end_text)
            if not (
                1 <= article <= ARTICLE_COUNT and 1 <= sentence_id <= SENTENCES_PER_ARTICLE
                and sentence.strip() and question.strip()
            ):
                raise ValueError(f"Invalid question record on source line {line_number}")
            stop = (article, sentence_id)
            if focal_texts.setdefault(stop, sentence) != sentence:
                raise ValueError(f"Focal sentence differs inside one reported stop on line {line_number}")
            row_counts[split(article)] += 1
            report_counts[stop] += 1
            question_counts[article, question] += 1
            status = span_status(sentence, span, start, end)
            span_counts[status] += 1
            if status == "exact":
                seen_spans.add((*stop, start, end))
            else:
                failures.append({
                    "source_line": line_number, "article_id": article,
                    "sentence_id": sentence_id, "reason": status,
                })
    if row_counts != RELEASE_COUNTS or len({article for article, _ in focal_texts}) != ARTICLE_COUNT:
        raise ValueError("Question or article count differs from the pinned release")
    return focal_texts, report_counts, span_counts, seen_spans, failures, question_counts


def require_digest(path, expected):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected:
        raise ValueError(f"Expected pinned author release for {path.name}")
    return digest


def dcqa_audit(path, question_counts):
    digest = require_digest(path, DCQA_SHA256)
    articles = set()
    ids = set()
    events = Counter()
    by_split = Counter()
    source_matches = Counter()
    answer_location = re.compile(r"[1-9][0-9]*(?:,[1-9][0-9]*)*")
    for record in json.loads(path.read_text(encoding="utf-8")):
        if record.keys() != {"Article"} or len(record["Article"]) != 1:
            raise ValueError("Unexpected DCQA article record")
        article = record["Article"][0]
        article_id = article["ArticleID"]
        if article_id in articles or not 1 <= article_id <= ARTICLE_COUNT:
            raise ValueError(f"Invalid or duplicate DCQA article {article_id}")
        articles.add(article_id)
        for qa in article["qas"]:
            identifier = qa["QuestionID"]
            question, separator, _ = qa["Question"].partition(" | ")
            exact_matches = question_counts[article_id, question]
            impossible = qa["is_impossible"]
            if identifier in ids or not separator or exact_matches < 1 or type(impossible) is not bool:
                raise ValueError(f"Invalid or unlinked DCQA question {identifier}")
            ids.add(identifier)
            by_split[split(article_id)] += 1
            source_matches["unique" if exact_matches == 1 else "ambiguous"] += 1
            if impossible:
                events["marked_unanswerable"] += 1
            else:
                events["marked_answerable"] += 1
                events["answerable_with_numeric_location"] += bool(answer_location.fullmatch(qa["AnswerSentenceID"]))
    if len(ids) != 719 or len(articles) != 61:
        raise ValueError("DCQA question or article counts differ from the pinned release")
    return {
        "source": {
            "revision": DCQA_RELEASE,
            "answers_url": f"{DCQA_REPO}/blob/{DCQA_RELEASE}/inquisitive.json",
            "article_terms_url": f"{DCQA_REPO}/tree/{DCQA_RELEASE}#article-sources",
            "method_url": "https://aclanthology.org/2022.emnlp-main.806.pdf#page=7",
            "bytes": path.stat().st_size,
            "sha256": digest,
        },
        "scope": "Retrospective expert answerability and candidate answer-sentence positions for the original question family.",
        "reader_linked_trajectory_available": False,
        "released_article_count": len(articles),
        "paper_article_count": 60,
        "question_rows": len(ids),
        "questions_by_original_article_split": dict(sorted(by_split.items())),
        "matches_by_article_id_and_exact_question": dict(source_matches),
        **events,
        "answerable_without_numeric_location": events["marked_answerable"] - events["answerable_with_numeric_location"],
    }


def audit(path, dcqa_path):
    digest = require_digest(path, SHA256)
    focal_texts, report_counts, span_counts, seen_spans, failures, question_counts = collect(path)
    scheduled = Counter(split(article) for article in range(1, ARTICLE_COUNT + 1))
    by_split = {}
    focal_folds = defaultdict(set)
    for (article, _), text in focal_texts.items():
        focal_folds[focal_key(text)].add(split(article))
    for name in RELEASE_COUNTS:
        stops = [(article, sentence) for article, sentence in focal_texts if split(article) == name]
        stop_count = scheduled[name] * SENTENCES_PER_ARTICLE
        by_split[name] = {
            "question_rows": sum(report_counts[stop] for stop in stops),
            "scheduled_article_ids": scheduled[name],
            "scheduled_article_sentence_stops": stop_count,
            "stops_with_a_released_question": len(stops),
            "stops_without_released_rows": stop_count - len(stops),
            "released_stops_with_at_least_one_preceding_sentence_unrepresented": sum(
                any((article, earlier) not in focal_texts for earlier in range(1, sentence))
                for article, sentence in stops
            ),
        }
    split_pairs = {
        f"{first}/{second}": sum({first, second} <= memberships for memberships in focal_folds.values())
        for first, second in itertools.combinations(RELEASE_COUNTS, 2)
    }
    return {
        "source": {
            "revision": RELEASE,
            "questions_url": f"{REPO}/blob/{RELEASE}/questions.txt",
            "article_terms_url": f"{REPO}/tree/{RELEASE}#article-sources",
            "protocol_url": "https://aclanthology.org/2020.emnlp-main.530.pdf#page=3",
            "sha256": digest,
            "bytes": path.stat().st_size,
        },
        "model_access": {
            "status": "article_access_and_reuse_unverified",
            "author_article_access": "Request by email. Newsela articles require prior Newsela permission.",
            "hugging_face_loader_article_archive_status": "Author upstream article URL returned HTTP 404 on verification.",
            "hugging_face_card": "https://huggingface.co/datasets/UT-CompLing/inquisitive_qg/tree/11b9160bec51fb01e2f1999f0de1c399aa81567a",
            "model_inputs_built": False,
        },
        "measurement": {
            "observed": "A validated reader question after a complete focal sentence was visible, paired with a focal-sentence highlight.",
            "unobserved": [
                "Within-sentence time of onset",
                "Prior-sentence source anchors",
                "Worker-linked zero-question trials",
                "Repeated uncertainty or resolution from the same reader",
            ],
            "denominator": "Unreleased stop rows can include zero reports or reports excluded by validation. They do not identify no-question trials.",
        },
        "by_official_article_split": by_split,
        "total": {
            "question_rows": report_counts.total(),
            "scheduled_article_sentence_stops": ARTICLE_COUNT * SENTENCES_PER_ARTICLE,
            "stops_with_a_released_question": len(report_counts),
            "stops_without_released_rows": ARTICLE_COUNT * SENTENCES_PER_ARTICLE - len(report_counts),
            "released_stops_with_at_least_one_preceding_sentence_unrepresented": sum(
                metrics["released_stops_with_at_least_one_preceding_sentence_unrepresented"]
                for metrics in by_split.values()
            ),
            "exact_span_rows": span_counts["exact"],
            "distinct_exact_highlights_at_stops": len(seen_spans),
        },
        "unusable_span_rows": failures,
        "cross_split_focal_sentences": {
            "normalization": "Unicode NFKC, case folding and whitespace normalization",
            "unique_normalized_focal_strings_in_multiple_splits": sum(
                len(memberships) > 1 for memberships in focal_folds.values()
            ),
            "by_split_pair_including_all_split_overlap": split_pairs,
        },
        "dcqa_inquisitive_reanalysis": dcqa_audit(dcqa_path, question_counts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--dcqa-answers", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.questions, args.dcqa_answers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
