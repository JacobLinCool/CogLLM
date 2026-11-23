"""Audit the sentence-level reader judgments in the Microsoft Hippocorpus V3 release."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import TextIOWrapper
import json
import math
from pathlib import Path
from zipfile import ZipFile


ARCHIVE_SHA256 = "cd27c69e43a3095eae98bad2d195b1f8ae20c30321160a904d65b9d1d43d385e"
SOURCE = "https://www.microsoft.com/en-us/download/details.aspx?id=105291"
FLAGS = ("majorBin", "minorBin", "expectedBin", "surprisingBin", "eventOrNot")
RAW_FLAGS = {
    "noEvent": (0, 0, 0, 0, 0),
    "minor-expected": (0, 1, 1, 0, 1),
    "major-expected": (1, 0, 1, 0, 1),
    "major-surprising": (1, 0, 0, 1, 1),
}
STORY_TYPES = {"imagined", "recalled", "retold"}


def rows(archive, name):
    with TextIOWrapper(archive.open(name), encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def sentence_key(row):
    return row["HippocorpusAssignmentId"], int(row["sentIx"])


def audit(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("Expected the fixed Microsoft Hippocorpus V3 archive")

    by_sentence = defaultdict(list)
    raw_labels = Counter()
    workers = set()
    with ZipFile(path) as archive:
        for row in rows(archive, "hcV3-eventAnnots.csv"):
            if tuple(int(row[key]) for key in FLAGS) != RAW_FLAGS[row["rawEventAnnot"]]:
                raise ValueError("Published flags differ from the original reader label")
            by_sentence[sentence_key(row)].append(row)
            raw_labels[row["rawEventAnnot"]] += 1
            workers.add(row["WorkerId"])
        aggregate = {}
        for row in rows(archive, "hcV3-eventAnnotsAggOverWorkers.csv"):
            key = sentence_key(row)
            if key in aggregate:
                raise ValueError("Duplicate sentence in the released aggregate file")
            aggregate[key] = row

    if by_sentence.keys() != aggregate.keys():
        raise ValueError("Raw responses and aggregate rows cover different sentences")
    topic_stories = defaultdict(dict)
    story_indices = defaultdict(set)
    response_counts = Counter()
    pluralities = Counter()
    for (story, index), responses in by_sentence.items():
        released = aggregate[(story, index)]
        topic, story_type = released["HITId"], released["memType"]
        metadata = ("HITId", "memType", "sent", "storyIx")
        if any(any(row[key] != released[key] for key in metadata) for row in responses):
            raise ValueError("Raters or aggregate disagree on the identified sentence")
        n = len(responses)
        if n not in {7, 8} or len({row["WorkerId"] for row in responses}) != n:
            raise ValueError("Unexpected number of distinct responses for a sentence")
        if topic_stories[topic].setdefault(story_type, story) != story:
            raise ValueError("A root topic has multiple stories of the same type")
        story_indices[story].add(index)
        response_counts[n] += 1
        means = {key: sum(int(row[key]) for row in responses) / n for key in FLAGS}
        if any(not math.isclose(means[key], float(released[key]), abs_tol=1e-12) for key in FLAGS):
            raise ValueError("Cannot reproduce a published aggregate from raw reader votes")
        if index > 0:
            categories = {
                "no_event": n - sum(int(row["eventOrNot"]) for row in responses),
                "expected": sum(int(row["expectedBin"]) for row in responses),
                "surprising": sum(int(row["surprisingBin"]) for row in responses),
            }
            winners = [label for label, count in categories.items() if count == max(categories.values())]
            pluralities[winners[0] if len(winners) == 1 else "tie"] += 1

    if len(topic_stories) != 80 or len(story_indices) != 240 or any(
        variants.keys() != STORY_TYPES for variants in topic_stories.values()
    ) or any(indices != set(range(len(indices))) for indices in story_indices.values()):
        raise ValueError("Unexpected topic variants or incomplete sentence sequences")

    return {
        "source": SOURCE,
        "archive_sha256": digest,
        "reader_judgments": sum(raw_labels.values()),
        "distinct_anonymized_reader_ids": len(workers),
        "sentences": len(by_sentence),
        "stories": len(story_indices),
        "topic_families": len(topic_stories),
        "raw_label_counts": dict(raw_labels),
        "sentences_by_number_of_responses": dict(response_counts),
        "released_binary_column_means_reproduced": True,
        "pluralities_after_excluding_initial_sentences": dict(pluralities),
        "published_table_1_for_comparison": {"no_event": 2255, "expected": 650, "surprising": 509, "tie": 511},
        "scope": "Prompted sentence-level narrative event judgments, including explicit no-event votes.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
