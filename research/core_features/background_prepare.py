"""Prepare grouped WikiNews lexical marking and annotator-background assays."""

import argparse
import csv
import hashlib
import io
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile


SOURCE = (
    "https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/"
    "complex-word-identification-dataset/cwishareddataset.zip"
)
SOURCE_SHA256 = "4a47b3f0b4589907e44f5cc19231dc24fee157636b65ba668ef1850224e52e3e"
MEMBERS = {
    "traindevset/english/WikiNews_Train.tsv": ("train", 7746),
    "traindevset/english/WikiNews_Dev.tsv": ("dev", 870),
    "testset/english/WikiNews_Test.tsv": ("test", 1287),
}
MARKER = re.compile(r"^#([1-9][0-9]*)-([1-9][0-9]*) ")
PARTITION_SEED = 43
HOLDOUT_GROUPS = 7


def normalized_sentence(sentence):
    return " ".join(unicodedata.normalize("NFKC", sentence).casefold().split())


def read_hits(archive):
    with archive.open("rb") as source:
        sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    if sha256 != SOURCE_SHA256:
        raise ValueError("The source must be the pinned official CWI 2018 archive")

    hits = defaultdict(dict)
    hit_partitions = {}
    source_partitions = defaultdict(set)
    exclusions = Counter()
    with ZipFile(archive) as corpus:
        for member, (split, expected_rows) in MEMBERS.items():
            with corpus.open(member) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
                    rows = list(csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE))
            if len(rows) != expected_rows:
                raise ValueError(f"Unexpected number of rows in {member}")
            for hit_id, raw_sentence, start, stop, target, nt, nnt, nm, nnm, label, probability in rows:
                marker = MARKER.match(raw_sentence)
                if marker is None:
                    raise ValueError("Missing WikiNews article and sentence marker")
                article, position = map(int, marker.groups())
                start, stop, nt, nnt, nm, nnm, label = map(int, (start, stop, nt, nnt, nm, nnm, label))
                if (nt, nnt) != (10, 10) or not (0 <= nm <= nt and 0 <= nnm <= nnt):
                    raise ValueError("Unexpected native or non-native annotation counts")
                if label != int(nm + nnm > 0) or abs(float(probability) - (nm + nnm) / (nt + nnt)) > 1e-12:
                    raise ValueError("Published labels disagree with native and non-native vote counts")
                if not marker.end() <= start < stop <= len(raw_sentence):
                    raise ValueError("The annotated target falls outside the displayed sentence")
                if hit_partitions.setdefault(hit_id, split) != split:
                    raise ValueError("A published HIT appears in multiple official splits")
                source_partitions[article].add(split)

                sentence = raw_sentence[marker.end():]
                previous_sentence, spans = hits[hit_id].setdefault((article, position), (sentence, {}))
                if sentence != previous_sentence:
                    raise ValueError("The same HIT and sentence marker has two sentence texts")
                if raw_sentence[start:stop] != target:
                    if (raw_sentence[start:stop], target) not in {("’s", "'s"), ("’ve", "'ve")}:
                        raise ValueError("Unrecognized exact-offset failure in the pinned source")
                    exclusions["offset_mismatch"] += 1
                    continue
                offsets = (start - marker.end(), stop - marker.end())
                if offsets in spans:
                    raise ValueError("Duplicate annotated candidate in a released sentence")
                spans[offsets] = (nt, nnt, nm, nnm)

    eligible = {}
    rejected_hits = Counter()
    for hit_id, sentences in hits.items():
        positions = sorted(sentences)
        articles = {article for article, _ in positions}
        n_rows = sum(len(spans) for _, spans in sentences.values())
        if len(articles) > 1:
            reason = "mixed_article_hit"
        elif [position for _, position in positions] != list(range(positions[0][1], positions[-1][1] + 1)):
            reason = "gapped_sentence_positions"
        else:
            eligible[hit_id] = sentences
            continue
        rejected_hits[reason] += 1
        exclusions[reason] += n_rows
    return eligible, exclusions, rejected_hits, source_partitions, len(hits)


def split_by_article(hits):
    """Group shared article markers and exactly repeated normalized sentences."""
    text_articles = defaultdict(set)
    for sentences in hits.values():
        for (article, _), (sentence, _) in sentences.items():
            text_articles[normalized_sentence(sentence)].add(article)
    by_article = {article: {article} for family in text_articles.values() for article in family}
    for family in text_articles.values():
        if len(family) > 1:
            component = set().union(*(by_article[article] for article in family))
            for article in component:
                by_article[article] = component
    families = sorted({tuple(sorted(component)) for component in by_article.values()})
    if len(families) <= 2 * HOLDOUT_GROUPS:
        raise ValueError("Too few independent WikiNews article groups for the locked split")
    random.Random(PARTITION_SEED).shuffle(families)
    partition = {
        article: (f"wikinews-article-{family[0]:02}", split)
        for split, partition_families in (
            ("test", families[:HOLDOUT_GROUPS]),
            ("dev", families[HOLDOUT_GROUPS:2 * HOLDOUT_GROUPS]),
            ("train", families[2 * HOLDOUT_GROUPS:]),
        )
        for family in partition_families
        for article in family
    }
    return partition, sum(len(articles) > 1 for articles in text_articles.values())


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare(archive, output):
    hits, exclusions, rejected_hits, original_partitions, original_hits = read_hits(archive)
    partition, duplicate_sentences = split_by_article(hits)
    stimuli, outcomes = [], []
    split_excerpts = Counter()
    split_anchors = Counter()
    split_marked = defaultdict(Counter)
    for hit_index, hit in enumerate(sorted(hits), start=1):
        sentences = hits[hit]
        excerpt_id = f"wikinews-{hit_index:03}"
        text_parts, stops = [], set()
        character = 0
        for (article, _), (sentence, spans) in sorted(sentences.items()):
            group_id, split = partition[article]
            text_parts.append(sentence)
            for (start, stop), (n_total, nn_total, n_marked, nn_marked) in sorted(spans.items()):
                end = character + stop
                stops.add(end)
                outcomes.append({
                    "id": excerpt_id, "group_id": group_id, "start": character + start, "stop": end,
                    "native_total": n_total, "nonnative_total": nn_total,
                    "native_marked": n_marked, "nonnative_marked": nn_marked, "split": split,
                })
                split_anchors[split] += 1
                split_marked[split]["native"] += n_marked
                split_marked[split]["nonnative"] += nn_marked
                split_marked[split]["any_group"] += int(n_marked + nn_marked > 0)
            character += len(sentence) + 1
        stimuli.append({"id": excerpt_id, "text": "\n".join(text_parts), "stops": sorted(stops)})
        split_excerpts[split] += 1

    output.mkdir(parents=True, exist_ok=True)
    files = {"stimuli.jsonl": stimuli, "outcomes.jsonl": outcomes}
    sha256s = {}
    for name, rows in files.items():
        path = output / name
        write_jsonl(path, rows)
        sha256s[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    group_counts = Counter(split for _, split in set(partition.values()))
    audit = {
        "source_url": SOURCE,
        "source_archive_sha256": SOURCE_SHA256,
        "author_release": "https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/complex-word-identification-dataset.html",
        "annotation_protocol": "https://sites.google.com/view/cwisharedtask2018/datasets",
        "paper": "https://aclanthology.org/W18-0507/",
        "release_license": "CC BY 4.0 per the author release, with WikiNews source text terms",
        "source_files": {member: count for member, (_, count) in MEMBERS.items()},
        "original_annotation_hits": original_hits,
        "original_article_markers": len(original_partitions),
        "original_articles_in_multiple_splits": sum(len(splits) > 1 for splits in original_partitions.values()),
        "offset_units": "Unicode code points, exclusive ends",
        "text_preparation": "Strip article and sentence markers, then join published consecutive sentences within the same article/HIT in source order with a newline.",
        "endpoint": "Votes to mark a released candidate as difficult for the task's named readerships, counted separately by annotators' native English status.",
        "excluded_anchors_by_reason": dict(exclusions),
        "excluded_annotation_hits_by_reason": dict(rejected_hits),
        "normalized_sentences_shared_by_two_or_more_articles": duplicate_sentences,
        "partition_seed": PARTITION_SEED,
        "article_markers_by_group_and_split": {
            str(article): {"group_id": group, "split": split}
            for article, (group, split) in sorted(partition.items())
        },
        "groups_by_split": dict(group_counts),
        "excerpts_by_split": dict(split_excerpts),
        "anchors_by_split": dict(split_anchors),
        "mark_counts_by_split": dict(split_marked),
        "output_sha256": sha256s,
    }
    (output / "source-audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    prepare(arguments.archive, arguments.output)
