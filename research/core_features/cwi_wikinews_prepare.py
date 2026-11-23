"""Prepare sentence-level WikiNews inputs and independent train, dev and test votes."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from research.core_features import background_prepare as wikinews
from research.core_features import cwi_news_prepare as news
from research.semantic_tracking.iirc_anchor_fit import WORD
from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3.run import save_json


PROTOCOL = Path(__file__).with_name("cwi-lexical-readout-protocol.md")
PRIOR_AUDIT = Path(__file__).resolve().parents[1] / "results/core-features-first-run/cwi-source-audit.json"
PRIOR_AUDIT_SHA256 = "83ec7c9e0aca89aa299e361b8bb71d36b3c9b3e2aef90f52fb0de432a9f6ce9a"
SPLITS = ("train", "dev", "test")


def original_sentence_counts(archive):
    """Count each released HIT/position once, including excluded HITs."""
    original = {}
    original_splits = {}
    with ZipFile(archive) as corpus:
        for member, (release_split, expected) in wikinews.MEMBERS.items():
            with corpus.open(member) as raw, io.TextIOWrapper(raw, encoding="utf-8", newline="") as stream:
                reader = csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
                seen = 0
                for hit, text, *_ in reader:
                    marker = wikinews.MARKER.match(text)
                    if marker is None:
                        raise ValueError(f"Missing original WikiNews sentence marker in {member}")
                    article, position = map(int, marker.groups())
                    key = hit, article, position
                    literal = text[marker.end():]
                    if (original.setdefault(key, literal) != literal
                            or original_splits.setdefault(hit, release_split) != release_split):
                        raise ValueError("Original HIT and sentence positions must have one literal text and split")
                    seen += 1
                if seen != expected:
                    raise ValueError("Original WikiNews row counts changed")
    return Counter(wikinews.normalized_sentence(text) for text in original.values())


def verified_partition(hits):
    if digest(PRIOR_AUDIT) != PRIOR_AUDIT_SHA256:
        raise ValueError("Prior WikiNews article split receipt changed")
    receipt = json.loads(PRIOR_AUDIT.read_text(encoding="utf-8"))
    partition, repeats = wikinews.split_by_article(hits)
    mapping = {str(article): {"group_id": group, "split": split} for article, (group, split) in partition.items()}
    if (mapping != receipt["article_markers_by_group_and_split"]
            or repeats != receipt["normalized_sentences_shared_by_two_or_more_articles"]):
        raise ValueError("The original project article families and split assignments could not be reproduced")
    stable = json.dumps(mapping, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return partition, hashlib.sha256(stable).hexdigest()


def word_records(text, spans, frequencies):
    locators = {match.span() for match in WORD.finditer(text)}
    outcomes = []
    for (start, end), (native_total, nonnative_total, native_votes, nonnative_votes) in sorted(spans.items()):
        if (start, end) not in locators:
            continue
        if (native_total, nonnative_total) != (10, 10):
            raise ValueError("Unexpected vote denominator within released annotations")
        forms = news.words(text[start:end])
        if len(forms) != 1:
            raise ValueError("A complete literal word must map to one CompLex frequency form")
        outcomes.append({"start": start, "end": end, "native_votes": native_votes,
                         "nonnative_votes": nonnative_votes, "form_count": frequencies[forms[0]]})
    return outcomes


def select(hits, partition, occurrence_counts, frequencies):
    stimuli = []
    layout = []
    outcomes = {split: [] for split in SPLITS}
    exclusions = {split: Counter() for split in SPLITS}
    source_counts = Counter()
    seen = set()
    for _, sentences in sorted(hits.items()):
        for (article, _), (text, spans) in sorted(sentences.items()):
            group, split = partition[article]
            source_counts[split] += len(spans)
            canonical = wikinews.normalized_sentence(text)
            if occurrence_counts[canonical] != 1:
                if not occurrence_counts[canonical]:
                    raise ValueError("Eligible sentence absent from the release-wide duplication audit")
                exclusions[split]["duplicate_normalized_sentence"] += len(spans)
                continue
            records = word_records(text, spans, frequencies)
            exclusions[split]["not_one_complete_served_word"] += len(spans) - len(records)
            if not records:
                continue
            if canonical in seen:
                raise ValueError("Normalized sentence reappeared after the release-wide duplication audit")
            seen.add(canonical)
            identifier = "cwi-wikinews-" + hashlib.sha256(text.encode("utf-8")).hexdigest()
            identity = {"id": identifier, "group_id": group, "split": split}
            stimuli.append({"id": identifier, "text": text})
            layout.append({**identity, "words": [[row["start"], row["end"], row["form_count"]] for row in records]})
            outcomes[split].append({**identity, "words": records})
    counts = {
        split: {
            "sentences": len(rows),
            "candidates": sum(len(row["words"]) for row in rows),
            "article_groups": len({row["group_id"] for row in rows}),
        } for split, rows in outcomes.items()
    }
    if len(stimuli) != len(seen) or any(
        counts[split]["candidates"] + exclusions[split].total() != source_counts[split] for split in SPLITS
    ):
        raise ValueError("Retained sentences and exclusions do not reconcile to source")
    for split in SPLITS:
        if not counts[split]["candidates"] or not counts[split]["article_groups"]:
            raise ValueError("Every project split must retain its own source article groups and candidate votes")
    return stimuli, layout, outcomes, counts, source_counts, exclusions


def prepare(archive, complex_stimuli, checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
    checkpoint_hash = news.checkpoint_hashes(checkpoint)
    protocol_hash = digest(PROTOCOL)
    hits, rejected_anchors, rejected_hits, _, total_hits = wikinews.read_hits(archive)
    partition, partition_hash = verified_partition(hits)
    occurrence_counts = original_sentence_counts(archive)
    _, frequencies = news.read_complex(complex_stimuli)
    stimuli, layout, outcomes, counts, source_counts, exclusions = select(hits, partition, occurrence_counts, frequencies)
    release_anchors = sum(count for _, count in wikinews.MEMBERS.values())
    if source_counts.total() + rejected_anchors.total() != release_anchors:
        raise ValueError("Eligible and rejected anchors must account for each original release row")
    files = {"stimuli.jsonl": stimuli, "layout.jsonl": layout, **{f"{key}.jsonl": value for key, value in outcomes.items()}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cwi-wikinews-", dir=output.parent) as tmp:
        staged = Path(tmp) / "prepared"
        staged.mkdir()
        for filename, rows in files.items():
            write_jsonl(staged / filename, rows)
        hashes = {name: digest(staged / name) for name in files}
        audit = {
            "schema": 1,
            "candidate_id": news.CANDIDATE,
            "lexical_weights_sha256": news.LEXICAL_SHA256,
            "checkpoint_manifest_sha256": checkpoint_hash,
            "protocol_sha256": protocol_hash,
            "preparation_code_sha256": digest(Path(__file__)),
            "stimuli_sha256": hashes["stimuli.jsonl"],
            "layout_sha256": hashes["layout.jsonl"],
            "partition_sha256": partition_hash,
            "partition_sha256_format": "SHA-256 of sorted compact JSON mapping article markers to their original group and split",
            "prior_source_audit_sha256": PRIOR_AUDIT_SHA256,
            "file_sha256": hashes,
            "source_url": wikinews.SOURCE,
            "source_sha256": wikinews.SOURCE_SHA256,
            "source_members": {member: {"original_split": split, "rows": rows}
                               for member, (split, rows) in wikinews.MEMBERS.items()},
            "source_hits": total_hits,
            "source_sentences": occurrence_counts.total(),
            "source_anchors": release_anchors,
            "rejected_source_hits": rejected_hits,
            "rejected_source_anchors": rejected_anchors,
            "duplicated_normalized_sentence_classes_across_full_source": sum(v > 1 for v in occurrence_counts.values()),
            "sentence_normalization": "Unicode NFKC, casefold, collapse whitespace, preserving punctuation",
            "duplication_scope": "All occurrence locations across released HITs and partitions, including excluded HITs",
            "candidate_selection": "Exact source offsets matching one complete served word in the literal sentence",
            "complex_stimuli_sha256": news.COMPLEX_SHA256,
            "frequency_reference_splits": ["train", "trial"],
            "frequency_forms": len(frequencies),
            "frequency_tokens": frequencies.total(),
            "eligible_source_anchors_by_partition_before_filter": source_counts,
            "excluded_candidate_counts_by_split": exclusions,
            "retained_sentences": len(stimuli),
            "retained_counts_by_split": counts,
            "grade_free_files": ["stimuli.jsonl", "layout.jsonl"],
            "votes_per_group": {"native": 10, "nonnative": 10},
            "license": "CC BY 4.0 per author release, with WikiNews source text terms",
            "presentation": "Literal released sentence with only the article/sentence marker stripped",
            "split_roles": {"train": "fit", "dev": "validation", "test": "evaluation"},
        }
        save_json(staged / "assay.json", audit)
        staged.rename(output)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "complex-stimuli", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = vars(parser.parse_args())
    print(json.dumps(prepare(**args)["retained_counts_by_split"], sort_keys=True))
