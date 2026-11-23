"""Prepare the held-out ContractNLI EDGAR dev corpus and seal its overlap audit."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import islice, product
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata
from urllib.parse import unquote, urlsplit

from research.semantic_tracking.contractnli_prepare import (
    ARCHIVE,
    EDGAR_TYPES,
    JUDGMENTS,
    PAPER,
    SOURCES,
    digest,
    normalized_text_hash,
    source_data,
)
from research.stage2.records import write_jsonl


SIMILARITY_THRESHOLD = 0.80
SHINGLE_SIZE = 5
EXPECTED_DEV_COUNTS = {
    "documents": 23,
    "documents_by_type": {"sec-html": 11, "sec-text": 12},
    "hypotheses": 17,
    "judgments": 391,
    "candidate_spans": 1739,
    "whitespace_only_candidate_spans": 54,
    "whitespace_only_evidence_memberships": 1,
    "labels": {"Entailment": 194, "Contradiction": 41, "NotMentioned": 156},
    "evidence_spans_by_label": {"Entailment": 427, "Contradiction": 82, "NotMentioned": 0},
}


def shingles(sequence):
    return set(zip(*(islice(sequence, offset, None) for offset in range(SHINGLE_SIZE))))


def shingle_inventory(documents):
    inventories = {}
    for document in sorted(documents, key=lambda row: row["id"]):
        normalized = " ".join(unicodedata.normalize("NFKC", document["text"]).split()).casefold()
        words = re.findall(r"[^\W_]+", normalized)
        inventories[document["id"]] = shingles(normalized), shingles(words)
    return inventories


def jaccard(left, right):
    intersection = len(left & right)
    return intersection / (len(left) + len(right) - intersection)


def canonical_url(row):
    parts = urlsplit(row["url"])
    return parts.hostname, unquote(parts.path)


def overlap_audit(fitting, evaluation):
    identity_keys = {
        "official_id": lambda row: row["id"],
        "raw_url": lambda row: row["url"],
        "canonical_url": canonical_url,
        "exact_text": lambda row: sha256(row["text"].encode("utf-8")).hexdigest(),
        "normalized_text": lambda row: normalized_text_hash(row["text"]),
    }
    overlaps = {}
    for name, key in identity_keys.items():
        fit_keys = {key(row): row["id"] for row in fitting}
        eval_keys = {key(row): row["id"] for row in evaluation}
        if len(fit_keys) != len(fitting) or len(eval_keys) != len(evaluation):
            raise ValueError(f"Within-split duplicate {name} in the EDGAR document inventory")
        overlaps[name] = sorted((fit_keys[match], eval_keys[match]) for match in fit_keys.keys() & eval_keys.keys())

    fit_shingles = shingle_inventory(fitting)
    eval_shingles = shingle_inventory(evaluation)
    metrics = ("character_jaccard", "word_jaccard")
    comparisons = [
        {
            "fitting_official_id": fit_id,
            "evaluation_official_id": eval_id,
            **{metric: jaccard(left, right)
               for metric, left, right in zip(metrics, fit_bags, eval_bags, strict=True)},
        }
        for (fit_id, fit_bags), (eval_id, eval_bags) in product(fit_shingles.items(), eval_shingles.items())
    ]
    strong_matches = [pair for pair in comparisons if any(pair[key] >= SIMILARITY_THRESHOLD for key in metrics)]
    if any(overlaps.values()) or strong_matches:
        raise ValueError(f"Cross-split EDGAR matches require review: {overlaps!r}, {strong_matches!r}")
    return {
        "identity_matches": overlaps,
        "comparisons": len(comparisons),
        "character_shingles": "5 adjacent Unicode code points of NFKC, whitespace-collapsed, casefolded text",
        "word_shingles": "5 adjacent Unicode alphanumeric tokens of the same normalized text",
        "metric": "intersection size divided by union size of the sets of shingles",
        "strong_match_threshold": SIMILARITY_THRESHOLD,
        "strong_matches": strong_matches,
        "most_similar_pairs": {key: max(comparisons, key=lambda pair: pair[key]) for key in metrics},
    }


def evaluation_rows(documents, labels):
    queries = [
        {"id": identifier, "text": labels[identifier]["hypothesis"]}
        for identifier in sorted(labels, key=lambda key: int(key.removeprefix("nda-")))
    ]
    if len({row["text"] for row in queries}) != 17 or any(
        not row["text"] or row["text"] != row["text"].strip() for row in queries
    ):
        raise ValueError("Expected 17 distinct literal hypotheses")

    stimuli, layout, gold = [], [], []
    whitespace_only = 0
    whitespace_evidence = 0
    labels_count = Counter()
    span_counts = Counter({judgment: 0 for judgment in JUDGMENTS})
    for document in sorted(documents, key=lambda row: row["id"]):
        official_id, text, spans = (document[name] for name in ("id", "text", "spans"))
        doc_id = f"contractnli-edgar-{official_id:04d}"
        group_id = normalized_text_hash(text)
        if (not spans or any(type(start) is not int or type(stop) is not int
                             or not 0 <= start < stop <= len(text) for start, stop in spans)
                or any(stop > next_start for (_, stop), (next_start, _) in zip(spans, spans[1:]))
                or text[spans[-1][1]:] != "\n"):
            raise ValueError(f"Invalid official spans for {doc_id}")
        blank = {index for index, (start, stop) in enumerate(spans) if not text[start:stop].strip()}
        whitespace_only += len(blank)
        identity = {"id": doc_id, "official_id": official_id, "group_id": group_id}
        stimuli.append({"id": doc_id, "text": text, "stops": [stop for _, stop in spans]})
        layout.append({**identity, "document_type": document["document_type"], "spans": spans})

        annotation_set, = document["annotation_sets"]
        annotations = annotation_set["annotations"]
        if annotation_set.keys() != {"annotations"} or annotations.keys() != labels.keys():
            raise ValueError(f"Unexpected merged annotation inventory for {doc_id}")
        for query in queries:
            query_id = query["id"]
            annotation = annotations[query_id]
            label, indices = annotation["choice"], annotation["spans"]
            if (annotation.keys() != {"choice", "spans"} or label not in JUDGMENTS
                    or not isinstance(indices, list)
                    or any(type(index) is not int or not 0 <= index < len(spans) for index in indices)
                    or len(set(indices)) != len(indices) or (indices and blank.issuperset(indices))
                    or (label == "NotMentioned") != (not indices)):
                raise ValueError(f"Invalid merged evidence for {doc_id} {query_id}")
            whitespace_evidence += len(blank.intersection(indices))
            gold.append({**identity, "query_id": query_id, "label": label, "evidence_indices": indices})
            labels_count[label] += 1
            span_counts[label] += len(indices)

    counts = {
        "documents": len(stimuli),
        "documents_by_type": Counter(row["document_type"] for row in layout),
        "hypotheses": len(queries),
        "judgments": len(gold),
        "candidate_spans": sum(len(row["spans"]) for row in layout),
        "whitespace_only_candidate_spans": whitespace_only,
        "whitespace_only_evidence_memberships": whitespace_evidence,
        "labels": labels_count,
        "evidence_spans_by_label": span_counts,
    }
    if counts != EXPECTED_DEV_COUNTS or len({row["group_id"] for row in layout}) != len(stimuli):
        raise ValueError(f"Unexpected dev inventory: {counts!r}")
    return {"stimuli.jsonl": stimuli, "layout.jsonl": layout, "queries.jsonl": queries, "gold.jsonl": gold}, counts


def prepare(raw, output):
    if output.exists():
        raise FileExistsError(output)
    fitting, labels, _ = source_data(raw)
    dev_path = raw / "dev.json"
    dev_bytes = dev_path.read_bytes()
    if sha256(dev_bytes).hexdigest() != SOURCES["dev"][2]:
        raise ValueError("Unexpected bytes in official dev.json")
    evaluation = [row for row in json.loads(dev_bytes)["documents"] if row["document_type"] in EDGAR_TYPES]
    overlap = overlap_audit(fitting, evaluation)
    tables, counts = evaluation_rows(evaluation, labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".contractnli-rerank-") as temporary:
        staged = Path(temporary) / "prepared"
        staged.mkdir(mode=0o700)
        for name, table in tables.items():
            write_jsonl(staged / name, table)
        audit = {
            "schema": 1,
            "paper": PAPER,
            "archive": ARCHIVE,
            "license": "CC BY 4.0",
            "source_code_sha256": digest(Path(__file__)),
            "fitting_split": "test",
            "retained_split": "dev",
            "source_json": {
                f"{split}.json": {"sha256": sha, "total_documents": count, "edgar_documents": edgar}
                for split, (count, edgar, sha) in SOURCES.items()
            },
            "evaluation_overlap_with_fitting": overlap,
            "counts": counts,
            "coordinates": "half-open Unicode code-point indices into the official literal source text",
            "stops": "all official candidate ends in order, excluding the final uncataloged newline",
            "evidence_indices": "complete unchanged zero-based lists into the corresponding layout spans",
            "group_id": "SHA-256 of Unicode NFKC then whitespace-collapsed then casefolded source text",
            "gold_endpoint": "merged offline human claim judgment and selected evidence membership",
            "grade_free_files": ["stimuli.jsonl", "layout.jsonl", "queries.jsonl"],
            "output_sha256": {name: digest(staged / name) for name in tables},
        }
        (staged / "source-audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        staged.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="directory containing the official split JSON files")
    parser.add_argument("--output", type=Path, required=True)
    prepare(**vars(parser.parse_args()))
