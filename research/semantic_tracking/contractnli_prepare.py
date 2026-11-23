"""Prepare exact EDGAR test bodies, hypotheses, and separate ContractNLI evidence labels."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unicodedata
from urllib.parse import urlsplit

from research.stage2.records import write_jsonl


ARCHIVE = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
PAPER = "https://doi.org/10.18653/v1/2021.findings-emnlp.164"
SOURCES = {
    "train": (423, 162, "dbceb356cd6203b35b27be94a5fa85e499a81c34c42c89ad53060b39f0257ba5"),
    "dev": (61, 23, "310af7d661d2ab50ee3700169cef524c75f39fb296bbf5a515c229eb0f42e68e"),
    "test": (123, 47, "460267b56052a2dc5aead98eb35eadef9e6734d5723d37b4a9790e410f812387"),
}
EDGAR_TYPES = {"sec-text", "sec-html"}
JUDGMENTS = {"Entailment", "Contradiction", "NotMentioned"}
EXPECTED_DUPLICATES = [[{"split": "train", "official_id": 538}, {"split": "test", "official_id": 537}]]


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def normalized_text_hash(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).split()).casefold()
    return sha256(normalized.encode("utf-8")).hexdigest()


def source_data(raw):
    documents, labels, official_ids = {}, None, set()
    by_text, by_url, by_normalized = defaultdict(set), defaultdict(set), defaultdict(list)
    for split, (official_count, edgar_count, checksum) in SOURCES.items():
        path = raw / f"{split}.json"
        payload = path.read_bytes()
        if sha256(payload).hexdigest() != checksum:
            raise ValueError(f"Unexpected official ContractNLI bytes: {path.name}")
        source = json.loads(payload)
        ids = {row["id"] for row in source["documents"]}
        if len(source["documents"]) != official_count or len(ids) != official_count or official_ids & ids:
            raise ValueError("Official document split IDs changed or overlap")
        official_ids.update(ids)
        if labels is None:
            labels = source["labels"]
        if source["labels"] != labels or len(labels) != 17:
            raise ValueError("The official splits must share the same 17 hypotheses")
        selected = [row for row in source["documents"] if row["document_type"] in EDGAR_TYPES]
        if len(selected) != edgar_count or any(urlsplit(row["url"]).hostname != "www.sec.gov" for row in selected):
            raise ValueError(f"Unexpected EDGAR source inventory in {split}")
        for row in selected:
            by_text[sha256(row["text"].encode("utf-8")).hexdigest()].add(split)
            by_url[row["url"]].add(split)
            by_normalized[normalized_text_hash(row["text"])].append({"split": split, "official_id": row["id"]})
        if split == "test":
            documents = selected

    duplicates = [group for group in by_normalized.values() if len({row["split"] for row in group}) > 1]
    if any(len(splits) > 1 for index in (by_text, by_url) for splits in index.values()) or duplicates != EXPECTED_DUPLICATES:
        raise ValueError("Unexpected raw or normalized inter-split EDGAR overlap")
    return documents, labels, duplicates


def prepare_rows(documents, labels):
    queries = [{"id": key, "text": labels[key]["hypothesis"]}
               for key in sorted(labels, key=lambda identifier: int(identifier.removeprefix("nda-")))]
    if len({row["text"] for row in queries}) != 17 or any(
            not row["text"] or row["text"] != row["text"].strip() for row in queries):
        raise ValueError("Expected 17 distinct literal hypotheses")

    stimuli, layout, gold = [], [], []
    counts, spans_by_label, disjoint = Counter(), Counter(), Counter()
    by_query = defaultdict(Counter)
    c_e_documents, blank_spans = 0, 0
    for document in sorted(documents, key=lambda row: row["id"]):
        official_id, text, spans = (document[name] for name in ("id", "text", "spans"))
        doc_id = f"contractnli-edgar-{official_id:04d}"
        group = normalized_text_hash(text)
        starts_and_ends = all(type(s) is int and type(e) is int and 0 <= s < e <= len(text) for s, e in spans)
        if (not starts_and_ends or not spans or any(e > next_start for (_, e), (next_start, _) in zip(spans, spans[1:]))
                or text[spans[-1][1]:] != "\n"):
            raise ValueError(f"Invalid candidate range inventory for {doc_id}")
        blank_spans += sum(not text[start:end].strip() for start, end in spans)
        stimuli.append({"id": doc_id, "text": text, "stops": [end for _, end in spans]})
        layout.append({"id": doc_id, "official_id": official_id, "group_id": group,
                       "document_type": document["document_type"], "spans": spans})

        annotation_set, = document["annotation_sets"]
        annotations = annotation_set["annotations"]
        if annotation_set.keys() != {"annotations"} or annotations.keys() != labels.keys():
            raise ValueError(f"Evidence judgments do not cover all hypotheses for {doc_id}")
        evidence = []
        for query in queries:
            key = query["id"]
            annotation = annotations[key]
            label, indices = annotation["choice"], annotation["spans"]
            if (annotation.keys() != {"choice", "spans"} or label not in JUDGMENTS
                    or not isinstance(indices, list) or len(set(indices)) != len(indices)
                    or any(type(i) is not int or not 0 <= i < len(spans) for i in indices)
                    or (label == "NotMentioned") != (not indices)):
                raise ValueError(f"Invalid evidence membership for {doc_id} and {key}")
            if any(not text[spans[i][0]:spans[i][1]].strip() for i in indices):
                raise ValueError(f"Unexpected blank evidence span for {doc_id} and {key}")
            gold.append({"id": doc_id, "group_id": group, "official_id": official_id, "query_id": key,
                         "label": label, "evidence_indices": indices})
            counts[label] += 1
            spans_by_label[label] += len(indices)
            by_query[key][label] += 1
            if indices:
                evidence.append((label, set(indices)))

        # All candidate source ranges are non-overlapping, hence index disjointness
        # is equivalent to disjoint unions of the complete labeled character ranges.
        pairs = Counter(tuple(sorted((left, right))) for (left, ranges), (right, other) in combinations(evidence, 2)
                        if ranges.isdisjoint(other))
        disjoint.update(pairs)
        c_e_documents += pairs["Contradiction", "Entailment"] > 0

    available_signs = [v for v in by_query.values() if v["Entailment"] and v["Contradiction"]]
    paired_documents = sum(v["Entailment"] * v["Contradiction"] for v in available_signs)
    span_inventory = sum(len(row["spans"]) for row in layout)
    if (len(stimuli) != 47 or len({row["group_id"] for row in layout}) != 47 or len(gold) != 799
            or span_inventory != 4318 or blank_spans != 102
            or counts != {"Entailment": 394, "Contradiction": 93, "NotMentioned": 312}
            or spans_by_label != {"Entailment": 883, "Contradiction": 172, "NotMentioned": 0}
            or disjoint != {("Entailment", "Entailment"): 1548, ("Contradiction", "Entailment"): 718,
                            ("Contradiction", "Contradiction"): 60}
            or (c_e_documents, len(available_signs), paired_documents) != (45, 10, 1158)):
        raise ValueError("EDGAR test inventory differs from the audited source")

    tables = {"stimuli.jsonl": stimuli, "layout.jsonl": layout, "queries.jsonl": queries, "gold.jsonl": gold}
    inventory = {
        "documents": len(stimuli), "hypotheses": len(queries), "judgments": len(gold),
        "documents_by_type": Counter(row["document_type"] for row in layout), "candidate_spans": span_inventory,
        "whitespace_only_candidate_spans": blank_spans, "labels": counts, "evidence_spans_by_label": spans_by_label,
        "same_document_disjoint_pairs": {"+".join(labels): count for labels, count in disjoint.items()},
        "documents_with_disjoint_entailment_contradiction_pairs": c_e_documents,
        "hypotheses_with_both_signs": len(available_signs), "same_hypothesis_cross_document_pairs": paired_documents,
    }
    return tables, inventory


def prepare(raw, output):
    if output.exists():
        raise FileExistsError(output)
    documents, labels, duplicates = source_data(raw)
    tables, inventory = prepare_rows(documents, labels)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".contractnli-") as temporary:
        stage = Path(temporary) / "prepared"
        stage.mkdir(mode=0o700)
        for name, table in tables.items():
            write_jsonl(stage / name, table)
        audit = {
            "schema": 1, "paper": PAPER, "archive": ARCHIVE, "license": "CC BY 4.0",
            "source_code_sha256": digest(Path(__file__)), "retained_split": "test",
            "source_json": {f"{split}.json": {"sha256": sha, "total_documents": count, "edgar_documents": edgar}
                            for split, (count, edgar, sha) in SOURCES.items()},
            "normalized_cross_split_duplicates": duplicates, "exact_cross_split_edgar_text_or_url_overlap": 0,
            "counts": inventory, "coordinates": "half-open Unicode code-point indices of the literal source text",
            "stops": "all official candidate ends in order, excluding the final uncataloged newline",
            "evidence_indices": "complete unmodified zero-based lists into the corresponding layout spans",
            "group_id": "SHA-256 of Unicode NFKC then whitespace-collapsed then casefolded source text",
            "gold_endpoint": "merged offline human claim judgment and selected evidence membership",
            "grade_free_files": ["stimuli.jsonl", "layout.jsonl", "queries.jsonl"],
            "output_sha256": {name: digest(stage / name) for name in tables},
        }
        (stage / "source-audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                                 encoding="utf-8")
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True, help="directory containing the official split JSON files")
    parser.add_argument("--output", type=Path, required=True)
    prepare(**vars(parser.parse_args()))
