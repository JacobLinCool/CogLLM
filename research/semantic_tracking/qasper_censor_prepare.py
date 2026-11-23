"""Prepare complete QASPER papers and annotation-level evidence arrival observations."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tarfile
from tempfile import TemporaryDirectory
import unicodedata

from research.semantic_tracking.expmrc_forecast import prepare_stimuli
from research.semantic_tracking.nq_evidence_prepare import load_tokenizer_and_segmenter
from research.semantic_tracking.qasper_prepare import ARCHIVES, RELEASE, TOKENIZER_SHA, digest, load_release, normalized


SALT = "cogllm-qasper-censor-v1:"
SPLIT_LIMITS = {"train": (45, 75_000), "test": (40, 65_000)}
MAX_INPUT_TOKENS = 6144
PRIOR_AUDITS = (
    "research/results/qasper-evidence-first-run/data-audit.json",
    "research/results/qasper-factorial-run/audit.json",
    ".cache/cogllm/qasper/input/audit.json",
    ".cache/cogllm/qasper/replay/audit.json",
    ".cache/cogllm/qasper/dev-holdout/audit.json",
)
PRIOR_STIMULI = (
    ".cache/cogllm/qasper/input/stimuli.jsonl",
    ".cache/cogllm/qasper/replay/stimuli.jsonl",
    ".cache/cogllm/qasper/dev-holdout/stimuli.jsonl",
)
PRIOR_UPLOADS = (
    ".cache/cogllm/qasper/runs/cogllm-semantic-qasper-20261120T201623Z-e9a06fa9/inputs.tar.gz",
    ".cache/cogllm/qasper/factorial-runs/cogllm-qasper-factorial-20261120T204203Z-d4aa99f6/inputs.tar.gz",
)


@dataclass(frozen=True)
class Material:
    paper_id: str
    title_family: str
    article: dict
    questions: tuple
    queries: tuple
    paragraphs: tuple
    abstract: tuple | None
    tokens: int


def title_family(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def selected_papers_from_stimuli(lines):
    return {
        identifier.removeprefix("paper-")
        for line in lines
        if (identifier := json.loads(line)["id"]).startswith("paper-")
    }


def prior_materials(repo, official):
    source_hashes = {}
    ids = set()
    for name in PRIOR_AUDITS:
        path = repo / name
        source_hashes[name] = digest(path)
        audit = json.loads(path.read_text(encoding="utf-8"))
        for identifier, metadata in audit["paper_metadata"].items():
            paper_id = identifier.removeprefix("paper-")
            if (
                identifier == paper_id or metadata["official_split"] not in official
                or paper_id not in official[metadata["official_split"]]
            ):
                raise ValueError(f"Invalid prior paper identity or split: {name}")
            ids.add(paper_id)
    for name in PRIOR_STIMULI:
        path = repo / name
        source_hashes[name] = digest(path)
        with path.open(encoding="utf-8") as source:
            ids.update(selected_papers_from_stimuli(source))
    for name in PRIOR_UPLOADS:
        path = repo / name
        source_hashes[name] = digest(path)
        with tarfile.open(path, "r:gz") as archive:
            with archive.extractfile("data/stimuli.jsonl") as source:
                ids.update(selected_papers_from_stimuli(source))
    prior_by_split = {split: sorted(ids.intersection(papers)) for split, papers in official.items()}
    if set().union(*map(set, prior_by_split.values())) != ids or len(ids) != 110:
        raise ValueError("The frozen prior scopes must resolve exactly to 110 official QASPER papers")
    titles = {title_family(paper["title"]) for papers in official.values() for key, paper in papers.items() if key in ids}
    if "" in titles:
        raise ValueError("A prior title family is missing")
    return ids, titles, {
        "input_sha256": source_hashes, "prior_paper_count": len(ids), "prior_title_count": len(titles),
        "prior_ids_by_official_split": prior_by_split,
        "title_normalization": "NFKC, casefold, then collapse whitespace",
    }


def complete_text(source):
    blocks, paragraphs = [], []
    end = 0

    def append(block):
        nonlocal end
        if blocks:
            end += 2
        blocks.append(block)
        end += len(block)
        return end

    title, abstract = source["title"], source["abstract"]
    if type(title) is not str or type(abstract) is not str:
        raise ValueError("Expected original QASPER title and abstract strings")
    if title.strip():
        append(title.strip())
    abstract = abstract.strip()
    abstract_span = (abstract, append(abstract)) if abstract else None
    for section in source["full_text"]:
        heading = (section["section_name"] or "").strip()
        if heading:
            append(heading)
        for paragraph in section["paragraphs"]:
            paragraph = paragraph.strip()
            if paragraph:
                paragraphs.append((paragraph, append(paragraph)))
    return "\n\n".join(blocks), tuple(paragraphs), abstract_span


def make_material(paper_id, paper, tokenizer, max_characters):
    text, paragraphs, abstract = complete_text(paper)
    if not paragraphs:
        return "no_released_body_paragraphs", None, {}
    if len(text) > max_characters:
        return "article_exceeds_production_characters", None, {}
    body_tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
    if not 0 < body_tokens <= MAX_INPUT_TOKENS:
        return "article_exceeds_backbone_tokens", None, {}
    article_id = f"paper-{paper_id}"
    questions, queries, seen = [], [], set()
    rejected_queries = Counter()
    tokens = body_tokens
    for record in paper["qas"]:
        question = record["question"]
        key = title_family(question)
        if not key:
            rejected_queries["empty_question"] += 1
            continue
        if key in seen:
            rejected_queries["duplicate_normalized_question"] += 1
            continue
        question_tokens = len(tokenizer.encode(question, add_special_tokens=False).ids)
        if not 0 < question_tokens <= MAX_INPUT_TOKENS:
            rejected_queries["question_exceeds_backbone_tokens"] += 1
            continue
        question_id = record["question_id"]
        if type(question_id) is not str or not question_id:
            raise ValueError("Missing official QASPER question ID")
        seen.add(key)
        tokens += question_tokens
        queries.append({"id": f"question-{question_id}", "article_id": article_id, "text": question})
        questions.append(record)
    if not queries:
        return "no_eligible_question_text", None, rejected_queries
    return None, Material(
        paper_id, title_family(paper["title"]), {"id": article_id, "text": text},
        tuple(questions), tuple(queries), paragraphs, abstract, tokens,
    ), rejected_queries


def choose_split(papers, old_ids, old_titles, tokenizer, max_characters, max_papers, budget):
    ordered = sorted(papers, key=lambda paper_id: sha256((SALT + paper_id).encode()).digest())
    selected = []
    new_titles = set()
    skipped_papers, skipped_questions = Counter(), Counter()
    cost, inspected, next_paper = 0, 0, None
    for paper_id in ordered:
        if len(selected) == max_papers:
            break
        inspected += 1
        paper = papers[paper_id]
        family = title_family(paper["title"])
        if not family:
            raise ValueError("A candidate article requires an original title for disjointness checks")
        if paper_id in old_ids or family in old_titles or family in new_titles:
            skipped_papers["previous_id_or_title_family"] += 1
            continue
        reason, material, query_exclusions = make_material(paper_id, paper, tokenizer, max_characters)
        if reason:
            skipped_papers[reason] += 1
            continue
        if cost + material.tokens > budget:
            next_paper = paper_id
            break
        cost += material.tokens
        selected.append(material)
        new_titles.add(family)
        skipped_questions.update(query_exclusions)
    if not selected:
        raise ValueError("No complete QASPER papers meet a registered split budget")
    return selected, {
        "official_papers": len(papers), "inspected_papers": inspected,
        "paper_cap": max_papers, "token_budget": budget, "input_tokens": cost,
        "first_eligible_over_budget": next_paper, "papers_after_stop_not_inspected": len(papers) - inspected,
        "excluded_papers_before_cutoff": dict(skipped_papers),
        "excluded_questions_in_selected_papers": dict(skipped_questions),
        "article_count": len(selected), "query_count": sum(len(row.queries) for row in selected),
        "selected_paper_ids_in_priority_order": [row.paper_id for row in selected],
    }


def annotation_endpoint(record, paragraph_lookup):
    answer = record.get("answer")
    if not answer:
        return None, "missing_annotation", None
    if answer.get("unanswerable") is None:
        return None, "missing_answerability_flag", None
    if type(answer["unanswerable"]) is not bool:
        raise ValueError("Expected explicit source unanswerable flag")
    evidence = answer.get("evidence") or []
    if not isinstance(evidence, list) or not all(isinstance(block, str) for block in evidence):
        raise ValueError("Expected QASPER v0.3 evidence strings")
    if answer["unanswerable"]:
        inconsistency = "unanswerable_with_nonempty_evidence" if any(block.strip() for block in evidence) else None
        return {"end": None}, "right_censored", inconsistency
    if not evidence or any(not block.strip() for block in evidence):
        return None, "missing_text_evidence", None
    if any(block.lstrip().startswith("FLOAT SELECTED") for block in evidence):
        return None, "figure_or_table_evidence", None
    matches = [paragraph_lookup.get(normalized(block), ()) for block in evidence]
    if any(not positions for positions in matches):
        return None, "unmatched_text_paragraph", None
    if any(len(positions) != 1 for positions in matches):
        return None, "nonunique_text_paragraph", None
    located = [positions[0] for positions in matches]
    support = "event_including_unique_abstract" if any(kind == "abstract" for _, kind in located) else None
    return {"end": max(end for end, _ in located)}, "event", support


def outcomes(materials):
    rows, disposition, details = [], Counter(), Counter()
    for material in materials:
        lookup = defaultdict(list)
        for paragraph, end in material.paragraphs:
            lookup[normalized(paragraph)].append((end, "paragraph"))
        if material.abstract is not None:
            abstract, end = material.abstract
            lookup[normalized(abstract)].append((end, "abstract"))
        for question, query in zip(material.questions, material.queries, strict=True):
            references, excluded = [], Counter()
            for annotation in question["answers"]:
                reference, result, detail = annotation_endpoint(annotation, lookup)
                disposition[result] += 1
                if detail is not None:
                    details[detail] += 1
                if reference is None:
                    excluded[result] += 1
                else:
                    references.append(reference)
            rows.append({
                "id": query["id"], "article_id": query["article_id"], "refs": references,
                "annotation_count": len(question["answers"]), "excluded": dict(excluded),
            })
    return rows, {
        "annotation_count": sum(row["annotation_count"] for row in rows),
        "disposition": dict(disposition),
        "events": disposition["event"],
        "right_censored": disposition["right_censored"],
        "unanswerable_with_nonempty_evidence": details["unanswerable_with_nonempty_evidence"],
        "events_including_unique_abstract": details["event_including_unique_abstract"],
        "questions_without_usable_annotation": sum(not row["refs"] for row in rows),
        "questions_with_event_and_censor": sum(
            any(reference["end"] is None for reference in row["refs"])
            and any(reference["end"] is not None for reference in row["refs"])
            for row in rows
        ),
    }


def jsonl(path, rows, *, private=False):
    mode = 0o600 if private else 0o644
    descriptor = os.open(path, os.O_EXCL | os.O_WRONLY | os.O_CREAT, mode)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def prepare(raw, tokenizer_path, checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
    repo = Path(__file__).resolve().parents[2]
    official = load_release(raw)
    old_ids, old_titles, prior = prior_materials(repo, official)
    tokenizer, segmenter, max_characters = load_tokenizer_and_segmenter(tokenizer_path, checkpoint)
    selected, splits = {}, {}
    for split, (max_papers, budget) in SPLIT_LIMITS.items():
        selected[split], splits[split] = choose_split(
            official[split], old_ids, old_titles, tokenizer, max_characters, max_papers, budget,
        )
        old_ids.update(row.paper_id for row in selected[split])
        old_titles.update(row.title_family for row in selected[split])
    train_ids = {row.paper_id for row in selected["train"]}
    train_titles = {row.title_family for row in selected["train"]}
    if train_ids.intersection(row.paper_id for row in selected["test"]) or train_titles.intersection(
        row.title_family for row in selected["test"]
    ):
        raise ValueError("A held-out paper shares an ID or title family with development")
    articles = [row.article for papers in selected.values() for row in papers]
    queries = [query for papers in selected.values() for row in papers for query in row.queries]
    if len({row["id"] for row in (*articles, *queries)}) != len(articles) + len(queries):
        raise ValueError("A paper or question identifier is repeated across the locked cohort")
    _, inventory = prepare_stimuli(articles, queries, tokenizer, segmenter)
    if (
        inventory["input_tokens"] != sum(row["input_tokens"] for row in splits.values())
        or "torch" in sys.modules or "transformers" in sys.modules
    ):
        raise ValueError("The fixed tokenizer preflight disagrees with the grade-free selection")

    # Read selected answer fields only after complete text, questions, cutoffs and stops are fixed.
    development, dev_disposition = outcomes(selected["train"])
    holdout, test_disposition = outcomes(selected["test"])
    for rows, status in ((development, dev_disposition), (holdout, test_disposition)):
        if (
            status["annotation_count"] != sum(status["disposition"].values())
            or any(len(row["refs"]) + sum(row["excluded"].values()) != row["annotation_count"] for row in rows)
        ):
            raise ValueError("The answer disposition does not reconcile with the original annotation denominator")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".qasper-censor-") as directory:
        staging = Path(directory) / "assay"
        staging.mkdir()
        names = ("articles.jsonl", "queries.jsonl", "article-metadata.jsonl", "development.jsonl", "holdout.jsonl")
        metadata = [
            {
                "id": row.article["id"], "source_paper_id": row.paper_id,
                "official_split": split, "title": official[split][row.paper_id]["title"],
                "title_family": row.title_family, "arxiv": f"https://arxiv.org/abs/{row.paper_id}",
                "body_paragraphs": len(row.paragraphs), "text_characters": len(row.article["text"]),
                "input_tokens_with_questions": row.tokens,
            }
            for split, papers in selected.items() for row in papers
        ]
        for name, rows in zip(names, (articles, queries, metadata, development, holdout), strict=True):
            jsonl(staging / name, rows, private=name in {"development.jsonl", "holdout.jsonl"})
        write_json(staging / "preflight.json", {
            "articles_sha256": digest(staging / "articles.jsonl"),
            "queries_sha256": digest(staging / "queries.jsonl"),
            **inventory,
        })
        audit = {
            "schema_version": 1,
            "source": "https://huggingface.co/datasets/allenai/qasper",
            "paper": "https://aclanthology.org/2021.naacl-main.365/",
            "source_license": "CC BY 4.0",
            "archives": {name: {"url": RELEASE + name, "sha256": sha} for name, sha in ARCHIVES.items()},
            "prepare_sha256": digest(Path(__file__)),
            "tokenizer_sha256": TOKENIZER_SHA,
            "segmentation_code_sha256": digest(Path(__file__).with_name("expmrc_forecast.py")),
            "prior_qasper": prior,
            "article_count": len(articles),
            "query_count": len(queries),
            "selection": {
                "hash_order": f"Ascending bytes of SHA-256({SALT!r} + official arXiv ID)",
                "budget_rule": "Stop at the first eligible complete paper and its questions that would "
                               "exceed the split budget, or when the paper cap is reached.",
                "max_input_tokens": MAX_INPUT_TOKENS,
                "max_article_characters": max_characters,
                "questions": "Original question text in source order, first occurrence per normalized "
                             "distinct question. Selection never uses answer fields.",
                "development_split": "train",
                "holdout_split": "test",
                "development_holdout_paper_id_matches": 0,
                "development_holdout_normalized_title_matches": 0,
                "title_exclusion": "Both splits exclude prior IDs and titles. Held-out selection also "
                                   "excludes each development title family.",
            },
            "source_view": "Original title, abstract, headings and every nonempty released full_text "
                           "paragraph in order, with two newlines between blocks and outer whitespace trimmed.",
            "observations": {
                "event": "A human annotation with all its selected text uniquely aligned to a released "
                         "body paragraph or the abstract. The event occurs at the latest matched Unicode end offset.",
                "right_censored": "An annotation explicitly marked unanswerable by its author; end=null, "
                                  "evaluated with survival at the final unchanged body stop.",
                "excluded": "Missing annotations or evidence, figure/table support, and incomplete or "
                            "ambiguous alignment. Excluded annotations remain in the published denominator.",
                "development": dev_disposition,
                "holdout": test_disposition,
            },
            "security": {
                "remote_inputs": ["articles.jsonl", "queries.jsonl"],
                "local_references": ["development.jsonl", "holdout.jsonl"],
            },
            "splits": splits,
            "file_sha256": {name: digest(staging / name) for name in (*names, "preflight.json")},
        }
        write_json(staging / "assay.json", audit)
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "tokenizer", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.raw, args.tokenizer, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
