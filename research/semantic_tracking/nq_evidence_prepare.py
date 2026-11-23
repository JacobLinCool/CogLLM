"""Prepare Natural Questions evidence inputs with all annotation outcomes sealed locally."""

import argparse
from collections import Counter
from html import unescape
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unicodedata
from urllib.parse import parse_qs, urlsplit

from research.semantic_tracking.expmrc_forecast import prepare_stimuli
from research.semantic_tracking.nq_evidence_fetch import EXPECTED_ROWS, GROUP_COUNT, SAMPLE_SALT, selected_groups
from research.semantic_tracking.nq_evidence_index import SOURCE_REVISION, digest, normalized_title
from research.semantic_tracking.nq_source_positions import TOKEN_COLUMNS, build_text, candidate_spans


DATASET = "https://huggingface.co/datasets/google-research-datasets/natural_questions"
LICENSE = "CC BY-SA 3.0"
CANDIDATE_INDEX_SHA256 = "6a4a3921b7fa142eaf34db148e0f93a44fd0da6313a2e9c2b63b8853cad247b5"
CHECKPOINT = "74c26ff02569b2b3bc28ed291bf15fd2f85db5b4c93a3ad7bfc34f6a5d333ea0"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
MAX_ARTICLE_TOKENS = 6144
MAX_QUESTION_TOKENS = 6144
TOTAL_TOKENS = 120_000
RAW_FIELDS = {"row_idx", "id", "document", "question", "long_answer_candidates"}
DOCUMENT_FIELDS = {"title", "url", "html", "tokens"}
QUESTION_FIELDS = {"text", "tokens"}


def json_line(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verified_intake(index, directory):
    manifest = read_json(directory / "manifest.json")
    order, grouped_rows = selected_groups(index)
    rows = [row for group in grouped_rows for row in group]
    if (
        manifest["dataset"] != "google-research-datasets/natural_questions"
        or manifest["config"] != "default" or manifest["split"] != "validation"
        or manifest["source_revision"] != SOURCE_REVISION or manifest["index_sha256"] != CANDIDATE_INDEX_SHA256
        or manifest["groups"] != GROUP_COUNT or manifest["records"] != EXPECTED_ROWS
        or manifest["selection"] != {"salt": SAMPLE_SALT, "revision_order": order, "rows": rows}
    ):
        raise ValueError("The intake receipt disagrees with the locked grade-free revision sample")
    source = directory / "grade-free.jsonl"
    source_sha256 = digest(source)
    if source_sha256 != manifest["file_sha256"]["grade-free.jsonl"]:
        raise ValueError("The grade-free data do not match their acquisition receipt")
    groups = dict(zip(order, grouped_rows, strict=True))
    offsets = indexed_offsets(source, {row["row_index"]: row for row in rows})
    return source, source_sha256, groups, order, offsets


def check_source_identity(row, index):
    if (
        row.keys() != RAW_FIELDS or row["row_idx"] != index["row_index"] or row["id"] != index["id"]
        or row["document"].keys() != DOCUMENT_FIELDS or row["document"]["url"] != index["document_url"]
        or row["question"].keys() != QUESTION_FIELDS
    ):
        raise ValueError("An input row differs from the locked grade-free source identity or schema")
    document = row["document"]
    titles = parse_qs(urlsplit(unescape(document["url"])).query).get("title", [])
    title = normalized_title(document["title"])
    if len(titles) != 1 or not title or title != normalized_title(titles[0]):
        raise ValueError("An archived page title disagrees with the indexed URL title")
    return title


def indexed_offsets(path, expected):
    offsets = {}
    with path.open("rb") as source:
        offset = source.tell()
        for line in source:
            row = json.loads(line)
            position = row["row_idx"]
            if type(position) is not int or position not in expected or position in offsets:
                raise ValueError("Duplicate, unexpected or non-integer intake row")
            check_source_identity(row, expected[position])
            offsets[position] = offset
            offset = source.tell()
    if offsets.keys() != expected.keys():
        raise ValueError("The grade-free intake must include every row of each selected revision")
    return offsets


def encode_inputs(stream, rows, offsets, tokenizer, max_characters):
    views, records, family_titles = [], [], set()
    for index in rows:
        stream.seek(offsets[index["row_index"]])
        row = json.loads(stream.readline())
        family_titles.add(check_source_identity(row, index))
        source_tokens = row["document"]["tokens"]
        if source_tokens.keys() != TOKEN_COLUMNS:
            raise ValueError("Unexpected Natural Questions token schema")
        for token, is_html in zip(source_tokens["token"], source_tokens["is_html"], strict=True):
            if type(token) is not str or type(is_html) is not bool:
                raise ValueError("Expected typed Natural Questions source tokens")
            if not is_html and (not token or any(character.isspace() for character in token)):
                return "uncertified_visible_token_geometry", None
        records.append(row)
        views.append(build_text(row["document"]))
    text = views[0].text
    if len(family_titles) != 1 or any(view.text != text for view in views[1:]):
        return "physical_visible_body_mismatch", None
    questions = [row["question"]["text"] for row in records]
    if any(type(question) is not str for question in questions):
        raise ValueError("Natural Questions question text must be a string")
    normalized = [" ".join(unicodedata.normalize("NFKC", question).casefold().split()) for question in questions]
    if any(not question for question in normalized) or len(set(normalized)) != len(normalized):
        return "questions_not_distinct", None
    if len(text) > max_characters:
        return "article_exceeds_production_characters", None
    body_tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
    if not 0 < body_tokens <= MAX_ARTICLE_TOKENS:
        return "article_exceeds_backbone_tokens", None
    query_tokens = [len(tokenizer.encode(text, add_special_tokens=False).ids) for text in questions]
    if any(not 0 < count <= MAX_QUESTION_TOKENS for count in query_tokens):
        return "query_exceeds_backbone_tokens", None
    return None, (records, views, text, questions, family_titles.pop(), body_tokens + sum(query_tokens))


def map_record(row, view, article_id, oldid, title_family):
    candidate_map = []
    coordinates = row["long_answer_candidates"]
    supports = candidate_spans(row, view)
    for position, span in enumerate(supports):
        candidate = {field: values[position] for field, values in coordinates.items()}
        start = candidate["start_token"]
        candidate["source_tag"] = row["document"]["tokens"]["token"][start]
        candidate["character_span"] = span
        candidate_map.append(candidate)
    return {
        "id": row["id"], "article_id": article_id, "row_index": row["row_idx"],
        "source_oldid": oldid, "document_title": row["document"]["title"], "title_family": title_family,
        "source_positions": view.source_positions, "character_spans": view.character_spans,
        "byte_spans": view.byte_spans, "candidate_spans": candidate_map,
    }


def load_tokenizer_and_segmenter(tokenizer_path, checkpoint_path):
    from tokenizers import Tokenizer

    from research.gradio_app.checkpoint import load_checkpoint
    from research.gradio_app.plaba import MAX_CHARACTERS

    if digest(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("The Rust tokenizer input differs from the fixed Qwen revision")
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.identifier != CHECKPOINT:
        raise ValueError("Expected the exact checkpoint's bundled English Punkt parameters")
    return Tokenizer.from_file(str(tokenizer_path)), checkpoint.segmenter(), MAX_CHARACTERS


def select_inputs(grade_free, groups, revision_order, offsets, tokenizer, max_characters):
    articles, queries, maps = {}, {}, {}
    failures = Counter()
    retained, inspected, used_tokens = [], 0, 0
    unique_body = set()
    cutoff = None
    with grade_free.open("rb") as source:
        for oldid in revision_order:
            inspected += 1
            reason, bundle = encode_inputs(source, groups[oldid], offsets, tokenizer, max_characters)
            if reason is not None:
                failures[reason] += 1
                continue
            records, views, body, questions, title_family, tokens = bundle
            if body in unique_body:
                raise ValueError("The same exact body occurs under two distinct selected source revisions")
            if used_tokens + tokens > TOTAL_TOKENS:
                cutoff = oldid
                break
            unique_body.add(body)
            used_tokens += tokens
            retained.append(oldid)
            article_id = f"nq-wiki-{oldid}"
            articles[min(row["row_idx"] for row in records)] = {"id": article_id, "text": body}
            for row, view, question in zip(records, views, questions, strict=True):
                row_index = row["row_idx"]
                queries[row_index] = {"id": row["id"], "article_id": article_id, "text": question}
                maps[row_index] = json_line(map_record(row, view, article_id, oldid, title_family))
    if not articles or len(queries) < len(articles) * 2:
        raise ValueError("No complete repeated-revision Natural Questions groups")
    inventory = {
        "revision_order": "Fixed grade-free intake hash priority. Process complete eligible groups in order.",
        "export_order": "Original validation source row order for questions and articles' first source rows.",
        "budget": "Stop before the first eligible whole group that would exceed the total token budget.",
        "revision_groups_inspected": inspected, "first_over_budget": cutoff,
        "uninspected_revision_groups": len(revision_order) - inspected,
        "rejected_revision_groups": dict(failures), "retained_revisions": retained,
    }
    return (
        [articles[index] for index in sorted(articles)],
        [queries[index] for index in sorted(queries)],
        [maps[index] for index in sorted(maps)],
        inventory, used_tokens,
    )


def prepare(index, intake, tokenizer_path, checkpoint_path, output):
    if output.exists():
        raise FileExistsError(output)
    grade_free, grade_free_sha256, groups, order, offsets = verified_intake(index, intake)
    tokenizer, segmenter, max_characters = load_tokenizer_and_segmenter(tokenizer_path, checkpoint_path)
    articles, queries, maps, selection, used_tokens = select_inputs(
        grade_free, groups, order, offsets, tokenizer, max_characters,
    )
    _, inventory = prepare_stimuli(articles, queries, tokenizer, segmenter)
    if inventory["input_tokens"] != used_tokens or used_tokens > TOTAL_TOKENS:
        raise ValueError("The exact exporter preflight does not match the selection token budget")
    if "torch" in sys.modules or "transformers" in sys.modules:
        raise RuntimeError("The preparation and preflight must use only the Rust tokenizer and bundled segmenter")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".nq-evidence-prepare-") as directory:
        staging = Path(directory) / "assay"
        staging.mkdir()
        for name, records in (
            ("articles.jsonl", map(json_line, articles)),
            ("queries.jsonl", map(json_line, queries)),
            ("span-maps.jsonl", maps),
        ):
            with (staging / name).open("x", encoding="utf-8", newline="\n") as stream:
                stream.writelines(records)
        preflight = {
            "articles_sha256": digest(staging / "articles.jsonl"),
            "queries_sha256": digest(staging / "queries.jsonl"),
            **inventory,
        }
        (staging / "preflight.json").write_text(json_line(preflight), encoding="utf-8")
        assay = {
            "schema_version": 1,
            "dataset": DATASET,
            "source_revision": SOURCE_REVISION,
            "source_license": LICENSE,
            "license_reference": "https://ai.google.com/research/NaturalQuestions/download",
            "candidate_index_sha256": CANDIDATE_INDEX_SHA256,
            "raw_manifest_sha256": digest(intake / "manifest.json"),
            "grade_free_sha256": grade_free_sha256,
            "prepare_sha256": digest(Path(__file__)),
            "positions_code_sha256": digest(Path(__file__).with_name("nq_source_positions.py")),
            "segmentation_code_sha256": digest(Path(__file__).with_name("expmrc_forecast.py")),
            "checkpoint": CHECKPOINT,
            "model_revision": MODEL_REVISION,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "source_view": "Exact upstream visible tokens joined with single ASCII spaces, excluding HTML tokens.",
            "coordinates": "Zero-based, half-open spans in source token indices, source UTF-8 bytes and visible "
                           "Unicode code points. Original long-answer candidates carry no selections.",
            "question_rule": "Emit original question text and ID. Distinctness uses NFKC, casefold and whitespace.",
            "limits": {
                "article_characters": max_characters, "article_tokens": MAX_ARTICLE_TOKENS,
                "question_tokens": MAX_QUESTION_TOKENS, "total_tokens": TOTAL_TOKENS,
            },
            "article_count": len(articles),
            "query_count": len(queries),
            "selection": selection,
            "security": {
                "remote_inputs": ["articles.jsonl", "queries.jsonl"],
                "local_geometry_only": ["span-maps.jsonl"],
                "reference_escrow_opened": False,
            },
            "file_sha256": {
                name: digest(staging / name)
                for name in ("articles.jsonl", "queries.jsonl", "span-maps.jsonl", "preflight.json")
            },
        }
        (staging / "assay.json").write_text(
            json.dumps(assay, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("index", "intake", "tokenizer", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.index, args.intake, args.tokenizer, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
