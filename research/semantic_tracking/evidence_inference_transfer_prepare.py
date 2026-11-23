"""Select official Evidence Inference test articles and export separate evidence coordinates."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata
from xml.etree import ElementTree

from research.semantic_tracking.contractnli_forecast import MAX_TOKENS, REVISION, TOKENIZER_SHA256, digest
from research.stage2.records import write_jsonl


SOURCE = "https://github.com/jayded/evidence-inference"
SOURCE_REVISION = "a661e8c14f973398380c8865cf2f27a535aaaf6d"
SOURCES = {
    "README.md": "b9d7599a666cada77cec4e52b459ec70425fab6f61604ecbe77d1bbf7a8297b1",
    "splits/test_article_ids.txt": "4fe6c633ba8f077a68a8feea48d4408c1749175e1c46f2b77d96727d291ef9d8",
    "prompts_merged.csv": "0d70d9e1e78d113fdbbd4919310eab7768d6a328368f7d8f9d5a3b48f99c25d3",
    "annotations_merged.csv": "a5dd1a3583e5105a0f77d3e1c695008879859425c9a93dfbc07054ef462e0255",
}
SALT = "cogllm-evidence-inference-transfer-v1:"
ARTICLE_COUNT = 24
TOKEN_BUDGET = 145_000
QUERY_TEMPLATE = "What was reported for {Outcome} when comparing {Intervention} with {Comparator}?"
CC_URL = re.compile(r"https?://(?:www\.)?creativecommons\.org(/[a-z0-9./-]+)", re.I)
OPEN_CC_PATH = re.compile(r"/(?:licenses/by(?:/(?:[1-4]\.0|2\.5))?|publicdomain/(?:zero|mark)/1\.0)/?")
EXCEPTION = re.compile(
    r"all rights reserved|(?:used|reprinted|reproduced|reproduced or adapted) (?:here )?with permission|"
    r"(?:excluded|excepted) from (?:the|this) (?:creative commons|licen[cs]e)|"
    r"(?:is|are) not (?:covered|included) (?:under|in) (?:the|this) (?:creative commons|licen[cs]e)|"
    r"copyright (?:of|held by) (?:a |the )?third part",
    re.I,
)
COMPONENT_CREDIT = re.compile(r"(?:reprinted|republished|reproduced|adapted) from\b", re.I)
RESTRICTED_TERMS = re.compile(r"\b(?:non[ -]?commercial|no[ -]+derivatives|share[ -]?alike)\b", re.I)


def verified_source(directory, name):
    path = directory / name
    if digest(path) != SOURCES[name]:
        raise ValueError(f"Author snapshot differs from the pinned bytes: {name}")
    return path


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def author_exclusions(readme):
    categories = {}
    for name in ("Incorrect", "Questionable", "Somewhat malformed"):
        match = re.search(rf"(?im)^### {name}:[ \t]*\n\s*([0-9][0-9, \t]*)", readme)
        if match is None:
            raise ValueError(f"Author's list of flagged prompts is missing: {name}")
        categories[name] = {int(value) for value in re.findall(r"\d+", match[1])}
    return categories


def prompt_inventory(path, article_ids, excluded):
    inventory = defaultdict(list)
    ignored = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != {"PromptID", "PMCID", "Outcome", "Intervention", "Comparator"}:
            raise ValueError("Unexpected author prompt CSV fields")
        seen = set()
        for row in reader:
            if row["PMCID"] not in article_ids:
                continue
            prompt_id = int(row["PromptID"])
            if prompt_id in seen:
                raise ValueError(f"Repeated official prompt ID {prompt_id}")
            seen.add(prompt_id)
            fields = {name: row[name].strip() for name in ("Outcome", "Intervention", "Comparator")}
            if prompt_id in excluded or not all(fields.values()):
                ignored[row["PMCID"]].append({
                    "prompt_id": prompt_id,
                    "reason": "author_flag" if prompt_id in excluded else "missing_query_field",
                })
            else:
                inventory[row["PMCID"]].append((prompt_id, QUERY_TEMPLATE.format(**fields)))
    distinct = {}
    for pmcid, queries in inventory.items():
        keys = {}
        rows = []
        for prompt_id, query in sorted(queries):
            key = normalized(query)
            if key in keys:
                ignored[pmcid].append({"prompt_id": prompt_id, "reason": "duplicate_normalized_query",
                                       "retained_prompt_id": keys[key]})
                continue
            keys[key] = prompt_id
            rows.append({"id": f"evidence-inference-{prompt_id}", "article_id": f"PMC{pmcid}", "text": query})
        distinct[pmcid] = rows
    return distinct, ignored


def jats_license(payload, pmcid):
    xml = ElementTree.fromstring(payload)
    meta = xml.find("front/article-meta")
    if meta is None:
        raise ValueError(f"Article metadata unavailable for PMC{pmcid}")
    ids = {entry.text.strip().removeprefix("PMC") for entry in meta.findall("article-id")
           if entry.get("pub-id-type") in {"pmc", "pmcid"} and entry.text}
    if ids and ids != {pmcid}:
        raise ValueError(f"Published JATS does not identify PMC{pmcid}")

    permissions = meta.findall("permissions")
    if not permissions:
        return None, "no_article_rights_statement"
    primary_nodes = {element for owner in permissions for element in owner.iter()}
    licenses = [element for owner in permissions for element in owner.findall("license")]
    url_inputs = [value for license_node in licenses for child in license_node.iter()
                  for value in child.attrib.values()]
    url_inputs.extend(" ".join(license_node.itertext()) for license_node in licenses)
    urls = {path.lower().rstrip("/") + "/" for value in url_inputs for path in CC_URL.findall(value)}
    license_text = " ".join(" ".join(" ".join(owner.itertext()).split()) for owner in permissions)
    if EXCEPTION.search(license_text) or RESTRICTED_TERMS.search(license_text):
        return None, "restrictive_article_rights_statement"
    if urls and all(OPEN_CC_PATH.fullmatch(path) for path in urls):
        rights = ["https://creativecommons.org" + path for path in sorted(urls)]
    elif not urls and re.search(r"\b(?:this|the) (?:article|work) is in the public domain\b", license_text, re.I):
        rights = ["public_domain_declared_in_article_permissions"]
    else:
        return None, "article_license_outside_allowed_set"

    # A separate component rights declaration or concrete permission credit
    # disqualifies the article rather than changing its published source text.
    for element in xml.iter():
        if element in primary_nodes:
            continue
        local_name = element.tag.rsplit("}", 1)[-1]
        if local_name in {"permissions", "license", "copyright-statement"}:
            return None, "separately_licensed_component"
        if local_name in {"attrib", "caption", "credit", "ack", "p"}:
            content = " ".join(" ".join(element.itertext()).split())
            if EXCEPTION.search(content):
                return None, "component_rights_credit"
            if local_name in {"attrib", "caption", "credit"} and (
                "©" in content or COMPONENT_CREDIT.search(content)
            ):
                return None, "component_rights_credit"
    return rights, None


def select_material(directory, tokenizer):
    split = verified_source(directory, "splits/test_article_ids.txt").read_text(encoding="utf-8").split()
    if len(split) != 449 or len(set(split)) != 449 or any(not re.fullmatch("[0-9]+", key) for key in split):
        raise ValueError("Unexpected official article split")
    named_flags = author_exclusions(verified_source(directory, "README.md").read_text(encoding="utf-8"))
    excluded = named_flags["Incorrect"] | named_flags["Somewhat malformed"]
    prompts, ignored = prompt_inventory(verified_source(directory, "prompts_merged.csv"), set(split), excluded)
    records = sorted((sha256((SALT + key).encode("ascii")).hexdigest(), key) for key in split)
    articles, queries, ledger = [], [], []
    tokens_total = 0
    for rank, (priority, pmcid) in enumerate(records, 1):
        article_id = f"PMC{pmcid}"
        article_queries = prompts.get(pmcid, [])
        receipt = {"article_id": article_id, "rank": rank, "priority_sha256": priority,
                   "distinct_queries": len(article_queries), "excluded_queries": ignored.get(pmcid, [])}
        ledger.append(receipt)
        if len(article_queries) < 2:
            receipt["status"] = "fewer_than_two_distinct_queries"
            continue

        # Open at most one pair of raw author files at a time.
        source_xml = directory / "xml_files" / f"{article_id}.nxml"
        xml_payload = source_xml.read_bytes()
        rights, rejection = jats_license(xml_payload, pmcid)
        receipt.update(jats_file=str(source_xml.relative_to(directory)), jats_sha256=sha256(xml_payload).hexdigest())
        if rejection is not None:
            receipt["status"] = rejection
            continue
        body_file = directory / "txt_files" / f"{article_id}.txt"
        raw_text = body_file.read_bytes()
        text = raw_text.decode("utf-8")
        body_tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
        query_tokens = [len(tokenizer.encode(row["text"], add_special_tokens=False).ids) for row in article_queries]
        receipt.update(article_license=rights, text_file=str(body_file.relative_to(directory)),
                       text_sha256=sha256(raw_text).hexdigest(), body_tokens=body_tokens, query_tokens=sum(query_tokens))
        if not text.strip() or not 0 < body_tokens <= MAX_TOKENS or any(
            not 0 < count <= MAX_TOKENS for count in query_tokens
        ):
            receipt["status"] = "token_length_outside_model_bounds"
            continue
        item_cost = body_tokens + sum(query_tokens)
        if tokens_total + item_cost > TOKEN_BUDGET:
            receipt["status"] = "does_not_fit_remaining_token_budget"
            continue
        receipt["status"] = "retained"
        tokens_total += item_cost
        articles.append({"id": article_id, "text": text})
        queries.extend(article_queries)
        if len(articles) == ARTICLE_COUNT:
            break
    if len(articles) != ARTICLE_COUNT:
        raise ValueError(f"Expected {ARTICLE_COUNT} eligible articles. Found {len(articles)} at {tokens_total} tokens")
    return articles, queries, ledger, tokens_total


def evidence_outcomes(path, articles, queries):
    text_by_id = {row["id"]: row["text"] for row in articles}
    query_lookup = {(row["article_id"], row["id"].removeprefix("evidence-inference-")): row["id"] for row in queries}
    references = defaultdict(list)
    counts = Counter()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"PMCID", "PromptID", "Valid Label", "Valid Reasoning", "Evidence Start", "Evidence End", "Annotations"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("Unexpected author annotation CSV fields")
        for row in reader:
            article_id = "PMC" + row["PMCID"]
            key = query_lookup.get((article_id, row["PromptID"]))
            if key is None:
                continue
            if row["Valid Label"] != "True" or row["Valid Reasoning"] != "True":
                counts["rejected_by_verifier"] += 1
                continue
            try:
                start, stop = int(row["Evidence Start"]), int(row["Evidence End"])
            except ValueError:
                counts["unusable_offsets"] += 1
                continue
            text = text_by_id[article_id]
            if not 0 <= start < stop <= len(text):
                counts["unusable_offsets"] += 1
                continue
            if not row["Annotations"].strip() or text[start:stop] != row["Annotations"]:
                counts["quote_does_not_match"] += 1
                continue
            references[key].append([start, stop])
            counts["accepted_references"] += 1
    outputs = [{"id": row["id"], "article_id": row["article_id"], "references": references[row["id"]]}
               for row in queries]
    counts["queries_with_verified_references"] = sum(bool(row["references"]) for row in outputs)
    counts["queries_without_verified_references"] = len(outputs) - counts["queries_with_verified_references"]
    return outputs, counts


def prepare(raw, tokenizer_path, output):
    from tokenizers import Tokenizer

    if output.exists():
        raise FileExistsError(output)
    if digest(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("Expected the pinned frozen Qwen3.5-2B Rust tokenizer")
    directory = raw / "annotations"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    articles, queries, ledger, tokens_total = select_material(directory, tokenizer)
    outcomes, counts = evidence_outcomes(verified_source(directory, "annotations_merged.csv"), articles, queries)
    tables = {"articles.jsonl": articles, "queries.jsonl": queries, "outcomes.jsonl": outcomes}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".evidence-inference-transfer-") as tmp:
        stage = Path(tmp) / "prepared"
        stage.mkdir(mode=0o700)
        for name, table in tables.items():
            write_jsonl(stage / name, table)
        audit = {
            "schema": 1,
            "source_repository": SOURCE,
            "source_revision": SOURCE_REVISION,
            "source_code_sha256": digest(Path(__file__)),
            "source_sha256": SOURCES,
            "retained_split": "test",
            "query_template": QUERY_TEMPLATE,
            "prompt_deduplication": "Lowest numeric PromptID per NFKC, collapsed-whitespace, casefolded query in each article",
            "author_excluded_query_categories": ["Incorrect", "Somewhat malformed"],
            "sampling_salt": SALT,
            "packing": "Ascending salted SHA-256. Skip ineligible articles and those that exceed remaining budget. Stop at 24.",
            "per_input_token_ceiling": MAX_TOKENS,
            "total_token_ceiling": TOKEN_BUDGET,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "model_revision": REVISION,
            "input_tokens": tokens_total,
            "article_count": len(articles),
            "query_count": len(queries),
            "grade_free_files": ["articles.jsonl", "queries.jsonl"],
            "source_selection_ledger": ledger,
            "evidence_coordinates": "Half-open Unicode code-point offsets into the exact author TXT",
            "evidence_endpoint": "Offline human evidence selections with both verifier flags true and exact quote matches",
            "empty_references": "Unknown. No verifier-accepted exact reference is observed for this query.",
            "outcome_counts": counts,
            "output_sha256": {name: digest(stage / name) for name in tables},
        }
        (stage / "source-audit.json").write_text(json.dumps(audit, sort_keys=True, indent=2, allow_nan=False) + "\n",
                                                 encoding="utf-8")
        stage.rename(output)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--raw", type=Path, required=True, help="author repository root with the annotations directory")
    cli.add_argument("--tokenizer", type=Path, required=True, dest="tokenizer_path")
    cli.add_argument("--output", type=Path, required=True)
    prepare(**vars(cli.parse_args()))
