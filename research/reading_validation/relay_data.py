"""Prepare label-free ReLay article inputs and retain deidentified judgments locally."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from research.reading_validation.local_forecast import WORD, digest
from research.reading_validation.local_jargon_prepare import sentence_bounds


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache/cogllm/reading-validation"
DATA = CACHE / "relay"
OUT = ROOT / "research/results/reading-validation-relay"
SOURCE_HASH = "51c7022e87809dde205c2a62e39a224fd89c3a9246a0b62e2431b292d7ef0501"
REV = "15852e8c16360a2fea060d615a32b45270f8a8fc"
WEIGHTS = ROOT / "research/results/reading-validation-local/help-choice/weights.json"
FROZEN = "44f93729a95b8c4cc8552c8a9237b27b8bbe27a1941a290458b8b7469a3e83fc"


def key(article, term):
    return hashlib.sha256((article + "\x1f" + term.casefold()).encode()).hexdigest()


def locate(text, term, segments):
    matches = list(re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I))
    if not matches:
        return None, "absent"
    start, end = matches[0].span()
    containing = [s for s in segments if s[0] <= start and end <= s[1]]
    if len(containing) != 1:
        return None, "crosses_sentence"
    words = list(WORD.finditer(text, *containing[0]))
    covered = [w for w in words if start <= w.start() and w.end() <= end]
    if not covered or covered[0].start() != start or covered[-1].end() != end:
        return None, "word_boundary"
    if len(covered) > 8:
        return None, "over_eight_words"
    return (start, end, len(matches)), "included"


def write_rows(path, rows):
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare():
    from tokenizers import Tokenizer

    source = CACHE / "relay-participant.json"
    if digest(source) != SOURCE_HASH or digest(WEIGHTS) != FROZEN:
        raise ValueError("Unexpected study source or source-trained estimator")
    tokpath = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots" / REV / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokpath))
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    articles, summaries, visits, inventory = {}, {}, [], {}
    counts = Counter()
    for reader, participant in enumerate(json.loads(source.read_bytes())):
        batches = participant["phases"]["static"]["batches"].values()
        seen = set()
        entries = [(k, v) for b in batches for k, v in b.get("abstracts", {}).items() if v["completed"]]
        for aid, entry in entries:
            if aid in seen:
                raise ValueError("Repeated article within reader")
            seen.add(aid)
            text, summary = entry["abstract"], entry["human_written_pls"]
            if articles.setdefault(aid, text) != text or summaries.setdefault(aid, summary) != summary:
                raise ValueError("Article identity has conflicting text")
            chosen = set()
            for record in entry["term_familarity"]:
                term = record["term"].strip()
                code = key(aid, term)
                if code in chosen:
                    raise ValueError("Repeated term within visit")
                chosen.add(code)
                inventory.setdefault(aid, {})[code] = term
                familiarity = record["familiarity_score"]
                if familiarity not in (1, 2, 3, 4, 5):
                    raise ValueError("Unknown elicited familiarity")
                answers = set(record["extra_information"])
                if answers - {"None", "Definition", "Example", "Background"}:
                    raise ValueError("Unknown support category")
                if not answers or ("None" in answers and len(answers) > 1):
                    counts["excluded_missing_or_conflicting_choices"] += 1
                    continue
                visits.append(dict(reader=reader, article=aid, term=code, regular=len(entries)==3,
                                   familiarity=familiarity-1, definition="Definition" in answers,
                                   example="Example" in answers))
        counts[f"readers_with_{len(entries)}_visits"] += 1
        counts["completed_visits"] += len(entries)
    geometry, packet, summary_packet = {}, [], []
    for aid, text in sorted(articles.items()):
        n = len(tokenizer.encode(text, add_special_tokens=False).ids)
        group = "native" if n <= 1024 else "extended" if n <= 2048 else "unsupported"
        counts["articles_" + group] += 1
        spans, segments = [], sentence_bounds(text)
        for code, term in sorted(inventory[aid].items()):
            match, why = locate(text, term, segments)
            counts["mentions_" + why] += 1
            if match is None or group == "unsupported":
                continue
            start, end, occurrences = match
            spans.append(dict(id=code, start=start, end=end, term=term))
            geometry[code] = dict(article=aid, occurrences=occurrences, tokens=n)
        if group != "unsupported":
            packet.append(dict(id="relay/" + aid, text=text, terms=spans))
        summary_packet.append(dict(id="relay-summary/" + aid, text=summaries[aid]))
    write_rows(DATA / "local-stimuli.jsonl", packet)
    write_rows(DATA / "summary-stimuli.jsonl", summary_packet)
    write_rows(DATA / "judgments.jsonl", visits)
    (DATA / "coordinates.json").write_text(json.dumps(geometry, sort_keys=True))
    counts["participant_term_rows"] = len(visits)
    counts["supported_article_term_types"] = len(geometry)
    counts["source_articles"] = len(articles)
    seal = dict(source_revision="860f6bc5c2a23e5ef8ed22bdf86ebc4f960ebec9",
                source_sha256=SOURCE_HASH, guo_sha256=FROZEN, counts=counts,
                tokenizer_revision=REV, protocol=digest(Path(__file__).with_name("relay-protocol.md")),
                hashes={name:digest(DATA/name) for name in ("local-stimuli.jsonl","summary-stimuli.jsonl",
                                                            "judgments.jsonl","coordinates.json")})
    (OUT / "input-seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    prepare()
