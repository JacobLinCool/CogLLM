"""Recover public, ungraded SemEval substitution contexts and screen earlier supervision."""

from collections import Counter, defaultdict
from html import unescape
from itertools import combinations
import json
from pathlib import Path
import re
import tarfile

from research.reading_validation.emtec_material import exposures
from research.reading_validation.local_forecast import WORD, digest
from research.reading_validation.nns_material import BASE, identifier
from research.reading_validation.nns_rewrite_material import focus

DATA = BASE / "semeval-lex"
OUT = Path("research/results/reading-validation-semlex")
ARCHIVE = "8a2b206a2581b35c02c3b64783fddf019a35a43acdc0a821be8cfa26e4aff245"
CHOICES = "4d60a97186235513a9694810d26abf9816acfffcc13288e307b338b83e333fde"
ENCODING = re.compile(r"&(?!apos;|quot;|amp;|lt;|gt;|#\d+;|#x[0-9a-fA-F]+;)")


def materials():
    archive = DATA / "original-2007.tar.gz"
    candidate = DATA / "candidates.txt"
    if digest(archive) != ARCHIVE or digest(candidate) != CHOICES:
        raise ValueError(
            "Unexpected original LexSub texts or lexical-simplicity candidate set"
        )
    with tarfile.open(archive) as z:
        raw = z.extractfile("test/lexsub_test.xml").read().decode()
    lexelts = re.findall(r'<lexelt item="([^"]+)">([\s\S]*?)</lexelt>', raw)
    if len(lexelts) != 171:
        raise ValueError("Wrong number of original test lexical groups")
    by_id = {}
    for name, part in lexelts:
        groups = re.findall(
            r'<instance id="(\d+)">\s*<context>([\s\S]*?)</context>\s*</instance>', part
        )
        if len(groups) != 10:
            raise ValueError("A source target is missing a context")
        for key, ctx in groups:
            if key in by_id:
                raise ValueError("Duplicated SemEval identifier")
            by_id[int(key)] = (name, ctx)
    no = Counter()
    cases = {}
    allq = []
    for expected, line in enumerate(candidate.read_text().splitlines(), 301):
        header, sep, rest = line.partition("::")
        if not sep:
            raise ValueError("Invalid SemEval candidate list")
        lemma, uid = header.split()
        idx = int(uid)
        if idx != expected:
            raise ValueError("2007 and 2012 author instance IDs disagree")
        if by_id[idx][0] != lemma:
            no["source_lemma_or_pos_disagrees_between_releases"] += 1
            continue
        base, pos = lemma.rsplit(".", 1)
        allowed = {base}
        for blob in rest.strip().rstrip(";").split(";"):
            if not blob.strip():
                continue
            c, h = blob.strip().rsplit(" ", 1)
            if not h.isnumeric():
                raise ValueError("Unrecognized original candidate count")
            allowed.add(c)
        original = by_id[idx][1]
        if "�" in original or ENCODING.search(original):
            no["invalid_source_character_or_xml_reference"] += 1
            continue
        head = re.fullmatch(r"([\s\S]*?)<head>([^<]*?)</head>([\s\S]*)", original)
        if not head or any("<" in x for x in head.groups()):
            no["head_tag_ambiguous"] += 1
            continue
        left, source, right = map(unescape, head.groups())
        if (
            base.casefold() != source.casefold()
            or source.upper() == source
            and len(source) > 1
        ):
            no["unmatched_source_inflection_or_case"] += 1
            continue
        words = sorted(
            s for s in allowed if WORD.fullmatch(s) and s[:1].islower() and s.isascii()
        )
        if len(words) < 2:
            no["fewer_than_two_verifiable_word_choices"] += 1
            continue
        variants = {}
        material = []
        for w in words:
            surface = w[:1].upper() + w[1:] if source[:1].isupper() else w
            text = left + surface + right
            span = (len(left), len(left) + len(surface))
            if span not in [m.span() for m in WORD.finditer(text)]:
                continue
            did = identifier(text)
            target = focus(did, *span)
            variants[w] = {"id": did, "start": span[0], "end": span[1], "word": target}
            material.append((idx, {"id": did, "text": text, "span": span}))
        if len(variants) < 2:
            no["ambiguous_target_boundary"] += 1
            continue
        cases[idx] = {
            "lemma": lemma,
            "candidates": sorted(allowed),
            "variants": variants,
        }
        allq.extend(material)
    if len(by_id) != 1710 or set(by_id) != set(range(301, 2011)):
        raise ValueError("Incomplete reference material")
    return cases, allq, no


def run():
    cases, full, drop = materials()
    bad, audit = exposures([r["text"] for _, r in full])
    exclude = {full[i][0] for i in bad}
    docs = {}
    spans = defaultdict(set)
    case = {k: v for k, v in cases.items() if k not in exclude}
    for i, r in full:
        if i not in case:
            continue
        doc = r["id"]
        docs[doc] = r["text"]
        spans[doc].add(tuple(r["span"]))
    OUT.mkdir(exist_ok=True, parents=True)
    packet = DATA / "stimuli.jsonl"
    geo = DATA / "geometry.json"
    with packet.open("w") as f:
        for k in sorted(docs):
            f.write(
                json.dumps(
                    {"id": k, "text": docs[k], "spans": sorted(spans[k])},
                    ensure_ascii=False,
                )
                + "\n"
            )
    geo.write_text(json.dumps(case, sort_keys=True) + "\n")
    pairs = defaultdict(list)
    total = 0
    for k, t in case.items():
        for a, b in combinations(sorted(t["variants"]), 2):
            pairs[t["lemma"], a, b].append(k)
            total += 1
    repeated = {k: v for k, v in pairs.items() if len(v) > 1}
    info = {
        "original_archive": ARCHIVE,
        "public_candidates": CHOICES,
        "held_ranking_sha256": digest(DATA / "outcomes-reserve.txt"),
        "author_supplement_revision": "7242f641933707840b2884eedc0a53784942a440",
        "original_instances": 1710,
        "prior_material_excluded_contexts": len(exclude),
        "exclusions_before_similarity": dict(drop),
        "exposure_source_counts": audit,
        "eligible_contexts": len(case),
        "eligible_lemmas": len({t["lemma"] for t in case.values()}),
        "proposed_word_pairs": total,
        "filled_texts": len(docs),
        "unique_focused_positions": sum(map(len, spans.values())),
        "repeated_candidate_pairs": len(repeated),
        "repeated_candidate_lemmas": len({k[0] for k in repeated}),
        "cross_context_same_word_pair_comparisons": sum(
            len(v) * (len(v) - 1) // 2 for v in repeated.values()
        ),
        "packet_sha256": digest(packet),
        "geometry_sha256": digest(geo),
        "outcomes_inspected": False,
        "registered_question_sha256": digest(
            Path(__file__).with_name("semlex-protocol.md")
        ),
        "material_code_sha256": digest(Path(__file__)),
    }
    (OUT / "material.json").write_text(json.dumps(info, indent=2) + "\n")
    print(
        {k: v for k, v in info.items() if isinstance(v, int) or k == "packet_sha256"},
        flush=True,
    )


if __name__ == "__main__":
    run()
