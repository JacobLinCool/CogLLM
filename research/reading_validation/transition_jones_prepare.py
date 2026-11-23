"""Preserve the original presentation order and experimental regions for Jones and Bergen."""

from collections import Counter
import csv
from difflib import SequenceMatcher
import json
from pathlib import Path
import re

from research.reading_validation.antecedent_inference import BUNDLE, load_tagger
from research.reading_validation.transition_architecture_prepare import load_tokenizer, word_indices
from research.reading_validation.transition_game_prepare import shingles
from research.reading_validation.transition_pointer_features import FORMS, sentence_ids
from research.reading_validation.transition_score import dump, sha
from research.reading_validation.transition_snac_prepare import ROOT, WORD


OUT = Path("research/results/reading-validation-transition-jones")
PROTOCOL = Path("research/reading_validation/transition-jones-protocol.md")
DETERMINERS = {"the", "a", "an", "his", "her", "their", "its"}


def read(name):
    with (ROOT / ("jones-" + name + ".csv")).open(newline="") as handle:
        return list(csv.DictReader(handle))


def regions(marked):
    parts, selected, length = [], [], 0
    for part in marked.split("/"):
        fragment = " ".join(part.split())
        measured = fragment.startswith("#")
        if measured:
            fragment = fragment[1:].strip()
        if not fragment:
            continue
        left = length + bool(parts)
        parts.append(fragment)
        length = left + len(fragment)
        if measured:
            selected.append([left, length])
    if len(selected) != 14:
        raise ValueError("An original passage does not identify 14 timed regions")
    return " ".join(parts), selected


def article(identifier, text, tok, tagger):
    enc = tok.encode(text, add_special_tokens=False)
    if len(enc.ids) > 2048:
        raise ValueError("An experimental article requires truncation")
    spans = [list(w.span()) for w in WORD.finditer(text)]
    table = word_indices(text, spans, tok, enc)
    words = [s for s, v in zip(spans, table, strict=True) if v is not None]
    tokens = [v for v in table if v is not None]
    forms = [text[a:b] for a, b in words]
    qs = [i for i, w in enumerate(forms) if i and w.casefold() in FORMS]
    tags = [t for _, t in tagger.tag(forms)]
    fixes = []
    for q in qs:
        prefix = [t for _, t in tagger.tag(forms[:q + 1])]
        if tags[:q - 1] != prefix[:q - 1]:
            raise ValueError("POS controls have uncaptured future dependence")
        fixes.append(prefix[q - 1])
    return {"id": identifier, "text": text, "words": words, "tokens": tokens, "queries": qs,
            "pos": tags, "pos_last": fixes, "sentences": sentence_ids(text, words), "token_count": len(enc.ids)}


def names(row, np1, np2, query, experiment):
    forms = [row["text"][a:b].casefold() for a, b in row["words"][:query]]
    result = []
    for name in (np1, np2):
        components = name.casefold().split()
        if components[0] in DETERMINERS:
            components = components[1:]
        if experiment == 2 and components == ["raquet"]:
            components = ["racquet"]
        hits = [i for i in range(len(forms) - len(components) + 1) if forms[i:i + len(components)] == components]
        if not hits:
            raise ValueError(f"Original referent {name} is missing in {row['id']}")
        i = hits[-1]
        start = i-1 if i and forms[i-1] in DETERMINERS else i
        result.append(list(range(start, i + len(components))))
    reversed_choices = experiment == 1 and row["id"] in {"e1-22_A", "e1-22_B"}
    in_source_order = max(result[0]) < min(result[1])
    if set(result[0]) & set(result[1]) or in_source_order == reversed_choices:
        raise ValueError("NP identity and the experiment's ordering do not agree")
    return result


def prepare():
    tok = load_tokenizer("qwen", json.loads(Path("research/backbone_transfer/models.json").read_text())["models"]["qwen"])
    tags = load_tagger(BUNDLE / "averaged_perceptron_tagger_eng.zip")
    source = {v["item_id"]: v for v in read("exp1-items")}
    masks = {v["item_id"]: v for v in read("exp1-gpt3")}
    raw = {v["item_id"]: v for group in json.loads((ROOT / "jones-exp2-stimuli.json").read_text()) for v in group}
    data, geometry = [], {}
    for item, spec in source.items():
        text = spec["expt_sent"]
        prefix, suffix = masks[item]["sentence_masked"].split("[MASK]")
        if not text.casefold().startswith(prefix.casefold()) or not text.casefold().endswith(suffix.casefold()):
            raise ValueError("Source masked and unmasked Exp. 1 critical sentences differ")
        row = article("e1-" + item, text, tok, tags)
        where = [q for q in row["queries"] if row["words"][q] == [len(prefix), len(text)-len(suffix)]]
        if len(where) != 1:
            raise ValueError("The original referring expression is not a certified pronoun")
        q = where[0]
        n = names(row, spec["expt_NP1"], spec["expt_NP2"], q, 1)
        geometry[row["id"]] = {"q": q, "end": len(row["words"])-1, "names": n}
        data.append(row)
    rows2, marks = {}, {}
    for key, description in raw.items():
        text, reg = regions(description["stimulus"])
        rows2[key] = article("e2-" + key, text, tok, tags)
        marks[key] = reg
    for key, row in rows2.items():
        src = raw[key]
        reg = marks[key]
        geom = {"regions": reg, "sent": int(src["sent_id"]), "order": src["order"],
                "continuation": int(src["continuation"][-1]), "explicit": int(src["unambiguous"])}
        if not src["unambiguous"]:
            other = rows2[key[:-3] + "un"]
            def forms(r):
                return [r["text"][a:b].casefold() for a, b in r["words"]]
            a, b = forms(row), forms(other)
            change = [v for v in SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if v[0] != "equal"]
            if len(change) != 1:
                raise ValueError("Unambiguous control differs outside a single replacement")
            _, q, q_end, start, stop = change[0]
            if q_end != q+1 or q not in row["queries"]:
                raise ValueError("The experiment's critical pronoun is not identified")
            spec = source["_".join(key.split("_")[:2])]
            choices = [spec["expt_NP1"], spec["expt_NP2"]]
            if src["sent_id"] == 22:
                choices.reverse()
            n = names(row, *choices, q, 2)
            end = max(i for i, (l, r) in enumerate(row["words"]) if reg[3][0] <= l and r <= reg[3][1])
            if not 0 <= end - q <= 24:
                raise ValueError("Ambiguous relation is outside the blinded readout window")
            other_idx = 1 if geom["continuation"] == 1 else 0
            observed = [w for w in b[start:stop] if w not in DETERMINERS]
            expected = [a[i] for i in n[other_idx] if a[i] not in DETERMINERS]
            if observed != expected:
                raise ValueError(f"Explicit control {key} names {observed}, expected {expected}")
            geom.update(q=q, end=end, names=n, counterfactual_key=other["id"],
                        explicit_np_end_char=other["words"][stop-1][1], explicit_event_end_char=marks[key[:-3]+"un"][3][1])
        geometry[row["id"]] = geom
        data.append(row)
    if len(data) != 300 or len(geometry) != 300:
        raise ValueError("Every original experiment presentation is required")
    out = ROOT / "jones-pointer-stimuli.jsonl"
    out.write_text("".join(json.dumps(v, separators=(",", ":"))+"\n" for v in data))
    OUT.mkdir(exist_ok=True, parents=True)
    dump(OUT/"geometry.json", geometry)
    known = set()
    for r in map(json.loads,(ROOT/"pointer-stimuli.jsonl").read_text().splitlines()):
        known |= shingles(r["text"])
    shared = [r["id"] for r in data if shingles(r["text"]) & known]
    if shared:
        raise ValueError(f"Game source overlap in {shared}")
    raw_hashes = {p.name: sha(p) for p in ROOT.glob("jones-*") if not p.name.endswith(("-files.json", "-data.json", ".jsonl"))}
    dump(OUT/"source-seal.json", {"protocol_sha256":sha(PROTOCOL),"input_sha256":sha(out),"source_files":raw_hashes,
         "geometry_sha256":sha(OUT/"geometry.json"),"original_articles":len(data),"word_support":sum(len(r["words"]) for r in data),
         "max_tokens":max(r["token_count"] for r in data),"overlap_20_words_game_sources":shared,
         "max_pronoun_to_event_words":max(v["end"]-v["q"] for v in geometry.values() if "q" in v),
         "manual_spelling_alignment":"Exp. 1 item 28 raquet; Exp. 2 racquet",
         "choice_coding":"In both orders of family 22 the Exp. 1 response names are reversed relative to local mention order. Use the listed names for Exp. 1. Exp. 2 numbering follows mention order and is verified against the actual NP substituted in each explicit control.",
         "eligible_pronouns":sum(len(r["queries"]) for r in data)})


if __name__ == "__main__":
    prepare()
