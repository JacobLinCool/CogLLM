"""Independently fix a word response alphabet and certify the original visible-prefix geometry."""
from collections import Counter, defaultdict
import csv, json, re
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tokenizers import Tokenizer
from research.backbone_transfer.extract import (
    digest,
    read_json,
    write_json,
    stream_rows,
)
from research.reading_validation.identity_prepare import DATA, TOKENIZER, sealed_rows
from research.reading_validation.convergent_word_prepare import ROOT as UCL

ROOT = DATA / "cloze-population"
OUT = Path("research/results/reading-validation-cloze-population")
PROVO = DATA / "provo-predictability.csv"
PLAN = Path(__file__).with_name("cloze-population-protocol.md")


def verified_json(path, content):
    if path.exists():
        if read_json(path)!=content:raise ValueError('Previously recovered public source differs: '+str(path))
    else:write_json(path,content)


def normalized(s):
    return str(s).strip().casefold().replace("’", "'")


def display(s):
    return "".join(
        c for c in normalized(s).strip(' .,?!:;()[]"') if c.isalnum() or c in "'-"
    )


def compatible(surface, source_word):
    s = display(surface)
    q = display(source_word)
    return bool(q and (q == s or s.startswith(q + "'") or s.startswith(q + "n'")))


def inventory():
    from wordfreq import top_n_list, word_frequency
    import importlib.metadata

    if importlib.metadata.version("wordfreq") != "3.1.1":
        raise ValueError("Unregistered lexicon revision")
    d = {}
    for w in top_n_list("en", 24000):
        w = normalized(w)
        if re.fullmatch("[a-z]+(?:'[a-z]+)*", w):
            d[w] = word_frequency(w, "en")
        if len(d) == 8192:
            break
    if len(d) != 8192:
        raise ValueError("Incomplete independent response alphabet")
    return list(d), list(d.values())


def source_counts():
    import xlrd
    from research.reading_validation.convergent_raw_audit import constants, download

    audit = read_json(
        Path(
            "research/results/reading-validation-convergent/independent-source-count-audit.json"
        )
    )
    gold = audit["items"]
    target = constants()["trgt_compregension"]
    sheet_items = defaultdict(list)
    for r in csv.DictReader((UCL / "items.csv").open()):
        sheet_items[int(r["list"])].append(r)
    out = {}
    for group in range(1, 9):

        def raw(s):
            b = download(s)
            if sha256(b).hexdigest() != audit["source_sha256"][s]:
                raise ValueError("Source response content changed")
            return b

        repairs = {
            (int(r["idx"]), r["Q_num"]): (r["word"], r["corrected"])
            for r in csv.DictReader(
                StringIO(raw(f"logs/logs{group}.csv").decode("utf-8-sig"))
            )
        }
        page = xlrd.open_workbook(
            file_contents=raw(f"cp/list{group}.xls")
        ).sheet_by_index(0)
        keys = page.row_values(0, 28)
        users = [
            (i - 1, page.row_values(i, 28))
            for i in range(2, page.nrows)
            if sum(a == b for a, b in zip(page.row_values(i, 18, 28), target)) >= 8
        ]
        for j, r in enumerate(sheet_items[group]):
            ix = r["item_id"]
            values = Counter()
            for i, row in users:
                word = row[j]
                if not str(word).strip():
                    continue
                if isinstance(word, str):
                    w = word.split(" ")[0].lower().replace("’", "'")
                    if (i, keys[j]) in repairs:
                        old, new = repairs[i, keys[j]]
                        if w != old:
                            raise ValueError("Source correction mismatch")
                        w = new
                else:
                    w = word
                values[normalized(w)] += 1
            if sum(values.values()) != gold[ix]["cloze_nonempty"]:
                raise ValueError("Distinct original response denominator")
            out[ix] = values
    verified_json(ROOT / "source-counts.json", out)


def prepare():
    lock = read_json(OUT / "registration.json")
    ROOT.mkdir(exist_ok=True)
    if (
        digest(PROVO) != lock["external_file_sha256"]
        or digest(PLAN) != lock["protocol"]
    ):
        raise ValueError("Unverified public source or protocol")
    source_counts()
    lex, freq = inventory()
    tok = Tokenizer.from_file(str(TOKENIZER))
    kw = []
    for w, f in zip(lex, freq):
        enc = tok.encode(" " + w, add_special_tokens=False)
        kw.append(dict(word=w, tokens=enc.ids, frequency=f))
    verified_json(ROOT / "dictionary.json", kw)
    materials = defaultdict(lambda: defaultdict(set))
    text_forms = defaultdict(set)
    for r in csv.DictReader(PROVO.open(encoding="mac_roman")):
        k = r["Text_ID"]
        i = int(r["Word_Number"])
        text_forms[k].add(r["Text"])
        materials[k][i].add((r["Word_Unique_ID"], r["Word"]))
    ambiguous_articles = {k: v for k, v in text_forms.items() if len(v) > 1}
    if set(ambiguous_articles) != {"27"}:
        raise ValueError("Unexpected external text ambiguity")
    texts = {k: next(iter(v)) for k, v in text_forms.items() if len(v) == 1}
    normative = {
        r["id"]: r for r in stream_rows(UCL / "map.jsonl") if r["language"] == "English"
    }
    packets = []
    entries = {}
    covered = {}

    def add(source, article, text, targets, complete):
        k = sha256(text.encode()).hexdigest()
        matches = list(re.finditer(r"\S+", text))
        enc = tok.encode(text, add_special_tokens=False)
        ends = []
        coords = []
        dropped = 0
        for i, m in enumerate(matches):
            a, b = m.span()
            ids = [
                j
                for j, (left, right) in enumerate(enc.offsets)
                if right > a and left < b and text[max(a, left) : min(b, right)].strip()
            ]
            if not ids:
                raise ValueError("Word has no visible native token")
            u, v = ids[0], ids[-1]
            ends.append([a, b, u, v])
            if i not in targets:
                continue
            prefix = text[: matches[i - 1].end()]
            before = tok.encode(prefix, add_special_tokens=False).ids
            if (
                i < 1
                or u < 1
                or before != enc.ids[:u]
                or text[matches[i - 1].end() : a] != " "
            ):
                dropped += 1
                continue
            coords.append(
                dict(
                    word=i,
                    source_id=str(targets[i]),
                    target=normalized(m.group()).strip('.,?!:;()[]"'),
                    before=u - 1,
                )
            )
        if k in entries:
            raise ValueError("Source/target overlap in full original text")
        packets.append({"id": k, "text": text, "ends": ends})
        entries[k] = {
            "source": source,
            "article": str(article),
            "items": coords,
            "complete_article_lexeme_alignment": complete,
            "excluded_token_boundaries": dropped,
        }

    for r in stream_rows(UCL / "stimuli.jsonl"):
        if r["id"] in normative:
            s = normative[r["id"]]
            add(
                "UCL",
                s["article"],
                r["text"],
                {i: z for i, z in enumerate(s["items"]) if z is not None},
                True,
            )
    for k, t in texts.items():
        raw = t.split()
        m = materials[k]
        selected = {}
        first = None
        for i in sorted(m):
            exact = [
                (key, w)
                for key, w in m[i]
                if 2 <= i <= len(raw) and compatible(raw[i - 1], w)
            ]
            if len(exact) != 1:
                first = i
                break
            selected[i - 1] = exact[0][0]
        covered[k] = {
            "original_outcomes": len(m),
            "verified_prefix_positions": len(selected),
            "first_unverified_position": first,
        }
        add("Provo", k, t, selected, first is None)
    covered["27"] = {
        "original_outcomes": len(materials["27"]),
        "verified_prefix_positions": 0,
        "distinct_source_spellings": 2,
    }
    sealed_rows(ROOT / "stimuli.jsonl", packets)
    write_json(ROOT / "geometry.json", entries)
    summary = {
        src: dict(
            articles=sum(v["source"] == src for v in entries.values()),
            positions=sum(
                len(v["items"]) for v in entries.values() if v["source"] == src
            ),
        )
        for src in ("UCL", "Provo")
    }
    write_json(
        OUT / "input-seal.json",
        dict(
            protocol=digest(PLAN),
            source_hash=digest(ROOT / "source-counts.json"),
            stimuli=digest(ROOT / "stimuli.jsonl"),
            geometry=digest(ROOT / "geometry.json"),
            dictionary=digest(ROOT / "dictionary.json"),
            dictionary_words=len(kw),
            independent_token_onsets=len({x["tokens"][0] for x in kw}),
            source_support=summary,
            provo_materials=covered,
            external_answers_read_for_scoring=False,
        ),
    )
    print(summary)


if __name__ == "__main__":
    prepare()
