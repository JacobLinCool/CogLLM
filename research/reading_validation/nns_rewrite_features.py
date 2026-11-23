"""Construct native word scores and separate masked-context preference geometry from outcomes."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from research.reading_validation.local_forecast import WORD, digest, rows
from research.reading_validation.local_inference import LocalInference
from research.reading_validation.nns_features import lexical, single_word_rank, stem
from research.reading_validation.nns_material import ARCHIVE, identifier
from research.reading_validation.nns_rewrite_material import DATA, MEMBER, OUT, focus

TIE = "The words are equally simple or equally complex."


def material():
    manifest = json.loads((OUT / "material.json").read_bytes())
    geo = DATA / "geometry.json"
    packet = DATA / "stimuli.jsonl"
    if (
        digest(geo) != manifest["geometry_sha256"]
        or digest(packet) != manifest["packet_sha256"]
    ):
        raise ValueError("Rewrite geometry differs from the locked public materials")
    return json.loads(geo.read_bytes()), manifest


def word_features(extracted):
    geometry, manifest = material()
    receipt = json.loads((extracted / "extraction.json").read_bytes())
    if receipt["packet_sha256"] != manifest["packet_sha256"] or receipt[
        "worker_sha256"
    ] != digest(Path(__file__).with_name("nns_rewrite_l4.py")):
        raise ValueError("Unexpected model extraction")
    for k, v in receipt["files_sha256"].items():
        if digest(extracted / k) != v:
            raise ValueError("Corrupt native word outputs")
    ids = json.loads((extracted / "ids.json").read_bytes())
    src = LocalInference.load()
    index = {k: i for i, k in enumerate(ids)}
    bank = defaultdict(list)
    specified = {
        kk: (key, geometry["families"][key], geometry["roles"][key])
        for key, vs in geometry["variants"].items()
        for kk in vs.values()
    }
    with np.load(extracted / "signals.npz", allow_pickle=False) as z:
        selected, primary, extra, lex_score = [
            z[k] for k in ("native", "plaba", "context", "lexical")
        ]
        surprisal, measured = z["surprisal"], z["measured_surprisal"]
        for r in rows(DATA / "stimuli.jsonl"):
            text = r["text"]
            doc = r["id"]
            targets = {tuple(pos) for pos in r["spans"]}
            words = list(WORD.finditer(text))
            positions = np.asarray([m.span() for m in words])
            n = len(words)
            ok = np.zeros(n, bool)
            p = np.zeros((n, 32), np.float32)
            c = np.zeros(n, np.float32)
            matches = []
            for j, m in enumerate(words):
                if m.span() not in targets:
                    continue
                key = focus(doc, *m.span())
                i = index[key]
                if selected[i] < 0:
                    continue
                matches.append((i, j, m, key))
                ok[j] = True
                p[j] = primary[i]
                c[j] = lex_score[i]
            rank = single_word_rank(src, text, doc, positions, ok, c, p)
            for i, j, m, key in matches:
                w = m.group().casefold()
                base, f, role = specified[key]
                bank["id"].append(key)
                bank["type"].append(w)
                bank["stem"].append(stem(w))
                bank["family"].append(f)
                bank["doc"].append(base)
                bank["role"].append(role)
                bank["form"].append(
                    [
                        *lexical(w),
                        float(m.group()[:1].isupper()),
                        float(m.group().isupper()),
                        np.log1p(j),
                    ]
                )
                bank["source"].append([c[j], rank[m.span()]])
                bank["context"].append(np.r_[primary[i], extra[i]])
                bank["surprisal"].append(surprisal[i])
                bank["measured_surprisal"].append(measured[i])
    arrays = {k: np.asarray(v) for k, v in bank.items()}
    if len(set(arrays["id"])) != len(arrays["id"]):
        raise ValueError("Nonunique target word")
    return arrays


def comparisons(bank, role, *, outcomes=False):
    codes, manifest = material()
    if digest(ARCHIVE) != manifest["release_sha256"]:
        raise ValueError("Unexpected author archive")
    ids = {k: i for i, k in enumerate(bank["id"])}
    tab = defaultdict(Counter)
    coverage = Counter()
    with ZipFile(ARCHIVE) as f, f.open(MEMBER) as stream:
        for raw in stream:
            cols = raw.decode().rstrip("\r\n").split("\t")
            s = identifier(cols[4])
            a, b = sorted(cols[7:9])
            if codes["roles"][s] != role:
                continue
            pair = (codes["variants"][s][a], codes["variants"][s][b])
            coverage["released_rows"] += 1
            if not all(k in ids for k in pair):
                coverage["missing_certified_word"] += 1
                continue
            if a == b:
                raise ValueError("Author supplied identical alternatives")
            key = (s, *pair)
            obj = tab[key]
            if outcomes:
                m = cols[9]
                if m not in (a, b, TIE):
                    raise ValueError("Unrecognized ternary reader response")
                obj[(a, b, TIE).index(m)] += 1
    keys = sorted(tab)
    pair = np.asarray([[ids[a], ids[b]] for _, a, b in keys])
    docs = np.asarray([s for s, _, _ in keys])
    family = np.asarray([codes["families"][s] for s in docs])
    cnt = (
        np.asarray([[tab[k][i] for i in range(3)] for k in keys]) if outcomes else None
    )
    return {
        "keys": np.asarray(keys),
        "pair": pair,
        "doc": docs,
        "family": family,
        "counts": cnt,
        "coverage": dict(coverage),
    }
