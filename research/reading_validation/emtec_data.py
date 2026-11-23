"""Join registered passage ratings and exact presentation-word IDs only to frozen forecasts."""

import csv
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from wordfreq import zipf_frequency

from research.reading_validation.emtec_material import DATA, OUT, identifier
from research.reading_validation.local_forecast import digest

EYE_SHA = "a9ce24cba3989f3ae351b15a770555fa61a1634f6e5055e3f55bb6eca580c205"
TINY = 1e-12


def native(extraction):
    material = json.loads((OUT / "material.json").read_bytes())
    rec = json.loads((extraction / "extraction.json").read_bytes())
    g = DATA / "geometry.json"
    if (
        rec["packet_sha256"] != material["public_packet_sha256"]
        or rec["worker_sha256"] != digest(Path(__file__).with_name("emtec_l4.py"))
        or digest(g) != material["local_geometry_sha256"]
        or any(digest(extraction / f) != h for f, h in rec["files_sha256"].items())
    ):
        raise ValueError("The human study does not match the locked native features")
    include = json.loads(g.read_bytes())
    info = json.loads((extraction / "summary.json").read_bytes())
    ids = {v["id"]: i for i, v in enumerate(info)}
    with np.load(extraction / "words.npz", allow_pickle=False) as z:
        matrix = z["matrix"]
        cuts = z["cuts"]
    col = {k: i for i, k in enumerate(rec["columns"])}
    words = {}
    full = {}
    alls = []
    article = []
    references = []
    aux = []
    with (DATA / "shown-stimuli.csv").open() as s:
        for r in csv.DictReader(s, delimiter="\t"):
            key = r["item_id"] + "/" + r["list"]
            if key not in include:
                continue
            i = ids[include[key]["id"]]
            words[key] = r["gen_seq_trunc"].split()
            full[i] = (r["gen_seq_trunc"], r["introductory_screen_item"])
    for i, v in enumerate(info):
        text = full[i][0]
        sl = slice(cuts[i], cuts[i + 1])
        blk = matrix[sl]
        tokens = list(re.finditer(r"\S+", text))
        wi = np.array([len(m.group()) for m in tokens])
        freq = np.array([zipf_frequency(m.group(), "en") for m in tokens])
        length = len(tokens)
        s = blk[:, col["word_surprisal"]]
        where = blk[:, col["native_end"]]
        stock = blk[:, col["stock"]]
        rise = blk[:, col["positive_increase"]]
        complex_ = blk[:, col["complex"]]
        ok = blk[:, col["complex_valid"]]
        counts = v["body_sentences"]
        ease = (
            float(v["mean_source_ease"]) if v["mean_source_ease"] is not None else 3.0
        )
        if (
            not np.array_equal(blk[:, :2], np.asarray([m.span() for m in tokens]))
            or length != v["words"]
        ):
            raise ValueError("Incorrect body word alignment")
        simple = [
            np.log1p(length),
            wi.mean(),
            freq.mean(),
            length / max(1, counts),
            where.mean(),
            s.mean(),
            np.quantile(s, 0.9),
        ]
        coverage = v["rated_sentences"] / max(1, counts)
        article.append([*simple, ease, coverage, stock.mean()])
        aux.append(
            [
                np.quantile(stock, 0.9),
                rise.mean(),
                blk[:, col["opening"]].mean(),
                blk[:, col["resolution"]].mean(),
                blk[:, col["transition"]].mean(),
            ]
        )
        references.append(
            [stock.mean(), 5 - ease, s.mean(), wi.mean(), length / max(1, counts)]
        )

        def lag(x):
            return np.r_[0, x[:-1]]

        # The original layout's line-end indicator is appended on the participant-word join.
        alls.append(
            np.column_stack(
                [
                    wi,
                    lag(wi),
                    freq,
                    lag(freq),
                    s,
                    lag(s),
                    where,
                    where**2,
                    np.log1p(where),
                    np.arange(length) / max(1, length - 1),
                    np.full(length, np.log1p(length)),
                    np.arange(length) == 0,
                    complex_,
                    ok,
                    stock,
                    rise,
                ]
            )
        )
    return {
        "id_index": ids,
        "info": info,
        "groups": include,
        "body_words": words,
        "cuts": cuts,
        "word": np.concatenate(alls),
        "article": np.asarray(article),
        "aux": np.asarray(aux),
        "raw_refs": np.asarray(references),
        "receipt": rec,
    }


def subject_folds(codes, prefix):
    keys = sorted(set(codes), key=lambda k: identifier(prefix + str(k)))
    pairs = {v: i % 4 for i, v in enumerate(keys)}
    return np.array([pairs[v] for v in codes])


def binary_value(v):
    if v.strip().casefold() in {"1", "1.0", "true"}:
        return 1
    if v.strip().casefold() in {"0", "0.0", "false"}:
        return 0
    raise ValueError("Unexpected binary coding in original experiment")


def observation_geometry(n, *, outcomes):
    mat = json.loads((OUT / "material.json").read_bytes())
    archive = DATA / "response-archive.zip"
    gaze = DATA / "corrected-reading.tsv"
    if digest(archive) != mat["response_archive_sha256"] or digest(gaze) != EYE_SHA:
        raise ValueError("Published reader observations changed")
    gd = n["groups"]
    ids = n["id_index"]
    subjects = []
    prompts = []
    docs = []
    keys = {}
    difficulty = []
    attention = []
    visits = []
    indices = []
    rr = []
    line = []
    with ZipFile(archive) as z, z.open("participant_info/participant_results.csv") as f:
        for v in csv.DictReader(io.TextIOWrapper(f), delimiter="\t"):
            key = v["item_id"] + "/" + v["list"]
            who = identifier("emtec-local-reader/" + v["subject_id"])
            pair = (who, key)
            if key not in gd:
                continue
            if pair in keys:
                raise ValueError("More than one trial for a reader and item list")
            keys[pair] = len(docs)
            docs.append(ids[gd[key]["id"]])
            subjects.append(who)
            prompts.append(v["item_id"])
            if outcomes:
                a = float(v["RATING_DIFFICULTY_VALUE"])
                b = binary_value(v["QUESTION_ACCURACY"])
                if a not in range(1, 6) or b not in (0, 1):
                    raise ValueError("Unexpected ordinal scale or attention index")
                difficulty.append(int(a))
                attention.append(b)
    observed_visits = np.zeros((len(docs), int(np.diff(n["cuts"]).max())), bool)
    with gaze.open(newline="") as f:
        for v in csv.DictReader(f, delimiter="\t"):
            key = v["item_id"] + "/" + v["list"]
            if key not in gd:
                continue
            who = identifier("emtec-local-reader/" + v["subject_id"])
            sid = keys.get((who, key))
            if sid is None:
                raise ValueError("Unmatched reader visit")
            j = int(v["word_id"])
            s = n["body_words"][key]
            if j < 0 or j >= len(s) or s[j] != v["word"]:
                raise ValueError("Reread word did not exactly match the published body")
            loc = n["cuts"][docs[sid]] + j
            if observed_visits[sid, j]:
                raise ValueError("Duplicated reader-word trial")
            observed_visits[sid, j] = True
            visits.append(sid)
            indices.append(loc)
            line.append(binary_value(v["last_in_line"]))
            if outcomes:
                rv = binary_value(v["RR"])
                if rv not in (0, 1):
                    raise ValueError("Unexpected reread coding")
                rr.append(rv)
    sindex = np.asarray(subjects)
    qindex = np.asarray(prompts)
    vidx = np.asarray(visits)
    wi = np.asarray(indices)
    lay = np.asarray(line)
    all_expected = n["cuts"][np.array(docs) + 1] - n["cuts"][docs]
    observed = np.bincount(vidx, minlength=len(docs))
    audit = {
        "rated_visits": len(docs),
        "eligible_readers": len(set(subjects)),
        "eligible_prompts": len(set(prompts)),
        "visits_with_word_trials": int((observed > 0).sum()),
        "exact_reread_word_positions": len(wi),
        "visits_missing_all_word_measures": int((observed == 0).sum()),
        "incomplete_word_visits_among_present": int(
            ((observed > 0) & (observed != all_expected)).sum()
        ),
        "observed_range_per_visit": [
            int(observed[observed > 0].min()),
            int(observed.max()),
        ],
    }
    return {
        "subjects": sindex,
        "prompts": qindex,
        "docs": np.array(docs),
        "v": vidx,
        "w": wi,
        "line": lay,
        "difficulty": np.asarray(difficulty),
        "attention": np.asarray(attention),
        "rr": np.asarray(rr),
        "audit": audit,
        "reader_fold": subject_folds(sindex, "reader-fold/"),
        "prompt_fold": subject_folds(qindex, "prompt-fold/"),
    }
