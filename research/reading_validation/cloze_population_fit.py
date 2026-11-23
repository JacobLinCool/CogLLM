"""Train on original completion counts, then lock all independent word distributions."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from research.backbone_transfer.extract import (
    digest,
    read_json,
    write_json,
    stream_rows,
)
from research.reading_validation.cloze_population_data import (
    ROOT,
    OUT,
    PLAN,
    normalized,
)
from research.reading_validation.cloze_population_model import (
    Choices,
    source_basis,
    project,
    packed_words,
    fit,
    material,
    predictions,
)
from research.reading_validation.meaning_score import folds


def data(folder):
    i = read_json(OUT / "input-seal.json")
    r = read_json(folder / "receipt.json")
    if (
        r["source"] != read_json(OUT / "semantic-input-seal.json")["combined"]
        or r["dictionary"] != i["dictionary"]
        or i["protocol"] != digest(PLAN)
        or digest(ROOT / "source-counts.json") != i["source_hash"]
        or any(digest(folder / k) != v for k, v in r["files"].items())
    ):
        raise ValueError("Scientific checkpoint mismatch")
    lex = read_json(ROOT / "dictionary.json")
    word = {r["word"]: j for j, r in enumerate(lex)}
    sources = read_json(ROOT / "source-counts.json")
    geo = read_json(ROOT / "geometry.json")
    ids = []
    articles = []
    positions = []
    targets = []
    internal = []
    xs = []
    ps = []
    indices = []
    counts = []
    complete = []
    words = len(word)
    with np.load(folder / "log-probabilities.npz") as L, np.load(
        folder / "prefix-states.npz"
    ) as S:
        for d, v in geo.items():
            if not len(v["items"]):
                continue
            ll = L[d]
            ss = S[d]
            q = np.asarray([i["word"] - 1 for i in v["items"]])
            ps.append(ll[q])
            xs.append(ss[q])
            for t in v["items"]:
                uid = t["source_id"]
                ids.append(uid)
                articles.append(v["article"])
                positions.append(t["word"])
                targets.append(word.get(t["target"], -1))
                is_source = v["source"] == "UCL"
                internal.append(is_source)
                complete.append(v["complete_article_lexeme_alignment"])
                c = {}
                if is_source:
                    for k, n in sources[uid].items():
                        j = word.get(k, words)
                        c[j] = c.get(j, 0) + n
                indices.append(np.array(sorted(c), int))
                counts.append(np.array([c[j] for j in sorted(c)], int))
    with np.load(folder / "lexicon.npz") as z:
        route = z["route"]
        share = z["share"]
        coords = z["coordinates"]
    if not np.array_equal(
        np.unique([v["tokens"][0] for v in lex], return_inverse=True)[1], route
    ):
        raise ValueError("Dictionary route changed")
    U, V = packed_words(coords, np.array([k["frequency"] for k in lex]))
    rows = np.vstack(ps)
    raw = material(np.arange(len(rows)), rows, route, share)
    X = np.vstack(xs).astype(float)
    meta = dict(
        ids=np.array(ids),
        articles=np.asarray(articles),
        pos=np.asarray(positions),
        targets=np.array(targets),
        source=np.array(internal),
        complete=np.array(complete),
        words=word,
    )
    return raw, X, U, V, Choices(indices, counts), meta


def assign(sent):
    base = Path("research/results/reading-validation-convergent/adaptive")
    known = {
        s: k
        for k in range(5)
        for s in read_json(base / f"context-fold-{k}.json")["held_articles"]
    }
    extra = folds(set(sent) - set(known), "Additional UCL population sources:")
    return {**known, **extra}


def ordinary(source):
    index = source.indices
    hits = source.counts
    total = np.bincount(
        np.concatenate(index), weights=np.concatenate(hits), minlength=8193
    )
    q = (total + 0.5) / (total.sum() + 0.5 * len(total))
    return dict(kind="contextless", logp=np.log(q).tolist())


def train(folder):
    raw, X, U, V, choices, m = data(folder)
    source = np.where(m["source"])[0]
    external = np.where(~m["source"])[0]
    part = assign(m["articles"][source])
    groups = np.array([part[a] for a in m["articles"][source]])
    receipts = []
    pack = {}
    for f in range(6):
        tr = source[groups != f] if f < 5 else source
        te = source[groups == f] if f < 5 else external
        file = OUT / f"fold-{f}.json"
        end = OUT / f"source-fold-{f}-scores.json"
        source_ids = m["ids"][tr].tolist()
        seal = digest(folder / "receipt.json")
        if file.exists():
            b = read_json(file)
            if b["input_receipt"] != seal or b["training_ids"] != source_ids:
                raise ValueError("Incompatible source training checkpoint")
            projection = b["geometry"]
            models = b["models"]
        else:
            projection = source_basis(X, tr)
            C = project(X[tr], projection)
            Y = choices.subset(tr)
            models = {"frequency": ordinary(Y), "raw": {"kind": "raw"}}
            training = raw[tr]
            for name, context in [("calibrated", C[:, :0]), ("context", C)]:
                models[name] = {"kind": name, **fit(training, context, Y, U, V)}
            write_json(
                file,
                dict(
                    input_receipt=seal,
                    protocol=digest(PLAN),
                    training_ids=source_ids,
                    geometry=projection,
                    models=models,
                ),
            )
        receipts.append(digest(file))
        hashes, measured = predictions(
            te,
            raw,
            X,
            projection,
            models,
            U,
            V,
            choices if f < 5 else None,
            m["targets"],
        )
        if f < 5:
            if not end.exists():
                write_json(end, dict(forecasts=hashes, scores=measured))
        else:
            pack = hashes
        print(
            "Word population source fold",
            f,
            "fit on",
            len(tr),
            "max per-choice derivative",
            max(models[k]["stationarity"] for k in ("calibrated", "context")),
            flush=True,
        )
    write_json(
        OUT / "forecast-lock.json",
        dict(
            protocol=digest(PLAN),
            native=digest(folder / "receipt.json"),
            external_forecast_sha256=pack,
            external_prefixes=len(external),
            alphabet=raw.shape[1],
            source_fits=receipts,
            external_human_values_scored=False,
            likelihood=digest(Path(__file__).with_name("cloze_population_model.py")),
        ),
    )


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("native", type=Path)
    train(p.parse_args().native)
