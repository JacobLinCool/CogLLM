"""Score fixed scientist-trained requests in an independent lay cohort."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from research.reading_validation.local_forecast import digest, rows
from research.reading_validation.local_help_inference import COLUMNS, DESIGNS, HelpReadout
from research.reading_validation.relay_data import DATA, FROZEN, OUT, WEIGHTS


def load_features(archive):
    receipt = json.loads((archive / "extraction.json").read_bytes())
    original = json.loads((OUT / "input-seal.json").read_bytes())
    file = archive / "features.jsonl"
    if (receipt["output_sha256"] != digest(file) or digest(WEIGHTS) != FROZEN
            or receipt["data_sha256"] != original["hashes"]["local-stimuli.jsonl"]
            or tuple(receipt["columns"]) != COLUMNS
            or receipt["backbone_forwards"] != receipt["unique_original_inputs"]):
        raise ValueError("Changed forecast or source estimator")
    for name, expected in original["hashes"].items():
        if digest(DATA / name) != expected:
            raise ValueError("Changed target material")
    if original["protocol"] != digest(Path(__file__).with_name("relay-protocol.md")):
        raise ValueError("Changed endpoint protocol")
    expected = {r["id"]:{x["id"] for x in r["terms"]} for r in rows(DATA / "local-stimuli.jsonl")}
    matrix = {}
    seen = set()
    for row in rows(file):
        article = row["id"]
        if article not in expected or article in seen:
            raise ValueError("Unexpected or duplicated target article")
        seen.add(article)
        included = {t["id"] for t in row["terms"]}
        excluded = {t["id"] for t in row["excluded_terms"]}
        if included & excluded or included | excluded != expected[article]:
            raise ValueError("Some designated terms were lost during extraction")
        for term in row["terms"]:
            f = np.array(term["features"])
            if f.shape != (len(COLUMNS),) or not np.isfinite(f).all() or term["id"] in matrix:
                raise ValueError("Invalid term feature vector")
            matrix[term["id"]] = f
    if seen != set(expected):
        raise ValueError("Target extraction missed a source article")
    return matrix


def forecast(matrix, records, endpoint):
    forms = np.stack([matrix[r["term"]] for r in records])
    prior = np.array([r["familiarity"] for r in records])
    model = HelpReadout()
    scores = {arm:model.predict(forms,endpoint=endpoint,arm=arm,familiarity_before=prior)["log_odds"]
              for arm in DESIGNS}
    if endpoint != "definition_over_example":
        scores["familiarity_only"] = -prior.astype(float)
    return list(scores), np.column_stack(list(scores.values()))


def observed(r, endpoint):
    if endpoint == "definition_over_example":
        return int(r["definition"]) if r["definition"] != r["example"] else None
    return int(r[endpoint])


def comparisons(records, scores, endpoint, group):
    across = group.startswith("between")
    cells = defaultdict(list)
    for i, r in enumerate(records):
        label = observed(r, endpoint)
        if label is not None:
            cells[r["article"],r["term"] if across else r["reader"]].append((i,label))
    groups, reader_a, reader_b, values = [], [], [], []
    for key, items in sorted(cells.items()):
        cell = []
        for (i,a),(j,b) in combinations(items,2):
            ri, rj = records[i], records[j]
            equal = ri["familiarity"] == rj["familiarity"]
            if a == b or (group.endswith("_same") and not equal) or (group.endswith("_different") and equal):
                continue
            signed = (scores[i]-scores[j])*(a-b)
            correct = np.where(abs(signed)<=1e-10,0.5,signed>0)
            if across:
                groups.append(key)
                reader_a.append(ri["reader"])
                reader_b.append(rj["reader"])
                values.append(correct)
            else:
                cell.append(correct)
        if cell:
            groups.append(key)
            reader_a.append(key[1])
            values.append((len(cell), np.mean(cell,axis=0)))
    if not values:
        return None
    pairs = len(values) if across else sum(n for n,_ in values)
    if not across:
        values = [v for _,v in values]
    return groups,np.array(reader_a),np.array(reader_b),np.array(values),pairs


def boundaries(labels):
    return np.flatnonzero([i==0 or v!=labels[i-1] for i,v in enumerate(labels)])


def division(num, den):
    return np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=den>0)


def bootstrap(info, names, tag, seed):
    groups, ra, rb, values, pairs = info
    cells = boundaries(groups)
    articles = [groups[i][0] for i in cells]
    edges = boundaries(articles)
    n_articles = len(edges)
    readers = sorted(set(ra)|set(rb))
    reader_index = {r:i for i,r in enumerate(readers)}
    ia = [reader_index[r] for r in ra]
    ib = [reader_index[r] for r in rb]
    n_readers = len(readers)

    def aggregate(wr, wa):
        if len(rb):
            weight = wr[:,ia]*wr[:,ib]
            num = np.add.reduceat(weight[:,:,None]*values, cells, axis=1)
            den = np.add.reduceat(weight, cells, axis=1)[:,:,None]
            cellscore = division(num,den)
            weight = np.isfinite(cellscore[:,:,0])
            num = np.nan_to_num(cellscore)
        else:
            weight = wr[:,ia]
            num = weight[:,:,None]*values
        article = division(np.add.reduceat(num,edges,axis=1),np.add.reduceat(weight,edges,axis=1)[:,:,None])
        valid = np.isfinite(article[:,:,0]) * wa
        return division((np.nan_to_num(article)*valid[:,:,None]).sum(1),valid.sum(1)[:,None])

    point = aggregate(np.ones((1,n_readers)),np.ones((1,n_articles)))[0]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(100):
        wr = rng.multinomial(n_readers,np.full(n_readers,1/n_readers),size=100)
        wa = rng.multinomial(n_articles,np.full(n_articles,1/n_articles),size=100)
        draws.append(aggregate(wr,wa))
    boot = np.concatenate(draws)
    finite = np.isfinite(boot).all(1)
    def result(p, b):
        return dict(estimate=float(p),interval95=np.quantile(b[finite],[.025,.975]).tolist() if finite.mean()>=.95 else None)
    stats = {m:result(point[i],boot[:,i]) for i,m in enumerate(names)}
    delta = {}
    for m, other in [("article_prior","lexical_prior"),("article","lexical"),("article_prior","article"),
                     ("article_prior","familiarity_only"),("article_prior","lexical_deft_prior")]:
        if other in names:
            i,j = names.index(m),names.index(other)
            delta[m+" minus "+other] = result(point[i]-point[j],boot[:,i]-boot[:,j])
    return dict(group=tag, articles=n_articles, readers=n_readers, informative_pairs=pairs,
                informative_cells=len(cells),bootstrap_valid=int(finite.sum()),scores=stats,differences=delta)


def run(archive):
    material = load_features(archive)
    geo = json.loads((DATA/"coordinates.json").read_bytes())
    reports = [r for r in rows(DATA/"judgments.jsonl") if r["term"] in material]
    out = {}
    for subset in ("native","extended_inclusive","native_unique_mentions","native_all_readers"):
        recs = [r for r in reports if (r["regular"] or subset=="native_all_readers")
                and (subset=="extended_inclusive" or geo[r["term"]]["tokens"]<=1024)
                and (subset!="native_unique_mentions" or geo[r["term"]]["occurrences"]==1)]
        endpoints = {}
        for endpoint in ("definition_over_example","definition","example"):
            names, pred = forecast(material,recs,endpoint)
            groups = ("within_same","within_all","between_different","between_same")
            for tag in groups:
                info = comparisons(recs,pred,endpoint,tag)
                endpoints[endpoint+"."+tag] = bootstrap(info,names,tag,611290) if info else dict(informative_pairs=0)
        out[subset] = dict(readers=len({r["reader"] for r in recs}),articles=len({r["article"] for r in recs}),
                           term_judgments=len(recs),endpoints=endpoints)
    result=dict(source_forecast=digest(archive/"features.jsonl"),source_weights=FROZEN,subsets=out)
    (OUT/"scores.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(out["native"]["endpoints"]["definition_over_example.within_same"],indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    run(parser.parse_args().archive)
