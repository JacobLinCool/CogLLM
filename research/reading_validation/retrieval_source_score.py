"""Score retrieval from independently written source triples and human-marked news spans."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from research.core_features.readback_score import average_precision
from research.reading_validation.memory_data import load_rows, sha
from research.reading_validation.retrieval_data import DATA, RESULT, TRACK


def interval(x):
    return dict(estimate=float(x[0]), interval95=np.quantile(x[1:], [.025, .975]).tolist())


def summarize(records, seed, focus="semantic", equal_cluster=True):
    """Report equal-unit means and paired intervals accounting for reused material."""
    if not records:
        return dict(groups=0, units=0)
    groups = sorted(records)
    methods = tuple(records[groups[0]][0])
    values = []
    for group in groups:
        matrix = np.array([[row[m] for m in methods] for row in records[group]])
        if not np.isfinite(matrix).all():
            raise ValueError("Nonfinite comparison")
        values.append(matrix.mean(0) if equal_cluster else matrix.sum(0))
    sizes = np.array([1 if equal_cluster else len(records[g]) for g in groups])
    n = len(groups)
    rng = np.random.default_rng(seed)
    draws = np.vstack((np.ones(n), rng.multinomial(n, np.full(n,1/n), size=10_000)))
    estimates = (draws @ np.stack(values))/(draws @ sizes)[:,None]
    primary = estimates[:,methods.index(focus)]
    return dict(groups=n, units=sum(map(len,records.values())),
                methods={m:interval(x) for m,x in zip(methods,estimates.T,strict=True)},
                contrasts={m:interval(primary-x) for m,x in zip(methods,estimates.T,strict=True) if m!=focus})


def tie_hit(scores, gold):
    a = np.array(scores)
    winners = a == a.max()
    return float(winners[gold] / winners.sum())


def source_results(forecast):
    lookup = {r["id"]:r["methods"] for r in forecast}
    sets = {}
    for split in ("dev","test"):
        items = [r for r in json.loads((DATA/"training.json").read_bytes()) if r["split"]==split]
        if not items:
            raise ValueError(f"No held-out triples for {split}")
        ranking, pointers = defaultdict(list),defaultdict(list)
        for r in items:
            predictions = [lookup[key] for key in r["ids"]]
            gold = r["y"].index(1)
            rankrow = {"chance":1/3}
            for name in ("lexical","semantic"):
                scores = [v[name]["conflict"] for v in predictions]
                rankrow[name] = float(all(scores[gold]>z for i,z in enumerate(scores) if i!=gold))
            ranking[r["family"]].append(rankrow)
            if r["n"]:
                ref, choices = r["evidence"], r["n"]+1
                for image in predictions:
                    row = {m:tie_hit(v["source"],ref) for m,v in image.items()}
                    row.update(nearest=float(ref==choices-1),first=float(ref==0),chance=1/choices)
                    pointers[r["family"]].append(row)
        sets[split] = dict(strict_rank=summarize(ranking,12222),source_hit=summarize(pointers,12223))
    return sets


def news_results(folder):
    directory = TRACK/"scarecrow-detection/prepared"
    layout = {r["id"]:r for r in load_rows(directory/"layout.jsonl") if r["split"]=="test"}
    outcomes = {r["id"]:r["counts"] for r in load_rows(directory/"outcomes.jsonl") if r["id"] in layout}
    historic = {r["id"]:r for r in load_rows(Path("research/results/scarecrow-detection-l4/forecasts.jsonl"))}
    annotations = load_rows(TRACK/"source/outcomes.jsonl")
    local, links = defaultdict(list),defaultdict(list)
    counts=Counter()
    pooled=defaultdict(list)
    method_cols = {"original_nli":2, "surprisal":3, "lexical":4, "semantic":5}
    with np.load(folder/"news.npz",allow_pickle=False) as dense:
        for key,row in layout.items():
            v=dense[key]
            ix_by_span={tuple(map(int,entry)):i for i,entry in enumerate(v[:,:2])}
            spans=list(map(tuple,row["words"]))
            if historic[key]["words"]!=row["words"] or any(s not in ix_by_span for s in spans):
                raise ValueError("A source or model word mismatches the eligible locations")
            ix=np.array([ix_by_span[s] for s in spans])
            totals=np.array([e["total"] for e in outcomes[key]])
            keep=totals>0
            yes=np.array([e["positive"]>0 for e in outcomes[key]])[keep]
            predictions={m:v[ix,col][keep] for m,col in method_cols.items()}
            for name in ("qwen","surface"):
                predictions["human_trained_"+name]=1-np.asarray(historic[key]["probabilities"][name])[keep,0]
            predictions["constant"]=np.zeros(len(yes))
            pooled["observed"].append(yes)
            for m,z in predictions.items():
                pooled[m].append(z)
            if yes.any() and not yes.all():
                local[row["group_id"]].append({m:average_precision(z,yes) for m,z in predictions.items()})
            counts["held_continuations"]+=1
            counts["eligible_words"]+=int(keep.sum())
            counts["marked_words"]+=int(yes.sum())
        for item in annotations:
            if item["split"]!="test":
                continue
            key=item["id"]
            if key not in layout or layout[key]["group_id"]!=item["group_id"]:
                raise ValueError("News split differs between pointing and word coverage")
            counts["held_pointer_marks"]+=1
            v=dense[key]
            hit=np.flatnonzero(v[:,1]==item["stop"])
            if len(hit)!=1:
                counts["nonendpoint_marks"]+=1
                continue
            i=hit.item()
            unit=int(v[i,7])
            linked={int(u) for a,b,u in v[:,[0,1,7]] if a<item["source_stop"] and b>item["source_start"] and u<unit}
            if not linked or unit<2:
                counts["insufficient_earlier_candidates"]+=1
                continue
            options={"lexical":8,"semantic":9,"native_attention":10,"word_overlap":11}
            picks={m:int(v[i,j]) for m,j in options.items()}
            if any(not 0<=q<unit for q in picks.values()):
                raise ValueError("Noncausal source pointer")
            result={m:float(pick in linked) for m,pick in picks.items()}
            result.update(nearest=float(unit-1 in linked),chance=len(linked)/unit)
            links[item["group_id"]].append(result)
    labels=np.concatenate(pooled.pop("observed"))
    aggregate={m:average_precision(np.concatenate(values),labels) for m,values in pooled.items()}
    return dict(inventory=dict(counts),pooled_word_average_precision=aggregate,
                within_mixed_continuation_ap=summarize(local,32703,equal_cluster=False),
                earlier_sentence_hit=summarize(links,32704))


def run(folder,dest):
    seal=json.loads((folder/"seal.json").read_bytes())
    if any(sha(folder/f)!=v for f,v in seal["files"].items()):
        raise ValueError("Forecasts no longer match the lock")
    results=dict(forecast=sha(folder/"seal.json"),source=source_results(json.loads((folder/"source.json").read_bytes())),
                 news=news_results(folder))
    dest.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    print("Held source and news endpoints scored",results["news"]["inventory"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,default=RESULT/"readout")
    parser.add_argument("--dest",type=Path,default=RESULT/"source-news-scores.json")
    run(**vars(parser.parse_args()))
