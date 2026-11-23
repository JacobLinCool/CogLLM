"""Score whether abstaining before reference prediction recovers real reader choices."""

from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_gate_prepare import DEST, PROTOCOL, ROOT
from research.reading_validation.transition_pointer_selective import calibrate, evaluate
from research.reading_validation.transition_score import dump, sha


def information(rows, constant):
    groups, idx, draws = grouped_distribution([r["doc"] for r in rows])
    records = {}
    matrix = {}
    y=np.array([r["prior"] for r in rows])
    for method in ("constant", "surface", "native"):
        p = np.full(len(y),constant) if method=="constant" else np.array([r[method] for r in rows])
        if np.any((p<=0)|(p>=1)):
            raise ValueError("A gate must provide finite probabilities on both actions")
        items=np.column_stack((y*(1-p)**2+(1-y)*p**2,-y*np.log(p)-(1-y)*np.log1p(-p),p-y))
        matrix[method]=np.stack([items[idx==i].mean(0) for i in range(len(groups))])
    for name, array in [*matrix.items(),("native_minus_surface",matrix["native"]-matrix["surface"]),
                       ("native_minus_constant",matrix["native"]-matrix["constant"])]:
        intervals=np.quantile(draws@array/len(groups),[.025,.975],axis=0)
        records[name]={k:[float(m), *intervals[:,j].tolist()] for j,(k,m) in
                       enumerate(zip(("reader_brier","reader_logloss","signed_calibration_error"),array.mean(0),strict=True))}
    return records


def observations(docs, forecast, labels):
    observed=[]; all_sites=[]
    for doc,r in docs.items():
        for q in r["queries"]:
            f=forecast[doc+"/"+str(q)]
            prior=f["conditional_confidence"]
            s={"doc":doc,"work":r["group"],"q":q,"raw":prior,"native_product":prior*f["native"],
               "surface_product":prior*f["surface"],"native":f["native"],"surface":f["surface"]}
            if str(q) in labels[doc]:
                actions=labels[doc][str(q)]["actions"]
                n=sum(count for _,_,_,count in actions)
                s.update(n=n,prior=sum(n for k,a,b,n in actions if k=="DO")/n,
                         hit=sum(n for k,a,b,n in actions if k=="DO" and a<=f["top"]<b)/n)
                observed.append(s)
            all_sites.append(s)
    return observed,all_sites


def eval_rows(observed, method, group="doc"):
    return [(r[group], r[method], r["hit"],r["n"]) for r in observed]


def bounds(all_sites, method, tau):
    selected=[r for r in all_sites if r[method]>=tau]
    if not selected:
        return {"selected":0,"total":len(all_sites),"unadjudicated":0,"precision":None,"wrong_per_100":[0.,0.]}
    unknown=sum("hit" not in r for r in selected)
    lo=sum(r.get("hit",0.) for r in selected); hi=lo+unknown; n=len(selected)
    return {"selected":n,"total":len(all_sites),"unadjudicated":unknown,"precision":[lo/n,hi/n],
            "wrong_per_100":[100*(n-hi)/len(all_sites),100*(n-lo)/len(all_sites)]}


def main(folder):
    seal=json.loads((DEST/"source-seal.json").read_text())
    receipt=json.loads((folder/"receipt.json").read_text()); states=folder/"word-forecasts.json"
    packet=ROOT/"gate-pointer-stimuli.jsonl"
    if (sha(states)!=receipt["forecasts_sha256"] or sha(folder/"gate-weights.npz")!=receipt["weights_sha256"]
            or sha(PROTOCOL)!=seal["protocol_sha256"] or sha(packet)!=seal["input_sha256"]
            or receipt["input_sha256"]!=seal["input_sha256"] or receipt["protocol_sha256"]!=sha(PROTOCOL)):
        raise ValueError("Source inputs or forecasts changed")
    all_docs={r["id"]:r for r in map(json.loads,packet.read_text().splitlines())}
    forecasts=json.loads(states.read_text()); dev={k:r for k,r in all_docs.items() if r["fold"]=="dev"}
    valid,_=observations(dev,forecasts,{k:r["actions"] for k,r in dev.items()})
    methods=("raw","surface_product","native_product")
    source={m:calibrate(eval_rows(valid,m,"work")) for m in methods}
    dev_prior=float(np.mean([np.mean([r["prior"] for r in valid if r["work"]==g]) for g in sorted({r["work"] for r in valid})]))
    dump(DEST/"selection.json",{"forecast_sha256":sha(states),"protocol_sha256":sha(PROTOCOL),
                                "all_action_source_dev_prior":dev_prior,"methods":source})
    gold_file=DEST/"external-actions.json"
    if sha(gold_file)!=seal["gold_sha256"]:
        raise ValueError("Held actions changed")
    test={k:r for k,r in all_docs.items() if r["fold"]=="test"}
    observed,sites=observations(test,forecasts,json.loads(gold_file.read_text()))
    if len(observed)!=631 or sum(r["n"] for r in observed)!=4880 or len(sites)!=757:
        raise ValueError("All-reader-action denominator changed")
    scored={}
    for m in methods:
        chosen=source[m]["selected"]; rules={"forced":0.,"fixed_80":.8}
        if chosen:rules["source_selected"]=chosen["threshold"]
        scored[m]={k:{"articles":evaluate(eval_rows(observed,m),t)[0],
                      "works":evaluate(eval_rows(observed,m,"work"),t)[0],"all_visible":bounds(sites,m,t)}
                   for k,t in rules.items()}
    dump(DEST/"scores.json",{"source_selection_sha256":sha(DEST/"selection.json"),"accelerator_receipt":receipt,
          "selection":scored,"action_probabilities":information(observed,dev_prior)})
    print({k:None if v["selected"] is None else v["selected"]["threshold"] for k,v in source.items()},flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--features",type=Path,required=True)
    main(p.parse_args().features)
