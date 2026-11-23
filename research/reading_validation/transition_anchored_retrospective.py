"""Evaluate the word-update identity on previously inspected Jones and authored materials."""
from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_ambicoref_prepare import OUT,PROTOCOL,ROOT
from research.reading_validation.transition_anchored_scores import probability,relation_scores,traces
from research.reading_validation.transition_dpr_score import open_hit,pick,same_prefix_type
from research.reading_validation.transition_jones_score import ORDERS,extract,interpretation,interval,pace,read
from research.reading_validation.transition_pointer_features import design
from research.reading_validation.transition_score import dump,sha


def compare_jones(jones,source,rows):
    prior_file=jones/"all-earlier-word-logits.npz";dfile=source/"all-word-utilities.npz"
    p0,b0,surprise,repeat,diff,receipt=extract(jones)
    comparisons=[("native_semantic_update",b) for b in ("native_prior","source_semantic","native_steps_update","native_attention_update","headwise_event","coin")]
    primary={};geometry_guards={}
    for meaning in ("region_mass","mean_word_utility"):
        cols,bels=p0.copy(),b0.copy();maximum=0.;curves={}
        with np.load(prior_file) as wp,np.load(dfile) as d:
            def forecast(identifier):
                geo=rows[identifier]["measurement"];q=geo["q"]
                a=wp[identifier+"/"+str(q)][[1,3],0,:]
                choices={k:probability(relation_scores(x,geo["names"],meaning=="mean_word_utility")) for k,x in traces(a,d[identifier]).items()}
                curves[identifier]={k:v.tolist() for k,v in choices.items()};return choices
            for i,item in enumerate(ORDERS):
                versions=["e1-"+item,"e2-"+item+"_NP1_amb","e2-"+item+"_NP2_amb"]
                z,bl,br=[forecast(v) for v in versions]
                for k,v in z.items():
                    if k not in cols:cols[k]=np.empty(60);bels[k]=np.empty(60)
                    cols[k][i]=v[-1];bels[k][i]=.5*(bl[k][-1]+br[k][-1])
                    maximum=max(maximum,float(abs(bl[k][-1]-br[k][-1])))
        geometry_guards[meaning]=maximum
        group=interpretation(cols,read("anonymous-exp1-script"),False,comparisons)
        no22=interpretation(cols,read("anonymous-exp1-script"),True,comparisons)
        primary[meaning]={"interpretations":group,"without_item22":no22}
        if meaning=="region_mass":
            pts=pace(read("anonymous-exp2"),bels,surprise,repeat,False,[(k,k) for k in ("native_semantic_update","native_attention_update","native_steps_update")])
            primary[meaning]["timing"]=pts
            dump(OUT/"jones-word-posteriors.json",curves)
    return {"calibration":"no fitted parameters","external_prior_location_forecasts":sha(prior_file),"dpr_forecasts":sha(dfile),
            "maximum_future_suffix_effect":geometry_guards,"analyses":primary}


def official(rows,source,weights):
    data={r["id"]:r for r in rows if r["fold"]=="test"}
    grouped=defaultdict(list)
    for r in data.values():grouped[r["group"]].append(r)
    checked=json.loads(Path("research/results/reading-validation-transition-dpr/authored-test-key.json").read_text())
    records=defaultdict(list);prior=np.load(weights)
    with np.load(source/"all-word-utilities.npz") as d:
        for g,twin in grouped.items():
            measurements=[r["measurement"] for r in twin];ys=[checked[r["id"]] for r in twin];readouts=[]
            for row,geo in zip(twin,measurements,strict=True):
                q=geo["q"];X=design(row,q,row["queries"].index(q));p=(X-prior["lexical__mean"])/prior["lexical__sd"]@prior["lexical__beta"]
                # The native prior was never saved for these authored items. Compute only the exact known lexical model.
                readouts.append({k:v for k,v in traces(np.stack([p,p]),d[row["id"]]).items() if not k.startswith("native_")})
            match=same_prefix_type(*twin)
            for k in readouts[0]:
                series=[relation_scores(ro[k],geo["names"])*(2*y-1) for ro,geo,y in zip(readouts,measurements,ys,strict=True)]
                last=np.array([x[-1] for x in series]);correct=pick(last)
                hit=[open_hit(to[k][-1:,:],geo["names"][y])[0] for to,geo,y in zip(readouts,measurements,ys,strict=True)]
                point=np.array([np.logaddexp(0,-last).mean(),correct.mean(),np.prod(correct),np.mean(hit)])
                records["all",k].append(point)
                if match is not None and match[1] is not None:
                    pos,update=match
                    z=np.sum([s[[pos,update,-1]] for s in series],axis=0)*.5
                    keys=["identical_through_query"]
                    if len(series[0])==len(series[1]):keys.append("identical_through_query_equal_length")
                    for subset in keys:records[subset,k].append(np.r_[point,z,pick(np.where(abs(z)<=.00005,0.,z))])
    by={};names=("two_option_nll","individual_accuracy","both_correct","unrestricted_top_word_hit","unchanged_signed_preference",
                 "first_changed_signed_preference","terminal_signed_preference","unchanged_direction","first_changed_direction","terminal_direction")
    for (group,k),v in records.items():
        x=np.stack(v);n=len(x);rng=np.random.default_rng(10029);draws=rng.multinomial(n,np.full(n,1/n),size=4000)
        def describe(z):
            return {name:interval(a,s) for name,a,s in zip(names,z.mean(0),(draws@z/n).T)}
        by.setdefault(group,{})[k]={"groups":n,"scores":describe(x)}
        if k=="lexical_semantic_update":
            for other in ("lexical_prior","source_semantic","lexical_steps_update","lexical_attention_update"):
                by[group][k]["minus_"+other]=describe(x-np.stack(records[group,other]))
    return by


def main(jones,source,pointer):
    receipt=json.loads((source/"receipt.json").read_text());p=ROOT/"dpr-pointer-stimuli.jsonl";w=pointer/"source-weights.npz"
    if (sha(source/"all-word-utilities.npz")!=receipt["forecast_sha256"] or sha(p)!=receipt["input_sha256"]
            or sha(w)!="9162700762a7574f45532bac88f8c018b1f2922118ac033bd1206a1edaecfaf5"):raise ValueError("Registered source data or readouts changed")
    rows=list(map(json.loads,p.read_text().splitlines()));lookup={r["id"]:r for r in rows}
    dump(OUT/"jones-retrospective.json",{"protocol_sha256":sha(PROTOCOL),**compare_jones(jones,source,lookup)})
    dump(OUT/"dpr-retrospective.json",{"protocol_sha256":sha(PROTOCOL),"lexical_click_prior_sha":sha(w),"scores":official(rows,source,w)})
    print("Retrospective evidence change scored before opening the independent human cohort",flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for nm in ("jones","source","pointer"):p.add_argument("--"+nm,type=Path,required=True)
    main(**vars(p.parse_args()))
