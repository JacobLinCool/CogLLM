"""Score the externally observed terminal interpretation and its change across twin clues."""

from argparse import ArgumentParser
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_dpr_prepare import OUT,PROTOCOL,ROOT
from research.reading_validation.transition_jones_prepare import OUT as JONES
from research.reading_validation.transition_jones_score import ORDERS,extract,interpretation,interval,logistic,pace,read
from research.reading_validation.transition_score import dump,sha


ARMS=("surface","steps","attention","semantic")


def mass(z,g):
    a,b=g["names"]
    return z[:,:,b].mean(axis=-1,dtype=float)-z[:,:,a].mean(axis=-1,dtype=float)


def pick(z):
    return (np.sign(z)+1)/2


def open_hit(z,words):
    mask=z==z.max(axis=1,keepdims=True)
    return mask[:,words].sum(1)/mask.sum(1)


def reduce(vector):
    if not vector:return {"pairs":0}
    ary=np.stack(vector);point=ary.mean(0);rng=np.random.default_rng(8325)
    draws=rng.multinomial(len(ary),np.full(len(ary),1/len(ary)),size=4000)
    boot=draws@ary/len(ary)
    rows={a:interval(point[j],boot[:,j]) for j,a in enumerate(ARMS)}
    paired={"semantic_minus_"+arm:interval(point[-1]-point[j],boot[:,-1]-boot[:,j]) for j,arm in enumerate(ARMS[:-1])}
    return {"pairs":len(ary),"arms":rows,"paired":paired}


def forms(row):
    return [row["text"][a:b] for a,b in row["words"]]


def same_prefix_type(a,b):
    ga,gb=a["measurement"],b["measurement"];qa,qb=ga["q"],gb["q"]
    if (qa!=qb or a["text"][:a["words"][qa][1]]!=b["text"][:b["words"][qb][1]]
            or sorted(ga["names"])!=sorted(gb["names"])):
        return None
    w1,w2=forms(a),forms(b)
    identical=qa
    for k in range(qa+1,min(len(w1),len(w2))):
        if a["text"][:a["words"][k][1]]!=b["text"][:b["words"][k][1]]:break
        identical=k
    # Censor a pair when finishing the shorter sentence is the first difference.
    if identical+1>=min(len(w1),len(w2)):return (identical-qa,None)
    return identical-qa,identical+1-qa


def official(values,records,external):
    fam=defaultdict(list);v=defaultdict(list);audit=Counter();max_equal_delta=np.zeros(len(ARMS));max_equal_probability=np.zeros(len(ARMS))
    for r in records:
        fam[r["group"]].append(r)
    for twin in fam.values():
        if len(twin)!=2:raise ValueError("Incomplete official source test family")
        rs=sorted(twin,key=lambda x:x["id"])
        signed=[];forecast=[];hit=[];preference=[]
        for r in rs:
            ident=r["id"];g=r["measurement"];z=values[ident]
            if z.shape!=(len(ARMS),g["k"]-g["q"]+1,g["q"]):
                raise ValueError("All original previous words and subsequent readings are required")
            y=external[ident]
            m=mass(z,g);signed.append((2*y-1)*m);forecast.append(z)
            preference.append(r["identities"][y]);hit.append(open_hit(z[:,-1],g["names"][y]))
        if preference[0]==preference[1]:audit["same_target_pairs"]+=1
        pair_terminal=(signed[0][:,-1]+signed[1][:,-1])/2
        terminal=np.stack([pick(s[:,-1]) for s in signed])
        for key,w in (("both_terminal_choices",terminal.prod(0)),("individual_terminal_accuracy",terminal.mean(0)),
                      ("terminal_logloss",np.mean([np.logaddexp(0,-s[:,-1]) for s in signed],0)),
                      ("all_prior_word_top1",np.mean(hit,0)),("signed_terminal_score",pair_terminal)):
            v["official_all/"+key].append(w)
        same=same_prefix_type(*rs)
        if same is None or preference[0]==preference[1]:continue
        pos,update=same; tag="identical_prefix_opposite_target"
        for k,l in (("all",tag),("equal_length",tag+"_equal_length")):
            if k=="equal_length" and signed[0].shape[1]!=signed[1].shape[1]:continue
            l+="/"
            v[l+"both_terminal_choices"].append(terminal.prod(0))
            v[l+"terminal_logloss"].append(np.mean([np.logaddexp(0,-s[:,-1]) for s in signed],0))
            for name,t in (("pronoun",0),("last_unchanged_word",pos),("first_changed_word",update),("terminal",-1)):
                if t is None:continue
                z=.5*(signed[0][:,t]+signed[1][:,t])
                v[l+name+"_signed_preference"].append(z)
                v[l+name+"_preferred_correct_direction"].append(pick(z))
                v[l+name+"_two_correct"].append(pick(signed[0][:,t])*pick(signed[1][:,t]))
                if name in ("pronoun","last_unchanged_word"):
                    # This additional numerical diagnostic treats a <1e-4 log-odds discrepancy as tied.
                    v[l+name+"_direction_with_numerical_ties"].append(pick(np.where(np.abs(2*z)<=1e-4,0.,z)))
                if name=="last_unchanged_word":
                    max_equal_delta=np.maximum(max_equal_delta,np.abs(2*z))
                    max_equal_probability=np.maximum(max_equal_probability,np.abs(logistic(signed[0][:,t])-logistic(-signed[1][:,t])))
            if update is not None:
                v[l+"first_change_minus_unchanged"].append(.5*sum(s[:,update]-s[:,pos] for s in signed))
            v[l+"terminal_minus_unchanged"].append(.5*sum(s[:,-1]-s[:,pos] for s in signed))
            if k=="all":audit["same_prefix_pairs"]+=1
    return {"official_families":len(fam),"audit":dict(audit),"predivergence_logodds_tie_sensitivity":1e-4,
            "max_predivergence_probability_difference":dict(zip(ARMS,max_equal_probability.tolist())),"max_predivergence_score_difference":dict(zip(ARMS,max_equal_delta.tolist())),
            "endpoints":{name:reduce(vs) for name,vs in v.items()}}


def main(features,original):
    seal=json.loads((OUT/"source-seal.json").read_text());receipt=json.loads((features/"receipt.json").read_text())
    packet=ROOT/"dpr-pointer-stimuli.jsonl";pf=features/"all-word-utilities.npz"
    if (sha(PROTOCOL)!=receipt["protocol_sha256"] or sha(pf)!=receipt["forecast_sha256"] or sha(packet)!=seal["input_sha256"]
            or receipt["input_sha256"]!=sha(packet) or receipt["external_supervision"]!=0 or receipt["arms"]!=list(ARMS)
            or sha(features/"source-semantic-weights.npz")!=receipt["weights_sha256"]):raise ValueError("Source-trained or external forecasts changed")
    key=OUT/"authored-test-key.json"
    if sha(key)!=seal["external_key_sha256"]:raise ValueError("An original test target changed")
    rows={r["id"]:r for r in map(json.loads,packet.read_text().splitlines())}
    orig,belief,surprise,repeat,original_delta,base_receipt=extract(original)
    additions=[a+"_"+s for a in ARMS for s in ("q","k")]
    cols=orig.copy();b=belief.copy()
    for label in additions:
        cols["dpr_"+label]=np.empty(60);b["dpr_"+label]=np.empty(60)
    maxdelta=0.
    with np.load(pf) as values:
        official_result=official(values,[r for r in rows.values() if r["fold"]=="test"],json.loads(key.read_text()))
        for i,k in enumerate(ORDERS):
            e1="e1-"+k;e2=["e2-"+k+"_"+s+"_amb" for s in ("NP1","NP2")]
            scores=lambda key: mass(values[key],rows[key]["measurement"] )[:,[0,-1]]
            p1=logistic(scores(e1));p2=np.stack([logistic(scores(key)) for key in e2])
            maxdelta=max(maxdelta,float(np.abs(p2[0]-p2[1]).max()))
            for j,a in enumerate(ARMS):
                for n,t in enumerate(("q","k")):
                    cols["dpr_"+a+"_"+t][i]=p1[j,n]
                    b["dpr_"+a+"_"+t][i]=p2.mean(0)[j,n]
    if maxdelta>.025:raise ValueError("Large future influence before disambiguating continuation")
    public={"protocol_sha256":sha(PROTOCOL),"receipt":receipt,"jones_frozen_baseline_sha256":base_receipt["forecast_sha256"],
           "max_same_prefix_posterior_difference":maxdelta}
    dump(OUT/"authored-pairs.json",{**public,**official_result})
    extra=[("dpr_semantic_k","dpr_"+s+"_k") for s in ARMS[:-1]]
    comparisons=extra+[("dpr_semantic_k",x) for x in ("dpr_semantic_q","headwise_event","published_gpt3_completion","coin")]
    main1=read("anonymous-exp1-script");main2=read("anonymous-exp2")
    dump(OUT/"independent-reader-choices.json",{**public,"script":interpretation(cols,main1,False,comparisons),
              "without_item_22":interpretation(cols,main1,True,comparisons),
              "paper_exclusion_rules":interpretation(cols,read("anonymous-exp1-paper"),False,comparisons)})
    extras=[("plus_dpr_"+a,"dpr_"+a+"_k") for a in ARMS]
    dump(OUT/"independent-reader-timing.json",{**public,"script":pace(main2,b,surprise,repeat,False,extras),
         "without_item_22":pace(main2,b,surprise,repeat,True,extras),"paper_exclusion_rules":pace(read("anonymous-exp2-paper-rules"),b,surprise,repeat,False,extras)})
    dump(OUT/"item-forecasts.json",{"orders":ORDERS,"standalone":{k:v.tolist() for k,v in cols.items()},
                                   "full_passage":{k:v.tolist() for k,v in b.items()}})
    print("Validated the held authored twins and both independent human-response experiments",flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for name in ("features","original"):p.add_argument("--"+name,type=Path,required=True)
    main(**vars(p.parse_args()))
