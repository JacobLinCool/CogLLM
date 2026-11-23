"""Score every observed AmbiCoref response in its original direction and certainty category."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from itertools import combinations
import json

import numpy as np

from research.reading_validation.transition_ambicoref_prepare import CATEGORIES,OUT,PROTOCOL,ROOT
from research.reading_validation.transition_anchored_scores import probability,relation_scores,traces
from research.reading_validation.transition_score import dump,sha


BOOT=4000
PRIMARY="native_semantic_update"
CONTROLS=("native_prior","source_semantic","native_steps_update","native_attention_update","source_surface","lexical_prior")
METRICS=("directional_logloss","directional_brier","directional_agreement","explicit_ambiguity_auroc","explicit_ambiguity_ap","explicit_ambiguity_prevalence")


def rank_order(score):
    ix=np.argsort(-score,kind="stable")
    return ix,np.r_[0,np.flatnonzero(np.diff(score[ix])!=0)+1]


def ranking(ix,starts,positive,negative):
    yp=np.add.reduceat(positive[:,ix],starts,axis=1);yn=np.add.reduceat(negative[:,ix],starts,axis=1)
    t=yp.cumsum(1);n=yn.cumsum(1)
    au=((t-.5*yp)*yn).sum(1)/(t[:,-1]*n[:,-1])
    top=np.divide(t,t+n,out=np.zeros_like(t),where=(t+n)>0)
    return au,(top*yp).sum(1)/t[:,-1]


def metrics(p,y,order):
    if p.ndim!=1 or len(p)!=y.shape[1] or y.ndim!=3 or y.shape[2]!=5:
        raise ValueError("Metrics require five independent human response categories at each displayed sentence")
    if not ((p>0)&(p<1)).all():raise ValueError("An observer assigned impossible directional certainty")
    a=y[:,:,0]+y[:,:,1];b=y[:,:,3]+y[:,:,4];u=y[:,:,2];n=(a+b).sum(1)
    point=np.sign(p-.5)/2+.5
    au,ap=ranking(*order,u,a+b)
    v=np.column_stack((-(a@np.log1p(-p)+b@np.log(p))/n,(a@np.square(p)+b@np.square(1-p))/n,
                       (b@point+a@(1-point))/n,au,ap,u.sum(1)/y.sum((1,2))))
    # Four mutually exclusive ordinal categories plus the explicit uncertainty action remain intact.
    counts=y.sum(1)
    mass=np.einsum("bsc,s->bc",y,p)
    uncertainty=np.einsum("bsc,s->bc",y,4*p*(1-p))
    return v,np.divide(mass,counts,out=np.zeros_like(mass),where=counts>0),np.divide(uncertainty,counts,out=np.zeros_like(mass),where=counts>0)


def summarize(z,burn):
    return [float(z),*np.quantile(burn,[.025,.975]).tolist()]


def crossed(geometry,models,family_equal=False):
    cells=list(geometry);index={v:i for i,v in enumerate(cells)}
    families=sorted({geometry[s]["family"] for s in cells});fid={s:i for i,s in enumerate(families)}
    owner=np.array([fid[geometry[s]["family"]] for s in cells]);group_counts=np.bincount(owner)
    events=sorted([(index[key],sheet,choice) for key,g in geometry.items() for sheet,choice in g["questionnaire_choices"]])
    items,assignment,category=np.asarray(events,dtype=int).T;sheet_ids,assign=np.unique(assignment,return_inverse=True)
    boundaries=np.r_[0,np.flatnonzero(np.diff(items))+1]
    if len(boundaries)!=len(cells):raise ValueError("Observed case lacked any original data")
    def measures(sheets,group):
        # Each sentence keeps equal mass regardless of the number of questionnaires where it appeared.
        weights=sheets[:,assign];counts=np.stack([np.add.reduceat(weights*(category==j),boundaries,axis=1) for j in range(5)],-1)
        norm=counts.sum(2,keepdims=True);v=counts/norm
        iw=group[:,owner]/group_counts[owner] if family_equal else group[:,owner]
        return v*iw[:,:,None]
    predictions={m:np.array([z[c] for c in cells]) for m,z in models.items()}
    orders={m:rank_order(4*v*(1-v)) for m,v in predictions.items()}
    y0=measures(np.ones((1,len(sheet_ids))),np.ones((1,len(families))))
    points={m:metrics(p,y0,orders[m]) for m,p in predictions.items()}
    rng=np.random.default_rng(91445);distribution={m:[] for m in predictions};ordinal={m:[] for m in predictions}
    for size in [min(64,BOOT-i) for i in range(0,BOOT,64)]:
        sw=rng.exponential(size=(size,len(sheet_ids)));gw=rng.exponential(size=(size,len(families)))
        y=measures(sw,gw)
        for m,p in predictions.items():
            values,prob,variance=metrics(p,y,orders[m]);distribution[m].append(values);ordinal[m].append(np.stack((prob,variance),axis=2))
    distributions={m:np.concatenate(z) for m,z in distribution.items()}
    stats={m:{k:summarize(points[m][0][0,j],distributions[m][:,j]) for j,k in enumerate(METRICS)} for m in predictions}
    comparison={}
    for m in CONTROLS:
        if m not in predictions:continue
        point=points[PRIMARY][0][0]-points[m][0][0];draws=distributions[PRIMARY]-distributions[m]
        comparison[PRIMARY+"_minus_"+m]={k:summarize(point[j],draws[:,j]) for j,k in enumerate(METRICS)}
    ords={}
    for m,p in points.items():
        d=np.concatenate(ordinal[m]);ords[m]={c:{"mean_predicted_second":summarize(p[1][0,i],d[:,i,0]),
                                                  "mean_forecast_variance":summarize(p[2][0,i],d[:,i,1])} for i,c in enumerate(CATEGORIES)}
    return {"sentences":len(cells),"lexical_skeletons":len(families),"questionnaires":len(sheet_ids),"judgments":len(events),
            "ordinal_response_counts":dict(zip(CATEGORIES,np.bincount(category,minlength=5).tolist(),strict=True)),
            "measurements":stats,"paired":comparison,"five_way_response_profiles":ords}


def descriptive(geometry,models):
    groups=defaultdict(list)
    for k,g in geometry.items():
        condition=g["condition"];suffix=g["k"]-g["q"]
        groups["terminal_pronoun" if not suffix else "has_later_evidence"].append(k)
        groups["after_pronoun_0" if not suffix else "after_pronoun_1_to_3" if suffix<=3 else "after_pronoun_over_3"].append(k)
        groups["declared_source_"+condition].append(k)
        if condition!="unmatched":groups["construct_"+condition.split("_")[0].split("-")[0]].append(k)
        if all(len(z)==1 for z in g["names"]):groups["two_single_word_names"].append(k)
    out={}
    for name,keys in groups.items():
        raw=np.stack([np.bincount([c for _,c in geometry[k]["questionnaire_choices"]],minlength=5) for k in keys])
        y=(raw/raw.sum(1,keepdims=True))[None,:,:]
        sub={}
        for m in (PRIMARY,*CONTROLS):
            p=np.array([models[m][k] for k in keys])
            if min(y[:,:,2].sum(),y[:,:,[0,1,3,4]].sum())<=0:continue
            v,_,_=metrics(p,y,rank_order(4*p*(1-p)))
            sub[m]={k:float(x) for k,x in zip(METRICS,v[0],strict=True)}
        out[name]={"sentences":len(keys),"decisions":int(raw.sum()),"methods":sub}
    return out


def exact_pairs(geometry,curves):
    groups=defaultdict(list)
    for k,g in geometry.items():
        if g["condition"]=="unmatched":continue
        vocab=g["visible_words"];q=g["q"];options=tuple(tuple(ix) for ix in g["names"]);construct=g["condition"].split("_")[0]
        for t in range(q+1,len(vocab)):
            groups[(construct,q,options,t,tuple(vocab[:t]),tuple(vocab[t+1:]))].append(k)
    out=[];seen=set()
    for system,items in groups.items():
        for a,b in combinations(items,2):
            if tuple(sorted((a,b))) in seen:continue
            if geometry[a]["condition"].endswith("_ambiguous")==geometry[b]["condition"].endswith("_ambiguous"):continue
            if not geometry[a]["condition"].endswith("_ambiguous"):a,b=b,a
            seen.add(tuple(sorted((a,b))));t=system[3]-system[1]
            z={}
            def obs(i):
                aa=np.bincount([c for _,c in geometry[i]["questionnaire_choices"]],minlength=5)
                return aa[2]/aa.sum()
            for m in (PRIMARY,"native_prior","native_attention_update","source_semantic"):
                x=curves[a][m];y=curves[b][m]
                z[m]=[float(np.max(abs(x[:t]-y[:t]))),float(4*(x[t]*(1-x[t])-y[t]*(1-y[t]))),
                      float(4*(x[-1]*(1-x[-1])-y[-1]*(1-y[-1])))]
            out.append({"ambiguous":a,"unambiguous":b,"changed_word":system[3],"human_explicit_ambiguity_difference":float(obs(a)-obs(b)),"models":z})
    return {"matched_pairs":len(out),"unique_sentences":len({s for p in seen for s in p}),"rows":out}


def main(folder):
    lock=json.loads((OUT/"external-seal.json").read_text());receipt=json.loads((folder/"receipt.json").read_text())
    gold=ROOT/"ambicoref-observed-choices.json";stim=ROOT/"ambicoref-reading-stimuli.jsonl"
    pw=folder/"prior-words.npz";du=folder/"evidence-words.npz"
    if (sha(PROTOCOL)!=lock["protocol_sha256"] or sha(stim)!=lock["input_sha256"] or receipt["input_sha256"]!=sha(stim)
        or receipt["protocol_sha256"]!=sha(PROTOCOL) or sha(gold)!=lock["outcomes_sha256"] or sha(pw)!=receipt["prior_forecast_sha256"]
        or sha(du)!=receipt["evidence_forecast_sha256"] or receipt["encoder_forwards"]!=lock["included_sentences"]
        or receipt["training_records"] or receipt["external_response_fields"]):raise ValueError("External five-way study provenance differs")
    geometry=json.loads(gold.read_text());models=defaultdict(dict);means=defaultdict(dict);curves={}
    with np.load(pw) as p,np.load(du) as d:
        if set(p.files)!=set(geometry) or set(d.files)!=set(geometry):raise ValueError("Missing external readings")
        for k,g in geometry.items():
            if p[k].shape[-1]!=g["q"] or d[k].shape[1]!=g["k"]-g["q"]+1:raise ValueError("Evidence alignment changed")
            curves[k]={}
            for m,z in traces(p[k],d[k]).items():
                trace=probability(relation_scores(z,g["names"]));curves[k][m]=trace
                models[m][k]=float(trace[-1]);means[m][k]=float(probability(relation_scores(z[-1],g["names"],True)))
    confirmed={k:models[k] for k in (PRIMARY,*CONTROLS,"lexical_semantic_update","source_attention")}
    direct_conditions=json.loads((OUT/"design-conditions.json").read_text())
    if direct_conditions["external_seal_sha256"]!=sha(OUT/"external-seal.json"):raise ValueError("Independent design categories changed")
    documented={k:{**g,"condition":direct_conditions["assignments"][k]} for k,g in geometry.items()}
    result={"receipt":receipt,"external_lock_sha256":sha(OUT/"external-seal.json"),"sentence_equal":crossed(geometry,confirmed),
            "lexical_family_equal":crossed(geometry,confirmed,True),"averaged_word_utility":crossed(geometry,{k:means[k] for k in (PRIMARY,*CONTROLS)}),
            "breakdowns":descriptive(geometry,models),"exact_matched_word_changes":exact_pairs(geometry,curves),
            "documented_verb_class_breakdowns":descriptive(documented,models),"documented_verb_class_word_changes":exact_pairs(documented,curves)}
    dump(OUT/"ambicoref-confirmation.json",result)
    print(result["sentence_equal"]["judgments"],result["sentence_equal"]["measurements"][PRIMARY],flush=True)


if __name__=="__main__":
    from pathlib import Path
    p=ArgumentParser(description=__doc__);p.add_argument("--features",type=Path,required=True);main(p.parse_args().features)
