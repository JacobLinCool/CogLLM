"""Test a game-trained pointer against experimental interpretation and later reading pace."""

from argparse import ArgumentParser
import csv
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_architecture_prepare import load_tokenizer
from research.reading_validation.transition_jones_prepare import OUT, ROOT, PROTOCOL
from research.reading_validation.transition_score import dump, sha


BOOT = 4000
ARMS = ("recency", "lexical", "pooled", "headwise")
ORDERS = [str(i)+"_"+o for i in range(1,31) for o in ("A","B")]
CONDITIONS = ("NP1_amb","NP1_un","NP2_amb","NP2_un")
# [ambiguous - explicit] for NP2 minus [ambiguous - explicit] for NP1.
DIF = np.array([-1,1,1,-1])
REGIONS = ["rt_cont_p3","rt_cont_p2","rt_cont_p1","rt_cont","rt_cont_sp1","rt_cont_sp2","rt_cont_sp3"]


def read(name):
    with (ROOT/("jones-"+name+".csv")).open(newline="") as f:
        return list(csv.DictReader(f))


def logistic(z):
    return np.exp(-np.logaddexp(0.,-np.asarray(z)))


def get_belief(x, indices):
    likelihood = [np.logaddexp.reduce(x[..., i],axis=-1) for i in indices]
    return logistic(likelihood[1]-likelihood[0])


def study_controls():
    priors={r["item_id"]:r for r in read("exp2-biases")}
    estimates={r["item_id"]:r for r in read("exp1-gpt3")}
    cols={"coin":np.full(60,.5)}
    for mode,nfield in (("physics","phys_NP2_mean"),("syntax","syn_NP2_mean")):
        counts=np.array([float(priors[x]["n_"+mode]) for x in ORDERS])
        prop=np.array([float(priors[x][nfield]) for x in ORDERS])
        cols[mode]=(counts*prop+.5)/(counts+1.)
    cols["published_gpt3_completion"]=logistic(np.array([
        float(estimates[x]["gpt3_davinci_partial_lp_target_a"])-float(estimates[x]["gpt3_davinci_partial_lp_target_b"])
        for x in ORDERS]))
    return cols


def sampling(npeople,nfamily,seed,strata=None):
    rng=np.random.default_rng(seed)
    fw=rng.exponential(size=(BOOT,nfamily))
    pw=rng.exponential(size=(BOOT,npeople))
    if strata is not None:
        for j in np.unique(strata):
            mask=strata==j
            pw[:,mask]/=pw[:,mask].mean(1,keepdims=True)
    return pw,fw


def interval(x,draws):
    return [float(x),*np.quantile(draws,[.025,.975]).tolist()]


def mean_items(y,draw,iw):
    # Input [orders] and [bootstrap,orders]; both orders receive equal weight.
    point=y.mean()
    fam=draw.reshape(len(iw),-1,2).mean(-1)
    return interval(point,(iw*fam).sum(1)/iw.sum(1))


def interpretation(pred,observed,skip22,comparators=None):
    mask=np.array([not skip22 or x.split("_")[0]!="22" for x in ORDERS])
    keep=[x for x,ok in zip(ORDERS,mask,strict=True) if ok]
    data=[r for r in observed if r["item_id"] in keep]
    people=sorted({r["reader"] for r in data})
    n=np.zeros((len(people),len(keep))); yes=np.zeros_like(n)
    for row in data:
        i,j=people.index(row["reader"]),keep.index(row["item_id"])
        n[i,j]+=1;yes[i,j]+=int(row["y"])
    pw,iw=sampling(len(people),len(keep)//2,92124)
    y=yes.sum(0)/n.sum(0)
    yb=(pw@yes)/(pw@n)
    out={}; matrices={}
    metrics=("choice_logloss","choice_brier","choice_agreement")
    for name,p_all in pred.items():
        p=p_all[mask]
        if not ((p>0)&(p<1)).all():
            raise ValueError("A nondegenerate binary forecast is required")
        a=(p>.5)+.5*(p==.5)
        def metric(r):
            return np.stack((-r*np.log(p)-(1-r)*np.log1p(-p),
                             r*(1-p)**2+(1-r)*p**2, r*a+(1-r)*(1-a)),axis=-1)
        point,reps=metric(y),metric(yb)
        out[name]={k:mean_items(point[:,j],reps[:,:,j],iw) for j,k in enumerate(metrics)}
        shapes=a.reshape(-1,2)
        identity=shapes[:,0]*(1-shapes[:,1])+(1-shapes[:,0])*shapes[:,1]
        out[name]["switch_mention_order_same_object"]=interval(identity.mean(),iw@identity/iw.sum(1))
        matrices[name]=(point,reps)
    comparisons={}
    if comparators is None:
        comparators=(("headwise_event","lexical_event"),("headwise_event","headwise_pronoun"),
                     ("headwise_event","coin"),("headwise_event","published_gpt3_completion"))
    for first,other in comparators:
        p,b=matrices[first];p0,b0=matrices[other]
        comparisons[first+"_minus_"+other]={k:mean_items(p[:,j]-p0[:,j],b[:,:,j]-b0[:,:,j],iw) for j,k in enumerate(metrics)}
    return {"readers":len(people),"responses":len(data),"families":len(keep)//2,"predictors":out,"paired":comparisons,
            "always_NP2_accuracy":mean_items(y,yb,iw)}


def region_value(text, offsets, series, left, right):
    chosen=[]
    for i,(a,b) in enumerate(offsets):
        nonspace=[j for j in range(a,b) if not text[j].isspace()]
        if not nonspace:continue
        inside=[left<=c<right for c in nonspace]
        if any(inside)!=all(inside):
            raise ValueError("A token has content on both sides of a registered region")
        if inside[0]:chosen.append(i)
    if not chosen:
        # No verbal predicate follows the noun in a minority of critical segments.
        if text[left:right].strip():
            raise ValueError("Experimental control lost its event characters")
        return 0.
    return float(np.sum(series[chosen]))


def repetitions(row,regions):
    forms=[row["text"][a:b].casefold() for a,b in row["words"]]
    seen=set();repeat=[]
    for f in forms:
        repeat.append(f in seen);seen.add(f)
    values=[]
    for a,b in regions:
        selected=[i for i,(x,y) in enumerate(row["words"]) if a<=x and y<=b]
        values.append(np.mean([repeat[i] for i in selected]) if selected else 0.)
    return values


def extract(folder):
    receipt=json.loads((folder/"receipt.json").read_text())
    seal=json.loads((OUT/"source-seal.json").read_text())
    packet=ROOT/"jones-pointer-stimuli.jsonl"
    wp,sp=folder/"all-earlier-word-logits.npz",folder/"surprisal.npz"
    if (sha(PROTOCOL)!=seal["protocol_sha256"] or receipt["protocol_sha256"]!=sha(PROTOCOL)
            or sha(packet)!=receipt["source_sha256"] or sha(wp)!=receipt["forecast_sha256"]
            or sha(sp)!=receipt["surprisal_sha256"] or receipt["arms"]!=list(ARMS) or receipt["encoder_forwards"]!=300):
        raise ValueError("Registered original stimuli or frozen forecasts changed")
    geomfile=OUT/"geometry.json"
    if sha(geomfile)!=seal["geometry_sha256"]:raise ValueError("Original region geometry changed")
    geom=json.loads(geomfile.read_text())
    packet={r["id"]:r for r in map(json.loads,packet.read_text().splitlines())}
    cols=study_controls(); physical=cols["physics"].copy(); syntax=cols["syntax"].copy()
    for a in ARMS:
        for label in ("pronoun","event"):
            cols[a+"_"+label]=np.empty(60)
    belief={n:np.empty(60) for n in cols if n not in ("coin","published_gpt3_completion")}
    belief["physics"]=physical;belief["syntax"]=syntax;belief["qwen_explicit_counterfactual"]=np.empty(60)
    maxdelta=0.; surprise=np.zeros((60,4,7));repeated=np.zeros_like(surprise)
    tokenizer=load_tokenizer("qwen",json.loads(Path("research/backbone_transfer/models.json").read_text())["models"]["qwen"])
    with np.load(wp) as forecasts,np.load(sp) as raw:
        for i,k in enumerate(ORDERS):
            id1="e1-"+k; g=geom[id1]
            z=forecasts[id1+"/"+str(g["q"])].astype(float)
            for j,a in enumerate(ARMS):
                for s,t in (("pronoun",0),("event",g["end"]-g["q"])):
                    cols[a+"_"+s][i]=get_belief(z[j,t],g["names"])
            event_nll=[]; evidence=[]
            for c,cond in enumerate(CONDITIONS):
                id2="e2-"+k+"_"+cond; g=geom[id2]; r=packet[id2]
                text=r["text"]; enc=tokenizer.encode(text,add_special_tokens=False)
                if len(raw[id2])!=len(enc.ids):raise ValueError("Original region tokenization changed")
                surprise[i,c]=[region_value(text,enc.offsets,raw[id2],a,b) for a,b in g["regions"][7:]]
                repeated[i,c]=repetitions(r,g["regions"][7:])
                if g["explicit"]:continue
                z=forecasts[id2+"/"+str(g["q"])].astype(float)
                evidence.append(np.stack([get_belief(z[:,t],g["names"]) for t in (0,g["end"]-g["q"])]))
                explicit=g["counterfactual_key"]
                txt=packet[explicit]["text"]; enc2=tokenizer.encode(txt,add_special_tokens=False)
                event_nll.append(region_value(txt,enc2.offsets,raw[explicit],g["explicit_np_end_char"],g["explicit_event_end_char"]))
            maxdelta=max(maxdelta,float(np.abs(evidence[0]-evidence[1]).max()))
            avg=np.mean(evidence,axis=0)
            for j,a in enumerate(ARMS):
                for t,stage in enumerate(("pronoun","event")):belief[a+"_"+stage][i]=avg[t,j]
            # The NP1-contradicting version explicitly specified NP2 and vice versa.
            belief["qwen_explicit_counterfactual"][i]=logistic(event_nll[1]-event_nll[0])
    return cols,belief,surprise,repeated,maxdelta,receipt


def weighted_corr(x,y,w):
    """First argument is [items], other two are [draw,items]."""
    nx=w.sum(1,keepdims=True); xm=(w*x).sum(1,keepdims=True)/nx; ym=(w*y).sum(1,keepdims=True)/nx
    xx=x-xm; yy=y-ym
    denom=np.sqrt((w*xx*xx).sum(1)*(w*yy*yy).sum(1))
    return np.divide((w*xx*yy).sum(1),denom,out=np.zeros(len(w)),where=denom>0)


def leave_item_out(x,y):
    out=np.empty(len(y)); family=np.arange(len(y))//2
    for name in np.unique(family):
        train=family!=name
        mx=x[train].mean(0);sd=x[train].std(0);scale=np.where(sd>1e-10,sd,1.)
        d=(x[train]-mx)/scale; baseline=y[train].mean(); target=y[train]-baseline
        cov=d.T@d/len(target)+np.eye(d.shape[1])*.01
        beta=np.linalg.solve(cov,d.T@target/len(target))
        out[~train]=baseline+(x[~train]-mx)/scale@beta
    return out


def pace(observed,belief,surprise,repeat,skip22,additional=()):
    mask=np.array([not skip22 or x.split("_")[0]!="22" for x in ORDERS])
    keep=[x for x,ok in zip(ORDERS,mask,strict=True) if ok]
    cases=[k+"_"+c for k in keep for c in CONDITIONS]
    lookup={k:j for j,k in enumerate(cases)}
    rows=[r for r in observed if r["item_id"] in lookup];people=sorted({r["reader"] for r in rows})
    pm={s:i for i,s in enumerate(people)}
    n=np.zeros((len(people),len(cases)))
    z=np.zeros((*n.shape,7)); milliseconds=np.zeros_like(z); strata=np.full(len(people),-1,dtype=int)
    for r in rows:
        i=pm[r["reader"]];j=lookup[r["item_id"]];group=int(r["item_id"].endswith("_amb"))
        if strata[i]!=-1 and strata[i]!=group:raise ValueError("Reader crossed the between-subject ambiguity condition")
        strata[i]=group
        rt=np.array([float(r[k]) for k in REGIONS]);logged=np.log(rt)
        n[i,j]+=1;z[i,j]+=logged-logged[:3].mean();milliseconds[i,j]+=rt
    if np.any(n.sum(0)==0):raise ValueError("A whole experimental item, order and condition cell is missing")
    pw,iw=sampling(len(people),len(keep)//2,18063,strata);aw=np.repeat(iw,2,axis=1)
    def calc(cells):
        return np.einsum("...icr,c->...ir",cells.reshape(*cells.shape[:-2],len(keep),4,7),DIF)
    y=calc(z.sum(0)/n.sum(0)[:,None])
    boot=np.empty((BOOT,len(keep),7));cond_count=pw@n
    for j in range(7):
        scaled=(pw@z[:,:,j])/cond_count
        boot[:,:,j]=scaled.reshape(BOOT,len(keep),4)@DIF
    rawrt=calc(milliseconds.sum(0)/n.sum(0)[:,None]);rawrt_boot=np.empty_like(boot)
    for j in range(7):
        rawrt_boot[:,:,j]=((pw@milliseconds[:,:,j])/cond_count).reshape(BOOT,len(keep),4)@DIF
    inputs={k:2*p[mask]-1 for k,p in belief.items()}
    dx=calc(surprise[mask].reshape(-1,7)); rx=calc(repeat[mask].reshape(-1,7))
    result=[]
    for j in range(7):
        assoc={}
        for k,x in inputs.items():
            signs=(np.sign(x)*np.sign(y[:,j])+1)/2
            ds=(np.sign(x)[None,:]*np.sign(boot[:,:,j])+1)/2
            corr=weighted_corr(x,y[:,j][None,:],np.ones((1,len(keep))))[0]
            bc=weighted_corr(x,boot[:,:,j],aw)
            assoc[k]={"pearson":interval(corr,bc),"direction_agreement":mean_items(signs,ds,iw),
                      "implied_raw_millisecond_effect":mean_items(np.sign(x)*rawrt[:,j],np.sign(x)*rawrt_boot[:,:,j],iw)}
        design=np.column_stack([np.tile([0.,1.],len(keep)//2),inputs["syntax"],rx[:,j],
                               *[dx[:,j-z] if j>=z else np.zeros(len(keep)) for z in range(4)]])
        fits={};boots={}
        for a,extra in (("control",None),("plus_headwise_event","headwise_event"),("plus_prior_headwise","headwise_pronoun"),
                        ("plus_counterfactual","qwen_explicit_counterfactual"),("plus_human_physics_norm","physics"),*additional):
            d=design if extra is None else np.column_stack((design,inputs[extra]))
            pred=leave_item_out(d,y[:,j])
            err=(pred-y[:,j])**2; be=(pred-boot[:,:,j])**2
            fits[a]={"held_family_mse":mean_items(err,be,iw)}
            boots[a]=(err,be)
            if extra is not None:
                c,cb=boots["control"]
                fits[a]["paired_mse_minus_control"]=mean_items(err-c,be-cb,iw)
        result.append({"region":j+1,"signed_association":assoc,"held_family_regression":fits})
    return {"readers":len(people),"trials":len(rows),"families":len(keep)//2,"min_trials_per_condition_cell":int(n.sum(0).min()),
            "max_trials_per_condition_cell":int(n.sum(0).max()),"regions":result}


def main(folder):
    cols,belief,surprise,repeat,delta,receipt=extract(folder)
    intake=json.loads((OUT/"human-intake.json").read_text())
    for name,expected in intake["study_local_anonymous_outcome_sha256"].items():
        if sha(ROOT/("jones-anonymous-"+name+".csv"))!=expected:raise ValueError("Original anonymous human response changed")
    main1=read("anonymous-exp1-script"); main2=read("anonymous-exp2")
    slope=Path("research/reading_validation/transition-jones-analysis-conventions.md")
    specific={"protocol_sha256":sha(PROTOCOL),"analysis_conventions_sha256":sha(slope),"accelerator_receipt":receipt,
              "crossed_multiplier_draws":BOOT,"max_same_prefix_cross_continuation_belief_delta":delta}
    if delta>.025:raise ValueError(f"Substantial illegitimate future influence at identical prefixes: {delta}")
    dump(OUT/"interpretation.json",{**specific,"released_script":interpretation(cols,main1,False),
         "exclude_order_discrepancy_family_22":interpretation(cols,main1,True),"paper_described_rules":interpretation(cols,read("anonymous-exp1-paper"),False)})
    dump(OUT/"later-reading.json",{**specific,"released_script":pace(main2,belief,surprise,repeat,False),
         "exclude_order_discrepancy_family_22":pace(main2,belief,surprise,repeat,True),
         "paper_described_rules":pace(read("anonymous-exp2-paper-rules"),belief,surprise,repeat,False)})
    dump(OUT/"item-forecasts.json",{"physical_families_and_orders":ORDERS,"experiment_1":{k:v.tolist() for k,v in cols.items()},
          "experiment_2":{k:v.tolist() for k,v in belief.items()},"surprisal":surprise.tolist(),"repeated_words":repeat.tolist()})
    print("Validated all 30 physical families and both independent participant cohorts",flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--features",type=Path,required=True)
    main(p.parse_args().features)
