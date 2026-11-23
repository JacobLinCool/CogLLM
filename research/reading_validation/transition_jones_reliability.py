"""Estimate how reproducible these outcomes are across disjoint experimental readers."""

import numpy as np

from research.reading_validation.transition_jones_score import CONDITIONS,DIF,ORDERS,OUT,REGIONS,read
from research.reading_validation.transition_score import dump


def interpretation():
    rows=read("anonymous-exp1-script");people=sorted({r["reader"] for r in rows})
    n=np.zeros((len(people),60));y=np.zeros_like(n)
    for r in rows:
        i,j=people.index(r["reader"]),ORDERS.index(r["item_id"])
        n[i,j]+=1;y[i,j]+=int(r["y"])
    if n.max()!=1:raise ValueError("Multiple responses to the same item-order by one reader")
    hit=[];ll=[];brier=[];agreement=[]
    for j in range(60):
        ratings=y[n[:,j]>0,j]
        ref=(ratings.sum()-ratings+.5)/len(ratings)
        decisions=(np.sign(ref-.5)+1)/2
        hit.append(np.mean(ratings*decisions+(1-ratings)*(1-decisions)))
        ll.append(np.mean(-ratings*np.log(ref)-(1-ratings)*np.log1p(-ref)))
        brier.append(np.mean((ratings-ref)**2))
        yes=ratings.sum();size=len(ratings)
        agreement.append((yes*(yes-1)+(size-yes)*(size-yes-1))/(size*(size-1)))
    rng=np.random.default_rng(130521);draw=rng.exponential(size=(4000,30));draw/=draw.sum(1,keepdims=True)
    def summary(values):
        v=np.asarray(values).reshape(30,2).mean(1);return [float(v.mean()),*np.quantile(draw@v,[.025,.975]).tolist()]
    return {"people":len(people),"responses":len(rows),"other_readers_same_item_choice_accuracy":summary(hit),
            "other_readers_probability_logloss":summary(ll),"other_readers_probability_brier":summary(brier),
            "random_pair_exact_choice_agreement":summary(agreement),"min_people_per_item_order":int(n.sum(0).min()),
            "max_people_per_item_order":int(n.sum(0).max()),"grouped_interval":"multiplier resampling 30 physical item families, conditional on observed panel"}


def timing():
    rows=read("anonymous-exp2");people=sorted({r["reader"] for r in rows}); cases=[s+"_"+c for s in ORDERS for c in CONDITIONS]
    z=np.zeros((len(people),len(cases),7));n=np.zeros(z.shape[:2]);groups=np.full(len(people),-1)
    for r in rows:
        i,j=people.index(r["reader"]),cases.index(r["item_id"]);g=int(r["item_id"].endswith("_amb"))
        if groups[i]>=0 and groups[i]!=g:raise ValueError("Mixed ambiguity assignment")
        groups[i]=g;time=np.log([float(r[k]) for k in REGIONS]);z[i,j]+=time-time[:3].mean();n[i,j]+=1
    rng=np.random.default_rng(75479)
    reps=5000;assignment=np.zeros((reps,len(people)))
    for i in range(reps):
        for g in np.unique(groups):
            pool=np.flatnonzero(groups==g);take=rng.choice(pool,len(pool)//2,replace=False);assignment[i,take]=1
    arms=(assignment,1-assignment)
    n0,n1=[a@n for a in arms]
    valid=((n0>0)&(n1>0)).reshape(reps,60,4).all(-1)
    report=[]
    for j in range(7):
        def contrasts(a,counts):
            out=np.divide(a@z[:,:,j],counts,out=np.zeros_like(counts),where=counts>0)
            return out.reshape(reps,60,4)@DIF
        a,b=[contrasts(m,nr) for m,nr in zip(arms,(n0,n1),strict=True)]
        # Averaging over intact item-order contrasts in both disjoint sets.
        s=valid.sum(1);mx=(a*valid).sum(1)/s;my=(b*valid).sum(1)/s
        xx=(a-mx[:,None])*valid;yy=(b-my[:,None])*valid
        rr=(xx*yy).sum(1)/np.sqrt((xx*xx).sum(1)*(yy*yy).sum(1))
        report.append({"region":j+1,"median_disjoint_half_correlation":float(np.median(rr)),
                       "random_splits_2_5_and_97_5_percentiles":np.quantile(rr,[.025,.975]).tolist(),
                       "fraction_positive":float((rr>0).mean())})
    return {"people":len(people),"trials":len(rows),"random_disjoint_half_splits":reps,"regions":report,
            "median_complete_item_orders_each_replicate":float(np.median(valid.sum(1))),
            "min_complete_item_orders":int(valid.sum(1).min()),"full_item_orders":60}


if __name__=="__main__":
    dump(OUT/"human-repeatability.json",{"choices":interpretation(),"timing":timing()})
