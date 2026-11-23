"""Grade the sealed source and independent judgments at each original sentence or argument."""
import argparse
from collections import Counter,defaultdict
import json

import numpy as np
from scipy.stats import rankdata

from research.reading_validation.argument_data import OUT,WORK
from research.reading_validation.argument_labels import bank,sufficiency
from research.reading_validation.memory_data import load_rows,sha


def ci(x,draw):
    return dict(estimate=float(x),interval95=np.quantile(draw,[.025,.975]).tolist())


def ap(y,score):
    ix=np.argsort(-score,kind='stable');y=np.asarray(y)[ix];s=score[ix]
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1];positive=np.cumsum(y)[ends]
    return float(np.diff(np.r_[0,positive])@(positive/(ends+1))/max(1,y.sum()))


def auc(y,p):
    y=np.asarray(y,bool);n=y.sum();m=(~y).sum()
    if min(n,m)==0:return None
    return float((rankdata(p)[y].sum()-n*(n+1)/2)/(n*m))


def decode(data,k,arm,n):
    if arm=='nearest':
        source=[np.empty(0),*[np.eye(1,j,j-1).ravel() for j in range(1,n)]]
        return source,None,None
    flat=data[k+'.'+arm+'.choice'];pos=data[k+'.'+arm+'.event'];grade=data[k+'.'+arm+'.grade']
    source=[flat[(j*(j-1))//2:((j+1)*j)//2].astype(float) for j in range(n)]
    if len(pos)!=n or any(abs(v.sum()-1)>5e-5 for v in source[1:]):raise ValueError('Causal sentence candidates changed')
    return source,pos,grade


def evaluate(data,k,n,r,arm):
    choices,p,grade=decode(data,k,arm,n);items=defaultdict(list)
    for kind,lig in [('all',r['links']),*r['types'].items()]:
        for j,true in lig.items():
            rank=choices[j];best=int(rank.argmax());hit=int(best in true)
            mass=float(rank[true].sum());start=float(np.max(rank[true]))
            items[kind+'.top1'].append(hit)
            items[kind+'.reciprocal_rank'].append(1/(1+np.sum(rank>start)+.5*(np.sum(rank==start)-1)))
            if arm!='nearest':items[kind+'.choice_nll'].append(-np.log(max(1e-15,mass)))
    opportunity=[]
    if arm!='nearest':
        for j,levels in r['grades'].items():
            for gold in levels:
                pr=grade[j];items['effectiveness.ordinal_nll'].append(-np.log(max(1e-15,pr[gold])))
                items['effectiveness.absolute_level_error'].append(abs(grade[j]@np.arange(3)-gold))
        if r['complete']:
            for j in range(1,n):
                if j in r['exclude']:continue
                truth=r['links'].get(j,[]);yes=bool(truth);pr=float(np.clip(p[j],1e-9,1-1e-9))
                correct=bool(yes and int(choices[j].argmax()) in truth)
                items['event.nll'].append(-np.log(pr if yes else 1-pr))
                opportunity.append((yes,correct,pr,pr*float(choices[j].max())))
    return items,opportunity


def choose_threshold(data,seals,labels,arm):
    o=[t for k,r in labels.items() if k in seals['keys'] for t in evaluate(data,k,seals['keys'][k],r,arm)[1]]
    y,correct,_,s=np.array(o).T;order=np.argsort(-s,kind='stable')
    ends=np.r_[np.flatnonzero(np.diff(s[order])),len(order)-1]
    cumulative=np.cumsum(correct[order])[ends];f1=2*cumulative/(ends+1+y.sum())
    j=ends[int(f1.argmax())]
    return dict(joint_threshold=float(s[order][j]),dev_f1=float(f1.max()),dev_positions=len(s),dev_links=int(y.sum()))


def panel(data,seal,labels,thresholds,metadata):
    rng=np.random.default_rng(930719);arms=['nearest',*seal['arms']]
    loaded=[k for k in sorted(labels) if k in seal['keys']]
    summary={};raw={};cases=Counter();ids_by=defaultdict(list)
    for name in arms:
        per=defaultdict(dict);flat=[]
        for k in loaded:
            r=labels[k];items,op=evaluate(data,k,seal['keys'][k],r,name)
            if name=='nearest':
                cases.update(r['audit'])
                if not r['complete']:cases['essays_with_unlocalizable_children']+=1
            if op:
                y,hit,p,s=np.array(op).T;mask=s>=thresholds[name]['joint_threshold'];tp=hit[mask].sum()
                if y.sum():
                    items['event.average_precision']=[ap(y,p)]
                    items['joint.retrieval_average_precision']=[ap(hit,s)*hit.sum()/y.sum()]
                items['joint.tp']=[tp];items['joint.emitted']=[mask.sum()];items['joint.gold']=[y.sum()]
                flat.extend(op)
            for target,v in items.items():
                per[target][k]=[float(np.mean(v)),len(v)]
        raw[name]=per;sm={}
        for metric,m in per.items():
            ids=sorted(m);vals=np.array([m[k][0] for k in ids]);n=len(ids)
            boot=vals[rng.integers(n,size=(2000,n))].mean(1)
            sm[metric]=dict(**ci(vals.mean(),boot),essays=n,measurements=sum(m[k][1] for k in ids))
            if metric=='all.top1':
                by=defaultdict(list)
                for k in ids:by[metadata[k].get('topic')].append(m[k][0])
                sm[metric]['topic_means']={str(g):float(np.mean(v)) for g,v in by.items()}
        if flat:
            y,hit,p,s=np.array(flat).T;mask=s>=thresholds[name]['joint_threshold']
            sm['pooled_unmarked']=dict(sentences=len(y),positive_sentences=int(y.sum()),event_average_precision=ap(y,p),
                joint_average_precision=ap(hit,s)*float(hit.sum()/y.sum()) if y.sum() else None,
                positives_emitted=int(mask.sum()),aligned_with_own_source=int(hit[mask].sum()),
                joint_f1=float(2*hit[mask].sum()/(mask.sum()+y.sum())),prevalence=float(y.mean()))
        summary[name]=sm
    delta={}
    for arm,base in [('form','nearest'),('native','form'),('fixed','isolated'),('joint','isolated'),('joint','fixed'),('adaptive','isolated'),('adaptive','joint'),('adaptive','half')]:
        if arm not in summary or base not in summary:continue
        for metric in raw[arm].keys()&raw[base].keys():
            a,b=raw[arm][metric],raw[base][metric];keys=sorted(a.keys()&b.keys())
            v=np.array([a[k][0]-b[k][0] for k in keys]);n=len(v)
            delta[f'{arm}_minus_{base}.{metric}']=ci(v.mean(),v[rng.integers(n,size=(3000,n))].mean(1))
    return dict(eligible_essays=len(loaded),audit=dict(cases),models=summary,differences=delta)


def paragraph_panel(data,seal):
    observed,excluded=sufficiency();texts={r['id']:r['text'] for r in load_rows(WORK/'stimuli.jsonl')}
    from research.reading_validation.argument_data import sentences
    primary=[];votes=[];majority=[];coverage=Counter()
    for r in observed:
        k=r['article']
        if k not in seal['keys']:coverage['native_exclusion']+=1;continue
        spans=sentences(texts[k]);jj=r['positions']
        scores={'shorter_paragraph':-len(texts[k][spans[jj[0]][0]:spans[jj[-1]][1]].split())}
        for a in seal['arms']:
            scores[a]=float(np.mean((data[k+'.'+a+'.grade'][jj]@np.array([1.,.5,0.]))))
        primary.append((k,r['label'],scores));votes.extend((k,v,scores) for v in r['votes'] or [])
        if r['votes'] is not None:majority.append((k,int(sum(r['votes'])>=2),scores))
    def summarize(rows):
        ordered=sorted({k for k,y,s in rows});m=len(ordered);index={k:i for i,k in enumerate(ordered)};rng=np.random.default_rng(91504)
        which=np.array([index[k] for k,y,s in rows]);y=np.array([y for k,y,s in rows])
        # One random multiplicity per original essay; all three raters of a given paragraph travel together.
        cluster=rng.multinomial(m,np.full(m,1/m),size=2500)
        cuts=[np.repeat(np.arange(len(rows)),w[which]) for w in cluster]
        vals={};est={}
        for arm in rows[0][2]:
            s=np.array([r[2][arm] for r in rows]);v=auc(y,s);draw=np.array([auc(y[ix],s[ix]) for ix in cuts])
            vals[arm]=(v,draw);est[arm]=dict(auc=ci(v,draw),ap=ap(y,s))
        for arm,base in [('form','shorter_paragraph'),('native','form'),('fixed','isolated'),('joint','fixed'),('joint','isolated'),('adaptive','half'),('adaptive','isolated')]:
            if arm in vals and base in vals:
                x,u=vals[arm];v,w=vals[base];est[arm+'_minus_'+base]=ci(x-v,u-w)
        return dict(essays=m,observations=len(y),insufficient=int(y.sum()),scores=est)
    return dict(published_label=summarize(primary),three_reader_majority=summarize(majority),individual_votes=summarize(votes),alignment_audit={**excluded,**coverage})


def run(mode):
    here=OUT/('forecast-'+mode);seal=json.loads((here/'seal.json').read_bytes())
    if any(sha(here/k)!=v for k,v in seal['files'].items()) or sha(OUT/'inputs.json')!=seal['source']:
        raise ValueError('The forecast changed before essay grading')
    meta=json.loads((OUT/'inputs.json').read_bytes())['articles']
    with np.load(here/'forecasts.npz') as data:
        dev=bank('dev');cut={a:choose_threshold(data,seal,dev,a) for a in seal['arms']}
        primary={r:panel(data,seal,bank(r),cut,meta) for r in ('held','external')}
        support=paragraph_panel(data,seal)
    (OUT/('scores-'+mode+'.json')).write_text(json.dumps(dict(forecast=sha(here/'seal.json'),
                  development=cut,panels=primary,sufficiency=support),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('form','native','joint-form','joint-native'));run(**vars(p.parse_args()))
