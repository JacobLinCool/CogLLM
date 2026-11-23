"""Score human adequacy forecasts in the origin and the independent expert-debate domains."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp
from scipy.stats import rankdata,spearmanr
from research.reading_validation.memory_data import sha
from research.reading_validation.reason_quality_data import OUT,DIMENSIONS
from research.reading_validation.reason_quality_features import Material
from research.reading_validation.reason_quality_fit import observations,expert_votes

def ci(x,b):return dict(estimate=float(x),interval95=np.nanquantile(b,[.025,.975]).tolist())

def within(labels,p,z):
    sign=np.sign(labels[:,None]-labels);ia,ib=np.tril_indices(len(labels),-1);ok=sign[ia,ib]!=0
    if not ok.any():return np.nan
    a=sign[ia,ib][ok]*np.sign(p[ia[ok]]-p[ib[ok]])
    return np.mean((a+1)/2)

def group_summary(y,models,topics,side,measure):
    groups=sorted(set(topics));subgroups=sorted(set(zip(topics,side,strict=True)));g=np.array([groups.index(k) for k in topics]);g2=np.array([subgroups.index(k) for k in zip(topics,side,strict=True)])
    rng=np.random.default_rng(815721);draw=rng.integers(len(groups),size=(5000,len(groups)));d2=rng.integers(len(subgroups),size=(5000,len(subgroups)))
    vals={};reports={}
    for key,pr in models.items():
        if measure=='ordering':
            arr=np.array([within(y[g==i],pr[g==i],None) for i in range(len(groups))]);unit=np.array([within(y[g2==i],pr[g2==i],None) for i in range(len(subgroups))]);rho=float(spearmanr(y,pr).statistic)
        else:
            scalar=-np.log(np.take_along_axis(pr,y,axis=1).clip(1e-12)).mean(1)
            arr=np.array([scalar[g==i].mean() for i in range(len(groups))]);unit=np.array([scalar[g2==i].mean() for i in range(len(subgroups))]);rho=float(spearmanr(y.mean(1),pr@np.arange(3)).statistic)
        value=np.nanmean(arr);b=np.nanmean(arr[draw],axis=1);vals[key]=(value,b)
        reports[key]=dict(primary=ci(value,b),spearman_all_unique_arguments=rho,debate_side_sensitivity=ci(np.nanmean(unit),np.nanmean(unit[d2],axis=1)))
    pairs=[]
    for method in ('isolated','joint','adaptive','paired-isolated','source_only','expert_only','expert_plus_source'):
        pairs.extend([('native.'+method,'form.'+method),('native.'+method,'surprisal.'+method)])
    for arm in ('form','surprisal','native'):
        pairs.extend((arm+'.'+a,arm+'.'+b) for a,b in [('joint','isolated'),('adaptive','isolated'),('adaptive','paired-isolated'),('expert_plus_source','expert_only'),('expert_plus_source','source_only')])
    comparison={}
    for a,b in pairs:
        if a in vals and b in vals:
            x,u=vals[a];z,v=vals[b];comparison[a+' minus '+b]=ci(x-z,u-v)
    return dict(models=reports,comparisons=comparison,topic_units=len(groups),debate_sides=len(subgroups))

def source_panel(obs,material,archive,fit):
    ix=material.index;binary,pairs=obs;arms=fit['fits'].keys();result={}
    for task,records in [('sufficiency',binary),('comparison',pairs)]:
        group=[r[1 if task=='sufficiency' else 2] for r in records];groups=sorted(set(group));g=np.array([groups.index(k) for k in group]);rng=np.random.default_rng(481);draw=rng.integers(len(groups),size=(5000,len(groups)))
        if task=='sufficiency':
            count=np.array([r[3] for r in records]);n=np.array([r[2] for r in records]);index=[ix[r[0]] for r in records]
        else:
            ballots=np.stack([r[3] for r in records]);count=ballots.sum(1);ia=[ix[r[0]] for r in records];ib=[ix[r[1]] for r in records]
        summary={};v={}
        for arm in arms:
            for mode in (('isolated','joint','adaptive') if task=='sufficiency' else ('isolated','joint')):
                if task=='sufficiency':
                    z=archive[arm+'.'+mode][index].astype(float);loss=count*np.logaddexp(0,z)-n*z
                else:
                    z=archive[arm+'.paired-'+mode].astype(float);s=.5*(z[ia]-z[ib]);tie=fit['models'][arm]['fits'][mode]['tie'];logits=np.column_stack((s,-s,np.full(len(s),tie)));loss=-(ballots*(logits-logsumexp(logits,axis=1,keepdims=True))).sum(1)
                per=np.bincount(g,weights=loss)/np.bincount(g,weights=count);mean=per.mean();b=per[draw].mean(1);summary[arm+'.'+mode]=ci(mean,b);v[arm+'.'+mode]=(mean,b)
        diffs={}
        for arm in arms:
            for a,b in [(arm+'.joint',arm+'.isolated'),(arm+'.adaptive',arm+'.isolated'),(arm+'.isolated','form.isolated')]:
                if a!=b and a in v and b in v:diffs[a+' minus '+b]=ci(v[a][0]-v[b][0],v[a][1]-v[b][1])
        result[task]=dict(source_units=len(groups),observations=int(count.sum()),unique_record_opportunities=len(records),nll=summary,contrasts=diffs)
    return result

def run(mode,archive):
    dest=OUT/('forecast-'+mode);seal=json.loads((dest/'seal.json').read_bytes());fit=json.loads((dest/'fits.json').read_bytes())
    if any(sha(dest/k)!=sig for k,sig in seal['files'].items()):raise ValueError('Expert-quality forecast changed')
    material=Material(archive);rows=material.expert
    if material.keys!=fit['keys'] or [r['key'] for r in rows]!=fit['expert']:raise ValueError('Different expert panel')
    ix=[material.index[r['key']] for r in rows];votes=expert_votes({r['record'] for r in rows});topics=[r['topic'] for r in rows];side=[r['side'] for r in rows]
    out={}
    with np.load(dest/'forecasts.npz') as z:
        source=source_panel(observations(material,'test'),material,z,fit)
        for dim in DIMENSIONS:
            observed=np.stack([np.array(votes[r['record']][dim])-1 for r in rows]);y=observed.mean(1)
            score={k:z[k][ix] for k in z if k.count('.')==1}
            grade={a+'.'+m:z[a+'.'+dim+'.'+m] for a in fit['fits'] for m in ('source_only','expert_only','expert_plus_source')}
            human={}
            for person in range(3):
                others=np.delete(observed,person,axis=1).mean(1)
                human[str(person+1)]=dict(correlation=float(spearmanr(others,observed[:,person]).statistic),
                   within_topic_ordering=float(np.mean([within(others[np.array(topics)==g],observed[np.array(topics)==g,person],None) for g in sorted(set(topics))])))
            out[dim]=dict(votes=int(observed.size),unique_passages=len(observed),direct_transfer=group_summary(y,score,topics,side,'ordering'),
                  ordinal_topic_holdout=group_summary(observed,grade,topics,side,'ordinal'),expert_vs_other_two=human)
    (OUT/('scores-'+mode+'.json')).write_text(json.dumps(dict(seal=sha(dest/'seal.json'),source_test=source,expert=out,excluded_source_repeats=len(material.near)),indent=2,allow_nan=False)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('native','form'));p.add_argument('--archive',type=Path);run(**vars(p.parse_args()))
