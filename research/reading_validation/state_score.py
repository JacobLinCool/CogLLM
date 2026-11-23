"""Grade sealed full-article state and conflict forecasts on the authors' original held story families."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from research.reading_validation.memory_data import sha
from research.reading_validation.state_events import accepted,materials,path
from research.reading_validation.state_transport_data import OUT
from research.reading_validation.state_words import ATTR, Archive,align

def unpack(flat,n):
    segs=[];start=0
    for j in range(1,n):
        p=flat[start:start+j+1].astype(float);start+=j+1
        if len(p)!=j+1 or abs(p.sum()-1)>1e-5:raise ValueError('Competing risk probabilities did not sum to one')
        segs.append(p)
    if start!=len(flat):raise ValueError('Missing causal opportunities')
    return segs

def distribution(pp):
    earlier=1.;locations=[]
    for x in pp:
        locations.append(earlier*x[1:]);earlier*=x[0]
    return earlier,locations

def scores(r,s,pp,original,threshold):
    none,joint=distribution(pp);p=1.-none;base=1.-distribution(original)[0]
    out={'path_nll':path(r,pp),'balanced_accuracy':float((p>=threshold)==(not r['plausible'])),
         'any_event_nll':-np.log(max(1e-12,none if r['plausible'] else p)),'event_probability':p}
    if r['plausible']:
        out['false_alarm']=float(p>=threshold);return out
    bp=r['breakpoint'];source=accepted(r);truth=sum(joint[bp-1][u] for u in source)
    forms={'all_edits':True,'changed_earlier':s['change'][0]<bp,'changed_here':s['change'][0]==bp}
    times=np.array([v.sum() for v in joint])
    j,i=max(((j,i) for j,a in enumerate(joint) for i in range(len(a))),key=lambda pair:joint[pair[0]][pair[1]])
    where=int(np.argmax(times))+1
    now=pp[bp-1];old=original[bp-1];score=float(sum(now[u+1] for u in source));base_source=float(sum(old[u+1] for u in source))
    metrics={'whole_story_pair':float(p>base)+.5*float(p==base),'whole_passage_source_and_breakpoint':float(j+1==bp and i in source),
             'breakpoint_only':float(where==bp),'antecedent_given_true_breakpoint':float(np.argmax(now[1:]) in source),
             'probability_correct_source_and_breakpoint':float(truth),'change_in_true_pair_hazard':score-base_source,
             'true_pair_hazard_direction':float(score>base_source)+.5*float(score==base_source),'change_in_target_hazard':old[0]-now[0]}
    for group,active in forms.items():
        if active:out.update({group+'/'+k:v for k,v in metrics.items()})
    return out

def summary(items,metrics,rs=6000):
    groups=sorted({s['family'] for r,s in items.values()});rng=np.random.default_rng(821213);draw=rng.integers(len(groups),size=(rs,len(groups)))
    sample={};uncertainty={}
    for model,records in metrics.items():
        names=sorted({name for value in records.values() for name in value})
        vals={}
        for name in names:
            cells=defaultdict(list)
            for ident,entry in records.items():
                if name in entry:
                    r,s=items[ident];cells[s['family'],r['plausible']].append(entry[name])
            family=np.array([np.mean([np.mean(cells[g,t]) for t in (False,True) if (g,t) in cells]) if any((g,t) in cells for t in (False,True)) else np.nan for g in groups])
            raw=float(np.nanmean(family));v=family[draw];bs=np.nanmean(v,axis=1)
            vals[name]=dict(estimate=raw,interval95=np.quantile(bs,[.025,.975]).tolist(),families=int(np.isfinite(family).sum()),articles=sum(map(len,cells.values())))
            uncertainty[model,name]=(raw,bs)
        sample[model]=vals
    comparisons={}
    for right,left in [('surprisal','form'),('context','surprisal'),('state','context'),('state','form'),('mixture','surprisal'),('mixture','context'),('mixture','form')]:
        if right not in metrics or left not in metrics:continue
        results={}
        for name in sample[right]:
            a,x=uncertainty[right,name];b,y=uncertainty[left,name]
            results[name]=dict(estimate=a-b,interval95=np.quantile(x-y,[.025,.975]).tolist())
        comparisons[right+'_minus_'+left]=results
    return dict(models=sample,comparisons=comparisons)

def frequencies(source,archive):
    yes=np.zeros((len(ATTR),2));counts=yes.copy()
    for r,s in source.values():
        y,_=align(r,archive.document(s['key']));yes+=(y==1).sum(0);counts+=(y>=0).sum(0)
    return (yes+.5)/(counts+1)

def state_score(source,items,archive,features):
    prior=frequencies(source,archive);y_all=[];p_all=[];family=[];counts=Counter()
    for r,s in items.values():
        y,cc=align(r,archive.document(s['key']));counts.update(cc);p=features[s['key']][:,:,:,0].astype(float)
        y_all.extend(y);p_all.extend(p);family.extend([s['family']]*len(y))
    y=np.array(y_all);p=np.clip(np.array(p_all),1e-7,1-1e-7);family=np.array(family);fams=np.unique(family)
    section={};numerator=defaultdict(lambda:np.zeros((len(fams),2)));discard=[]
    for a,name in enumerate(ATTR):
        for coordinate,side in enumerate(('pre','post')):
            idx=y[:,a,coordinate]>=0;yy=y[idx,a,coordinate];pp=p[idx,a,coordinate];ff=family[idx];key=name+'.'+side
            if not len(yy) or not (yy==0).any() or not (yy==1).any():discard.append(key);continue
            q=prior[a,coordinate];truth=yy.astype(bool);native=-np.log(np.where(truth,pp,1-pp));base=-np.log(np.where(truth,q,1-q))
            section[key]=dict(known=len(yy),affirmed=int(truth.sum()),source_rate=q,state_nll=native.mean(),prior_nll=base.mean(),
                              balanced_accuracy=.5*((pp[truth]>=.5).mean()+(pp[~truth]<.5).mean()))
            for n,v in [('native',native),('prior',base)]:
                out=np.zeros((len(fams),2));out[:,0]=np.bincount(np.searchsorted(fams,ff),weights=v,minlength=len(fams));out[:,1]=np.bincount(np.searchsorted(fams,ff),minlength=len(fams))
                numerator[n,key]=out
    rng=np.random.default_rng(231);weights=rng.multinomial(len(fams),np.ones(len(fams))/len(fams),size=3000);stats={}
    for name in ('native','prior'):
        ss=[];est=[]
        for key in section:
            data=numerator[name,key];b=weights@data;ss.append(np.divide(b[:,0],b[:,1],out=np.full(len(b),np.nan),where=b[:,1]>0));est.append(data[:,0].sum()/data[:,1].sum())
        stats[name]=(float(np.mean(est)),np.nanmean(ss,axis=0))
    def ci(m,v):return {'estimate':float(m),'interval95':np.quantile(v,[.025,.975]).tolist()}
    a,x=stats['native'];b,z=stats['prior']
    return dict(annotation_alignment=dict(counts),endpoint_details=section,one_or_zero_class=discard,macro_known_truth_nll=dict(state=ci(a,x),source_prior=ci(b,z),difference=ci(a-b,x-z)))

def run(arm,native):
    out=OUT/('forecast-'+arm);seal=json.loads((out/'seal.json').read_bytes())
    if any(sha(out/f)!=sig for f,sig in seal['files'].items()):raise ValueError('Altered held event forecasts')
    archive=Archive(native);items,skip=materials(archive,'test');train,_=materials(archive,'train');fit=json.loads((out/'model.json').read_bytes());metrics=defaultdict(dict)
    with np.load(out/'events.npz') as all_p:
        for mode,threshold in fit['thresholds'].items():
            cache={}
            for ident,(r,s) in items.items():
                for key in (s['key'],s['original']):
                    if key not in cache:cache[key]=unpack(all_p[key+'.'+mode],len(archive.document(key)['regions']))
                metrics[mode][ident]=scores(r,s,cache[s['key']],cache[s['original']],threshold)
    with np.load(out/'states.npz') as f:states=state_score(train,items,archive,f)
    audit=OUT/'family-exposure.json'
    coverage=json.loads(audit.read_bytes())
    if coverage['inputs']!=sha(OUT/'inputs.json'):
        raise ValueError('Held story exposure belongs to a different split')
    clean=set(coverage['clean_families'])
    independent={key:row for key,row in items.items() if row[1]['family'] in clean}
    independent_metrics={m:{k:v for k,v in rows.items() if k in independent} for m,rows in metrics.items()}
    result=dict(forecast=sha(out/'seal.json'),excluded=skip,attributes=states,events=summary(items,metrics),
                exposure=sha(audit),nonoverlapping_family_events=summary(independent,independent_metrics))
    (OUT/('scores-'+arm+'.json')).write_text(json.dumps(result,allow_nan=False,indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('arm',choices=('native','form'));p.add_argument('--native',type=Path)
    run(**vars(p.parse_args()))
