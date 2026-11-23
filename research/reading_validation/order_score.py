"""Compare sealed whole-passage forecasts with original reader responses and factorial effects."""
from collections import Counter,defaultdict
import json
from math import log,pi
from pathlib import Path

import numpy as np

from research.reading_validation.memory_data import sha
from research.reading_validation.order_data import OUT
from research.reading_validation.order_fit import identity,weights
from research.reading_validation.order_measure import ARMS,TASKS,Study


def interval(x,draws):
    return dict(estimate=float(x),interval95=np.quantile(draws,[.025,.975]).tolist())


def family_summary(values,rows,draws):
    groups=sorted({r[0] for r in rows})
    w=weights(rows);index=np.array([groups.index(r[0]) for r in rows])
    per=np.bincount(index,weights=w*values,minlength=len(groups))*len(groups)
    return interval(per.mean(),per[draws].mean(1))


def contrast(values,rows,study,seed):
    # Two positive crossed multiplier weights preserve observable sparse cells.
    # Within-family condition means use exactly the readers who received that condition.
    rng=np.random.default_rng(seed);n_boot=3000
    readers=sorted({r[2] for r in rows});groups=sorted({r[0] for r in rows})
    rw=np.vstack((np.ones(len(readers)),rng.exponential(size=(n_boot,len(readers)))))
    gw=np.vstack((np.ones(len(groups)),rng.exponential(size=(n_boot,len(groups)))))
    by=defaultdict(list)
    for i,(g,c,p,_) in enumerate(rows):
        meta=study.conditions[g,c]
        by[g,meta['structure'],meta['order'],meta['given']].append(i)
    curves={};family_curves={};numeric={};coverage={}
    for structure in ('m-s','s-m'):
        per=[];valid=[];cells=0
        for j,g in enumerate(groups):
            diffs=[]
            for chronology in ('e1-e2','e2-e1'):
                arms=[]
                for direction in ('given-new','new-given'):
                    indices=by.get((g,structure,chronology,direction),[])
                    if not indices:break
                    who=[readers.index(rows[i][2]) for i in indices]
                    wt=rw[:,who];arms.append((wt@values[indices])/wt.sum(1))
                if len(arms)==2:diffs.append(arms[0]-arms[1]);cells+=1
            if diffs:
                valid.append(j);per.append(np.mean(diffs,axis=0))
        if per:
            denom=gw[:,valid].sum(1);vec=(np.stack(per,axis=1)*gw[:,valid]).sum(1)/denom
            curves[structure]=vec;numeric[structure]=interval(vec[0],vec[1:])
            family_curves[structure]={g:float(v[0]) for g,v in zip([groups[j] for j in valid],per,strict=True)}
        coverage[structure]=dict(materials=len(valid),matched_strata=cells,possible_strata=2*len(groups))
    if len(curves)==2:
        vec=curves['m-s']-curves['s-m'];numeric['interaction_main_minus_subordinate']=interval(vec[0],vec[1:])
    return numeric,coverage,family_curves


def score():
    out=OUT/'forecast';seal=json.loads((out/'seal.json').read_bytes());study=Study()
    if any(sha(out/name)!=v for name,v in seal['files'].items()) or sha(OUT/'inputs.json')!=seal['input']:
        raise ValueError('Order effects must be graded against sealed forecasts')
    pred=json.loads((out/'predictions.json').read_bytes());fits=json.loads((out/'fits.json').read_bytes())
    result={}
    for i,task in enumerate(TASKS):
        result[task]={}
        for panel in ('held_both','held_material'):
            rows=study.panel(task,panel);groups=sorted({r[0] for r in rows})
            y=np.array([r[3] for r in rows]);lookup={key:j for j,key in enumerate(pred[task]['keys'])}
            ix=[lookup[identity(r)] for r in rows];rng=np.random.default_rng(10931+i)
            draws=rng.integers(len(groups),size=(5000,len(groups)))
            human=y if task=='question' else np.exp(y)
            diffs,coverage,human_family=contrast(human,rows,study,193+i)
            outpanel=dict(trials=len(y),materials=len(groups),participants=len({r[2] for r in rows}),
                          units='proportion correct' if task=='question' else 'milliseconds',observed_contrast=diffs,
                          coverage=coverage,models={},comparisons={});losses={}
            for arm in ARMS:
                estimate=np.array(pred[task]['predictions'][arm])[ix];spec=fits[task][arm]
                if task=='question':
                    p=np.clip(estimate,1e-9,1-1e-9);loss=-(y*np.log(p)+(1-y)*np.log1p(-p));natural=p
                else:
                    sd=spec['residual_sd'];loss=.5*((y-estimate)/sd)**2+log(sd)+.5*log(2*pi)
                    natural=np.exp(estimate)*spec['source_smearing']
                effects,_,family=contrast(natural,rows,study,193+i)
                delta_effect,_,_=contrast(natural-human,rows,study,193+i)
                agreement={}
                for order,vals in human_family.items():
                    both=np.array([[v,family[order][g]] for g,v in vals.items() if abs(v)>1e-12])
                    agreement[order]=dict(eligible=len(both),matched=int(np.sum(both[:,0]*both[:,1]>0)) if len(both) else 0)
                losses[arm]=loss
                outpanel['models'][arm]=dict(nll=family_summary(loss,rows,draws),
                      absolute_error=family_summary(abs(estimate-y),rows,draws),predicted_contrast=effects,
                      discrepancy_in_original_units=delta_effect,family_direction=agreement)
            for arm,base in (('context','surprisal'),('context','form'),('profile','context'),('context','design')):
                outpanel['comparisons'][arm+'_minus_'+base]=family_summary(losses[arm]-losses[base],rows,draws)
            result[task][panel]=outpanel
        all_rows=study.observations[task];actual=np.array([r[3] for r in all_rows])
        if task!='question':actual=np.exp(actual)
        effect,cover,_=contrast(actual,all_rows,study,297+i)
        result[task]['original_all_eligible']=dict(trials=len(actual),effects=effect,coverage=cover)
    (OUT/'scores.json').write_text(json.dumps(dict(seal=sha(out/'seal.json'),tasks=result),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':score()
