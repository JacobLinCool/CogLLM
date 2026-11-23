"""Evaluate human judgments with independent held-out words, carriers and readers."""
from argparse import ArgumentParser
from collections import Counter
from hashlib import sha256
from pathlib import Path
import csv
import re

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import rankdata

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.meaning_data import DIR, OUT, ROOT
from research.reading_validation.identity_causal_score import confidence


def normalize(s):
    return ' '.join(s.casefold().split())


def folds(names, salt):
    ordered = sorted(set(names), key=lambda x: sha256((salt+x).encode()).digest())
    return {k:i % 5 for i,k in enumerate(ordered)}


def outcomes(cases):
    source = next(ROOT.glob('Experiment 3*csv'))
    match = {(r['word'],normalize(r['sentence'])):i for i,r in enumerate(cases)}
    rows=[];exclusions=Counter()
    with source.open(encoding='cp1252',newline='') as f:
        for r in csv.DictReader(f):
            i=match.get((r['Word'],normalize(r['PresentedSentenceDuringTest'])))
            if i is None:
                exclusions['unregistered_sentence']+=1;continue
            design=cases[i];fits=r['Fit']=='TRUE';familiar=r['Familiarity']=='TRUE'
            if fits!=design['fit'] or familiar!=design['familiar']:raise ValueError('Mismatched experimental assignment')
            mirrored=int(r['ResponseMirrored']);judgment=mirrored if fits else 7-mirrored
            if not 1<=judgment<=6 or ((judgment>=4)==fits)!=(r['Accuracy']=='TRUE') or int(abs(judgment-3.5)+.5)!=int(r['Confidence']):
                raise ValueError('Signed endorsements and confidence do not agree with official coding')
            rows.append((i,int(r['PP']),r['Cond']=='retrieval',judgment))
    return np.asarray(rows,dtype=int),dict(exclusions)


def native_features(bank,item):
    key=item['id'];x=bank[0][key].astype('float64');s=bank[1][key]
    if len(x)!=item['length'] or len(x)!=len(s):raise ValueError('Native input changed')
    return np.r_[x[-1],x.mean(0),s[1:].mean(),s[1:].max()]


def assemble(folder,cases):
    receipt=read_json(folder/'receipt.json');seal=read_json(OUT/'input-seal.json')
    if receipt['stimuli']!=seal['stimuli'] or receipt['onset_vocabulary']!=seal['vocabulary'] or digest(DIR/'cases.json')!=seal['mapping']:
        raise ValueError('Judgment inputs or forecast mapping changed')
    for name,sha in receipt['sha256'].items():
        if digest(folder/(name+'.npz'))!=sha:raise ValueError('Native forecast is corrupt')
    code=read_json(DIR/'vocabulary.json')['onsets']
    bank=[np.load(folder/(name+'.npz')) for name in ('states','surprisal','onsets')]
    position=[];semantic={k:[] for k in ('raw','known','native')};prior=[]
    for r in cases:
        item=r['ids']['raw'];s=r['sentence'];words=re.findall(r"\w+",s.casefold())
        position.append([np.log1p(len(s)),np.log1p(len(words)),np.log1p(len(r['word'])),item['first']/item['length'],len(set(words))/len(words)])
        for k in semantic:semantic[k].append(native_features(bank,r['ids'][k]))
        if item['first']==0:
            prior.append(np.nan)
        else:
            z=bank[2][item['id']][item['first']-1].astype(float);prior.append(z[code.index(r['onset'])]-logsumexp(z))
    for b in bank:b.close()
    return np.array(position),{k:np.asarray(v) for k,v in semantic.items()},np.array(prior),receipt


def designs(cases,position,semantic):
    fits=np.array([r['fit'] for r in cases],dtype=float);fam=np.array([r['familiar'] for r in cases],dtype=float)
    # Rows are stable case-condition combinations, independent of the participant and outcome.
    c=np.tile([0.,1.],len(cases))
    def repeat(x):return np.repeat(x,2,axis=0)
    f=repeat(fits);h=repeat(fam);s=repeat(position)
    offline=np.column_stack((s,h,c,h*c))
    oracle=np.column_stack((offline,f,f*h,f*c,f*h*c))
    m=len(c)
    return {'constant':np.empty((m,0)), 'surface':s, 'design_surface':offline,
            **{k:repeat(v) for k,v in semantic.items()},
            'oracle':oracle,'oracle_with_known':np.column_stack((oracle,repeat(semantic['known'])))}


def fit_predict(x, tr, y, weights, test, ridge=.03):
    w=weights/weights.sum()
    mu=w@x[tr];sd=np.sqrt(w@np.square(x[tr]-mu))
    sd[sd<1e-6]=1
    train=np.column_stack((np.ones(len(tr)),(x[tr]-mu)/sd));query=np.column_stack((np.ones(len(test)),(x[test]-mu)/sd))
    k=train.shape[1];cut=np.arange(1,6);observed=y[:,None]>cut
    # Rows with the same case and practice condition have identical input.
    # Their label distribution and mass are sufficient statistics.
    keys,inverse=np.unique(tr,return_inverse=True)
    mass=np.bincount(inverse,weights=w,minlength=len(keys))
    sums=np.stack([np.bincount(inverse,weights=w*observed[:,j],minlength=len(keys)) for j in range(5)],axis=1)
    A=np.column_stack((np.ones(len(keys)),(x[keys]-mu)/sd))
    def objective(flat):
        coef=flat.reshape(k,5);z=A@coef;reg=coef.copy();reg[0]=0
        nll=(mass[:,None]*np.logaddexp(0.,z)-sums*z).sum()
        grad=A.T@(mass[:,None]*expit(z)-sums)+ridge*reg
        return float(nll+.5*ridge*np.square(reg).sum()),grad.ravel()
    opt=minimize(objective,np.zeros(k*5),method='L-BFGS-B',jac=True,
                 options={'maxiter':350,'ftol':1e-14,'gtol':1e-7})
    grad=float(abs(opt.jac).max())
    if grad>3e-5 or not np.isfinite(opt.fun):raise ValueError(f'Ordinal regression failed stationarity check: {grad}')
    p=np.sort(expit(query@opt.x.reshape(k,5)),axis=1)[:,::-1]
    return p,{'gradient':grad,'iterations':int(opt.nit),'loss':float(opt.fun)}


def crossfit(cases,rows,design):
    ix,people,practice,y=rows.T;groups=np.array([r['word'] for r in cases])[ix]
    frames=np.array([r['frame'] for r in cases]);wx=folds(groups,'20261122-semantic-word:');px=folds(list(map(str,people)),'20261122-semantic-reader:')
    wf=np.array([wx[g] for g in groups]);pf=np.array([px[str(v)] for v in people]);casestr=ix*2+practice
    predictions={n:np.full((len(rows),5),np.nan) for n in design};checks=[]
    for i in range(5):
        for j in range(5):
            test=(wf==i)&(pf==j);train=(wf!=i)&(pf!=j);before=int(sum(train))
            testframes=set(frames[ix[test]]);train &= ~np.isin(frames[ix],list(testframes))
            loads=Counter(groups[train]);weights=np.array([1/loads[g] for g in groups[train]])
            if set(groups[train])&set(groups[test]) or set(people[train])&set(people[test]):raise ValueError('Pair component leakage')
            unique,inv=np.unique(casestr[test],return_inverse=True)
            fits={}
            for arm,x in design.items():
                q,fits[arm]=fit_predict(x,casestr[train],y[train],weights,unique)
                predictions[arm][test]=q[inv]
            checks.append({'word_fold':i,'reader_fold':j,'train_rows':int(train.sum()),'test_rows':int(test.sum()),
                           'train_rows_excluded_for_carrier_overlap':before-int(train.sum()),'fits':fits})
    if any(not np.isfinite(a).all() for a in predictions.values()):raise ValueError('Incomplete held-out forecasts')
    return predictions,checks


def rank_corr(a,b):
    a=rankdata(a);b=rankdata(b);a-=a.mean();b-=b.mean();den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den else None


def summarize(cases,rows,pred,prior):
    ix,people,practice,y=rows.T;words=np.array([s['word'] for s in cases]);g=words[ix]
    truth=y>=4;observed=y[:,None]>np.arange(1,6);grps=np.unique(ix);golds=np.array([r['fit'] for r in cases])[grps]
    means=np.array([truth[ix==k].mean() for k in grps])
    est={};losses={}
    for arm,p in pred.items():
        q=p[:,2].clip(1e-9,1-1e-9)
        ll=-(truth*np.log(q)+(1-truth)*np.log1p(-q));rps=np.square(p-observed).mean(axis=1)
        pm=np.array([q[ix==k].mean() for k in grps])
        est[arm]={'endorsement_nll':float(ll.mean()),'six_point_rps':float(rps.mean()),'endorsement_accuracy':float(np.mean((q>=.5)==truth)),
                  'mean_ordinal_prediction':float((1+p.sum(1)).mean()),'item_endorsement_rho':rank_corr(means,pm),
                  'rho_within_fit':rank_corr(means[golds],pm[golds]),'rho_within_nofit':rank_corr(means[~golds],pm[~golds])}
        losses[arm]=np.c_[ll,rps]
    tests={}
    for first,base in [('known','raw'),('known','surface'),('raw','surface'),('native','raw'),('known','design_surface'),('oracle_with_known','oracle')]:
        d=losses[base]-losses[first]
        tests[first+'__over__'+base]={k:{axis:confidence([v.mean() for v in (d[groups==k,j] for k in np.unique(groups))]) for axis,groups in [('word',g),('reader',people)]} for j,k in enumerate(('endorsement_log_loss_reduction','six_point_rps_reduction'))}
    early={}
    for familiarity in (False,True):
        items={}
        for i,case in enumerate(cases):
            if case['familiar']==familiarity and np.isfinite(prior[i]):items.setdefault(case['word'],{})[case['fit']]=prior[i]
        dif=[d[True]-d[False] for d in items.values() if len(d)==2]
        early['familiar' if familiarity else 'unfamiliar']={'paired_words':len(dif),'fit_minus_no_fit_true_meaning_log_probability':confidence(dif),
                    'paired_concordance':confidence([float(x>0)+.5*(x==0) for x in dif])}
    behavior={}
    for fit in (False,True):
        for familiar in (False,True):
            take=np.array([cases[c]['fit']==fit and cases[c]['familiar']==familiar for c in ix]);t=y[take]
            behavior[f'fit={fit}, familiar={familiar}']={'trials':len(t),'endorsement':float((t>=4).mean()),'ordinal_mean':float(t.mean())}
    return {'fitted_arms':est,'comparisons':tests,'pre_noun_independent_scores':early,'unavailable_first_word':int((~np.isfinite(prior)).sum()),
            'human_endorsement_by_experimental_condition':behavior,'scored_trials':len(rows),'readers':len(set(people)),'words':len(set(g)),'sentences':len(grps)}


def main(folder):
    cases=read_json(DIR/'cases.json');rows,exclusions=outcomes(cases);surface,sem,prior,receipt=assemble(folder,cases)
    pred,checks=crossfit(cases,rows,designs(cases,surface,sem));target=DIR/'human-crossfit-predictions.npz';np.savez_compressed(target,**pred)
    write_json(OUT/'forecast-receipt.json',{'features':receipt,'protocol':digest(Path(__file__).with_name('meaning-protocol.md')),
            'scores_code':digest(Path(__file__)),'forecasts':digest(target),'convergence_and_overlap':checks})
    report=summarize(cases,rows,pred,prior);report.update(exclusions=exclusions,forecasts=digest(target));write_json(OUT/'scores.json',report)
    print('Matched:',report['readers'],report['words'],report['scored_trials'])
    print({k:v['endorsement_nll'] for k,v in report['fitted_arms'].items()})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('forecasts',type=Path);main(p.parse_args().forecasts)
