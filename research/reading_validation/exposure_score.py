"""Test prospective Dutch-meaning forecasts against the actual person-by-word practice responses."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,log_softmax
from scipy.stats import rankdata

from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.exposure_data import OUT,ROOT,key,read_csv
from research.reading_validation.identity_causal_score import confidence


def forecast(bank_dir):
    seal=read_json(OUT/'input-seal.json');receipt=read_json(bank_dir/'receipt.json');codes=read_json(ROOT/'vocabulary.json')['onsets']
    if receipt['input']!=seal['stimuli'] or receipt['vocabulary']!=seal['vocabulary'] or digest(bank_dir/'onsets.npz')!=receipt['archive']:
        raise ValueError('Unexpected public stimuli or forecasts')
    items=read_json(ROOT/'items.json');pred={};max_error=0.
    with np.load(bank_dir/'onsets.npz',allow_pickle=False) as tensors:
        for row in items:
            ix=codes.index(row['onset']);arms={}
            for arm,v in row['arms'].items():
                z=tensors[v['id']][v['end']].astype('float64');logp=log_softmax(z);rank=float(rankdata(-z)[ix])
                arms[arm]={'logp':float(logp[ix]),'rr':1./rank,'rank':rank,'entropy':float(-np.exp(logp)@logp)}
            early=row['arms']['sparse'];late=row['arms']['after']
            drift=float(abs(tensors[early['id']][early['end']]-tensors[late['id']][late['end']]).max());max_error=max(max_error,drift)
            if drift>.003:raise ValueError(f'An unseen future sentence changed logits by {drift}')
            pred[row['word']]={'arms':arms,'unique':row['unique_onset'],'cue_tokens':row['cue_tokens'],'foil_tokens':row['foil_tokens']}
    write_json(OUT/'forecasts.json',pred)
    write_json(OUT/'forecast-lock.json',{'forecasts':digest(OUT/'forecasts.json'),'source_seal':digest(OUT/'input-seal.json'),
        'max_future_logit_error':max_error,'grade_opened':False,'accelerator_receipt':digest(bank_dir/'receipt.json')})


def rho(x,y):
    a=rankdata(x);b=rankdata(y);a-=a.mean();b-=b.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den else np.nan


def summary_dist(value,dist):
    dd=np.asarray(dist);dd=dd[np.isfinite(dd)]
    return {'estimate':float(value),'bootstrap_95':np.quantile(dd,[.025,.975]).tolist(),'finite_draws':len(dd)} if np.isfinite(value) and len(dd) else None


def crossed(y,rich,w,r,features,reps=10000):
    # All designs include multiple participants per noun and several nouns per participant.
    nw=int(w.max()+1);nr=int(r.max()+1);ones=np.ones(len(y));shape=(nw,nr)
    pos=np.zeros(shape);ct=np.zeros(shape);neg=np.zeros(shape);nt=np.zeros(shape)
    np.add.at(pos,(w[rich],r[rich]),y[rich]);np.add.at(ct,(w[rich],r[rich]),ones[rich])
    np.add.at(neg,(w[~rich],r[~rich]),y[~rich]);np.add.at(nt,(w[~rich],r[~rich]),ones[~rich])
    def item_gain(v):
        a=ct@v;b=nt@v;return (pos@v)/np.maximum(a,1.)-(neg@v)/np.maximum(b,1.),(a>0)&(b>0)
    truth,valid=item_gain(np.ones(nr));keys=list(features);v0=[]
    for k in keys:v0.append(rho(features[k][valid],truth[valid]))
    v0.append(float(truth[valid].mean()));rng=np.random.default_rng(40539);trials=[[],[]]
    for _ in range(reps):
        ids=rng.integers(0,nw,size=nw);personal=np.bincount(rng.integers(0,nr,size=nr),minlength=nr)
        for j,(g,ok) in enumerate([(truth,valid),item_gain(personal)]):
            keep=ids[ok[ids]];trials[j].append([*[rho(features[k][keep],g[keep]) for k in keys],float(g[keep].mean())])
    return {kind:{k:summary_dist(value,np.asarray(d)[:,i]) for i,(k,value) in enumerate(zip([*keys,'rich_minus_sparse_accuracy'],v0,strict=True))}
            for kind,d in zip(['word_cluster','crossed_word_reader'],trials,strict=True)}


def cross_validated(records):
    y=np.array([r['y'] for r in records]);z=np.array([r['logp'] for r in records]);kn=np.array([r['prior'] for r in records]);rich=np.array([r['rich'] for r in records],dtype='float64')
    rw=[r['person'] for r in records];wd=[r['word'] for r in records]
    rf=np.array([int(key('reader '+v)[:8],16)%5 for v in rw]);wf=np.array([int(key('word '+v)[:8],16)%5 for v in wd])
    X=np.column_stack((kn,rich,kn*rich,z));preds={};grads={}
    for k,dim in [('prior_and_condition',3),('plus_Qwen',4)]:
        out=np.full(len(y),np.nan);grad=[]
        for i in range(5):
            for j in range(5):
                train=(rf!=i)&(wf!=j);test=(rf==i)&(wf==j)
                if not test.any():continue
                xx=X[:,:dim];mean=xx[train].mean(0);sig=xx[train].std(0).clip(.001);xx=np.column_stack((np.ones(len(y)),(xx-mean)/sig))
                x=xx[train];t=y[train]
                def obj(v):
                    a=x@v;reg=v.copy();reg[0]=0.
                    return np.logaddexp(0.,a).sum()-t@a+.5*(reg@reg),x.T@(expit(a)-t)+reg
                sol=minimize(obj,np.zeros(dim+1),jac=True,method='BFGS',options={'gtol':1e-7})
                g=float(np.abs(sol.jac).max());grad.append(g)
                if g>.0005:raise ValueError(f'Crossed-fold logistic fit did not converge: {g}')
                logits=xx[test]@sol.x;out[test]=np.logaddexp(0.,logits)-y[test]*logits
        if np.isnan(out).any():raise ValueError('Unscored reader/word cell')
        preds[k]=out;grads[k]=float(max(grad))
    delta=preds['prior_and_condition']-preds['plus_Qwen'];ur,ri=np.unique(rw,return_inverse=True);uw,wi=np.unique(wd,return_inverse=True)
    sums=np.zeros((len(ur),len(uw)));n=np.zeros_like(sums);np.add.at(sums,(ri,wi),delta);np.add.at(n,(ri,wi),1)
    rng=np.random.default_rng(55813);boot=[]
    for _ in range(10000):
        a=np.bincount(rng.integers(0,len(ur),len(ur)),minlength=len(ur));b=np.bincount(rng.integers(0,len(uw),len(uw)),minlength=len(uw))
        boot.append((a@sums@b)/(a@n@b))
    return {'nll':{k:float(v.mean()) for k,v in preds.items()},'joint_clustered_nll_reduction':summary_dist(delta.mean(),boot),'max_gradient':grads}


def score():
    lock=read_json(OUT/'forecast-lock.json')
    if digest(OUT/'forecasts.json')!=lock['forecasts']:raise ValueError('Forecasts changed after observing outcomes')
    pred=read_json(OUT/'forecasts.json');seal=read_json(OUT/'input-seal.json');result={};src=read_csv(ROOT/'stimuli_exp2.csv')
    structural={}
    for group,words in [('all',list(pred)),('unique_onset',[w for w,row in pred.items() if row['unique']])]:
        comparisons={}
        for comp in ('sparse','foil','after'):
            for q in ('logp','rr','entropy'):
                diff=[pred[w]['arms']['before'][q]-pred[w]['arms'][comp][q] for w in words]
                comparisons[q+'_before_minus_'+comp]={'difference':confidence(diff),'positive_fraction':float(np.mean(np.array(diff)>0))}
        structural[group]={'items':len(words),'estimates':comparisons,'winner':{arm:float(np.mean([pred[w]['arms'][arm]['rank']==1 for w in words])) for arm in ('before','sparse','foil')}}
    for exp in (1,2):
        path=next(ROOT.glob(f'Experiment {exp}. Information at WordLevel*.csv'))
        if digest(path)!=seal['source_outcome_hashes'][path.name]:raise ValueError('Trial source changed')
        rows=read_csv(path);records=[];excluded=Counter()
        richtexts={r['Word']:r['Context'].strip() for r in src}
        for r in rows:
            w=r['Word'];cond=r['Condition'];person=r['PP']
            if w not in pred:excluded['no_eligible_stimulus']+=1;continue
            if r['PractSentence'].strip()!=richtexts[w]:excluded['actual_rich_sentence_mismatch']+=1;continue
            if cond not in ('context','retrieval','retrievalc','retrievalf'):raise ValueError(f'Unknown practice arm {cond}')
            val=r['PractCorrectRespB1']
            if val not in ('TRUE','FALSE'):
                if val in ('','NA'):excluded['missing_response']+=1;continue
                raise ValueError(f'Unknown published binary response field {val!r}')
            arm='before' if cond=='context' else 'sparse'
            row={'word':w,'person':person,'condition':cond,'rich':cond=='context','y':int(val=='TRUE'),'logp':pred[w]['arms'][arm]['logp']}
            if exp==2:row['prior']=float(r['RatingKnowlPretr3'])
            records.append(row)
        if len({(x['person'],x['word']) for x in records})!=len(records):raise ValueError('Repeated first practice trial')
        out={}
        for kind,rec in [('all',records),('unique_onset',[x for x in records if pred[x['word']]['unique']])]:
            names,wi=np.unique([r['word'] for r in rec],return_inverse=True);people,ri=np.unique([r['person'] for r in rec],return_inverse=True)
            y=np.asarray([r['y'] for r in rec]);rich=np.asarray([r['rich'] for r in rec])
            full={k:np.array([pred[w]['arms']['before'][k]-pred[w]['arms']['sparse'][k] for w in names]) for k in ('logp','entropy')};full['entropy']*=-1
            stats=crossed(y,rich,wi,ri,full)
            out[kind]={'trials':len(rec),'participants':len(people),'words':len(names),'associations':stats,
                'observed_accuracy':{c:{'accuracy':float(np.mean([r['y'] for r in rec if r['condition']==c])),'n':sum(r['condition']==c for r in rec)}
                                     for c in sorted({r['condition'] for r in rec})}}
        result[f'experiment_{exp}']={'scopes':out,'exclusions':dict(excluded),'source_rows':len(rows)}
        if exp==2:result['experiment_2']['heldout_people_and_items']=cross_validated(records)
    write_json(OUT/'scores.json',{'forecasts':digest(OUT/'forecasts.json'),'model_specificity':structural,'human_responses':result})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('phase',choices=('forecast','score'));p.add_argument('bank',type=Path,nargs='?');a=p.parse_args()
    if a.phase=='forecast':forecast(a.bank)
    else:score()
