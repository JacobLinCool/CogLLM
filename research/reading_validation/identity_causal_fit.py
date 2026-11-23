"""Fit a causal antecedent classifier and separate cue-free and lexical controls."""

import argparse
from collections import Counter
from pathlib import Path
import re
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.identity_prepare import DATA, TOKENIZER


OUT=Path('research/results/reading-validation-causal'); SCALE=[.03,.15,.75]
IGNORE=set('a an the and who what when how why where was were be is are of to in at by with for he she it their as i his her him its had have on from this that do only so did they we'.split())
WORD=re.compile(r'[^\W_]+',re.UNICODE)


def normalized(v):
    return v / max(float(np.linalg.norm(v)/np.sqrt(v.size)),.0001)


def lexical(a,b):
    extract=lambda text:[w for w in WORD.findall(text.casefold()) if w not in IGNORE]
    aa,bb=extract(a),extract(b); A,B=set(aa),set(bb); both=len(A&B)
    chars=lambda s:set().union(*[{'^'+w[:2],w[-2:]+'$',*(w[j:j+3] for j in range(len(w)-2))} for w in s]) if s else set()
    ac,bc=chars(A),chars(B)
    return np.array([both/max(1,len(A)),both/max(1,len(B)),both/max(1,len(A|B)),len(ac&bc)/max(1,len(ac|bc)),
                     np.log1p(len(aa))/4,np.log1p(len(bb))/4],dtype='float32')


def design(a,b,rel,distance,la,lb):
    aa,ae=map(normalized,a); ba,be=map(normalized,b); d=1./distance; words=lexical(la,lb)
    text=np.r_[words,d].astype('float32'); prior=np.r_[aa,ae,d,words[4]].astype('float32')
    full=np.concatenate((aa,ae,ba,be,aa*ba,ae*be,aa*be,abs(aa-ba),d*ba,d*be,rel,text)).astype('float32')
    return {'semantic':full,'lexical':text,'prior':prior}


class Bank:
    def __init__(self,root):
        meta=read_json(root/'geometry.json');self.ids=meta['ids']; self.index={k:i for i,k in enumerate(self.ids)}
        self.spans={k:v for k,v in zip(self.ids,meta['sentence_tokens'],strict=True)}
        self.starts=meta['sentence_offsets']; self.dense=set(meta['dense']); self.baseline=set(meta['baseline'])
        self.tokenizer=Tokenizer.from_file(str(TOKENIZER)); self.root=root
        self.text={r['id']:r['text'] for r in stream_rows(DATA/'causal-stimuli.jsonl')}; self.chunks={}; self.offsets={}
        for key,text in self.text.items():
            off=self.tokenizer.encode(text,add_special_tokens=False).offsets
            self.offsets[key]=off; self.chunks[key]=[text[off[a][0]:off[b-1][1]] for a,b in self.spans[key]]
        with np.load(root/'sentence-states.npz',allow_pickle=False) as parsed:
            self.bank=parsed['states']; self.rel=parsed['relations']; self.old=parsed['old']
        if set(self.ids)!=self.text.keys() or len(self.bank)!=self.starts[-1]:raise ValueError('Misaligned causal feature bank')

    def info(self,key):
        i=self.index[key];return self.bank[self.starts[i]:self.starts[i+1]].astype('float32'),self.rel[i].astype('float32')

    def candidates(self,key):
        states,rel=self.info(key);j=len(states)-1;text=self.chunks[key]
        return [design(states[i],states[j],rel[j,i],j-i,text[i],text[j]) for i in range(j)]


def vectors(bank,trials,arm):
    seen_rows=sum(len(r['views']) for r in trials); X=None;t=0;types=[]
    for r in trials:
        for view in r['views']:
            candidates=[d[arm] for d in bank.candidates(view['id'])]
            outcome=view['targets'];types.append('choice' if len(candidates)==2 else view['kind'])
            if len(candidates)!=len(outcome):raise ValueError('An annotated statement is not a separate parsed sentence')
            if X is None:X=np.empty((seen_rows,len(candidates[0])+1),dtype='float32')
            if len(candidates)==2:
                target=outcome.index(1);X[t,:-1]=candidates[target]-candidates[1-target]; X[t,-1]=0
            else:
                sign=2*outcome[0]-1; X[t,:-1]=sign*candidates[0];X[t,-1]=sign
            t+=1
    counts=Counter(types);weights=np.array([{'choice':.5,'single':.4,'repeat':.1}[name]/counts[name] for name in types],dtype='float32')
    return X,weights


def loss_and_grad(theta,X,weights,penalty):
    # SGEMV avoids a copy of the training matrix on each gradient evaluation.
    vector=theta.astype('float32'); z=(X@vector).astype('float64');
    ridge=theta.copy();ridge[-1]=0
    obj=weights@np.logaddexp(0.,-z)+.5*penalty*ridge@ridge
    grad=X.T@(-weights*expit(-z)).astype('float32')
    return float(obj),grad.astype(float)+penalty*ridge


def fit(bank,trials,validation,arm):
    X,w=vectors(bank,trials,arm); valid,vw=vectors(bank,validation,arm)
    scale=np.sqrt(np.einsum('ij,ij->j',X,X,dtype='float64')/len(X)).astype('float32');scale[scale<.01]=1;scale[-1]=1
    X/=scale;valid/=scale;models=[]
    for lam in SCALE:
        v=np.zeros(X.shape[1]); stamp=time.monotonic()
        res=minimize(loss_and_grad,v,args=(X,w,lam),method='L-BFGS-B',jac=True,options={'maxiter':500,'gtol':3e-6,'ftol':1e-12,'maxls':40})
        maxgrad=float(abs(res.jac).max())
        if not np.isfinite(res.fun) or maxgrad>8e-5:raise ArithmeticError(f'{arm} stationarity {maxgrad}, {res.message}')
        valid_loss=float(vw@np.logaddexp(0.,-(valid@res.x.astype('float32')).astype(float)))
        models.append({'arm':arm,'weights':(res.x[:-1]/scale[:-1]).tolist(),'bias':float(res.x[-1]),'train_loss':float(res.fun),
                       'calibration_loss':valid_loss,'penalty':lam,'iterations':int(res.nit),'max_abs_gradient':maxgrad,
                       'optimizer_status':str(res.message),'elapsed_seconds':time.monotonic()-stamp})
    selected=min(models,key=lambda m:m['calibration_loss']);selected['selection']=[{k:v for k,v in m.items() if k not in ('weights','bias','arm','selection')} for m in models]
    return selected


def flat_scores(models,data):
    return {arm:float(vector@np.asarray(models[arm]['weights'])+models[arm]['bias']) for arm,vector in data.items()}


def trace(bank,models,key,states,att,anchors=None):
    n=len(states);spans=bank.spans[key];text=bank.text[key];off=bank.offsets[key]
    if anchors is None:anchors={str(i):(a,b) for i,(a,b) in enumerate(spans)}
    else:anchors={k:tuple(v) for k,v in anchors.items()}
    level={a:np.zeros(n,dtype='float64') for a in anchors}; related={a:np.zeros(n,dtype='float64') for a in anchors}
    states=states.astype('float32');full_attn=att.shape[1]==n
    if anchors and not full_attn and any(x not in list(map(tuple,spans)) for x in anchors.values()):
        raise ValueError('Supplied anchors require genuine span-level native attention')
    tau=models.get('alert_threshold',0.)
    for i,(begin,end) in anchors.items():
        start_sentence=next(j for j,(s,e) in enumerate(spans) if s<=begin<e);a_text=text[off[begin][0]:off[end-1][1]]
        a=np.stack((states[begin:end].mean(0),states[end-1])); maximum=0.
        for j,(s,e) in enumerate(spans):
            if j<=start_sentence or s<end: continue
            total=np.zeros_like(states[0]);cur_len=0
            for t in range(s,e):
                current=states[t];total+=current;cur_len+=1
                if full_attn:
                    rel=np.log1p(np.array([att[t,begin:end,0].astype('float32').mean(),att[t,begin:end,1].max()])*(t+1))
                else:rel=att[t,start_sentence].astype('float32')
                b_text=text[off[s][0]:off[t][1]]
                features=design(a,(total/cur_len,current),rel,j-start_sentence,a_text,b_text)
                scores=flat_scores(models,features);p=expit(scores['semantic']);prior=expit(scores['prior'])
                gain=max(0.,(p-prior)/max(1-prior,1e-9));related[i][t]=gain
                maximum=max(maximum,max(0.,gain-tau)/max(1.-tau,1e-9));level[i][t]=maximum
        level[i]=np.maximum.accumulate(level[i])
    return level,related


def fit_all(root):
    bank=Bank(root);source=read_json(OUT/'input-seal.json')
    if digest(DATA/'causal-ecare-train.jsonl')!=source['source_grades_sha256']['ecare-train'] or digest(DATA/'causal-calibration.jsonl')!=source['source_grades_sha256']['calibration']:
        raise ValueError('Changed training or calibration source')
    train=list(stream_rows(DATA/'causal-ecare-train.jsonl'));val=list(stream_rows(DATA/'causal-calibration.jsonl'))
    models={}
    for name in ('lexical','prior','semantic'):
        models[name]=fit(bank,train,val,name);print(name,models[name]['calibration_loss'],models[name]['max_abs_gradient'],flush=True)
    null=[]
    with np.load(root/'token-states.npz',allow_pickle=False) as ss,np.load(root/'token-attention.npz',allow_pickle=False) as aa:
        for item in val:
            for view in item['views']:
                key=view['id'];
                if key in bank.dense and view['kind'] in ('single','repeat') and not view['targets'][0]:
                    _,curve=trace(bank,models,key,ss[key],aa[key]);null.append(float(max(curve['0'])))
    if len(null)<80:raise ValueError('Too few eligible causal nulls to calibrate')
    models['alert_threshold']=float(np.quantile(null,.90,method='higher'));models['calibration_null_views']=len(null)
    write_json(OUT/'weights.json',models)
    # Forecast every stimulus with two prior statements. Held-out grades remain unopened.
    preds={}
    for key in bank.ids:
        if len(bank.spans[key])!=3:continue
        rows=[flat_scores(models,v) for v in bank.candidates(key)]
        preds[key]={arm:[v[arm] for v in rows] for arm in ('semantic','lexical','prior')}
        if key in bank.baseline:
            ix=bank.index[key];preds[key]['joint']=bank.old[ix,2,:2].astype(float).tolist()
    write_json(OUT/'choice-forecasts.json',preds)
    source_layout=read_json(Path('.cache/cogllm/token-readout/linked-prepared/qwen.json'))['rows'];art={}
    with np.load(root/'token-states.npz',allow_pickle=False) as ss,np.load(root/'token-attention.npz',allow_pickle=False) as aa:
        for key,g in source_layout.items():
            if g['kind'] not in ('ab','controls'):continue
            s=ss[key];a=aa[key];auto,_=trace(bank,models,key,s,a);given,_=trace(bank,models,key,s,a,g['anchor_tokens'])
            # Evaluation maps the source window to its previously generated antecedent candidate.
            mapped={}
            for name,(lo,hi) in g['anchor_tokens'].items():
                j=next(j for j,(start,end) in enumerate(bank.spans[key]) if start<=lo<end)
                mapped[name]=auto[str(j)]
            def hazard(levels):
                res={}
                for k,v in levels.items():
                    before=np.r_[0.,v[:-1]];res[k]=(np.maximum(0.,v-before)/np.maximum(1.-before,1e-9)).tolist()
                return {'hazard':res}
            art[key]={'automatic':hazard(mapped),'supplied':hazard(given),'all_candidates':hazard(auto),'candidate_sentences':bank.spans[key]}
    write_json(OUT/'authored-forecasts.json',art)
    write_json(OUT/'forecast-seal.json',{'data_sha256':source['input_sha256'],'weights_sha256':digest(OUT/'weights.json'),
                                        'semantic_forecast_sha256':digest(OUT/'choice-forecasts.json'),'authored_forecast_sha256':digest(OUT/'authored-forecasts.json'),
                                        'protocol_sha256':source['protocol_sha256'],'test_outcomes_loaded_during_fitting':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);fit_all(p.parse_args().root)
