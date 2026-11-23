"""Fit and seal supervised question-specific evidence hazards before joining test labels."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json


DATA=Path('.cache/cogllm/reading-validation')
RESULT=Path('research/results/reading-validation-identity')
ARMS=('joint','lexical','position','new_readout')
RIDGE=0.015
PAIR_WEIGHT=2.0


class Features:
    def __init__(self, root):
        receipt=read_json(root/'geometry.json')
        self.channels=receipt['channels']
        self.values={}
        self.anchors={}
        for row in receipt['documents']:
            key=row['id']
            with np.load(root/f'{key}.npz',allow_pickle=False) as archive:
                self.values[key]=archive['pair']
            self.anchors[key]=row['anchors']
            if self.values[key].shape != (len(row['anchors']),row['tokens'],len(self.channels)):
                raise ValueError(f'Feature array does not match receipt: {key}')
        if len(self.values)!=receipt['forward_count'] or receipt['input_sha256']!=digest(DATA/'identity-stimuli.jsonl'):
            raise ValueError('Inference did not cover every registered input')


@dataclass
class RiskSet:
    x: np.ndarray
    inside: np.ndarray
    owner: np.ndarray
    weight: np.ndarray
    pairs: np.ndarray
    pair_weights: np.ndarray
    exposure: int
    events: int
    mean: np.ndarray
    sd: np.ndarray

    @classmethod
    def make(cls, data, rows, columns):
        groups=defaultdict(Counter)
        for r in rows:
            groups[r['source']][r['group_id']] += 1
        segments, bands, owners, weights=[],[],[],[]
        contrasts, contrast_weights=[],[]
        exposure, events=0,0
        for row in rows:
            key=row['id']; windows=row['window_tokens']; first=row['risk_start']; names=data.anchors[key]
            if set(names)!=set(windows):
                raise ValueError(f'Training anchor mismatch: {key}')
            docs=groups[row['source']]
            doc_weight=1.0/(len(groups)*len(docs)*docs[row['group_id']])
            matrices={}
            for name,array in zip(names,data.values[key],strict=True):
                matrices[name]=array[:,columns]
                window=windows[name]
                start,end=(len(array),len(array)) if window is None else window
                if not first <= start <= end <= len(array) or first>=end:
                    raise ValueError(f'Invalid first-hit region: {key}')
                segment=matrices[name][first:end]
                owners.append(np.full(len(segment),len(weights),dtype=np.int32))
                bands.append(np.arange(first,end)>=start)
                weights.append(doc_weight/len(names))
                segments.append(segment)
                exposure+=len(segment); events+=window is not None
            if len(names)==2 and all(windows.values()):
                # Conditional identity of an annotation within a shared physical paragraph.
                left,right=sorted(windows.values())
                if left[1]<=right[0]:
                    for target,other in (names,names[::-1]):
                        a,b=windows[target]
                        contrasts.append(np.mean(matrices[target][a:b]-matrices[other][a:b],axis=0,dtype=np.float64))
                        contrast_weights.append(doc_weight/2)
        x=np.ascontiguousarray(np.concatenate(segments),dtype=np.float64)
        mean=x.mean(0); sd=x.std(0); sd[sd<1e-6]=1.
        x-=mean; x/=sd
        if not contrasts:
            raise ValueError('Need paired assignments for the contrastive objective')
        w=np.array(contrast_weights); w/=w.sum()
        return cls(x,np.concatenate(bands),np.concatenate(owners),np.array(weights),
                   np.ascontiguousarray(np.array(contrasts)/sd),w,exposure,events,mean,sd)

    def objective(self, theta):
        coefs=theta[1:]; z=self.x@coefs+theta[0]
        per_step=np.logaddexp(0.,z)
        counts=len(self.weight)
        observed=np.bincount(self.owner, weights=self.inside,minlength=counts)>0
        before=np.bincount(self.owner,weights=per_step*(~self.inside),minlength=counts)
        inside=np.bincount(self.owner,weights=per_step*self.inside,minlength=counts)
        inside[observed]=np.maximum(inside[observed],1e-220)
        l=before.copy()
        l[observed]-=np.log(-np.expm1(-inside[observed]))
        factor=np.zeros(counts)
        # e^{-s}/(1-e^{-s}) is stable for both large and small interval hazard.
        factor[observed]=-np.exp(-inside[observed])/(-np.expm1(-inside[observed]))
        delta=self.weight[self.owner]*np.where(self.inside,factor[self.owner],1.)*expit(z)
        margins=self.pairs@coefs
        pair_loss=self.pair_weights@np.logaddexp(0.,-margins)
        loss=float(self.weight@l+PAIR_WEIGHT*pair_loss+0.5*RIDGE*(coefs@coefs))
        grad=self.x.T@delta-PAIR_WEIGHT*self.pairs.T@(self.pair_weights*expit(-margins))+RIDGE*coefs
        return loss,np.r_[delta.sum(),grad]


def fit_arm(data,rows,columns):
    risk=RiskSet.make(data,rows,columns)
    start=np.zeros(len(columns)+1)
    prior=(risk.events+.5)/(risk.exposure+1.)
    start[0]=np.log(prior/(1.-prior))
    result=minimize(risk.objective,start,method='L-BFGS-B',jac=True,
                    options={'maxiter':700,'gtol':2e-7,'ftol':2e-13,'maxls':40})
    grad=float(np.max(np.abs(result.jac)))
    if not result.success or not np.isfinite(result.fun) or grad>2e-5:
        raise ArithmeticError(f'Insufficient stationarity: {result.message}, gradient={grad}')
    return {'coef':result.x[1:].tolist(),'bias':float(result.x[0]),'columns':columns,
            'mean':risk.mean.tolist(),'sd':risk.sd.tolist(),'loss':float(result.fun),
            'max_abs_gradient':grad,'iterations':int(result.nit),'position_probability':prior,
            'questions':len(risk.weight),'at_risk_tokens':risk.exposure,'events':risk.events,
            'optimizer_status':str(result.message)}


def fit_models(data,rows):
    indices=list(range(len(data.channels)))
    lex=[i for i,k in enumerate(data.channels) if k.partition('/')[0] in ('exact','character')]
    return {'new_readout':fit_arm(data,rows,indices), 'lexical':fit_arm(data,rows,lex)}


def predict(models,array):
    probability=models['new_readout']['position_probability']
    outputs={'position':np.full(array.shape[:2],probability), 'joint':expit(array[:,:,0].astype(np.float64))}
    for key,m in models.items():
        terms=(array[:,:,m['columns']]-np.array(m['mean']))/np.array(m['sd'])
        outputs[key]=expit(terms@np.array(m['coef'])+m['bias'])
    return np.stack([outputs[arm] for arm in ARMS])


def fold(key):
    return int.from_bytes(sha256(('CogLLM identity OOF v1 '+key).encode()).digest()[:4],'big')%5


def fit(data_dir,cache,output):
    data=Features(data_dir)
    train=list(stream_rows(DATA/'identity-train.jsonl'))
    marker=read_json(RESULT/'input-seal.json')
    if digest(DATA/'identity-train.jsonl')!=marker['files']['train']:
        raise ValueError('Training input label seal mismatch')
    output.mkdir(parents=True,exist_ok=True); cache.mkdir(parents=True,exist_ok=True)
    fitted={}; forecasts={}
    keys={r['id'] for r in train}; qasper={r['id'] for r in train if r['source']=='qasper'}
    for split in ['official',*range(5)]:
        if split=='official':
            rows=train
            evaluate=set(data.values)-keys
        else:
            evaluate={k for k in qasper if fold(k)==split}
            rows=[r for r in train if r['id'] not in evaluate]
        if evaluate & {r['id'] for r in rows}:
            raise ValueError('A forecast paper is also used for its own fit')
        models=fit_models(data,rows); fitted[str(split)]=models
        for k in sorted(evaluate):
            if k in forecasts:
                raise ValueError('Forecast was produced twice')
            forecasts[k]=predict(models,data.values[k])
        print(f'Frozen identity models {split}, {len(rows)} fitting documents, {len(evaluate)} forecast documents',flush=True)
    np.savez_compressed(cache/'forecasts.npz',**forecasts)
    write_json(output/'readout.json',{'version':1,'features':data.channels,'arms':ARMS,'models':fitted,
                                      'ridge':RIDGE,'pair_weight':PAIR_WEIGHT,'protocol_sha256':marker['protocol_sha256']})
    write_json(output/'forecast-seal.json',{'version':1,'input_seal_sha256':digest(RESULT/'input-seal.json'),
                                           'protocol_sha256':marker['protocol_sha256'],'features_sha256':digest(data_dir/'geometry.json'),
                                           'weights_sha256':digest(output/'readout.json'),'forecasts_sha256':digest(cache/'forecasts.npz'),
                                           'model_code_sha256':digest(Path(__file__)),'cases':sorted(forecasts),
                                           'test_labels_opened_during_fit':False,'scoring_required':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_dir',type=Path)
    parser.add_argument('--cache',type=Path,default=DATA/'identity-fit');parser.add_argument('--output',type=Path,default=RESULT)
    fit(**vars(parser.parse_args()))
