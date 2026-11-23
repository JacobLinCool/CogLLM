"""Optimize evidence sharing on actual source essays and estimate endpoint borrowing before evaluation."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from research.reading_validation.argument_data import OUT
from research.reading_validation.argument_features import Corpus
from research.reading_validation.argument_labels import bank
from research.reading_validation.argument_shared import Distribution,features
from research.reading_validation.memory_data import sha

PLAN=Path(__file__).with_name('argument-joint-protocol.md')


def expert_readout(path,arm):
    fit=json.loads(path.read_bytes())[arm]
    return {name:{k:np.array(v) if k in ('mu','sd','v') else v for k,v in spec.items()} for name,spec in fit.items()}


def coefficients(dev,first,shared,metadata):
    # Each observed opportunity returns its exact probability (including accepted sets).
    observations=defaultdict(list);counts=Counter();topics=defaultdict(set)
    for k,record in dev.items():
        if k not in first:continue
        a,b=first[k],shared[k];topic=metadata[k]['topic']
        def add(unit,pa,pb):
            observations[unit].append((k,topic,float(pa),float(pb)));counts[unit,k]+=1;topics[unit,topic].add(k)
        for j in range(len(a['event'])):
            true=record['links'].get(j,[])
            if j:
                if true:
                    ix=(j*(j-1))//2+np.array(true);add('choice',a['choice'][ix].sum(),b['choice'][ix].sum())
                if record['complete'] and j not in record['exclude']:
                    py=bool(true);add('event',a['event'][j] if py else 1-a['event'][j],b['event'][j] if py else 1-b['event'][j])
            for y in record['grades'].get(j,[]):add('grade',a['grade'][j,y],b['grade'][j,y])
    result={}
    for name,rows in observations.items():
        aa=np.array([x[2] for x in rows]);bb=np.array([x[3] for x in rows])
        w=np.array([1/(len(topics[name,t])*counts[name,k]) for k,t,*_ in rows])
        # One equally weighted vote per essay, with equal total weight per topic.
        essays=len({k for k,_,_,_ in rows});w*=essays/w.sum()
        def loss(q):
            return -(w@np.log(((1-q)*aa+q*bb).clip(1e-15)))-np.log(q)-np.log1p(-q)
        opt=minimize_scalar(loss,method='bounded',bounds=(1e-7,1-1e-7),options={'xatol':1e-8})
        if not opt.success:raise ArithmeticError('Borrowing calibration failed')
        result[name]=dict(shared_fraction=float(opt.x),development_essays=essays,observations=len(rows))
    return result


def run(arm,archive):
    out=OUT/('forecast-joint-'+arm)
    if out.exists() and any(out.iterdir()):raise FileExistsError('A nonempty joint forecast is already present')
    out.mkdir(exist_ok=True)
    folder=OUT/('forecast-native' if arm=='native' else 'forecast-form')
    original=json.loads((folder/'seal.json').read_bytes())
    if sha(folder/'model.json')!=original['files']['model.json']:raise ValueError('Source-only independent readouts changed')
    corpus=Corpus(archive);meta=corpus.meta['articles'];expert=expert_readout(folder/'model.json',arm)
    training=bank('train');measure=Distribution(features(corpus,sorted(training),arm,expert,training))
    fits={name:measure.fit(opened) for name,opened in [('fixed',False),('joint',True)]}
    print('Actual source converged',[(k,v[1]) for k,v in fits.items()],flush=True)
    keys=[k for k,r in meta.items() if r['role']!='train' and corpus.eligible(k)]
    measure_test=Distribution(features(corpus,keys,arm,expert),measure.scales)
    preds={m:measure_test.unpack(v,m=='joint') for m,(v,_) in fits.items()}
    with np.load(folder/'forecasts.npz') as f:
        basic={k:{y:f[k+'.'+arm+'.'+y].astype(float) for y in ('choice','grade','event')} for k in keys}
    borrow=coefficients(bank('dev'),basic,preds['joint'],meta)
    forecasts={};variants=['isolated','fixed','joint','half','adaptive']
    for k in keys:
        for name in ('choice','grade','event'):
            left,right=basic[k][name],preds['joint'][k][name]
            alpha=borrow[name]['shared_fraction']
            for m,a in zip(variants,(left,preds['fixed'][k][name],right,.5*(left+right),(1-alpha)*left+alpha*right),strict=True):
                forecasts[k+'.'+m+'.'+name]=a.astype('float32')
    np.savez_compressed(out/'forecasts.npz',**forecasts)
    model=dict(fits={k:dict(parameters=v.tolist(),**diag) for k,(v,diag) in fits.items()},
               scales=[v.tolist() for v in measure.scales],borrowing=borrow)
    (out/'model.json').write_text(json.dumps(model,indent=2,allow_nan=False)+'\n')
    (out/'seal.json').write_text(json.dumps(dict(protocol=sha(PLAN),source=sha(OUT/'inputs.json'),base=sha(folder/'seal.json'),
             archive=original['archive'],keys=original['keys'],arms=variants,files={f:sha(out/f) for f in ('model.json','forecasts.npz')}),indent=2)+'\n')
    print('Sealed original/indirect/joint held predictions',len(keys),borrow,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('arm',choices=('form','native'));p.add_argument('--archive',type=Path)
    run(**vars(p.parse_args()))
