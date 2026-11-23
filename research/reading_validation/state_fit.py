"""Seal competing first-event forecasts before joining any test responses or state outcomes."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.reading_validation.memory_data import h,sha
from research.reading_validation.state_events import EventEstimator,balance,build_pairs,combine,materials,mixture,pair_rows
from research.reading_validation.state_transport_data import OUT
from research.reading_validation.state_words import Archive,StateEstimator,state_training

PLAN=Path(__file__).with_name('state-transport-estimator.md')

def save_model(model):
    return {k:getattr(model,k).tolist() for k in ('mu','sd','v')}|{'fit':model.diag}

def store_fold(out,tag,rows,archive):
    p=out/('attribute-'+tag+'.npz');certificate=p.with_suffix('.seal.json')
    shape=h('|'.join(sorted(k for r,k in rows)))
    if certificate.exists():
        saved=json.loads(certificate.read_bytes())
        if sha(p)!=saved['sha'] or saved['source']!=shape or saved['plan']!=sha(PLAN) or saved['archive']!=archive.certificate:
            raise ValueError('Incompatible partial model cache')
        with np.load(p) as f:
            m=StateEstimator.__new__(StateEstimator);m.mu=f['mu'];m.sd=f['sd'];m.v=f['v'];m.diag=saved['fit'];return m
    m=state_training(rows,archive);np.savez_compressed(p,mu=m.mu,sd=m.sd,v=m.v)
    certificate.write_text(json.dumps(dict(sha=sha(p),source=shape,plan=sha(PLAN),fit=m.diag,archive=archive.certificate),indent=2)+'\n')
    print('Converged attribute model',tag,m.diag,flush=True);return m

def run(arm,native):
    out=OUT/('forecast-'+arm);out.mkdir(exist_ok=True)
    if (out/'seal.json').exists():raise FileExistsError('A verified physical forecast already exists')
    archive=Archive(native);training,ex1=materials(archive,'train');dev,ex2=materials(archive,'dev');held,ex3=materials(archive,'test')
    families=sorted({s['family'] for r,s in training.values()},key=lambda k:h('physical-crossfit-'+str(k)))
    folds={g:i%3 for i,g in enumerate(families)};modes=['form','state'] if native is None else ['form','surprisal','context','state']
    rows=[(r,s['key']) for r,s in training.values()];cross=[]
    for fold in range(3):cross.append(store_fold(out,str(fold),[(r,k) for r,k in rows if folds[r['story_id']]!=fold],archive))
    full=store_fold(out,'full',rows,archive);bank=build_pairs(training,archive,cross,folds,True)
    extractors={m:EventEstimator(training,bank,m) for m in modes};print('Event readouts',[(m,a.diag) for m,a in extractors.items()],flush=True)
    del bank
    probe=build_pairs({**dev,**{('held-'+k):v for k,v in held.items()}},archive,full,None)
    # Independent study texts enter with the same full-text-only mechanism.
    for k in archive.spec['groups']['frick'].values():
        if archive.eligible(k):
            doc=archive.document(k);probe[k]=pair_rows(doc,full.predict(doc['x']))
    forecasts={m:{k:f.predict(t,m) for k,t in probe.items()} for m,f in extractors.items()}
    coeff=mixture(dev,forecasts,modes);forecasts['mixture']=combine(forecasts,modes,coeff)
    arrays={};state_arrays={};weights=balance(dev);thresholds={}
    for m,values in forecasts.items():
        review=[]
        for ident,(r,s) in dev.items():
            pp=values[s['key']];review.append((1.-np.prod([a[0] for a in pp]),float(not r['plausible']),weights[ident]))
        prob,y,w=np.array(review).T;cuts=np.unique(np.r_[0.,prob,1.]);acc=np.array([w@((prob>=t)==y) for t in cuts])
        thresholds[m]=float(cuts[np.flatnonzero(acc>=max(acc)-1e-12)[-1]])
        for k,v in values.items():arrays[k+'.'+m]=np.concatenate(v).astype('float32')
    for key in {s['key'] for r,s in held.values()}:
        state_arrays[key]=full.predict(archive.document(key)['x']).astype('float32')
    np.savez_compressed(out/'events.npz',**arrays);np.savez_compressed(out/'states.npz',**state_arrays)
    fits=dict(modes=modes,dev_mixture=dict(zip(modes,coeff)),thresholds=thresholds,event={m:save_model(a) for m,a in extractors.items()},
              excluded={'train':ex1,'dev':ex2,'held':ex3},source_family_folds=folds)
    (out/'model.json').write_text(json.dumps(fits,indent=2,allow_nan=False)+'\n')
    files=['model.json','events.npz','states.npz']+['attribute-'+i+'.seal.json' for i in ('0','1','2','full')]
    (out/'seal.json').write_text(json.dumps(dict(protocol=sha(PLAN),source=sha(OUT/'inputs.json'),archive=archive.certificate,
                                       files={f:sha(out/f) for f in files}),indent=2)+'\n')
    print('Original unseen story predictions sealed',len(arrays),coeff,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('arm',choices=('form','native'));p.add_argument('--native',type=Path)
    a=p.parse_args()
    if (a.arm=='native')!=(a.native is not None):p.error('Native mode requires the original verified L4 archive')
    run(**vars(a))
