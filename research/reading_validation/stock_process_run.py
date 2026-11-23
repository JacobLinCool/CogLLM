"""Estimate conserved issue curves and their independent suspense observation in four intact stories."""
import argparse
import json
from pathlib import Path
import numpy as np
from research.reading_validation.memory_data import sha
from research.reading_validation.narrative_course_score import count_outcomes,identity,spearman
from research.reading_validation.narrative_survival_fit import pages
from research.reading_validation.narrative_survival_score import observation_loss,uncued
from research.reading_validation.stock_process import Design,Fit,fit_affect,suspense

OUT=Path('research/results/reading-validation-stock-process')
PLAN=Path(__file__).with_name('stock-process-protocol.md')
ARMS=('lifetime','count','joint')
def forecast(archive,old):
    bank=pages(archive,old);OUT.mkdir(exist_ok=True);folder=OUT/'forecast';folder.mkdir(exist_ok=True);models={}
    if (folder/'seal.json').exists():raise FileExistsError('Complete trajectory experiment already exists')
    for name in sorted(bank):
        path=folder/(name+'.npz');receipt=folder/(name+'.json')
        if receipt.exists():
            prior=json.loads(receipt.read_bytes())
            if prior['hash']!=sha(path) or prior['protocol']!=sha(PLAN) or prior['archive']!=sha(archive/'receipt.json'):raise ValueError('Incompatible partly completed story')
            models[name]=prior;continue
        source=[q for k,q in bank.items() if k!=name];design=Design(source);train=[design.page(p) for p in source];probe=design.page(bank[name]);optimizer=Fit(train)
        initial=None;diagnostics={};arrays={}
        for mode in ARMS:
            v,diag=optimizer.fit(train,mode,design.birth_scale,initial);initial=v.copy()
            if mode!='joint':v=fit_affect(train,v,optimizer)
            val=probe.forecast(optimizer.unpack(v));diag['conservation']=val.pop('conservation');diag['parameters']=v.tolist();val['suspense_change']*=design.response_scale
            arrays.update({mode+'.'+k:z for k,z in val.items()});diagnostics[mode]=diag
            print('Completed external story forecast',name,mode,diag['iterations'],diag['gradient'],flush=True)
        np.savez_compressed(path,**arrays);result=dict(hash=sha(path),protocol=sha(PLAN),archive=sha(archive/'receipt.json'),reader_residual_scale=design.response_scale,fits=diagnostics)
        receipt.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');models[name]=result
    (folder/'seal.json').write_text(json.dumps(dict(protocol=sha(PLAN),archive=sha(archive/'receipt.json'),old=sha(old/'receipt.json'),stories={k:v['hash'] for k,v in models.items()}),indent=2)+'\n')
def score(archive,old):
    bank=pages(archive,old);folder=OUT/'forecast';seal=json.loads((folder/'seal.json').read_bytes());records={}
    for k,p in bank.items():
        if sha(folder/(k+'.npz'))!=seal['stories'][k]:raise ValueError('Altered story predictions')
        y=np.diff(suspense(p),prepend=np.nan);eligible=np.isfinite(y);records[k]={}
        with np.load(folder/(k+'.npz')) as f:
            for arm in ARMS:
                hz,event,stock,delta=[f[arm+'.'+label] for label in ('hazard','event','stock','suspense_change')]
                who=identity(p.story,{'conditional':hz,'unmarked':event})['methods'];num=count_outcomes(p.story,{'state':stock})['state']
                records[k][arm]=dict(count_interval_mse=num['interval_mse'],change_direction=num['direction'],count_rank=num['rank_midpoint'],
                     conditional_source=who['conditional']['paired_accuracy'],source_at_known_closure=who['unmarked']['top1_any_earlier'],
                     lifetime_nll=observation_loss(p,hz),uncued_ap=uncued(p,event)['ap'],
                     suspense_change_rmse=float(np.sqrt(np.mean((delta[eligible]-y[eligible])**2))),suspense_change_rho=spearman(delta,y))
    equal={m:{a:float(np.mean([r[m][a] for r in records.values() if r[m][a] is not None])) for a in next(iter(records.values()))[m]} for m in ARMS}
    (OUT/'scores.json').write_text(json.dumps(dict(forecast=sha(folder/'seal.json'),stories=records,equal_story=equal),indent=2,allow_nan=False)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('fit','score'));p.add_argument('archive',type=Path);p.add_argument('old',type=Path);a=p.parse_args();(forecast if a.action=='fit' else score)(a.archive,a.old)
