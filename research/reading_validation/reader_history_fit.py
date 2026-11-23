"""Fit source-only population-precision effects with held questionnaire rows and sentences."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.reader_history_materials import OUT,DIR,PLAN
from research.reading_validation.reader_history_model import response_law,scores,fit_prior,predict,report
from research.reading_validation.cloze_population_fit import data,assign
from research.reading_validation.cloze_population_model import Choices,fit,distribution


def event_table(native):
    raw,_,U,V,_,meta=data(native)
    sel=np.flatnonzero(meta['source'])
    pos=meta['ids'][sel];groups=meta['articles'][sel]
    species=assign(groups);fold=np.array([species[k] for k in groups])
    with np.load(DIR/'source-observations.npz') as f:events=f['events']
    byid={int(k):i for i,k in enumerate(pos)}
    readers,list_id,id0,choices=events.T
    ev=np.column_stack((readers,[byid[k] for k in id0],choices))
    lists={r:int(min(list_id[readers==r])) for r in np.unique(readers)}
    fitread=set();testread=set()
    for k in set(lists.values()):
        identities=sorted([r for r in lists if lists[r]==k],key=lambda r:sha256(f'UCL reader partition {r}'.encode()).digest())
        testread.update(identities[::4]);fitread.update(r for r in identities if r not in testread)
    if fitread&testread:raise ValueError("Reader split contamination")
    return raw[sel],U,V,groups,fold,ev,fitread,testread,lists


def counts(events):
    by=sorted(set(events[:,1]));ids=[];hits=[]
    for i in by:
        w,n=np.unique(events[events[:,1]==i,2],return_counts=True)
        ids.append(w);hits.append(n)
    return np.array(by),Choices(ids,hits)


def calibrate(raw,U,V,model):
    extra=np.empty((len(raw),0))
    return distribution(raw,extra,model,U,V).astype(np.float32)


def run(native):
    meta=read_json(OUT/'source-readers.json')
    if meta['canonical_source_file']!=digest(DIR/'source-observations.npz') or meta['protocol']!=digest(PLAN):
        raise ValueError("Source responses changed")
    raw,U,V,sg,fold,ev,fitread,testread,lists=event_table(native)
    fitr=np.isin(ev[:,0],list(fitread));heldr=~fitr;history=[];scored=[]
    for k in range(6):
        training=ev[(fitr&(fold[ev[:,1]]!=k)) if k<5 else np.ones(len(ev),bool)]
        q=OUT/f'fold-{k}.json'
        if q.exists():
            done=read_json(q);cal=done['single'];prior=np.asarray(done['prior'])
            if done['source']!=digest(OUT/'source-readers.json'):raise ValueError("Wrong prior")
            lp=calibrate(raw,U,V,cal)
        else:
            positions,obs=counts(training)
            cal={'kind':'calibrated',**fit(raw[positions],np.empty((len(positions),0)),obs,U,V)}
            lp=calibrate(raw,U,V,cal)
            z,_=response_law(lp)
            prior,info=fit_prior(scores(lp,z,training),training[:,0])
            write_json(q,dict(source=digest(OUT/'source-readers.json'),fitted_word_positions=len(positions),
                training_sessions=len(set(training[:,0])),training_responses=len(training),single=cal,prior=prior.tolist(),inference=info))
        history.append(digest(q));print('Reader prior',k,len(training),'max KKT residual',read_json(q)['inference']['stationarity'],flush=True)
        if k==5:continue
        histories={};quer={};material={}
        for r in sorted(testread):
            x=ev[ev[:,0]==r]
            train=x[fold[x[:,1]]!=k];new=x[fold[x[:,1]]==k]
            order=sorted(range(len(train)),key=lambda j:sha256(f'fixed questionnaire history {sg[train[j,1]]} {train[j,1]}'.encode()).digest())
            if len(order)<20 or not len(new):continue
            histories[r]=train[order[:20]]
            quer[r]=new
            material[r]=set(train[:,1])
            if set(histories[r][:,1])&set(new[:,1]):raise ValueError("Personal calibration uses later target items")
        E,L,O,match=predict(lp,prior,histories,quer,material)
        scored.append((E,L,O));np.savez_compressed(DIR/f'fold-{k}-readouts.npz',event=E,NLL=L,other=O)
    e,l,o=[np.vstack(v) for v in zip(*scored)]
    results=report(e,l,o,materials=sg)
    results['held_questionnaire_sessions']=len(testread);results['source_training_sessions']=len(fitread)
    results['history']='Twenty answered items from other source sentences in the same questionnaire. Individual field response times are unavailable.'
    results['independent_identity_confirmation']=meta['verified_nonmissing_worker_keys']
    write_json(OUT/'source-scores.json',results)
    write_json(OUT/'source-lock.json',dict(fits=history,protocol=digest(PLAN),chronology=digest(OUT/'chronology-seal.json'),prior_model=digest(Path(__file__).with_name('reader_history_model.py')),
               external_response_choices_consumed=False))
    print('Held UCL',results['scored_responses'],results['paired'])


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('native',type=Path);run(p.parse_args().native)
