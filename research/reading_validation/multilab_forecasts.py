"""Seal all nine-laboratory predictions before opening either EEG outcome column."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from scipy.special import expit
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.convergent_adaptive import BASE
from research.reading_validation.convergent_word_score import predict
from research.reading_validation.multilab_materials import ROOT,OUT,PLAN


def apply_english(X,ctype):
    fit=read_json(BASE/'adaptive'/(ctype+'-fold-5.json'));spec=fit['projection'];m=fit['models']['shared']
    z=[np.ones((len(X),1)),(X[:,:6]-spec['mean'])/spec['sd']]
    if ctype=='context':z.append(((X[:,6:]-spec['context_mean'])/spec['context_scale'])@np.asarray(spec['basis']))
    value=np.column_stack(z)@np.asarray(m['B']);value[:,1]=expit(value[:,1]);return value*m['scale']+m['mu']


def lock(folder):
    seal=read_json(OUT/'input-seal.json');receipt=read_json(folder/'receipt.json');p=folder/'word-states.npz'
    if digest(PLAN)!=seal['protocol'] or receipt['input']!=seal['stimuli'] or receipt['word_states']!=digest(p) or digest(ROOT/'conditions.jsonl')!=seal['conditions']:raise ValueError('Native forecast archive differs from certified original materials')
    spanish=read_json(BASE/'scores.json')['external_spanish_no_updates']['spanish_coefficients'];rows=[];prefixes={};prefix_checks=[]
    with np.load(p) as states:
        for r in stream_rows(ROOT/'conditions.jsonl'):
            article=states[r['id']].astype(float)
            prefix=article[:r['article']]
            if r['item'] in prefixes:
                other=prefixes.pop(r['item'])
                if prefix.shape!=other.shape:raise ValueError('Original invariant context changed in model input')
                delta=abs(prefix-other).max(0)
                if delta[5]>.04 or delta[6:].max()>.012:raise ValueError('A future word leaked into common-prefix readout')
                prefix_checks.append([float(delta[5]),float(delta[6:].max())])
            else:prefixes[r['item']]=prefix
            x=article[[r['article'],r['noun']]]
            en=apply_english(x,'context');surf=apply_english(x,'surface');es=predict(x,spanish['context'])
            signals={'raw_surprisal':x[:,5],'unresolved':x[:,6],'english_context_EEG':en[:,0],'english_surface_EEG':surf[:,0],'english_cloze_miss':en[:,1],'english_unpredictability':en[:,2],'spanish_context_EEG':es[:,0],'spanish_early_control':es[:,1],'spanish_surface_EEG':predict(x[:,:6],spanish['qwen_surface'])[:,0]}
            rows.append(dict(id=r['id'],item=r['item'],condition=r['condition'],forecasts={name:value.tolist() for name,value in signals.items()}))
    if prefixes or len(prefix_checks)!=80:raise ValueError('Incomplete common-prefix validation')
    if len(rows)!=160 or len(set(r['id'] for r in rows))!=160:raise ValueError('Incomplete independent printed contrasts')
    path=OUT/'blinded-forecasts.json';write_json(path,rows);write_json(OUT/'forecast-lock.json',dict(forecasts=digest(path),source_inputs=digest(OUT/'input-seal.json'),physiological_coefficients=digest(BASE/'scores.json'),english_readouts=[digest(BASE/'adaptive'/(name+'-fold-5.json')) for name in ('surface','context')],native_receipt=digest(folder/'receipt.json'),native_artifact=digest(p),program=digest(Path(__file__)),stored_locations=2*len(rows),common_prefix_families=len(prefix_checks),prefix_max_abs_surprisal_and_native=np.max(prefix_checks,axis=0).tolist()));print('Sealed',2*len(rows),'word forecasts per readout')


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);lock(p.parse_args().folder)
