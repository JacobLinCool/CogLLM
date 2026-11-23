"""Evaluate the frozen Spanish N400 coefficients on every exact English EEG word exposure."""
from argparse import ArgumentParser
from pathlib import Path
import csv
import numpy as np
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.convergent_word_score import OUT,ROOT,assemble,predict,table_comparisons,correlations


def score(old,new):
    src=read_json(OUT/'erp-exposure-seal.json');model=read_json(OUT/'scores.json')['external_spanish_no_updates']['spanish_coefficients'];newmeta=read_json(new/'receipt.json');npz=new/'word-states.npz'
    if digest(npz)!=newmeta['word_states'] or newmeta['input']!=src['stimuli'] or digest(ROOT/'erp-map.jsonl')!=src['map']:raise ValueError('Unverified original-exposure replay')
    (_,_,_,X,H,c,_,_),_=assemble(old);old_by={int(v[1]):(x,y[0],v[0]) for x,y,v in zip(X,H,c)};new_by={};parity=[];gold={int(row['item_id']):row for row in csv.DictReader((ROOT/'outcomes.csv').open())};group=[]
    with np.load(npz) as z:
        for d in stream_rows(ROOT/'erp-map.jsonl'):
            native=z[d['id']].astype(float)
            for item,idx,same in d['items']:
                if same:
                    prior,_,_=old_by[item];parity.append(np.max(abs(prior-native[idx])))
                else:
                    if item in old_by or item in new_by:raise ValueError('Repeated original EEG outcome')
                    new_by[item]=(native[idx],-float(gold[item]['N400']),d['sentence'])
    if len(new_by)!=180 or set(new_by)|set(old_by)!=set(gold) or len(gold)!=1726 or max(parity)>.04:raise ValueError('Original stimulus join is incomplete or future text changed a prediction')
    results={};rows={**old_by,**new_by}
    for name,lookup in [('missing_180',new_by),('entire_1726',rows)]:
        items=sorted(lookup);x=np.vstack([lookup[i][0] for i in items]);y=np.array([lookup[i][1] for i in items])[:,None];ids=np.array([lookup[i][2] for i in items]);P={'context':predict(x,model['context'])[:,0],'context_early_control':predict(x,model['context'])[:,1],'qwen_surface':predict(x[:,:6],model['qwen_surface'])[:,0],'surface':predict(x[:,:5],model['surface'])[:,0],'qwen_surprisal':x[:,5],'pending':x[:,6]};c=correlations(P,y,ids);comp=table_comparisons(c,[('context','qwen_surface'),('context','qwen_surprisal'),('context','surface'),('context','context_early_control')]);results[name]=dict(words=len(x),sentences=len(set(ids)),rho={k:v[0][0] for k,v in c.items()},comp=comp)
    out=dict(coefficient_checkpoint=digest(OUT/'scores.json'),replay=digest(new/'receipt.json'),join=digest(OUT/'erp-exposure-seal.json'),exact_overlap_positions=len(parity),max_absolute_overlap_feature_drift=max(parity),observations=results);write_json(OUT/'original-exposure-scores.json',out);print('EEG replay complete',[(k,v['rho']['context']['rho']) for k,v in results.items()])


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('original',type=Path);p.add_argument('replay',type=Path);a=p.parse_args();score(a.original,a.replay)
