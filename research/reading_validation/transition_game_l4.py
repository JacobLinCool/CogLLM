"""Fit one frozen Qwen reader to measured human antecedent disagreement in disjoint texts."""
from argparse import ArgumentParser
from collections import Counter
import gc
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.bridge_forecast import STOP
from research.reading_validation.transition_architecture_l4 import native_losses,verify_logits
from research.reading_validation.transition_direct_l4 import learn
from research.reading_validation.transition_game_optimizer import learn_shared,self_test
from research.token_readout.backbone import Backbone

FORMS=tuple(sorted('he him his himself she her hers herself they them their theirs themself themselves it its itself'.split()))
CONTROL=('word_surprisal','word_length','is_capitalized','log_word_order','log_block','block_first','log_within_block',
         'log_same_form_repetitions','log_previous_form_distance','previous_capitalized','capitalized_rate',
         'recent_distinct_capitalized_forms','distance_to_nearest_capitalized','distance_to_second_capitalized',
         'recency_competition','recent_personal_pronoun_count','same_form_in_recent_window')+tuple('form_'+x for x in FORMS)
ARMS=('surface','direct','shared')
PROTOCOL='373bb5cf8cd007f08613d1b7846bd3e9fe47bde2977a815d366bf3086bcb89b6'


def controls(row,loss):
    names=[row['text'][a:b] for a,b,_,_ in row['words']]
    at={j:(start,end) for j,(start,end) in zip(row['save'],row['token'],strict=True)}
    sums=np.r_[0.,np.cumsum(loss)];last={};freq=Counter();names_seen={};cap_count=0;prior_pronouns=[]
    x=[]
    for j,(word,bounds) in enumerate(zip(names,row['words'],strict=True)):
        key=word.casefold();cap=word[0].isupper()
        if j in at:
            a,b=at[j];dist=sorted(j-i for i in names_seen.values() if i>=j-60)
            q=np.exp(-np.array(dist,dtype=float)/20.)
            if q.size:q/=q.sum()
            x.append([sums[b+1]-sums[a],math.log1p(len(word)),float(cap),math.log1p(j+1),math.log1p(bounds[2]+1),
                      float(bounds[3]==0),math.log1p(bounds[3]+1),math.log1p(freq[key]),math.log1p(j-last[key]) if key in last else 0.,
                      float(j>0 and names[j-1][0].isupper()),cap_count/max(1,j),math.log1p(len(dist)),
                      math.log1p(dist[0]) if dist else 0.,math.log1p(dist[1]) if len(dist)>1 else 0.,
                      1-float(q@q) if q.size else 0.,math.log1p(sum(i>=j-60 for i in prior_pronouns)),float(key in last and j-last[key]<=60)]
                     +[float(key==form) for form in FORMS])
        if cap and key not in STOP:names_seen[key]=j
        if key in FORMS:prior_pronouns.append(j)
        cap_count+=cap;last[key]=j;freq[key]+=1
    result=np.asarray(x,dtype=np.float32).reshape(len(at),len(CONTROL))
    if not np.isfinite(result).all():raise ValueError('Nonfinite causal form baseline')
    return result


@torch.inference_mode()
def extract(model,rows,temp):
    runtime=Backbone.load('qwen',model,native_attention=False);reference=verify_logits(runtime)
    width=runtime.width;size=sum(len(r['save']) for r in rows)
    disk=np.lib.format.open_memmap(temp/'raw.npy',mode='w+',dtype='float32',shape=(size,len(CONTROL)+2*width))
    spans={};start=0;calls=0
    def observe(*args):
        nonlocal calls
        calls+=1
    with runtime.encoder.register_forward_pre_hook(observe):
        for r in rows:
            f=runtime.encode(r['text']);k=r['save'];positions=r['token']
            if len(f.token_ids)!=r['token_count'] or any(f.offsets[last][1]!=r['words'][j][1] for j,(_,last) in zip(k,positions,strict=True)):
                raise ValueError('Original pronoun word ends changed')
            ix=torch.tensor([v[1] for v in positions],device=runtime.device,dtype=torch.long)
            hidden=torch.cat((F.layer_norm(f.mid[ix],(width,)),F.layer_norm(f.hidden[ix],(width,))),dim=1)
            x=controls(r,native_losses(runtime,f));end=start+len(k)
            disk[start:end]=np.column_stack((x,hidden.cpu().numpy()));spans[r['id']]=(start,end);start=end
            if calls!=len(spans):raise ValueError('Reading forecast required more than one full article pass')
            if calls%40==0:print(f'Game native word features {calls}/{len(rows)}',flush=True)
    disk.flush()
    return disk,spans,width,{'revision':runtime.revision,'forwards':calls,'saved_words':size,'native_logits':reference}


def source_weights(labels,groups):
    mass=np.zeros_like(labels,dtype=np.float64)
    for t in range(labels.shape[1]):
        visible=labels[:,t]>=0;contributing=np.unique(groups[visible])
        if not len(contributing):raise ValueError('Source task lacks visible responses')
        for key in contributing:
            here=visible&(groups==key)
            mass[here,t]=1/(labels.shape[1]*len(contributing)*here.sum())
    return mass


def fit(X,labels,groups,dims,shared):
    y=torch.tensor(np.maximum(labels,0),device=X.device,dtype=X.dtype)
    mass=torch.tensor(source_weights(labels,groups),device=X.device,dtype=X.dtype)
    return learn_shared(X,y,mass,dims) if shared else learn(X,y,mass,dims,False)


def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-50,50)))


def run(model,stimuli,output,protocol_sha256,input_sha256):
    if protocol_sha256!=PROTOCOL or digest(stimuli)!=input_sha256:raise ValueError('Wrong silver training or registered protocol')
    self_test()
    rows=list(map(json.loads,stimuli.read_text().splitlines()))
    if any(('labels' not in r)!=(r['fold']=='test') for r in rows):raise ValueError('Gold player outcome leaked into native extraction')
    output.mkdir(parents=True,exist_ok=True);start=perf_counter()
    with TemporaryDirectory(dir='/content') as tmp:
        disk,spans,width,meta=extract(model,rows,Path(tmp));extraction=perf_counter()-start
        gc.collect();torch.cuda.empty_cache()
        n=len(disk);observations=np.full((n,4),-1.,dtype=float);groups=np.full(n,-1,dtype=int)
        domain=np.zeros(n,dtype=bool);source=domain.copy();dev=domain.copy()
        names=sorted({r['group'] for r in rows if r['fold']=='train'})
        for r in rows:
            a,b=spans[r['id']];domain[a:b]=r['domain']=='game'
            if 'labels' in r:observations[a:b]=r['labels']
            if r['fold']=='train':source[a:b]=True;groups[a:b]=names.index(r['group'])
            if r['fold']=='dev':dev[a:b]=True
        controls_end=len(CONTROL);all_p=np.zeros((n,len(ARMS)));fits={};states={}
        observed_dev=dev&(observations[:,3]>=0)
        for arm_id,name in enumerate(ARMS):
            cols=controls_end if name=='surface' else controls_end+2*width
            mask=source if name=='shared' else source&domain
            X=torch.tensor(disk[:,:cols],device='cuda',dtype=torch.float64)
            mu=X[mask].mean(0);sd=X[mask].std(0,unbiased=False);sd=torch.where(sd>1e-10,sd,torch.ones_like(sd))
            if cols>controls_end:sd[controls_end:]*=math.sqrt(width/48)
            X.sub_(mu).div_(sd)
            ys=observations[mask] if name=='shared' else observations[mask,3:4]
            w,b,record=fit(X[mask],ys,groups[mask],2*width if cols>controls_end else 0,name=='shared')
            eta=(X@w[:,-1]+b[-1]).cpu().numpy();low=-40.;high=40.
            target=observations[observed_dev,3].mean()
            for iteration in range(80):
                middle=(low+high)/2
                if sigmoid(eta[observed_dev]+middle).mean()<target:low=middle
                else:high=middle
            offset=(low+high)/2;risk=sigmoid(eta+offset);all_p[:,arm_id]=risk
            if abs(risk[observed_dev].mean()-target)>1e-9:raise ValueError('Source-only calibration did not converge')
            record.update(offset=offset,quintile_edges=np.quantile(risk[observed_dev],[.2,.4,.6,.8]).tolist(),
                          observed_development_rate=float(target),uncalibrated_development_rate=float(sigmoid(eta[observed_dev]).mean()))
            fits[name]=record
            for key,v in [('weights',w),('bias',b),('mean',mu),('sd',sd)]:states[name+'__'+key]=v.cpu().numpy()
            del X
            # Preserve validated individual fits even if a later, coupled objective needs more work.
            partial=output/(name+'-source-stage.npz')
            np.savez_compressed(partial,**{key:val for key,val in states.items() if key.startswith(name+'__')},
                                heldout_probabilities=all_p[dev|~source,arm_id])
            write_json(output/(name+'-source-stage.json'),{'input_sha256':input_sha256,'source_fit':record,
                       'arm':name,'partial_sha256':digest(partial)})
            print(name,record,flush=True)
        preds=output/'game-forecasts.npz';coeff=output/'readouts.npz'
        np.savez_compressed(preds,**{r['id']:all_p[slice(*spans[r['id']])] for r in rows if r['fold'] in ('test','dev')})
        np.savez_compressed(coeff,**states)
        write_json(output/'receipt.json',{'protocol_sha256':PROTOCOL,'input_sha256':input_sha256,'native':meta,
             'arms':ARMS,'surface_columns':CONTROL,'fits':fits,'forecasts_sha256':digest(preds),'weights_sha256':digest(coeff),
             'extraction_seconds':extraction,'seconds':perf_counter()-start})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for k in ('model','stimuli','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('protocol-sha256','input-sha256'):p.add_argument('--'+k,required=True)
    run(**vars(p.parse_args()))
