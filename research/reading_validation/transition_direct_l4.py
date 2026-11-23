"""One original forward, source-balanced low-rank training, and blinded forecasts."""
from argparse import ArgumentParser
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
from research.reading_validation.transition_architecture_l4 import CHECKPOINTS, native_losses, verify_logits
from research.reading_validation.transition_snac_forecast import design_from_features
from research.token_readout.backbone import Backbone
from research.token_readout.readout import PortableReadout

PACKET='5ba394da96bd18b4859aed77d5c0b154980f18bd0eed5908855ff12ef0ce4467'
PROTOCOL='7701cbe2cea263e635aeeab518314e108dd5be2560e256137bcff79e0ca1dcb4'
NER='9f906e30f9f998385c4fb4a76c67be95c2c98e7822a7f191a4920362400d64be'
TASKS=('CharE','RefE','CorefE')
ARMS=('surface','ner','historic','direct','shared')


@torch.inference_mode()
def extract(model,candidate,weights,rows,ner,folder):
    runtime=Backbone.load(candidate,model,native_attention=True)
    packet=torch.load(weights,map_location='cpu',weights_only=True)
    if runtime.revision!=packet['revision'] or candidate!=packet['candidate']:raise ValueError('Wrong text snapshot')
    reader=PortableReadout(runtime.width,packet['variant'],packet['relations']).to(runtime.device)
    reader.load_state_dict(packet['reader'],strict=True);reader.eval().requires_grad_(False)
    parity=verify_logits(runtime);stations={};n=sum(len(r['words']) for r in rows)
    width=runtime.width;old=2*reader.final.out_features;ng=ner[rows[0]['id']].shape[1]
    disk=np.lib.format.open_memmap(folder/'states.npy',mode='w+',dtype='float32',shape=(n,32+old+ng+2*width))
    ncall=0
    def observe(*args):
        nonlocal ncall
        ncall+=1
    start=0
    with runtime.encoder.register_forward_pre_hook(observe):
        for r in rows:
            d=r['id'];w=[s[:2]+t+s[2:] for s,t in zip(r['words'],r['token_ranges'][candidate],strict=True)]
            f=runtime.encode(r['text'])
            if len(f.token_ids)!=r['token_counts'][candidate] or any(f.offsets[last][1]!=end for _,end,_,last,_,_ in w):
                raise ValueError('Token geometry differs from the human source word')
            end=start+len(w);stations[d]=(start,end)
            traditional=design_from_features(runtime,reader,{'text':r['text'],'words':w},f,native_losses(runtime,f))
            ix=torch.tensor([a[3] for a in w],device=runtime.device)
            hidden=torch.cat((F.layer_norm(f.mid[ix],(width,)),F.layer_norm(f.hidden[ix],(width,))),dim=1)
            # Game NER is outside the registered external comparators. Keep zero placeholders out of that forecast.
            tag=ner[d][r['indices']] if r['split']!='independent' else np.zeros((len(w),ng),dtype='float32')
            disk[start:end]=np.column_stack((traditional[:,:17],traditional[:,-15:],traditional[:,17:17+old],tag,
                                             hidden.float().cpu().numpy()))
            start=end
            if len(stations)!=ncall:raise ValueError('Unexpected extra article forward')
            if ncall%40==0:print(f'{candidate}: native features {ncall}/{len(rows)} articles',flush=True)
    disk.flush()
    return disk,stations,{'forwards':ncall,'words':n,'native_width':width,'historic_width':old,'ner_width':ng,
                           'official_logits':parity,'revision':runtime.revision,'global_layers':len(runtime.global_layers)}


def learn(X, labels, mass, semantic, pooled):
    d=X.shape[1];k=labels.shape[1]; lam=.01
    v=torch.zeros((d,k),device=X.device,dtype=X.dtype,requires_grad=True)
    priors=(labels*mass).sum(0)/mass.sum(0)
    b=torch.special.logit(priors.clamp(1e-8,1-1e-8)).detach().requires_grad_()
    solver=torch.optim.LBFGS((v,b),max_iter=220,tolerance_grad=5e-9,tolerance_change=1e-14,
                              history_size=40,line_search_fn='strong_wolfe')
    initial=None;calls=0
    def closure():
        nonlocal initial,calls
        solver.zero_grad()
        logits=X@v+b
        loss=(F.binary_cross_entropy_with_logits(logits,labels,reduction='none')*mass).sum()
        ridge=lam/(2*k)*v.square().sum()
        if pooled:
            w=v[-semantic:]
            singular=torch.linalg.eigvalsh(w.T@w+torch.eye(k,device=X.device,dtype=X.dtype)*1e-12).clamp_min(1e-12).sqrt()
            ridge=ridge+lam/(4*k)*(singular.sum().square()-w.square().sum())
        objective=loss+ridge
        objective.backward()
        if initial is None:initial=float(objective.detach())
        calls+=1
        return objective
    solver.step(closure);loss=float(closure().detach())
    grad=max(float(v.grad.abs().max()),float(b.grad.abs().max()))
    if not math.isfinite(loss+grad) or grad>1e-6 or loss>initial:
        raise ValueError(f'Stationarity check failed, objective={loss}, gradient={grad}, calls={calls}')
    return v.detach(),b.detach(),{'initial_objective':initial,'objective':loss,'max_abs_gradient':grad,'evaluations':calls}


def run(model,candidate,stimuli,checkpoint,ner,output,protocol_sha256):
    if (digest(stimuli)!=PACKET or digest(ner)!=NER or digest(checkpoint)!=CHECKPOINTS[candidate]
            or protocol_sha256!=PROTOCOL):raise ValueError('Unexpected direct-readout input or protocol')
    rows=list(map(json.loads,stimuli.read_text().splitlines()))
    if any(('labels' in r)!=(r['split'] in ('train','dev')) for r in rows):raise ValueError('Test outcome in accelerator packet')
    output.mkdir(exist_ok=True,parents=True);started=perf_counter()
    with TemporaryDirectory(dir='/content') as tmp:
        disk,spans,meta=extract(model,candidate,checkpoint,rows,np.load(ner),Path(tmp))
        extraction=perf_counter()-started
        gc.collect();torch.cuda.empty_cache()
        n=len(disk);tasks=len(TASKS);labels=np.full((n,tasks),-1,dtype=np.int8)
        train=np.zeros(n,dtype=bool); dev=train.copy();group=np.full(n,-1,dtype=int)
        titles=sorted({r['group'] for r in rows if r['split']=='train'})
        for r in rows:
            a,b=spans[r['id']]
            if r['split'] in ('train','dev'):
                labels[a:b]=r['labels']
                if r['split']=='train':train[a:b]=True;group[a:b]=titles.index(r['group'])
                else:dev[a:b]=True
        mass=np.zeros_like(labels[train],dtype=float);target=labels[train]
        for j in range(tasks):
            for g in range(len(titles)):
                here=(group[train]==g)&(target[:,j]>=0)
                if not here.any():raise ValueError('Source title lacks observable task')
                mass[here,j]=1/(here.sum()*len(titles)*tasks)
        y=torch.tensor(target.clip(0),device='cuda',dtype=torch.float64)
        scales=torch.tensor(mass,device='cuda',dtype=torch.float64)
        nw=meta['native_width'];old=meta['historic_width'];ng=meta['ner_width']; raw=32+old+ng
        columns={'surface':np.arange(17),'ner':np.r_[0:17,32+old:raw],'historic':np.arange(32+old),
                 'native':np.r_[0:32,raw:raw+2*nw]}
        all_prob=np.zeros((n,len(ARMS),tasks),dtype='float32')
        checks={};models={};limits={}
        for block,idx in columns.items():
            feature=torch.tensor(disk[:,idx].T.copy().T,device='cuda',dtype=torch.float64)
            fit=feature[train];mean=fit.mean(0);sd=fit.std(0,unbiased=False)
            sd=torch.where(sd>1e-10,sd,torch.ones_like(sd))
            if block=='native':sd[-2*nw:]*=math.sqrt(nw/48)
            feature.sub_(mean).div_(sd);del fit
            fit=feature[train]
            arms=('direct','shared') if block=='native' else (block,)
            for name in arms:
                w,b,check=learn(fit,y,scales,2*nw if block=='native' else 0,name=='shared')
                p=(feature@w+b).sigmoid().cpu().numpy()
                limits[name]=[float(np.quantile(p[dev&(labels[:,j]==0),j],.95,method='higher')) for j in range(tasks)]
                all_prob[:,ARMS.index(name),:]=p
                checks[name]=check
                for key,arr in [('weights',w),('bias',b),('mean',mean),('sd',sd)]:
                    models[name+'__'+key]=arr.cpu().numpy()
                print(candidate,name,check,flush=True)
            del feature,fit
        packets={r['id']:all_prob[slice(*spans[r['id']])] for r in rows if r['split'] in ('test','independent')}
        # NER features are available only on SNaC; mark missing independent NER explicitly.
        for r in rows:
            if r['split']=='independent':packets[r['id']][:,ARMS.index('ner'),:]=np.nan
        preds=output/'word-probabilities.npz';fitted=output/'weights.npz'
        np.savez_compressed(preds,**packets);np.savez_compressed(fitted,**models)
        write_json(output/'receipt.json',{'candidate':candidate,'input_sha256':PACKET,'protocol_sha256':PROTOCOL,
            'checkpoint_sha256':digest(checkpoint),'ner_sha256':NER,'arms':ARMS,'tasks':TASKS,'cutoffs':limits,
            'optimizers':checks,'feature_metadata':meta,'training_titles':titles,'extraction_seconds':extraction,
            'elapsed_seconds':perf_counter()-started,'weights_sha256':digest(fitted),'predictions_sha256':digest(preds)})


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--candidate',required=True,choices=tuple(CHECKPOINTS))
    for f in ('model','stimuli','checkpoint','ner','output'):parser.add_argument('--'+f,required=True,type=Path)
    parser.add_argument('--protocol-sha256',required=True);run(**vars(parser.parse_args()))
