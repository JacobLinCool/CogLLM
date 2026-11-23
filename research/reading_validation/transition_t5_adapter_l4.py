"""Fit the same-generative T5-Large specialist with resumable low-rank updates."""
from argparse import ArgumentParser
from copy import deepcopy
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import transformers
from transformers import T5ForConditionalGeneration,T5Tokenizer,set_seed

from research.reading_validation.memory_train import LowRank,RANK
from research.reading_validation.transition_t5_l4 import (
    INPUT_SHA,REPO,REVISION,SEED,WEIGHTS,batch,digest,evaluate,examples,generate,model_path,schedule,write_json,
)

PROTOCOL='4d212bcfc6bd0039c7d82601df0230deca700a6080e714ca69e221df4fbc0eb3'
STEPS=1840
CHECK_EVERY=25


def optimizer_for(parameters):
    opt=torch.optim.Adam(parameters,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.,fused=True)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda t:max(0.,(STEPS-t)/STEPS))
    return opt,sched


def state_cpu(obj):
    if isinstance(obj,torch.Tensor):return obj.detach().to('cpu',copy=True)
    if isinstance(obj,dict):return {k:state_cpu(v) for k,v in obj.items()}
    if isinstance(obj,list):return [state_cpu(v) for v in obj]
    if isinstance(obj,tuple):return tuple(state_cpu(v) for v in obj)
    return deepcopy(obj)


def adapter_state(params):
    return {k:p.detach().to('cpu',copy=True) for k,p in params.items()}


def install(params,stored):
    if params.keys()!=stored.keys():raise ValueError('Unknown T5 adapter projection names')
    with torch.no_grad():
        for k,p in params.items():p.copy_(stored[k])


def attach(model):
    modules=[]
    for q in model.encoder.block:modules.append(q.layer[0].SelfAttention)
    for q in model.decoder.block:modules.extend((q.layer[0].SelfAttention,q.layer[1].EncDecAttention))
    if len(modules)!=72 or RANK!=8:raise ValueError('Expected 72 distinct T5-Large attention blocks')
    for att in modules:
        for name in ('q','v'):
            setattr(att,name,LowRank(getattr(att,name)))
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    if len(params)!=288 or sum(p.numel() for p in params.values())!=2359296 or any(p.dtype!=torch.float32 for p in params.values()):
        raise ValueError('Wrong adaptation capacity or trainable dtype')
    return params


def microbatches(group,budget):
    pile=[];width=0
    for row in group:
        n=len(row['input']);new=max(n,width)
        if pile and ((len(pile)+1)*new>budget or len(pile)==8):
            yield pile;pile=[];width=0
        pile.append(row);width=max(width,n)
    if pile:yield pile


def update(model,params,opt,sched,group,budget):
    model.train();opt.zero_grad(set_to_none=True);tokens=sum(len(r['target']) for r in group);nll=0.
    for small in microbatches(group,budget):
        f=batch(small,True);n=sum(len(r['target']) for r in small)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model(**f,use_cache=False).loss
        if not torch.isfinite(loss):raise ArithmeticError('Nonfinite LoRA source loss')
        (loss*(n/tokens)).backward();nll+=float(loss.detach())*n
    g=float(torch.nn.utils.clip_grad_norm_(params.values(),1.,error_if_nonfinite=True))
    opt.step();sched.step()
    return nll,tokens,g


def snapshot(meta,params,opt,sched,details):
    return {'meta':meta,'details':state_cpu(details),'adapter':adapter_state(params),'optimizer':state_cpu(opt.state_dict()),
            'scheduler':state_cpu(sched.state_dict()),'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}


def restore(blob,meta,params,opt,sched):
    if blob['meta']!=meta:raise ValueError('Snapshot differs in data, model, capacity, runtime or approved microbatch size')
    install(params,blob['adapter']);opt.load_state_dict(state_cpu(blob['optimizer']));sched.load_state_dict(state_cpu(blob['scheduler']))
    torch.set_rng_state(blob['cpu_rng']);torch.cuda.set_rng_state(blob['cuda_rng'])
    return state_cpu(blob['details'])


def advance_slot(out,meta,params,opt,sched,details):
    number=details['updates'];slot=(number//CHECK_EVERY)%2;final=out/f'resume-{slot}.pt'
    temporary=out/f'resume-{slot}.partial'
    torch.save(snapshot(meta,params,opt,sched,details),temporary);temporary.replace(final)
    write_json(out/'resume-manifest.json',{'step':number,'name':final.name,'sha256':digest(final),'bytes':final.stat().st_size,
                                         'epoch':details['epoch'],'protocol_sha256':PROTOCOL})


def preflight(model,params,data,requested=None):
    longest=sorted(data,key=lambda r:(len(r['input']),len(r['target'])),reverse=True)[:8]
    results={};rng=torch.cuda.get_rng_state();cpu_rng=torch.get_rng_state()
    for budget in ([requested] if requested else [3072,2048,1024]):
        if budget not in (3072,2048,1024):raise ValueError('Unknown source-length token budget')
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();model.train();model.zero_grad(set_to_none=True)
        failed=False
        try:
            # Backpropagate the worst admitted padded microbatch before any optimizer update.
            for small in microbatches(longest,budget):
                f=batch(small,True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=model(**f,use_cache=False).loss
                loss.backward();del f,loss
                break
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            failed=True
        model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.set_rng_state(rng);torch.set_rng_state(cpu_rng)
        used=torch.cuda.max_memory_allocated()
        results[str(budget)]={'reached_backward':not failed,'peak_bytes':used}
        if not failed and used<=21*1024**3:return budget,results
    raise RuntimeError(f'No registered microbatch budget met measured L4 memory requirements: {results}')


def run(stimuli,output,protocol_sha256,model=None,resume=None,resume_sha256=None):
    if digest(stimuli)!=INPUT_SHA or protocol_sha256!=PROTOCOL or bool(resume)!=bool(resume_sha256):
        raise ValueError('Unexpected T5-LoRA input, protocol or resume hash')
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_name()!='NVIDIA L4':
        raise RuntimeError('This experiment requires one Colab NVIDIA L4')
    output.mkdir(exist_ok=True,parents=True);start=perf_counter();set_seed(SEED);torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    saved=None
    if resume:
        if digest(resume)!=resume_sha256:raise ValueError('Resume artifact checksum mismatch')
        saved=torch.load(resume,map_location='cpu',weights_only=True)
    pre=model_path(model);tokenizer=T5Tokenizer.from_pretrained(pre,local_files_only=True)
    backbone=T5ForConditionalGeneration.from_pretrained(pre,local_files_only=True,use_safetensors=True,
        dtype=torch.bfloat16).to('cuda').eval().requires_grad_(False)
    params=attach(backbone);data,geometry=examples(stimuli,tokenizer)
    budget,mem=preflight(backbone,params,data['train'],saved['meta']['microbatch_budget'] if saved else None)
    meta={'protocol':PROTOCOL,'source':INPUT_SHA,'revision':REVISION,'base_weights':WEIGHTS,'seed':SEED,
          'torch':str(torch.__version__),'transformers':transformers.__version__,'microbatch_budget':budget,
          'steps':STEPS,'rank':RANK,'trainable_parameters':2359296}
    opt,sched=optimizer_for(params.values())
    if saved:
        log=restore(saved,meta,params,opt,sched);del saved
        print(f"Restored source fit at update {log['updates']}/{STEPS}",flush=True)
    else:
        initial=evaluate(backbone,data['dev'])
        log={'updates':0,'epoch':1,'next_group':0,'initial_dev':initial,'history':[],'best_loss':float('inf'),
             'best_epoch':None,'best_adapter':None,'epoch_nll':0.,'epoch_tokens':0,'gradient_sum':0.,
             'last_grad':0.,'recent_train':[],'training_seconds':0.}
    # Verify current momentum, dropout RNG, schedule and trainables by two discarded identical updates.
    verify=snapshot(meta,params,opt,sched,log);short=sorted(data['train'],key=lambda r:len(r['input']))[:2]
    first=update(backbone,params,opt,sched,short,budget);after=adapter_state(params)
    restore(verify,meta,params,opt,sched)
    second=update(backbone,params,opt,sched,short,budget)
    drift=max(float((p.detach().cpu()-after[k]).abs().max()) for k,p in params.items())
    if abs(first[0]-second[0])>1e-5 or drift>1e-7:raise RuntimeError(f'T5 optimizer restoration failed: {drift}')
    log=restore(verify,meta,params,opt,sched);del verify,after
    if log['updates']!=sched.last_epoch:raise ValueError('Scheduler and saved optimizer location differ')
    advance_slot(output,meta,params,opt,sched,log);stamp=perf_counter()
    print(f"LoRA resumed/initialized; development NLL {log['initial_dev']:.4f}, start step {log['updates']}, token budget {budget}",flush=True)
    while log['epoch']<=5:
        epoch=log['epoch'];groups=schedule(data['train'],SEED+epoch)
        for i in range(log['next_group'],len(groups)):
            nll,tokens,grad=update(backbone,params,opt,sched,groups[i],budget)
            log['epoch_nll']+=nll;log['epoch_tokens']+=tokens;log['gradient_sum']+=grad;log['last_grad']=grad
            log['recent_train']=(log['recent_train']+[nll/tokens])[-30:]
            log['updates']+=1;log['next_group']=i+1
            if log['updates']%CHECK_EVERY==0:
                now=perf_counter();log['training_seconds']+=now-stamp;stamp=now
                advance_slot(output,meta,params,opt,sched,log)
            if log['updates']%100==0:
                print(f"T5 LoRA {log['updates']}/{STEPS}, training NLL {np.mean(log['recent_train']):.5f}, recovery step sealed",flush=True)
        dev=evaluate(backbone,data['dev']);now=perf_counter();log['training_seconds']+=now-stamp;stamp=now
        if dev<log['best_loss']:
            log['best_loss']=dev;log['best_epoch']=epoch;log['best_adapter']=adapter_state(params)
        log['history'].append({'epoch':epoch,'training_nll':log['epoch_nll']/log['epoch_tokens'],
                               'development_nll':dev,'mean_gradient_norm':log['gradient_sum']/len(groups),
                               'last_gradient_norm':log['last_grad'],'end_learning_rate':sched.get_last_lr()[0]})
        log.update(epoch=epoch+1,next_group=0,epoch_nll=0.,epoch_tokens=0,gradient_sum=0.)
        advance_slot(output,meta,params,opt,sched,log)
        write_json(output/'training.json',{'initial_development_nll':log['initial_dev'],'best_epoch':log['best_epoch'],'epochs':log['history']})
    if log['updates']!=STEPS or log['best_adapter'] is None:raise ValueError('Incomplete T5-LoRA book fit')
    install(params,log['best_adapter']);out=output/'movie-generations.jsonl'
    generate(backbone,tokenizer,data['test'],out)
    best=output/'selected-adapter.pt';torch.save(log['best_adapter'],best)
    write_json(output/'receipt.json',{'variant':'T5-Large all-attention Q/V LoRA','protocol_sha256':PROTOCOL,
        'parent_protocol_sha256':'7aa3d0c1dc825755862ed91097c13bc7d9d7b069b2ed6a4ab61ffbb3f2a4d51f','input_sha256':INPUT_SHA,
        'pretrained':REPO,**meta,'optimizer_updates':log['updates'],'initial_development_nll':log['initial_dev'],
        'best_epoch':log['best_epoch'],'best_development_nll':log['best_loss'],'external_sentences':len(data['test']),
        'forecast_sha256':digest(out),'best_adapter_sha256':digest(best),'training_seconds':log['training_seconds'],
        'allocation_elapsed_seconds':perf_counter()-start,'restore_replay_loss_abs_error':abs(first[0]-second[0]),
        'restore_replay_parameter_abs_error':drift,'preflight':mem,'geometry':geometry})
    # The final archive only requires the selected adapter. Completed resumption slots remain in a dedicated subdirectory.
    completed=output/'completed-resume';completed.mkdir(exist_ok=True)
    for name in ('resume-0.pt','resume-1.pt','resume-manifest.json'):
        f=output/name
        if f.exists():f.replace(completed/name)


if __name__=='__main__':
    c=ArgumentParser(description=__doc__)
    for k in ('stimuli','output'):c.add_argument('--'+k,type=Path,required=True)
    for k in ('model','resume'):c.add_argument('--'+k,type=Path)
    c.add_argument('--resume-sha256');c.add_argument('--protocol-sha256',required=True)
    run(**vars(c.parse_args()))
