"""Extract causal projected states and exact native attention to visible notation, one pass per text."""
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from research.backbone_transfer.extract import digest, stream_rows, write_json, read_json
from research.reading_validation.symbol_recovery import publish
from research.reading_validation.definition_forecast import projection
from research.reading_validation.identity_causal_features import encode_batch


def native(mean,peak,keys,n):
    keys=torch.as_tensor(keys,dtype=torch.long,device=mean.device)
    scale=1+torch.arange(n,device=mean.device)[:,None,None]
    r=torch.stack((mean[:n,keys],peak[:n,keys]),-1)
    if torch.any(r[torch.arange(n,device=mean.device)[:,None]<keys[None,:]]):
        raise ArithmeticError('An unseen mathematical expression received nonzero attention')
    return torch.log1p(r*scale)


@torch.no_grad()
def run(runtime,reader,checkpoint,input_path,output):
    source=Path(input_path);out=Path(output);out.mkdir(parents=True);start=perf_counter();basis=projection(runtime)
    rows=list(stream_rows(source));texts=[runtime.tokenizer.encode(r['text'],add_special_tokens=False) for r in rows]
    for r,t in zip(rows,texts,strict=True):
        k=r['keys'];n=len(t.ids)
        if not k or not 1<=n<=1024 or n!=r['tokens'] or sorted(set(k))!=k or not 0<=k[0]<=k[-1]<n or sha256(r['text'].encode()).hexdigest()!=r['id']:
            raise ValueError('Invalid symbol or paragraph coordinates')
    order=sorted(range(len(rows)),key=lambda i:(rows[i]['tokens'],rows[i]['id']));count=0;expected=0;checks=[];files={};ids={}
    resume=read_json(Path('symbol-inherited.json')) if Path('symbol-inherited.json').exists() else {};inherited_calls=0
    def counted(_m,_a):
        nonlocal count
        count+=1
    with runtime.encoder.register_forward_pre_hook(counted):
        for shard,lo in enumerate(range(0,len(order),100)):
            batch={};indices=order[lo:lo+100];peaklen=rows[indices[-1]]['tokens'];size=16 if peaklen<=200 else 8 if peaklen<=380 else 4 if peaklen<=600 else 2
            file=f'shard-{shard:03}.npz';members=[rows[i]['id'] for i in indices];prior=resume.get(file)
            if prior:
                if prior['ids']!=members or prior['stimuli']!=digest(source) or prior['revision']!=runtime.revision:raise ValueError('Prior scientific feature shard does not match this complete cohort')
                ids.update({key:shard for key in members});files[file]=prior['sha256'];checks.extend(prior['prefix_checks']);inherited_calls+=prior['production_calls'];continue
            prior_checks=len(checks);prior_expected=expected
            for begin in range(0,len(indices),size):
                ii=indices[begin:begin+size];z,_,_,m,p=encode_batch(runtime,[texts[i].ids for i in ii],basis);expected+=1
                for j,i in enumerate(ii):
                    r=rows[i];key=r['id'];n=r['tokens'];v=native(m[j],p[j],r['keys'],n);quant=(32*v).round()
                    if not torch.isfinite(quant).all() or quant.min()<0 or quant.max()>255:raise ArithmeticError('Unrepresentable native relative weights')
                    batch[key+'_s']=z[j,:n].half().cpu().numpy();batch[key+'_a']=quant.to(torch.uint8).cpu().numpy();ids[key]=shard
                if shard in (len(order)//100//3,2*(len(order)//100)//3) and begin==0:
                    i=ii[0];r=rows[i];cut=r['tokens']*2//3;keys=[k for k in r['keys'] if k<cut]
                    if not keys:continue
                    v,_,_,m0,p0=encode_batch(runtime,[texts[i].ids[:cut]],basis);actual=native(m0[0],p0[0],keys,cut).cpu().numpy();ref=batch[r['id']+'_a'][:cut,:len(keys)].astype('float32')/32
                    a=v[0].cpu().numpy();d=batch[r['id']+'_s'][:cut].astype('float32');error=float(np.linalg.norm(a-d)/np.linalg.norm(a));att=float(np.abs(actual-ref).max())
                    if error>.002 or att>.017:raise ArithmeticError(f'Suffix invariance failed {error}, {att}')
                    checks.append({'id':r['id'],'cut':cut,'state_relative':error,'attention_max':att})
            file=f'shard-{shard:03}.npz';np.savez_compressed(out/file,**batch);files[file]=digest(out/file)
            meta={'ids':members,'stimuli':digest(source),'revision':runtime.revision,'production_calls':expected-prior_expected,'prefix_checks':checks[prior_checks:]};publish(out/file,meta)
            write_json(out/f'shard-{shard:03}.json',{**meta,'sha256':files[file]})
            print(f'Archived symbol paragraphs {min(lo+100,len(rows))}/{len(rows)}',flush=True)
    if count!=expected+len(checks)-sum(len(v['prefix_checks']) for v in resume.values()) or not checks or len(ids)!=len(rows):raise ValueError('Incomplete feature extraction')
    write_json(out/'receipt.json',{'stimuli':digest(source),'files':files,'shards':ids,'revision':runtime.revision,'paragraphs':len(rows),'tokens':sum(r['tokens'] for r in rows),'production_calls':expected+inherited_calls,'new_production_calls':expected,'inherited_shards':sorted(resume),'prefix_checks':checks,'device':torch.cuda.get_device_name(),'seconds':perf_counter()-start})
