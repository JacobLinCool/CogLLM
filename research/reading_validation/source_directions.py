"""Fit a bounded common between-study direction to each authorized reading state."""
from collections import Counter,defaultdict
from copy import deepcopy
from hashlib import sha256
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from research.backbone_transfer.extract import digest,stream_rows
from research.reading_validation.borrowing_data import PACKET,OUTPUT,FIT,TUNE,CALIBRATE,EVALUATE,data
from research.reading_validation.borrowing_fit import Private,train
from research.reading_validation.borrowing_score import GeometricPool,material_weights
from research.union_training.fit_observations import Readouts,weights,COUNT_MODELS,observation_scales,task_responses

OUT=Path("research/results/reading-validation-source-directions")
PLAN=Path("research/reading_validation/source-directions-protocol.md")

class Directions(nn.Module):
    def __init__(self,base,tasks,split):
        super().__init__()
        self.measurements=deepcopy(base.measurements);self.shared=deepcopy(base.shared)
        self.direction=nn.ParameterDict({k:nn.Parameter(torch.randn(v.out_features,v.in_features+1)) for k,v in base.shared.items()})
        self.coef=nn.ParameterDict();self.source_keys={};self.family_counts={};self.sizes={}
        for source in sorted({t.source for t in tasks}):
            subtasks=[t for t in tasks if t.source==source];states=sorted({s for t in subtasks for s in t.states})
            self.sizes[source]=len(states);self.source_keys[source]={}
            for state in states:
                key=source+"__"+state;self.source_keys[source][state]=key;self.coef[key]=nn.Parameter(torch.zeros(()))
                self.family_counts[key]=len({g for t in subtasks if state in t.states for g in t.family[split[t.name]==FIT]})
    def forward(self,t,cache,rows):
        locations=t.coordinates[rows];h=cache[locations.clamp_min(0)];out=[]
        for state in t.states:
            raw=self.direction[state];d=raw*(raw.shape[0]**.5)/raw.norm().clamp_min(1e-6)
            b=self.shared[state];amount=self.coef[self.source_keys[t.source][state]]
            out.append(F.linear(h,b.weight+amount*d[:,:-1],b.bias+amount*d[:,-1]).tanh())
        return self.measurements[t.name](torch.cat(out,dim=-1),locations>=0)
    def penalty(self):
        # Source likelihoods average over studies. Put the corresponding prior in identical units.
        by=[]
        for s,keys in self.source_keys.items():
            by.append(torch.stack([self.coef[k].square()/(2*self.family_counts[k]) for k in keys.values()]).sum())
        return torch.stack(by).mean()

def fresh(tasks):
    families=defaultdict(set)
    for t in tasks:families[t.source].update(t.family[t.role==0])
    def order(s,g,salt):
        return sha256(("source-direction-replication-v1/"+salt+"/"+s+"/"+g).encode()).digest()
    sets={}
    for source,original in families.items():
        ranked=sorted(original,key=lambda g:order(source,g,"held"));count=min(max(1,round(len(ranked)*.2)),len(ranked)-1)
        held=set(ranked[:count])
        # New train-only inner partitions may use original development material.
        complete={g for t in tasks if t.source==source for g in t.family}
        rest=sorted(complete-held,key=lambda g:order(source,g,"inner"));n=min(max(1,round(len(rest)*.15)),(len(rest)-1)//2)
        for j,g in enumerate(rest):sets[source,g]=TUNE if j<n else CALIBRATE if j<2*n else FIT
        for g in held:sets[source,g]=EVALUATE
    splits={};exposure={}
    for t in tasks:
        r=np.array([sets[t.source,g] for g in t.family]);splits[t.name]=r
        rows=list(stream_rows(PACKET/(t.name+".jsonl")))
        for entry,part in zip(rows,r,strict=True):
            for doc,_ in entry["coordinates"]:
                if exposure.setdefault(doc,(t.source,entry["family"],int(part)))!=(t.source,entry["family"],int(part)):
                    raise ValueError("An article appeared in two partitions")
        if t.likelihood not in COUNT_MODELS:
            values=task_responses(t.likelihood,rows,t.coordinates.shape[1])
            t.trials,t.center,t.spread=observation_scales(t.likelihood,values,r)
            t.response=torch.as_tensor(values,dtype=torch.float32)
    return splits

def run(mode,seed):
    torch.set_num_threads(1);torch.manual_seed(seed)
    out=OUT/mode/f"seed-{seed}";out.mkdir(parents=True,exist_ok=True)
    cache,tasks,oldsplit=data();split=oldsplit if mode=="historical" else fresh(tasks)
    baseline=Readouts(cache.shape[1],tasks);private=Private(baseline,tasks);adaptive=Directions(baseline,tasks,split)
    for t in tasks:
        ix=torch.as_tensor(np.flatnonzero(split[t.name]==FIT)[:3])
        if not torch.equal(adaptive(t,cache,ix),baseline(t,cache,ix)):raise ValueError("Initial source deviation is not zero")
    fits={}
    if mode=="fresh":
        for name,expert in (("shared",baseline),("private",private)):fits[name]=train(expert,name,tasks,cache,split,seed,out)
    fits["directions"]=train(adaptive,"directions",tasks,cache,split,seed,out,penalty=adaptive.penalty)
    if any(s["stop"]!="inner_likelihood_plateau" for s in fits.values()):raise ArithmeticError("Unresolved convergence")
    receipt={"seed":seed,"mode":mode,"plan":digest(PLAN),"packet":digest(PACKET/"receipt.json"),"cache_shape":list(cache.shape),
             "fits":fits,"roles":{t.name:np.bincount(split[t.name],minlength=4).tolist() for t in tasks},
             "coefficients":{s:{st:float(adaptive.coef[k].detach()) for st,k in v.items()} for s,v in adaptive.source_keys.items()},
             "files":{v.name:digest(v) for v in out.glob("*.npz")}}
    (out/"fit.json").write_text(json.dumps(receipt,indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=("all","historical","fresh"),default="all")
    p.add_argument("--seed",type=int,nargs="+",default=[1729,2718]);a=p.parse_args()
    for mode in (("historical","fresh") if a.mode=="all" else (a.mode,)):
        for seed in a.seed:run(mode,seed)
