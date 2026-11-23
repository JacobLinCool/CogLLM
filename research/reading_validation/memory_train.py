"""Train an actual query/value attention adapter on human counterfactual premises."""

from dataclasses import replace
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from research.backbone_transfer.extract import digest, forward
from research.token_readout.runner import trace_rows


PAD = 248044
SEED = 280173
RANK = 8


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_json(path, data):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=1)+"\n")
    tmp.replace(path)


def save_state(file,state):
    temporary=file.with_suffix(".tmp")
    torch.save(state,temporary)
    temporary.replace(file)


class LowRank(nn.Module):
    """A trainable additive rank-eight change to a real, frozen Qwen projection."""
    def __init__(self, original):
        super().__init__()
        self.original=original
        self.a=nn.Parameter(torch.empty(RANK, original.in_features,device=original.weight.device))
        self.b=nn.Parameter(torch.zeros(original.out_features,RANK,device=original.weight.device))
        nn.init.kaiming_uniform_(self.a,a=5**.5)

    def forward(self,x):
        return self.original(x)+F.linear(F.linear(x,self.a),self.b)*2.0


def variants(rows, tokenizer):
    for r in rows:
        for key in ("texts","adjacent"):
            seqs=[tokenizer.encode(t,add_special_tokens=False).ids for t in r[key]]
            if len(seqs)!=3 or not all(1<len(s)<2048 for s in seqs):
                raise ValueError("Invalid matched counterfactual triplet")
            r[key+"_ids"]=seqs
    return rows


def batch_ids(sequences, device):
    lengths=[len(s) for s in sequences]
    n=max(lengths)
    ids=torch.tensor([s+[PAD]*(n-len(s)) for s in sequences],device=device)
    positions=torch.arange(n,device=device)
    stop=torch.tensor(lengths,device=device)-1
    return ids,positions[None,:]<=stop[:,None],stop


def last_train(encoder,seqs):
    ids,mask,stop=batch_ids(seqs,next(encoder.parameters()).device)
    y=encoder(input_ids=ids,attention_mask=mask.long(),use_cache=False,output_hidden_states=False,
              output_attentions=False).last_hidden_state
    return F.layer_norm(y[torch.arange(len(seqs),device=y.device),stop],(y.shape[-1],))


def ordered_terminal(runtime,rows,key):
    seqs=[seq for r in rows for seq in r[key+"_ids"]]
    order=sorted(range(len(seqs)),key=lambda i:len(seqs[i]))
    out=torch.empty((len(seqs),runtime.width),device=runtime.device)
    for start in range(0,len(order),18):
        ix=order[start:start+18]
        ss=[seqs[i] for i in ix]
        states=forward(runtime.encoder,ss,PAD,torch)
        stops=[len(s)-1 for s in ss]
        out[ix]=F.layer_norm(states[torch.arange(len(ix),device=runtime.device),stops],(runtime.width,))
    return out.reshape(len(rows),3,runtime.width)


def data_loss(z,y):
    targets=y.to(z)
    positive=z[targets.bool()].reshape(-1,1)
    others=z[~targets.bool()].reshape(-1,2)
    return F.binary_cross_entropy_with_logits(z,targets)+.5*F.softplus(others-positive).mean()


def strict(z,y):
    return float((z.argmax(1)==y.argmax(1)).float().mean())


def fit_frozen(x,y):
    data=x.double()
    y=y.double()
    w=nn.Parameter(torch.zeros(x.shape[-1],device=x.device,dtype=torch.float64))
    b=nn.Parameter(torch.zeros((),device=x.device,dtype=torch.float64))
    opt=torch.optim.LBFGS([w,b],lr=1,max_iter=420,tolerance_grad=2e-7,tolerance_change=1e-12,
                          history_size=40,line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        z=data@w+b
        loss=data_loss(z,y)+.005*w.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    loss=closure()
    maxgrad=max(float(p.grad.abs().max()) for p in (w,b))
    if not np.isfinite(maxgrad) or maxgrad>2e-4:
        raise ArithmeticError(f"Frozen comparison failed convergence, max gradient {maxgrad}")
    return w.detach().float(),b.detach().float(),{"loss":float(loss.detach()),"max_gradient":maxgrad,
                                                   "iterations":int(opt.state[w]["n_iter"])}


def decision_features(runtime,reader,packet,stories,head):
    """Derive both the reading controls and premise score from one article forward."""
    predictions={}
    calls=0
    def count(_module,_args):
        nonlocal calls
        calls+=1
    w,b=head
    with torch.no_grad(),runtime.encoder.register_forward_pre_hook(count):
        for n,r in enumerate(stories,1):
            text="\n\n".join(r["sentences"])
            features=runtime.encode(text)
            scores=reader(features)
            scale=packet["scale"]
            scores=replace(scores,opening=scores.opening*scale["opening_gain"],
                           resolution=scores.resolution*scale["resolution_gain"]+scale["resolution_bias"])
            curve=trace_rows(text,features.offsets,scores,packet["priors"],reader.anchor_tokens)
            logits=(F.layer_norm(features.hidden,(runtime.width,))@w+b).cpu().double().numpy()
            ends=np.cumsum([len(s)+2 for s in r["sentences"]])-2
            token_ends=[]
            for stop in ends:
                prefix=runtime.tokenizer.encode(text[:stop],add_special_tokens=False).ids
                if tuple(prefix)!=features.token_ids[:len(prefix)]:
                    raise ValueError("Sentence marker is not a certified causal token boundary")
                token_ends.append(len(prefix)-1)
            predictions[r["id"]]=dict(logits=logits[token_ends].tolist(),
                signals=[[curve[k][q] for q in ("unresolved","opened","resolved","transition")] for k in token_ends],
                all_token_logits=logits.tolist(),ends=token_ends)
            if calls!=n:
                raise ValueError("Inference used an extra encoder forward")
    return predictions,calls


def models_state(modules,w,b):
    return {**{name+"."+k:p.detach().clone().cpu() for name,layer in modules.items()
               for k,p in (("a",layer.a),("b",layer.b))},"readout.weight":w.detach().clone().cpu(),
            "readout.bias":b.detach().clone().cpu()}


def restore(modules,w,b,params):
    with torch.no_grad():
        for name,layer in modules.items():
            layer.a.copy_(params[name+".a"])
            layer.b.copy_(params[name+".b"])
        w.copy_(params["readout.weight"])
        b.copy_(params["readout.bias"])


def run(runtime,reader,packet,source,output):
    """One fitted LoRA; training enables gradients only for the last Q/V updates and readout."""
    output=Path(output)
    output.mkdir(exist_ok=True)
    root=Path(source).parent
    train_file=Path(source)
    test_file=root/"cad-test.jsonl"
    story_file=root/"stories.jsonl"
    torch.manual_seed(SEED)
    items=variants(read_rows(train_file),runtime.tokenizer)
    train=[r for r in items if r["split"]=="train"]
    dev=[r for r in items if r["split"]=="dev"]
    test=variants(read_rows(test_file),runtime.tokenizer)
    if (any("y" in r for r in test) or len(train)+len(dev)!=len(items)
            or set(r["id"] for r in train)&{r["id"] for r in dev+test}):
        raise ValueError("Training and blinded test schema differ from protocol")
    studies=read_rows(story_file)
    y=torch.tensor([r["y"] for r in train],device=runtime.device,dtype=torch.float32)
    yd=torch.tensor([r["y"] for r in dev],device=runtime.device,dtype=torch.float32)
    t0=perf_counter()
    print(f"Fitting frozen and adapted attention on {len(train)} matched families.",flush=True)
    x=ordered_terminal(runtime,train,"texts")
    dv=ordered_terminal(runtime,dev,"texts")
    with torch.enable_grad():
        fw,fb,conv=fit_frozen(x,y)
    del x
    zbase=dv@fw+fb
    dev_loss=float(data_loss(zbase,yd))
    conv["dev_loss"]=dev_loss
    conv["dev_strict"]=strict(zbase,yd)
    frozen_story,base_calls=decision_features(runtime,reader,packet,studies,(fw,fb))
    write_json(output/"stories-frozen.json",frozen_story)
    write_json(output/"frozen-solver.json",conv)
    test_logits={"id":[r["id"] for r in test]}
    for key in ("texts","adjacent"):
        test_logits["frozen."+key]=(ordered_terminal(runtime,test,key)@fw+fb).cpu().numpy()
    attn=runtime.encoder.layers[-1].self_attn
    if runtime.encoder.config.layer_types[-1]!="full_attention":
        raise ValueError("Expected a final native global-attention layer")
    replaced={name:getattr(attn,name) for name in ("q_proj","v_proj")}
    modules={name:LowRank(layer) for name,layer in replaced.items()}
    w=nn.Parameter(fw.clone())
    b=nn.Parameter(fb.clone())
    for name,layer in modules.items():
        setattr(attn,name,layer)
    try:
        with torch.no_grad():
            recheck=ordered_terminal(runtime,dev[:8],"texts")@w+b
        zero_diff=float((recheck-zbase[:8]).abs().max())
        if zero_diff>.002:
            raise ArithmeticError(f"Zero adapter changed pretrained output: {zero_diff}")
        best=dev_loss
        best_epoch=0
        best_state=models_state(modules,w,b)
        adapter_params=[p for layer in modules.values() for p in (layer.a,layer.b)]
        parameters=[*adapter_params,w,b]
        optimizer=torch.optim.AdamW([{"params":adapter_params,"lr":2e-4,"weight_decay":0.01},
                                     {"params":[w,b],"lr":2e-5,"weight_decay":0}],betas=(.9,.98))
        if set(id(p) for p in runtime.encoder.parameters() if p.requires_grad)!=set(map(id,adapter_params)):
            raise ValueError("Unexpected trainable backbone parameters")
        history=[]
        since=0
        start_epoch=1
        checkpoint=output/"training-state.pt"
        bindings={k.name:digest(k) for k in (train_file,test_file,story_file)}
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
            if saved["inputs"]!=bindings or saved["revision"]!=runtime.revision:
                raise ValueError("Refusing to resume a different experiment")
            restore(modules,w,b,saved["weights"])
            optimizer.load_state_dict(saved["optimizer"])
            best,best_epoch,best_state,since,history=(saved[k] for k in ("best","best_epoch","best_state","since","history"))
            start_epoch=saved["epoch"]+1
        stop_reason="maximum_epochs"
        for epoch in range(start_epoch,13):
            if since>=3:
                stop_reason="development_patience"
                break
            # Batch whole matched families; shuffle length buckets for speed without outcome sorting.
            order=sorted(range(len(train)),key=lambda i:max(map(len,train[i]["texts_ids"])))
            rng=random.Random(SEED+epoch)
            chunks=[order[i:i+64] for i in range(0,len(order),64)]
            for chunk in chunks: rng.shuffle(chunk)
            rng.shuffle(chunks)
            batches=[c[i:i+5] for c in chunks for i in range(0,len(c),5)]
            total=0.
            max_norm=0.
            before=perf_counter()
            with torch.enable_grad():
                for ix in batches:
                    ss=[s for i in ix for s in train[i]["texts_ids"]]
                    x=last_train(runtime.encoder,ss).reshape(len(ix),3,runtime.width)
                    z=x@w+b
                    loss=data_loss(z,y[ix])+.005*w.square().sum()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad=nn.utils.clip_grad_norm_(parameters,1.0,error_if_nonfinite=True)
                    optimizer.step()
                    total+=float(loss.detach())*len(ix)
                    max_norm=max(max_norm,float(grad))
            with torch.no_grad():
                zdev=ordered_terminal(runtime,dev,"texts")@w+b
                score=float(data_loss(zdev,yd))
            history.append(dict(epoch=epoch,train_loss=total/len(train),dev_data_loss=score,
                                dev_strict=strict(zdev,yd),max_unclipped_grad=max_norm,seconds=perf_counter()-before))
            if score<best-.002:
                best,best_epoch,best_state,since=score,epoch,models_state(modules,w,b),0
            else:
                since+=1
            to_save=dict(inputs=bindings,revision=runtime.revision,epoch=epoch,weights=models_state(modules,w,b),
                         optimizer=optimizer.state_dict(),best=best,best_epoch=best_epoch,best_state=best_state,
                         since=since,history=history)
            save_state(checkpoint,to_save)
            write_json(output/"history.json",history)
            print(json.dumps(history[-1]),flush=True)
        if since>=3:
            stop_reason="development_patience"
        restore(modules,w,b,best_state)
        with torch.no_grad():
            for key in ("texts","adjacent"):
                test_logits["adapted."+key]=(ordered_terminal(runtime,test,key)@w+b).cpu().numpy()
            adapted_story,adapt_calls=decision_features(runtime,reader,packet,studies,(w,b))
            write_json(output/"stories-adapted.json",adapted_story)
            # Directly certify a shared arbitrary prefix after attaching the learned attention update.
            all_seq=test[0]["texts_ids"][0]; half=len(all_seq)//2
            a=forward(runtime.encoder,[all_seq],PAD,torch)[0,half-1]
            c=forward(runtime.encoder,[all_seq[:half]],PAD,torch)[0,-1]
            prefix_error=float((a-c).norm()/c.norm())
            if prefix_error>1e-3:
                raise ValueError("Adapted hidden state depends on a future suffix")
        np.savez_compressed(output/"nli-test.npz",**test_logits)
        save_state(output/"adapter.pt",best_state)
        save_state(output/"frozen.pt",{"weight":fw.cpu(),"bias":fb.cpu()})
        result=dict(model=runtime.revision,source_sha256=bindings,train=len(train),dev=len(dev),test=len(test),
                    frozen_solver=conv,zero_adapter_max_abs_error=zero_diff,prefix_relative_diff=prefix_error,
                    frozen_article_forwards=base_calls,adapted_article_forwards=adapt_calls,
                    attention_parameters=sum(p.numel() for p in adapter_params),readout_parameters=w.numel()+1,
                    best_epoch=best_epoch,best_development_loss=best,stop=stop_reason,
                    epochs_completed=len(history),wall_seconds=perf_counter()-t0,peak_device_bytes=torch.cuda.max_memory_allocated())
        result["files"]={k:digest(output/k) for k in ("stories-frozen.json","stories-adapted.json","nli-test.npz",
                                                      "adapter.pt","frozen.pt","history.json")}
        write_json(output/"receipt.json",result)
    finally:
        for name,layer in replaced.items():
            setattr(attn,name,layer)
    return result
