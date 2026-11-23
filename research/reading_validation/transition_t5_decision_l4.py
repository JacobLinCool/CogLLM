"""Recover type ranking and conditional span generation from the already fitted T5 specialist."""
from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

from research.reading_validation.transition_t5_adapter_l4 import attach,install
from research.reading_validation.transition_t5_l4 import REVISION,WEIGHTS,batch,batches,digest,model_path,write_json


@torch.inference_mode()
def run(stimuli,adapter,output,input_sha256,protocol_sha256,model=None):
    if digest(stimuli)!=input_sha256 or digest(adapter)!="a82d523cd771dca02cbaf55cc67d75f6da454be05276da0d2d670d6e70113c7c":
        raise ValueError("Data or selected T5 adapter changed")
    snapshot=model_path(model);tok=T5Tokenizer.from_pretrained(snapshot,local_files_only=True)
    backbone=T5ForConditionalGeneration.from_pretrained(snapshot,local_files_only=True,use_safetensors=True,dtype=torch.bfloat16).to("cuda").eval().requires_grad_(False)
    params=attach(backbone);install(params,torch.load(adapter,map_location="cpu",weights_only=True));backbone.eval()
    classes=[tok(s,add_special_tokens=False)["input_ids"] for s in ("true","false")]
    prefix=tok("false character",add_special_tokens=False)["input_ids"]
    if not all(classes) or classes[0][0]==classes[1][0] or tok("false character Mary",add_special_tokens=False)["input_ids"][:len(prefix)]!=prefix:
        raise ValueError("The fixed tokenizer has unexpected class or span-prefix geometry")
    pad=backbone.config.decoder_start_token_id;eos=tok.eos_token_id;sentinel=tok.convert_tokens_to_ids("<extra_id_0>")
    rows=[]
    for r in map(json.loads,stimuli.read_text().splitlines()):
        if set(r)!={"id","sentence","context"}:raise ValueError("The T5 diagnostic accepts grade-free words only")
        seq=tok(" ".join((r["context"],"<extra_id_0>",r["sentence"])),return_attention_mask=False)["input_ids"]
        if seq.count(sentinel)!=1:raise ValueError("The original target delimiter is ambiguous")
        ix=seq.index(sentinel);visible=max(0,min(len(seq)-1,1023)-ix-1);required=len(seq)-2-ix
        geometry="full" if visible==required else "partial" if visible>0 else "hidden"
        encoded=seq if len(seq)<=1024 else seq[:1023]+[eos]
        rows.append({"id":r["id"],"input":encoded,"visibility":geometry})
    output.mkdir(parents=True,exist_ok=True);forecasts=[];enc_rows=0;enc_calls=0;started=perf_counter()
    def count(_mod,args,kwargs):
        nonlocal enc_rows,enc_calls
        arr=kwargs.get("input_ids",args[0] if args else None)
        enc_rows+=int(arr.shape[0]);enc_calls+=1
    with backbone.encoder.register_forward_pre_hook(count,with_kwargs=True),torch.autocast("cuda",dtype=torch.bfloat16):
        for group in batches(sorted(rows,key=lambda r:len(r["input"])),8):
            data=batch(group,False);mask=data["attention_mask"];B=len(group)
            enc=backbone.encoder(**data,return_dict=True)
            probs=[]
            for ids in classes:
                frames=torch.tensor([pad,*ids[:-1]],device="cuda")[None,:].expand(B,-1)
                logits=backbone(encoder_outputs=enc,attention_mask=mask,decoder_input_ids=frames,use_cache=False).logits.float().log_softmax(-1)
                selected=logits[:,torch.arange(len(ids),device="cuda"),torch.tensor(ids,device="cuda")].sum(1)
                probs.append(selected.cpu().numpy())
            log_odds=probs[1].astype(float)-probs[0].astype(float)
            seed=torch.tensor([pad,*prefix],device="cuda")[None,:].expand(B,-1).contiguous()
            decoded=backbone.generate(encoder_outputs=enc,attention_mask=mask,decoder_input_ids=seed,max_length=80,
                                      do_sample=False,num_beams=1,use_cache=True)
            for r,odds,ids in zip(group,log_odds,decoded.tolist(),strict=True):
                if ids[:len(prefix)+1]!=[pad,*prefix]:raise ValueError("Span generation lost its declared category")
                keep=[t for t in ids if t not in (pad,eos)]
                forecasts.append({"id":r["id"],"class_log_odds":float(odds),"visibility":r["visibility"],
                                  "conditional_generation":tok.decode(keep,skip_special_tokens=False)})
    if len(rows)!=2377 or enc_rows!=len(rows) or len({r["id"] for r in forecasts})!=len(rows):raise ValueError("More than one encoder invocation per original sentence")
    out=output/"source-and-movie-forecasts.jsonl"
    out.write_text("".join(json.dumps(row,separators=(",",":"))+"\n" for row in forecasts))
    write_json(output/"receipt.json",{"input_sha256":input_sha256,"protocol_sha256":protocol_sha256,"source_adapter_sha256":digest(adapter),
               "base_revision":REVISION,"base_weights":WEIGHTS,"external_labels":0,"decoded_original_sentences":len(rows),"encoder_rows":enc_rows,
               "encoder_batches":enc_calls,"type_label_ids":classes,"span_prefix_ids":prefix,"output_sha256":digest(out),"seconds":perf_counter()-started})


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for k in ("stimuli","adapter","output","model"):p.add_argument("--"+k,type=Path,required=k!="model")
    for k in ("input-sha256","protocol-sha256"):p.add_argument("--"+k,required=True)
    run(**vars(p.parse_args()))
