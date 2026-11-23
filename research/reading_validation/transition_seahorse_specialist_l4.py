"""Run the released SEAHORSE Q1 metric on the original article-summary strings in one fixed classifier step."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL="google/seahorse-large-q1"
REV="a39a2c372923817c2a857b7953cb10827ec0e810"
PIN="c3cc21518164b750a34f6d0ab05291b1cf60322fe275dd57a7f5104e9a45acc0"
TOKENIZER="99cc999819aaabf74898a252863b10d86fbcd86e8b3f65c118ff334ff85c5ea5"


def digest(p):
    h=sha256()
    with open(p,"rb") as fin:
        while chunk:=fin.read(4<<20):h.update(chunk)
    return h.hexdigest()


@torch.inference_mode()
def run(stimuli,output,input_sha256,protocol_sha256,model=None):
    if digest(stimuli)!=input_sha256:raise ValueError("Unregistered source/summary inputs")
    snap=Path(model) if model else Path(snapshot_download(MODEL,revision=REV,allow_patterns=["config.json","pytorch_model.bin","generation_config.json","spiece.model","special_tokens_map.json","tokenizer.json","tokenizer_config.json"]))
    if digest(snap/"pytorch_model.bin")!=PIN or digest(snap/"tokenizer.json")!=TOKENIZER:
        raise ValueError("Public native SEAHORSE Q1 checkpoint or tokenizer mismatch")
    tok=AutoTokenizer.from_pretrained(snap,local_files_only=True,use_fast=True)
    net=AutoModelForSeq2SeqLM.from_pretrained(snap,local_files_only=True,use_safetensors=False,weights_only=True,
              dtype=torch.bfloat16).to("cuda").eval().requires_grad_(False)
    labels=[tok(s,add_special_tokens=False)["input_ids"] for s in ("0","1")]
    if any(len(v)!=1 for v in labels) or labels[0]==labels[1]:raise ValueError("Expected distinct single native SEAHORSE class tokens")
    started=perf_counter();rows=[];seen=set();eos=tok.eos_token_id
    with stimuli.open(encoding="utf-8", newline="\n") as original:
        presentations = [json.loads(line) for line in original]
    for row in presentations:
        if set(row)!={"id","article","summary"}:raise ValueError("Specialist accepts only original grade-free text")
        a,s=row["article"],row["summary"]; uid=sha256(json.dumps([a,s],ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
        if uid in seen or uid!=row["id"]:raise ValueError("Broken public article/summary identity")
        seen.add(uid);prefix="premise: "+a+" hypothesis: "; encoding=tok(prefix+s,return_offsets_mapping=True,
                           return_attention_mask=False,add_special_tokens=True)
        source=encoding["input_ids"];off=encoding["offset_mapping"]
        if source[-1]!=eos:raise ValueError("Unknown original tokenizer EOS")
        seq=source if len(source)<=2048 else source[:2047]+[eos]
        visible=sum(hi>len(prefix) and hi>lo for lo,hi in off[:min(len(source)-1,2047)])
        expected=sum(hi>len(prefix) and hi>lo for lo,hi in off)
        rows.append(dict(id=uid,input=seq,total=len(source),visible=visible,expected=expected))
    if len(rows)!=4355:raise ValueError("Full English test required")
    output.mkdir(parents=True,exist_ok=True);out=output/"specialist-forecasts.jsonl"
    ordered=sorted(rows,key=lambda r:(len(r["input"]),r["id"]));n=0;calls=0
    with out.open("w") as f:
        i=0
        while i<len(ordered):
            batch=ordered[i:i+8]
            pad=tok.pad({"input_ids":[r["input"] for r in batch]},padding=True,return_tensors="pt")
            inp=pad["input_ids"].to("cuda"); mask=pad["attention_mask"].to("cuda")
            decoder_start=torch.full((len(batch),1),net.config.decoder_start_token_id,dtype=torch.long,device="cuda")
            logits=net(input_ids=inp,attention_mask=mask,decoder_input_ids=decoder_start,
                       use_cache=False,return_dict=True).logits[:,0,:].float()
            logits=logits[:,[labels[0][0],labels[1][0]]].cpu().numpy()
            for r, v in zip(batch,logits,strict=True):
                f.write(json.dumps(dict(id=r["id"],log_odds_no=float(v[0]-v[1]),
                              input_tokens=r["total"],visible_summary_pieces=r["visible"],all_summary_pieces=r["expected"]),
                              allow_nan=False,separators=(",",":"))+"\n")
            n+=len(batch);calls+=1;i+=len(batch)
            if calls%50==0:
                f.flush();print("SEAHORSE original Q1",n,len(ordered),round(perf_counter()-started,1),flush=True)
    if n!=4355:raise ValueError("Incomplete Q1 specialist")
    info=dict(model=MODEL,revision=REV,weights_sha256=PIN,tokenizer_sha256=TOKENIZER,
              source_sha256=input_sha256,protocol_sha256=protocol_sha256,forecasts_sha256=digest(out),label_tokens=labels,
              articles_and_summaries=n,model_calls=calls,seen_grades=0,input_length=2048,seconds=perf_counter()-started,
              hardware=torch.cuda.get_device_name(),dtype="bfloat16")
    (output/"receipt.json").write_text(json.dumps(info,indent=2)+"\n")


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for flag in ("stimuli","output","model"):p.add_argument("--"+flag,type=Path,required=flag!="model")
    for flag in ("input-sha256","protocol-sha256"):p.add_argument("--"+flag,required=True)
    run(**vars(p.parse_args()))
