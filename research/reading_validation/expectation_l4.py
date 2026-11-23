"""Capture single-pass lexical expectations on the sealed original passages."""

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from research.backbone_transfer.extract import digest, stream_rows, write_json
from research.reading_validation.expectation_native import COLUMNS,native
from research.reading_validation.identity_run_l4 import run_phases


def run(runtime,reader,checkpoint,source,output):
    rows=list(stream_rows(source))
    if any(set(r)!={"id","text"} or hashlib.sha256(r["text"].encode()).hexdigest()!=r["id"] for r in rows):
        raise ValueError("Expected only the original text and its digest")
    order=sorted(range(len(rows)),key=lambda i:len(rows[i]["text"]))
    output.mkdir()
    arrays={};info={};calls=0;start=perf_counter()
    def count(module, args):
        nonlocal calls
        calls+=1
    with runtime.encoder.register_forward_pre_hook(count):
        for lo in range(0,len(order),16):
            batch=[rows[j] for j in order[lo:lo+16]]
            for row,(coords,z,n) in zip(batch,native(runtime,[r["text"] for r in batch]),strict=True):
                key=row["id"]
                arrays[key+".word"]=coords
                arrays[key+".features"]=z
                info[key]=dict(words=len(z),tokens=n)
    if calls!=(len(rows)+15)//16:
        raise ArithmeticError("Reading alternatives required extra encoder calls")
    checks=[]
    # A genuine full-prefix truncation must have identical source tokens, positions and alternatives.
    for original in [rows[order[len(order)//2]],rows[order[-1]]]:
        key=original["id"];r=arrays[key+".word"]
        ix=len(r)//2
        text=original["text"][:r[ix,1]]
        coords,z,_=native(runtime,[text])[0]
        eligible=(coords[:,4]!=0)&(r[:len(z),4]!=0)
        if not np.array_equal(r[:len(z),:2],coords[:,:2]) or not eligible.any():
            raise ArithmeticError("Prefix lexical positions have changed")
        error=float(np.abs(z[eligible]-arrays[key+".features"][:len(z)][eligible]).max())
        if error>.008:
            raise ArithmeticError(f"Suffix changed semantic expectation on an unchanged prefix by {error}")
        checks.append(dict(words=int(eligible.sum()),max_abs=error))
    np.savez_compressed(output/"features.npz",**arrays)
    write_json(output/"receipt.json",dict(articles=len(info),words=sum(v["words"] for v in info.values()),
         tokens=sum(v["tokens"] for v in info.values()),revision=runtime.revision,device=torch.cuda.get_device_name(),
         encoder_batches=calls,technical_checks=checks,columns=COLUMNS,source=digest(source),seconds=perf_counter()-start,
         file=digest(output/"features.npz")))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("model","stimuli","output"):
        p.add_argument("--"+key,type=Path,required=True)
    for key in ("candidate","protocol-sha256"):
        p.add_argument("--"+key,required=True)
    args=p.parse_args()
    run_phases(args.model,args.candidate,args.output,args.protocol_sha256,[("expectation",run,args.stimuli)])
