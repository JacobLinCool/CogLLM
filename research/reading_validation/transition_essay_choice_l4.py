"""Extract source-identical, grade-free probabilities' native features at each expert alternative."""
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch
from research.reading_validation.transition_external_forecast import project
from research.reading_validation.transition_inference import REVISION


@torch.inference_mode()
def run(runtime,reader,checkpoint,source,out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if runtime.candidate!="qwen" or runtime.revision!=REVISION:
        raise ValueError("Expert choice requires the original fixed Qwen features")
    started=perf_counter();n=[0];seen=set()
    def observe(*_):n[0]+=1
    f=out/"sentence-features.jsonl"
    with runtime.encoder.register_forward_pre_hook(observe),f.open("w") as fp:
        for r in map(json.loads,source.read_text().splitlines()):
            if set(r)!={"id","text","start","end","index"} or r["end"]!=len(r["text"]) or len(r["text"][:r["start"]].split())<2:
                raise ValueError("Expected one complete, unmarked passage with a final candidate")
            uid=r["id"]
            if sha256(r["text"].encode()).hexdigest()!=uid or uid in seen:raise ValueError("Nonunique expert alternative")
            seen.add(uid);before=n[0]
            s={k:r[k] for k in ("id","start","end","index")};s["archive_final"]=False
            rows,unused=project(runtime,reader,checkpoint,{"text":r["text"],"stops":[s]})
            if n[0]!=before+1 or len(rows)!=1 or rows[0]["id"]!=uid or unused:
                raise ValueError("Additional forward or missing candidate")
            fp.write(json.dumps(rows[0],allow_nan=False)+"\n")
            if len(seen)%100==0:print("Essay alternatives",len(seen),flush=True)
    receipt={"input_sha256":sha256(source.read_bytes()).hexdigest(),"file_sha256":sha256(f.read_bytes()).hexdigest(),
             "documents":len(seen),"forwards":n[0],"external_outcomes":0,"revision":runtime.revision,"seconds":perf_counter()-started}
    (out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
