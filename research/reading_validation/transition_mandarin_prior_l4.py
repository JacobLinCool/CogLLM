"""Read frozen word inputs and isolated-predicate states for an offline Mandarin dictionary."""
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import torch

PACKET="ec90fab2e5aa2a2aea2a4485891309d911edc61cde31a81a02b949d0b091cfe2"


def run(runtime,reader,checkpoint,input_path,output_dir):
    raw=Path(input_path).read_bytes()
    if sha256(raw).hexdigest()!=PACKET or runtime.candidate!="qwen":
        raise ValueError("Expected the fixed Mandarin vocabulary and selected Qwen backbone")
    verbs=[json.loads(s)["verb"] for s in raw.splitlines()]
    if len(verbs)!=188 or len(set(verbs))!=188:raise ValueError("Unexpected Mandarin vocabulary")
    output=Path(output_dir);output.mkdir(exist_ok=True,parents=True);clock=time.monotonic();calls=[0];states=[]
    def observe(_m,_x):calls[0]+=1
    with torch.inference_mode(),runtime.encoder.register_forward_pre_hook(observe):
        for i,v in enumerate(verbs):
            f=runtime.encode(v);token_ids=torch.tensor(f.token_ids,device=f.hidden.device)
            embed=runtime.encoder.get_input_embeddings()(token_ids).float().mean(0)
            states.append(torch.stack((embed,f.mid[-1].float(),f.hidden[-1].float())).cpu().numpy())
            if calls[0]!=i+1:raise ValueError("Each dictionary entry requires one isolated text pass")
    weights=output/"word-representations.npz";a=np.stack(states)
    if a.shape!=(188,3,2048) or not np.isfinite(a).all():raise ValueError("Native lexical vector dimensions changed")
    np.savez_compressed(weights,features=a)
    (output/"receipt.json").write_text(json.dumps({"input_sha256":PACKET,"weights_sha256":sha256(weights.read_bytes()).hexdigest(),
       "revision":runtime.revision,"predicates":len(verbs),"dictionary_forwards":calls[0],"external_outcome_fields":0,
       "extra_article_forwards_for_dictionary_lookup":0,"seconds":time.monotonic()-clock},indent=2)+"\n")
