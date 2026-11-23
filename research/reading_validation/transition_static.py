"""A compact text-only semantic control on original article and sentence coordinates."""
from argparse import ArgumentParser
import json
from pathlib import Path
import re
import struct

import numpy as np
from tokenizers import Tokenizer

from research.reading_validation.transition_external_fit import binomial, predict
from research.reading_validation.transition_external_outcomes import votes
from research.reading_validation.transition_external_prepare import OUT, ROOT
from research.reading_validation.transition_score import dump, sha


def norm(word):
    return re.sub(r"[^a-z0-9]+","",word.lower())


def project_rows(snapshot, ids):
    path=snapshot/"model.safetensors-00001-of-00001.safetensors"
    if path.resolve().name!="aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1":
        raise ValueError("Expected immutable HF cache for pinned Qwen revision")
    with path.open("rb") as f:
        header=struct.unpack("<Q",f.read(8))[0]
        schema=json.loads(f.read(header))["model.language_model.embed_tokens.weight"]
        if schema["dtype"]!="BF16" or schema["shape"]!=[248320,2048]:
            raise ValueError("Unsupported Qwen embedding matrix")
        base=8+header+schema["data_offsets"][0]
        # Only read the requested rows. With these datasets this is < 40 MiB of tensors.
        matrix=np.empty((len(ids),2048),dtype=np.float32)
        for i,num in enumerate(ids):
            f.seek(base+int(num)*2048*2)
            arr=np.frombuffer(f.read(2048*2),dtype="<u2").astype("<u4")
            matrix[i]=(arr<<16).view("<f4")
    matrix/=np.maximum(np.linalg.norm(matrix,axis=1,keepdims=True),1e-20)
    rng=np.random.default_rng(513)
    projection=rng.normal(size=(2048,64)).astype(np.float32)/8
    return matrix@projection


def make(snapshot):
    pkt=ROOT/"cohesion-stimuli.jsonl"
    stops=json.loads((OUT/"stimuli.json").read_text())
    if sha(pkt)!=stops["packet_sha256"]:
        raise ValueError("Stimulus seal mismatch")
    tokenizer_file=snapshot/"tokenizer.json"
    if sha(tokenizer_file)!="5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42":
        raise ValueError("Unknown tokenizer revision")
    tokenizer=Tokenizer.from_file(str(tokenizer_file))
    docs=[json.loads(l) for l in pkt.open()]
    encoded=[tokenizer.encode(x["text"],add_special_tokens=False) for x in docs]
    ids=sorted({num for d in encoded for num in d.ids})
    emb=project_rows(snapshot,ids)
    lookup={v:j for j,v in enumerate(ids)}
    # Independent replication of the locked GPU lexical preprocessing for parity checking.
    stopwords=set(("a an the am is are was were be been being has have had do did does can could would should "
                   "may might will shall of by on at in to from and or but for as it its if this that these those "
                   "he she they them their his her him hers you i we our my me us with into out").split())
    features={}
    for record,enc in zip(docs,encoded,strict=True):
        txt=record["text"]; offsets=np.array(enc.offsets)
        words=list(re.finditer(r"\S+",txt))
        vals=np.array([norm(m.group()) for m in words])
        word_starts=np.array([m.start() for m in words]);word_ends=np.array([m.end() for m in words])
        content=np.zeros(len(enc.ids),dtype=bool)
        for match,w in zip(words,vals,strict=True):
            if w and w not in stopwords:
                content|=(offsets[:,0]<match.end())&(offsets[:,1]>match.start())
        content &= np.array([any(ch.isalnum() for ch in txt[s:e]) for s,e in offsets])
        tokens=emb[[lookup[i] for i in enc.ids]]
        for stop in record["stops"]:
            s,e=stop["start"],stop["end"]
            c=content & (offsets[:,1]<=s); t=content & (offsets[:,0]<e)&(offsets[:,1]>s)
            a=tokens[c].mean(0) if c.any() else np.zeros(64)
            b=tokens[t].mean(0) if t.any() else np.zeros(64)
            den=np.linalg.norm(a)*np.linalg.norm(b)
            wc=vals[word_ends<=s].tolist();wt=vals[(word_starts<e)&(word_ends>s)].tolist()
            ac={w for w in wc if w and w not in stopwords};bc={w for w in wt if w and w not in stopwords}
            overlap=next((k/len(wt) for k in range(min(len(wc),len(wt)),0,-1) if wc[-k:]==wt[:k]),0.)
            lexical=[np.log1p(stop["index"]),np.log1p(len(wc)),np.log1p(len(wt)),len(ac&bc)/max(1,len(bc)),overlap]
            features[stop["id"]]=np.r_[lexical,float(a@b/den) if den else 0.,a,b,a*b]
    return stops["material"],features,len(ids)


def run(snapshot):
    records,features,n_ids=make(snapshot)
    x=np.stack([features[r["id"]] for r in records])
    train=np.array([r["split"]=="train" for r in records])
    dev=np.array([r["split"]=="dev" for r in records])
    gold=np.array([votes(r)[:2] if a or b else (0,0) for r,a,b in zip(records,train,dev,strict=True)])
    train &= gold[:,1]>0; dev &= gold[:,1]>0
    arms={"lexical_cpu":x[:,:5],"static":x}
    result={};forecasts={}
    for name,f in arms.items():
        m=binomial(f[train],gold[train,0],gold[train,1]);p=predict(f,m)
        cutoff=float(np.quantile(np.repeat(p[dev],gold[dev,1]-gold[dev,0]),.9,method="higher"))
        result[name]={"model":m,"threshold":cutoff}
        forecasts[name]=p
    pred=[]
    for i,r in enumerate(records):
        if r["split"] in ("holdout","external"):
            pred.append({**{k:r[k] for k in ("id","source","split","group","index")},
                         **{k:float(v[i]) for k,v in forecasts.items()}})
    output=OUT/"static-predictions.json"
    dump(output,pred)
    dump(OUT/"static-lock.json",{"protocol_sha256":sha(Path("research/reading_validation/transition-static-control.md")),
        "forecast_sha256":sha(output),"feature_source":"Qwen3.5-2B static BF16 input embedding",
        "vocabulary_rows":n_ids,"embedding_f32_bytes":n_ids*2048*4,"arms":result})


if __name__=="__main__":
    parser=ArgumentParser()
    parser.add_argument("snapshot",type=Path)
    run(parser.parse_args().snapshot)
