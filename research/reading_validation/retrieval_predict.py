"""Freeze all new reading trajectories before joining marked spans or human keys."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.reading_validation.memory_data import sha
from research.reading_validation.retrieval_data import DATA, RESULT
from research.reading_validation.retrieval_features import Corpus
from research.reading_validation.retrieval_fit import Readout


def source_forecasts(corpus,models):
    meta=json.loads((DATA/"training.json").read_bytes())
    rows=[]
    for item in meta:
        if item["split"]=="train":
            continue
        for key in item["ids"]:
            page=corpus.page(key)
            v=page.pairs(len(page.sections)-1,terminal=True)
            methods={}
            for name,model in models.items():
                logit,p,_=model(v)
                methods[name]=dict(conflict=float(logit[0]),source=p[0].tolist())
            methods["native_attention"]=dict(source=v[0,:,8].tolist())
            methods["word_overlap"]=dict(source=v[0,:,5].tolist())
            rows.append(dict(id=key,family=item["family"],methods=methods))
    return rows


def read_page(page,models):
    count=len(page.coords)
    word=np.zeros((count,6))
    pointers=np.full((count,4),-1,dtype=np.int16)
    code,prior=None,None
    curve,surprisals=[],[]
    for j,ix in enumerate(page.sections):
        n=len(ix)
        source_surprisal=page.signals[ix,1]
        per_token=source_surprisal/page.signals[ix,2]
        surprisals.append([float(per_token.mean()),float(np.quantile(per_token,.9)),float(per_token.max())])
        # No reading can report a contrast with an earlier sentence in the first unit.
        code=np.full(n,-np.log(2))
        strength=np.zeros(n)
        word[ix,0]=page.signals[ix,0]
        word[ix,1]=per_token
        if j:
            x=page.pairs(j)
            for i,(name,head) in enumerate(models.items()):
                z,p,posterior=head(x)
                if name=="semantic":
                    code=z
                    strength=p.max(-1)
                word[ix,2+i]=z
                pointers[ix,i]=posterior.argmax(-1)
            word[ix,4]=strength
            pointers[ix,2]=x[:,:,8].argmax(-1)
            pointers[ix,3]=x[:,:,5].argmax(-1)
        else:
            word[ix,2:4]=code[:,None]
        word[ix,5]=j
        delta=0. if prior is None else code[-1]-prior
        curve.append([float(code[-1]),float(delta),float(code.max()),float(strength[-1])])
        prior=code[-1]
    dense=np.column_stack((page.coords,word,pointers))
    return dict(retrieval=curve,surprisal=surprisals),dense


def run(archive,fit,output):
    if output.exists():
        raise FileExistsError(output)
    seal=json.loads((fit/"seal.json").read_bytes())
    if sha(fit/"weights.json")!=seal["weights"] or sha(archive/"receipt.json")!=seal["archive"]:
        raise ValueError("Native extraction or fitted maps changed")
    weights=json.loads((fit/"weights.json").read_bytes())
    models={k:Readout(weights[k]) for k in ("lexical","semantic")}
    corpus=Corpus(archive)
    source=source_forecasts(corpus,models)
    mapping=json.loads((DATA/"mapping.json").read_bytes())
    output.mkdir(parents=True)
    (output/"source.json").write_text(json.dumps(source,allow_nan=False)+"\n")
    for kind,items in mapping.items():
        records,dense={},{}
        for original,key in items.items():
            records[original],dense[original]=read_page(corpus.page(key),models)
        (output/(kind+".json")).write_text(json.dumps(records,allow_nan=False)+"\n")
        np.savez_compressed(output/(kind+".npz"),**dense)
    files={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    header=["start","end","original_nli","surprisal","lexical","semantic","max_relevance","unit",
            "lexical_pointer","semantic_pointer","attention_pointer","overlap_pointer"]
    (output/"seal.json").write_text(json.dumps(dict(source_fit=sha(fit/"seal.json"),columns=header,files=files),
                                              indent=2)+"\n")
    print("Sealed source and reading forecasts", {k:len(v) for k,v in mapping.items()})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--fit",type=Path,default=RESULT/"fit")
    parser.add_argument("--output",type=Path,default=RESULT/"readout")
    args=parser.parse_args()
    run(args.archive,args.fit,args.output)
