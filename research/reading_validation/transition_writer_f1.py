"""Calibrate already fitted word and sentence observers using only the same book-development cases."""
from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.narrative_inference import MarginalNarrativeInference
from research.reading_validation.transition_external_fit import predict
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import RESULTS,ROOT,source
from research.reading_validation.transition_t5_prepare import RESULT as T5

PROTOCOL=Path("research/reading_validation/transition-writer-f1-calibration.md")


def threshold(score,y):
    score=np.asarray(score,dtype=float);y=np.asarray(y,bool)
    if len(y)!=len(score) or len(np.unique(y))!=2 or np.any((score<0)|(score>1)):raise ValueError("Unavailable source calibration class")
    order=np.argsort(-score,kind="stable");yy=y[order];s=score[order]
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1];ends=ends[s[ends]>0]
    if not len(ends):return {"cutoff":None,"tp_fp_fn_tn":[0,0,int(y.sum()),int((~y).sum())],"f1":0.}
    tp=np.cumsum(yy)[ends];fp=ends+1-tp;fn=y.sum()-tp;f1=2*tp/(2*tp+fp+fn)
    j=int(f1.argmax())
    if not f1[j]>0:return {"cutoff":None,"tp_fp_fn_tn":[0,0,int(y.sum()),int((~y).sum())],"f1":0.}
    return {"cutoff":float(s[ends[j]]),"tp_fp_fn_tn":[int(tp[j]),int(fp[j]),int(fn[j]),int((~y).sum()-fp[j])],"f1":float(f1[j])}


def main(folder):
    f=folder/"word-features.npz";rec=json.loads((folder/"receipt.json").read_text())
    if sha(f)!=rec["feature_sha256"]:raise ValueError("Original word features changed")
    pack=json.loads((T5/"input-seal.json").read_text());cutrows=defaultdict(list)
    for s in pack["sentences"]:
        if s["split"]=="dev":cutrows[s["doc"]].append(s)
    docs,meta=inputs();raw=source();nerfile=ROOT/"snac-ner.npz";nerlock=json.loads((RESULTS/"entity-control-lock.json").read_text())
    if sha(nerfile)!=nerlock["local_ner_sha256"]:raise ValueError("Named-entity book features changed")
    with np.load(f) as x,np.load(nerfile) as ent:
        m=MarginalNarrativeInference.load();wsrc=defaultdict(list);ssrc=defaultdict(list);yw=[];ys=[];titles=set()
        for item in meta["documents"]:
            if item["split"]!="dev":continue
            d=item["doc"];titles.add(item["group"]);ok,vs,_=outcomes(docs[d],item,raw[d],"CharE")
            scores={"served_marginal":m.probabilities(x[d]),"ner_surprisal":predict(np.column_stack((x[d][:,:17],ent[d])),nerlock["model"])}
            for s in cutrows[d]:
                ix=slice(s["word_index_start"],s["word_index_stop"])
                if not ok[ix].all():raise ValueError("Development word lacks identified human supervision")
                y=vs[ix]>0;yw.extend(y);ys.append(bool(y.any()))
                for k,p in scores.items():wsrc[k].extend(p[ix]);ssrc[k].append(float(p[ix].max()))
    if len(ys)!=716 or sum(ys)!=131 or len(titles)!=11:raise ValueError("The source-selected 716-sentence population changed")
    res={unit:{m:threshold(p,y) for m,p in a.items()} for unit,a,y in (("words",wsrc,yw),("sentences",ssrc,ys))}
    dump(T5/"source-equal-f1-thresholds.json",{"protocol_sha256":sha(PROTOCOL),"source_observer_features_sha256":sha(f),"development_books":len(titles),
         "development_sentences":len(ys),"development_positive_sentences":sum(ys),"development_words":len(yw),"development_positive_words":int(sum(yw)),
         "model_decisions":res})
    print(res)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--features",type=Path,required=True);main(p.parse_args().features)
