"""Score a source-calibrated T5 character rule and source-matched causal writer rules."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_author_precision import counts,measures
from research.reading_validation.transition_external_score import evaluate_one,grouped_distribution
from research.reading_validation.transition_score import dump,sha,with_interval
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import RESULTS,ROOT,source
from research.reading_validation.transition_t5_score import OUT,mark
from research.reading_validation.transition_t5_serving import DIRECT,PARTIAL
from research.reading_validation.transition_writer_f1 import threshold


PROTOCOL=Path("research/reading_validation/transition-t5-decision-protocol.md")
DEST=OUT/"type-calibration"


def score_panel(y,grp,models,cutoffs):
    y=np.asarray(y,dtype=int);titles,g,draws=grouped_distribution(grp);out={};boot={}
    for m,v in models.items():
        p=np.array(v,dtype=float);cut=cutoffs[m];flags=p>=cut if cut is not None else np.zeros(len(p),bool)
        # The ranking uncertainty uses the same resampled source units for all models.
        r,b=evaluate_one(y,np.ones_like(y),p,np.nextafter(cut,-np.inf) if cut is not None else np.inf,g,draws)
        c=counts(y>0,flags,g);point=measures(c.sum(axis=0,keepdims=True));intervals=measures(draws@c)
        out[m]={"threshold":cut,"tp_fp_fn_tn":c.sum(0).astype(int).tolist(),"average_precision":r["ap"],"auroc":r["auroc"],
                **{k:with_interval(point[k][0],intervals[k]) for k in ("precision","recall","f1","false_alarms_per_100_negatives")}}
        boot[m]={"average_precision":b["ap"],"auroc":b["auroc"],**intervals}
    compare={}
    pairs=[("served_marginal","ner_surprisal","served_marginal_minus_ner_surprisal")]
    if "t5_calibrated" in models:
        pairs.extend(("t5_calibrated",m,"t5_minus_"+m) for m in models if m!="t5_calibrated")
    for left,right,label in pairs:
        compare[label]={k:with_interval(out[left][k]["estimate"]-out[right][k]["estimate"],boot[left][k]-boot[right][k])
                       for k in ("average_precision","auroc","precision","recall","f1")}
    return {"complete_groups":len(titles),"locations":len(y),"human_positives":int(y.sum()),"methods":out,"paired":compare}


def source_calibration(material,docs,folds,raw_by_doc,forecasts):
    sent=[];word=[];sy=[];wy=[];tally=Counter()
    for s in material:
        if s["split"]!="dev":continue
        d=s["doc"];ix=slice(s["word_index_start"],s["word_index_stop"]);doc=docs[d]
        valid,gold,_=outcomes(doc,folds[d],raw_by_doc[d],"CharE")
        if not valid[ix].all():raise ValueError("T5 book case lacks fully scored words")
        f=forecasts[s["id"]];words=[doc["text"][a:b] for a,b,*_ in doc["words"][ix]]
        mask,count=mark(f["conditional_generation"],words);tally.update(count)
        prob=float(np.exp(-np.logaddexp(0.,-f["class_log_odds"])))
        sent.append(prob);sy.append(bool((gold[ix]>0).any()));word.extend(prob*mask);wy.extend(gold[ix]>0)
    if len(sent)!=716 or sum(sy)!=131:raise ValueError("Development population differs")
    return {"sentences":threshold(sent,sy),"words":threshold(word,wy),"sentences_n":len(sy),"words_n":len(wy),"conditional_spans":dict(tally)}


def main(folder=None):
    meta=json.loads((OUT/"input-seal.json").read_text());old=OUT/"source-equal-f1-thresholds.json";rules=json.loads(old.read_text())["model_decisions"]
    docs,folds=inputs();rows={m["doc"]:m for m in folds["documents"]};gold=source()
    cuts={k:{m:v["cutoff"] for m,v in d.items()} for k,d in rules.items()};material=meta["sentences"]
    DEST.mkdir(exist_ok=True);lock=DEST/"book-operating-points.json";forecasts={};receipt=None
    if folder:
        receipt=json.loads((folder/"receipt.json").read_text());fn=folder/"source-and-movie-forecasts.jsonl";stim=ROOT/"snac-t5-calibration-stimuli.jsonl"
        if (sha(PROTOCOL)!=receipt["protocol_sha256"] or sha(fn)!=receipt["output_sha256"] or sha(stim)!=receipt["input_sha256"] or
                receipt["source_adapter_sha256"]!=sha(OUT/"adapter/selected-adapter.pt") or receipt["external_labels"]!=0 or receipt["encoder_rows"]!=2377):
            raise ValueError("Fixed T5 diagnostic predictions changed")
        forecasts={v["id"]:v for v in map(json.loads,fn.read_text().splitlines())};source_dec=source_calibration(material,docs,rows,gold,forecasts)
        dump(lock,{"protocol_sha256":sha(PROTOCOL),"t5_receipt":receipt,"source_calibration":source_dec,"comparison_thresholds_sha256":sha(old)})
        for u in ("words","sentences"):cuts[u]["t5_calibrated"]=source_dec[u]["cutoff"]
        cuts["words"]["t5_forced_spans"]=.5
    direct=np.load(DIRECT/"word-probabilities.npz");dr=json.loads((DIRECT/"receipt.json").read_text())
    labels=dr["tasks"];arms=dr["arms"];cut=dr["cutoffs"]["direct"][labels.index("CharE")]
    cuts["words"]["direct_fixed"]=cut;cuts["sentences"]["direct_fixed"]=cut
    marg=np.load(PARTIAL/"locked-forecasts.npz")["set_marginal"];marg=np.exp(-np.logaddexp(0.,-marg));offset=0;ner=np.load(RESULTS/"entity-control-probabilities.npz")
    by=defaultdict(list)
    for s in material:
        if s["split"]=="test":by[s["doc"]].append(s)
    group=defaultdict(list);outcomes_by_unit=defaultdict(list);models={u:defaultdict(list) for u in ("words","sentences")}
    visibility=defaultdict(list);tally=Counter();nviews=Counter()
    for row in folds["documents"]:
        if row["split"]!="test":continue
        d=row["doc"];doc=docs[d];valid,answer,_=outcomes(doc,row,gold[d],"CharE")
        count=len(doc["words"]);qwen=marg[offset:offset+count];offset+=count
        words={"served_marginal":qwen,"ner_surprisal":ner[d][:,0],"direct_fixed":direct[d][:,arms.index("direct"),labels.index("CharE")]}
        for s in by[d]:
            ix=slice(s["word_index_start"],s["word_index_stop"]);can=valid[ix];truth=answer[ix]>0
            word_methods={k:v[ix] for k,v in words.items()}
            sent_methods={k:float(z.max()) for k,z in word_methods.items()};v="unknown"
            if forecasts:
                f=forecasts[s["id"]];v=f["visibility"];nviews[v]+=1
                p=float(np.exp(-np.logaddexp(0.,-f["class_log_odds"])))
                aus=[doc["text"][a:b] for a,b,*_ in doc["words"][ix]]
                mask,matches=mark(f["conditional_generation"],aus);tally.update(matches)
                word_methods.update(t5_calibrated=p*mask,t5_forced_spans=mask.astype(float));sent_methods["t5_calibrated"]=p
            if can.all():
                group["sentences"].append(row["group"]);outcomes_by_unit["sentences"].append(truth.any());visibility["sentences"].append(v)
                for k,z in sent_methods.items():models["sentences"][k].append(z)
            group["words"].extend([row["group"]]*int(can.sum()));outcomes_by_unit["words"].extend(truth[can]);visibility["words"].extend([v]*int(can.sum()))
            for k,z in word_methods.items():models["words"][k].extend(z[can])
    if len(group["sentences"])!=1416 or len(group["words"])!=22133 or offset!=26417:raise ValueError("The common movie denominator differs")
    result={}
    for u,z in models.items():
        result[u]=score_panel(outcomes_by_unit[u],group[u],z,cuts[u])
        if folder:
            for v in ("full","partial","hidden"):
                mask=np.array(visibility[u])==v;y=np.asarray(outcomes_by_unit[u])[mask];g=np.asarray(group[u])[mask]
                if len(np.unique(g))<2 or len(set(y))!=2:continue
                result[u+"_"+v]=score_panel(y,g,{m:np.asarray(vp)[mask] for m,vp in z.items()},cuts[u])
    out=DEST/("movie-scores.json" if folder else "qwen-ner-source-matched-movies.json")
    dump(out,{"source_thresholds_sha256":sha(old),"t5_source_decision_sha256":sha(lock) if folder else None,
              "sentence_visibility":dict(nviews),"conditional_generation_format":dict(tally),"scores":result})
    print({u:{k:round(a["f1"]["estimate"],3) for k,a in res["methods"].items()} for u,res in result.items()},flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--features",type=Path);main(p.parse_args().features)
