"""Compare source-selected writer flags with an outcome-blind specialist on the same movies."""
from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.narrative_inference import MARGINAL_SHA, MarginalNarrativeInference
from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_score import dump, sha
from research.reading_validation.transition_sentence_comparator import record,summarize
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import RESULTS as SNAC,source
from research.reading_validation.transition_t5_score import OUT,compare,mark


PROTOCOL=Path("research/reading_validation/transition-t5-serving-comparison.md")
RESULTS=Path("research/results")
PARTIAL=RESULTS/"reading-validation-transition-partial"
DIRECT=RESULTS/"reading-validation-transition-direct/qwen"


def supplemental(variant):
    docs,folds=inputs();gold=source();model=MarginalNarrativeInference.load()
    mat=json.loads((OUT/"input-seal.json").read_text());raw_by_doc=defaultdict(list)
    for s in mat["sentences"]:
        if s["split"]=="test":raw_by_doc[s["doc"]].append(s)
    dr=json.loads((DIRECT/"receipt.json").read_text());dp=DIRECT/"word-probabilities.npz"
    replay=json.loads((PARTIAL/"serving/cached-replay.json").read_text());mp=PARTIAL/"locked-forecasts.npz"
    if sha(dp)!=dr["predictions_sha256"] or sha(mp)!=replay["forecast_sha256"] or replay["bundle_sha256"]!=MARGINAL_SHA:
        raise ValueError("Word probabilities did not match their certified original")
    direct=np.load(dp);marginal=np.exp(-np.logaddexp(0.,-np.load(mp)["set_marginal"]))
    cutoff=dr["cutoffs"]["direct"][dr["tasks"].index("CharE")];arm=dr["arms"].index("direct");task=dr["tasks"].index("CharE")
    t5={};receipt=None
    if variant:
        f=OUT/variant/"movie-generations.jsonl";receipt=json.loads((OUT/variant/"receipt.json").read_text())
        if sha(f)!=receipt["forecast_sha256"] or receipt["input_sha256"]!=mat["input_sha256"]:raise ValueError("T5 generation provenance differs")
        t5={r["id"]:r["generation"] for r in map(json.loads,f.read_text().splitlines())}
    idx=0;word_gold=[];word_group=[];first=[];word_masks=defaultdict(list);sentence_masks=defaultdict(list);sentence_group=[]
    entity=SNAC/"entity-control-probabilities.npz"
    if sha(entity)!=json.loads((SNAC/"entity-control-lock.json").read_text())["forecast_sha256"]:raise ValueError("First-PERSON geometry changed")
    ner=np.load(entity)
    for row in folds["documents"]:
        if row["split"]!="test":continue
        d=row["doc"];doc=docs[d];n=len(doc["words"])
        valid,votes,_=outcomes(doc,row,gold[d],"CharE");positive=votes>0
        flags={"served_marginal":marginal[idx:idx+n]>model.threshold,"direct_native":direct[d][:,arm,task]>cutoff}
        if variant:flags["t5_span"]=np.zeros(n,dtype=bool)
        idx+=n
        for s in raw_by_doc[d]:
            region=slice(s["word_index_start"],s["word_index_stop"]);eligible=valid[region].all();predicted={}
            if variant:
                words=[doc["text"][a:b] for a,b,*_ in doc["words"][region]]
                mask,stats=mark(t5[s["id"]],words);flags["t5_span"][region]=mask
                predicted["t5_span"]=bool(stats["well_formed_character_sentence"] or stats["empty_character_sentence"])
            if eligible:
                sentence_group.append(row["group"])
                for k,f in flags.items():sentence_masks[k].append(record(positive[region],f[region],predicted.get(k,bool(f[region].any()))))
        word_gold.extend(positive[valid]);word_group.extend([row["group"]]*int(valid.sum()));first.extend(ner[d][valid,2]>0)
        for k,v in flags.items():word_masks[k].extend(v[valid])
    if idx!=replay["words"]:raise ValueError("Certified word denominator changed")
    titles,ids,draws=grouped_distribution(sentence_group)
    sentences={k:summarize(np.stack(v),ids,draws) for k,v in sentence_masks.items()}
    # Reuse the original movie-level uncertainty implementation with both explicit competitors.
    def words(selection):
        names={"t5_span":"t5_span","qwen_word":"served_marginal","ner_control":"direct_native"}
        y=np.asarray(word_gold);g=np.asarray(word_group)
        if not variant:names["t5_span"]="served_marginal"
        v=compare(y,{k:np.asarray(word_masks[ident]) for k,ident in names.items()},g,np.asarray(selection))
        v["methods"]={names[k]:x for k,x in v["methods"].items() if k!="t5_span" or variant}
        if variant:v["paired_differences"]={k.replace("qwen_word","served_marginal").replace("ner_control","direct_native"):x for k,x in v["paired_differences"].items()}
        else:v.pop("paired_differences")
        if not variant:v.pop("word_alert_overlap")
        return v
    dest=OUT/(variant or "")
    dump(dest/"serving-comparison.json",{"protocol_sha256":sha(PROTOCOL),"variant":variant,"direct_source_cut":cutoff,
        "marginal_source_cut":model.threshold,"direct_forecasts_sha256":sha(dp),"marginal_forecasts_sha256":sha(mp),
        "t5_forecasts_sha256":receipt["forecast_sha256"] if receipt else None,"all_words":words(np.ones(len(word_group),bool)),
        "first_person_mention":words(first),"sentence_metrics":sentences,"fully_measured_sentences":len(sentence_group),"movie_groups":len(titles)})
    print(len(word_gold),len(sentence_group),{k:round(v["f1"]["estimate"],3) for k,v in sentences.items()})


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--variant",choices=("adapter","full"));supplemental(p.parse_args().variant)
