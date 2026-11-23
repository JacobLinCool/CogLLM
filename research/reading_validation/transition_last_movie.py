"""Prepare or score the sole previously unevaluated movie's supported prefix."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from research.reading_validation.narrative_inference import MarginalNarrativeInference, certified_words
from research.reading_validation.transition_balanced_bounds import counts_by_segment, sharp_bounds
from research.reading_validation.transition_external_fit import predict
from research.reading_validation.transition_full_movie_bounds import segment_solutions
from research.reading_validation.transition_score import dump, sha
from research.reading_validation.transition_snac_ner import annotate_document, taggers
from research.reading_validation.transition_snac_prepare import DIGEST, RESULTS, ROOT, WORD, source
from research.reading_validation.transition_writer_export import SOURCE

ID = "tripod21"
DEST = RESULTS / "unseen-movie"
PROTOCOL = Path("research/reading_validation/transition-last-movie-protocol.md")
PACKET = ROOT / "last-movie-stimuli.jsonl"


def prepare():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer.json"))
    old=json.loads((RESULTS / "stimuli.json").read_text())
    if old["excluded_complete_summaries"] != [{"doc":ID,"group":"movie-21","tokens":2487}]:
        raise ValueError("The external case does not match the original exclusion")
    src=source()[ID]
    sections=[" ".join(src[str(i)]["text"].split()) for i in range(len(src))]
    complete=" ".join(sections); text=""; segments=[]
    for i, part in enumerate(sections):
        new=text+" "+part if text else part
        if len(tok.encode(new,add_special_tokens=False).ids)>2048:
            break
        segments.append({"index":i,"start":len(text)+bool(text),"end":len(new)});text=new
    tokens=tok.encode(text,add_special_tokens=False)
    words=certified_words(text,tok,tuple(tokens.ids),list(map(tuple,tokens.offsets)),[(s["start"],s["end"]) for s in segments])
    if len(words)!=len(list(WORD.finditer(text))) or len(segments)>=len(src):
        raise ValueError("Incomplete certification of evaluated word positions")
    row={"id":ID,"text":text,"words":words}
    PACKET.write_text(json.dumps(row,sort_keys=True)+"\n")
    DEST.mkdir(parents=True,exist_ok=True)
    dump(DEST/"seal.json",{"protocol_sha256":sha(PROTOCOL),"source_sha256":DIGEST,"input_sha256":sha(PACKET),
         "reference_source_decision_sha256":sha(SOURCE),"full_tokens":len(tok.encode(complete,add_special_tokens=False).ids),
         "full_original_words":len(list(WORD.finditer(complete))),"full_original_segments":len(sections),
         "presented_tokens":len(tokens.ids),"presented_words":len(words),"presented_segments":segments,"external_outcomes_read":False})
    print(sha(PACKET),len(tokens.ids),len(words),len(segments),len(sections))


def score(folder):
    seal=json.loads((DEST/"seal.json").read_text())
    receipt=json.loads((folder/"receipt.json").read_text()); f=folder/"word-features.npz"
    if (sha(PROTOCOL)!=seal["protocol_sha256"] or sha(PACKET)!=seal["input_sha256"] or receipt["input_sha256"]!=sha(PACKET)
            or sha(f)!=receipt["feature_sha256"] or receipt["documents"]!=1 or receipt["forwards"]!=1
            or sha(SOURCE)!=seal["reference_source_decision_sha256"]):
        raise ValueError("The original prefix or its forecasts changed")
    row=json.loads(PACKET.read_text()); seg=seal["presented_segments"]
    with np.load(f) as data: x=data[ID]
    head=MarginalNarrativeInference.for_writers()
    nerd=json.loads((RESULTS/"entity-control-lock.json").read_text())
    tagger,ner,tok=taggers()
    ent=annotate_document(row,seg,tagger,ner,tok)
    cut=json.loads(SOURCE.read_text())["model_decisions"]["words"]["ner_surprisal"]["cutoff"]
    pp=head.probabilities(x);pn=predict(np.column_stack((x[:,:17],ent)),nerd["model"])
    pred=DEST/"frozen-forecasts.npz";np.savez_compressed(pred,served=pp,ner=pn)
    flags=head.flags(pp);base=pn>=cut;human=source()[ID];words=np.array(row["words"]);groups=[];placements=[];n_records=0
    for s in seg:
        i=s["index"]; ix=np.flatnonzero(words[:,4]==i);record=human[str(i)]
        e=[e for e in record["errors"] if e["error_type"]=="CharE"]
        possible,multiple=segment_solutions(words[ix,:2]-s["start"],row["text"][s["start"]:s["end"]],e)
        n_records+=len(e);placements.extend(multiple);groups.append((ix,possible))
    res=sharp_bounds(counts_by_segment(groups,flags,base),[flags.sum(),base.sum()],len(pp))
    dump(DEST/"scores.json",{"protocol_sha256":sha(PROTOCOL),"source_sha256":DIGEST,"forecasts_sha256":sha(pred),
         "receipt":receipt,"recorded_character_spans":n_records,"ambiguous_span_records":sum(n>1 for n in placements),
         "eligible_words":len(pp),"excluded_tail_words":seal["full_original_words"]-len(pp),"source_thresholds":[head.threshold,cut],
         "all_placements":res})
    print(res)


if __name__=="__main__":
    cli=ArgumentParser(description=__doc__);cli.add_argument("--features",type=Path)
    args=cli.parse_args()
    if args.features:score(args.features)
    else:prepare()
