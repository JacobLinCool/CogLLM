"""Certify that the shared batched essay readout reproduces original article encodings."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from research.reading_validation.argument_data import h, sentences
from research.reading_validation.essay_hazard_inference import EssayHazardInference, WEIGHTS
from research.reading_validation.transition_essay_reasons import DEST
from research.reading_validation.transition_external_prepare import OUT, ROOT
from research.reading_validation.transition_score import dump, sha


def prepare(folder):
    historic = {r["id"]:r for r in map(json.loads,(folder/"features.jsonl").read_text().splitlines())}
    data = json.loads((OUT/"stimuli.json").read_text())
    docs = {r["id"]:r for r in map(json.loads,(ROOT/"cohesion-stimuli.jsonl").read_text().splitlines())}
    ids = sorted({r["doc"] for r in data["material"] if r["source"]=="decor"}, key=lambda k:h("source-parity/"+k))
    model = EssayHazardInference.load()
    expected = {}; packet = []
    for key in ids:
        doc = docs[key]; text = doc["text"]
        spans = sentences(text)
        if not 60 <= len(text.split()) <= 400 or len(spans)<5:
            continue
        stops = {(s["start"],s["end"]):s for s in doc["stops"]}
        matched = []
        for i, (a,b) in enumerate(spans,1):
            if i==1 or (a,b) not in stops:
                continue
            row = historic[stops[a,b]["id"]]
            features = np.r_[np.log1p(i),row["features"],row["representation"]]
            risk, lexical = model.probabilities(features[None,:])
            matched.append(dict(position=i,start=a,end=b,essay_hazard=float(risk[0]),essay_lexical=float(lexical[0])))
        if len(matched)>=4:
            expected[key]=matched; packet.append({"id":key,"text":text})
        if len(packet)==3:break
    if len(packet)!=3:raise ValueError("Insufficient independently archived essay stops")
    path=ROOT/"essay-batched-parity-stimuli.jsonl"
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in packet))
    dump(DEST/"batch-source-reference.json",{"packet_sha256":sha(path),"weights_sha256":sha(WEIGHTS),
         "single_source_feature_sha256":sha(folder/"features.jsonl"),"expected":expected,
         "max_probability_error":1e-4})
    print(sha(path),sum(len(s) for s in expected.values()),flush=True)


def verify(folder):
    ref=json.loads((DEST/"batch-source-reference.json").read_text());receipt=json.loads((folder/"receipt.json").read_text())
    if (receipt["source"]!=ref["packet_sha256"] or receipt["encoder_batches"]!=1 or
        sha(folder/"features.npz")!=receipt["file"] or set(receipt["articles"])!=set(ref["expected"])):
        raise ValueError("A source exposure was skipped or changed")
    errors={k:[] for k in ("essay_hazard","essay_lexical")}
    widths=[]
    for k,expect in ref["expected"].items():
        observed={(p["start"],p["end"],p["position"]):p for p in receipt["articles"][k]["essay"]}
        matched=0
        for v in expect:
            key=(v["start"],v["end"],v["position"])
            if key not in observed:continue
            matched+=1
            for name,values in errors.items():values.append(abs(v[name]-observed[key][name]))
        widths.append((len(expect),matched))
    if any(n!=m for n,m in widths):raise ValueError("The batch skipped an originally archived aligned stop")
    delta={m:max(v) for m,v in errors.items()}
    if max(delta.values())>ref["max_probability_error"]:raise ArithmeticError(f"Source native parity exceeded: {delta}")
    summary={"original_sha256":sha(DEST/"batch-source-reference.json"),"batch_receipt_sha256":sha(folder/"receipt.json"),
             "source_exposures":len(widths),"source_stops":sum(n for n,_ in widths),"encoder_batches":receipt["encoder_batches"],
             "max_probability_error":delta,"maximum_tolerance":ref["max_probability_error"]}
    dump(DEST/"batch-source-replay.json",summary);print(summary)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("stage",choices=["prepare","verify"]);p.add_argument("folder",type=Path)
    args=p.parse_args(); (prepare if args.stage=="prepare" else verify)(args.folder)
