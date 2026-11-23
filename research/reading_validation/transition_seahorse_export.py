"""Export the compact Q1 observer and independently replay it on all original held forecastable texts."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from research.reading_validation.memory_data import h
from research.reading_validation.summeval_forecast import open_old
from research.reading_validation.summary_understanding import BUNDLE, SCOPE, SummaryUnderstanding
from research.reading_validation.transition_seahorse_fit import key
from research.reading_validation.transition_seahorse_data import DEST, WORK, sha, source_rows
from research.reading_validation.transition_seahorse_features import load, save

DEMO=("The city will replace the footbridge after inspectors found cracks in its supports. "
      "A shuttle will carry pedestrians across the river during the repairs.")


def export(archive):
    fit=load(DEST/"fit-seal.json")
    p,_=open_old()
    original=DEST/"supervised-weights.npz"
    if sha(original)!=fit["weights_sha256"]:raise ValueError("The original source Q1 fit changed")
    rev=load(archive/"receipt.json")
    if sha(archive/"features.npz")!=rev["file"] or rev["source"]!=load(DEST/"exposure-seal.json")["packets"]["test"]["sha256"]:
        raise ValueError("Expected exact official held Qwen archive")
    BUNDLE.mkdir(exist_ok=True,parents=True)
    with np.load(original,allow_pickle=False) as a:
        np.savez_compressed(BUNDLE/"weights.npz",center=a["native.mean"],scale=a["native.scale"],theta=a["native.theta"],
                       calibration=a["native.calibration"],region_center=p["english.center"],region_scale=p["english.scale"])
    spec={"schema":"english-summary-q1-v1","scope":SCOPE,"backbone_revision":rev["revision"],
          "threshold":fit["thresholds"]["native"],"weights_sha256":sha(BUNDLE/"weights.npz"),
          "trained_weights_sha256":fit["weights_sha256"],"readout":"P(professional SEAHORSE English rater answers No, summary alone)",
          "source_levels":["XSum","English XL-Sum","WikiLingua English"],"max_tokens":1024,
          "source_heads":"validation-calibrated native standardized binary logistic, no system ID or reference",
          "feature_schema_sha256":sha(Path(__file__).with_name("summary_measurement.py"))}
    save(BUNDLE/"spec.json",spec)
    inst=SummaryUnderstanding.load()
    rows=list(source_rows("test"))
    texts={h(r["summary"]):r["summary"] for r in rows}
    canonical={key(r):h(r["summary"]) for r in rows}
    src=DEST/"forecasts-before-held-responses.npz"
    with np.load(src,allow_pickle=False) as a:
        expected=dict(zip(a["keys"],a["native"],strict=True))
    result={}
    with np.load(archive/"features.npz",allow_pickle=False) as a:
        for k,details in rev["articles"].items():
            native={n:a[k+"."+n] for n in ("words","surprisal","rel","char")}
            result[k]=inst.from_arrays(texts[k],native)
        for k,why in rev["skipped"].items():result[k]=inst.unavailable(why)
    if set(result)!=set(texts):raise ValueError("Replay DID NOT cover all held strings")
    diffs=[]
    for row,k in canonical.items():
        real=result[k]
        if row in expected:
            if not real["available"]:raise ValueError("Known scorable item became unavailable")
            diffs.append(abs(expected[row]-real["probability_difficult"]))
            if real["review"]!=(expected[row]>=inst.threshold):raise ValueError("Published binary decision changed")
        elif real["available"]:
            raise ValueError("A previously excluded source acquired an unvalidated estimate")
    if max(diffs)>1e-10:raise ArithmeticError(("Unsigned downstream preprocessing",max(diffs)))
    # First hash within a fixed domain and source generator; votes do not affect the selected example.
    valid=sorted({h(r["summary"]) for r in rows if r["gem_id"].startswith("wiki_") and r["model"]=="mt5_small" and result[h(r["summary"])]["available"]})
    example=valid[0]
    held_example=next(r for r in rows if h(r["summary"])==example and r['model']=="mt5_small")
    selection=sorted(result,key=lambda k:h("seahorse-native-writer-replay-v1/"+k))[:64]
    selection=sorted(set(selection)|{example}|{k for k,v in result.items() if not v["available"]})
    output=WORK/"seahorse-writer-stimuli.jsonl"
    with output.open("w") as f:
        for k in [h(DEMO),example,*[k for k in selection if k!=example]]:
            f.write(json.dumps({"id":k,"text":texts.get(k,DEMO)},ensure_ascii=False)+"\n")
    lock={"scope":SCOPE,"receipt":sha(BUNDLE/"spec.json"),"held_original_rows":len(canonical),
          "held_original_unique":len(texts),"held_probabilities_verified":len(diffs),"max_absolute_error":max(diffs),
          "identical_screening_decisions":len(diffs),"abstained":dict(Counter(r["reason"] for r in result.values() if not r["available"])),
          "example":{"text":texts[example],"gem_id":held_example["gem_id"],"generator":"mt5_small","text_sha256":example,
                     "official_row_key":key(held_example),"grade_used_to_select":False},
          "synthetic_demo":DEMO,"native_packet":sha(output),"native_rows":len(selection)+1,
          "native_expected":{k:result[k] for k in selection}}
    save(DEST/"serving-replay.json",lock)
    print(sha(BUNDLE/"spec.json"),sha(BUNDLE/"weights.npz"),sha(output),len(diffs),max(diffs),len(selection)+1,example)
    print("Selected example:",held_example['gem_id'],repr(texts[example]))


if __name__=="__main__":
    from argparse import ArgumentParser
    p=ArgumentParser(description=__doc__);p.add_argument("archive",type=Path);export(p.parse_args().archive)
