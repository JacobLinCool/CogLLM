"""Export the source-trained physical-state estimator and ten outcome-blind replay cases."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from research.reading_validation.memory_data import h, sha
from research.reading_validation.state_inference import BUNDLE, PhysicalContinuity
from research.reading_validation.state_score import unpack
from research.reading_validation.state_transport_data import OUT, TRIP
from research.reading_validation.state_words import Archive


def run(archive):
    folder=OUT/"forecast-native";seal=json.loads((folder/"seal.json").read_bytes())
    if any(sha(folder/name)!=sig for name,sig in seal["files"].items()):
        raise ValueError("State readout differs from the original scientific evaluation.")
    spec=json.loads((folder/"model.json").read_bytes())
    weights={}
    source=folder/"attribute-full.npz"
    if sha(source)!=json.loads((folder/"attribute-full.seal.json").read_bytes())["sha"]:
        raise ValueError("Attribute parameters differ from the original calibrated model.")
    with np.load(source) as params:
        for key in ("mu","sd","v"):weights["attribute."+key]=params[key]
    for name in spec["modes"]:
        for key in ("mu","sd","v"):weights[name+"."+key]=np.asarray(spec["event"][name][key])
    BUNDLE.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(BUNDLE/"weights.npz",**weights)
    receipt=json.loads((archive/"receipt.json").read_bytes())
    params=dict(schema="physical-first-conflict-v1", revision=receipt["revision"], weights=sha(BUNDLE/"weights.npz"),
                source=sha(folder/"seal.json"), arms=spec["modes"], mixture=[spec["dev_mixture"][s] for s in spec["modes"]],
                threshold=spec["thresholds"]["mixture"], source_sentences=[5,7])
    (BUNDLE/"spec.json").write_text(json.dumps(params,indent=2)+"\n")
    reader=PhysicalContinuity.load();bank=Archive(archive)
    held=bank.spec["groups"]["physical"]["test"]
    families=defaultdict(list)
    for item in held.values():
        if item["change"]:families[item["family"]].append(item)
    ordered=sorted(families,key=lambda i:h("physical-continuity-replay-v1."+str(i)))
    rows=[];checks={}
    with np.load(folder/"events.npz") as scores:
        for g in ordered[:5]:
            edited=min(families[g],key=lambda x:h("physical-continuity-replay-v1."+x["key"]))
            for key in (edited["original"],edited["key"]):
                text=bank.text[key];arrays={p:bank.native[key+"."+p] for p in ("char","rep","links","words","surprisal")}
                output=reader.from_arrays(text,arrays);rows.append(dict(id=key,text=text))
                raw=unpack(scores[key+".mixture"],len(output["sentences"]));expected=[]
                alive=1.
                for p,v in zip(raw,output["observations"],strict=True):
                    expected.append(float(np.max(np.abs(alive*p[1:]-v["source_probabilities"]))))
                    alive*=p[0]
                if max(expected)>2e-6 or abs(1-alive-output["probability_any"])>2e-6:
                    raise ValueError("Export failed replay of the scientifically evaluated readout.")
                checks[key]=output
    if len({r["id"] for r in rows})!=10:raise ValueError("Ten unique original cases were requested.")
    stimuli=TRIP/"serving-stimuli.jsonl"
    stimuli.write_text("".join(json.dumps(r)+"\n" for r in rows))
    (BUNDLE/"expected.json").write_text(json.dumps(checks,indent=2)+"\n")
    print("All ten archived text forecasts reproduced. Inputs",sha(stimuli),"bundle",sha(BUNDLE/"spec.json"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("archive",type=Path)
    run(parser.parse_args().archive)
