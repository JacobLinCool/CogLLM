"""Build a production artifact only when the source operating point transfers to all actions."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from research.reading_validation.antecedent_inference import BUNDLE,BUNDLE_SHA,REVISION
from research.reading_validation.transition_gate_features import NAMES
from research.reading_validation.transition_gate_prepare import DEST
from research.reading_validation.transition_score import dump,sha


def export(folder, directory):
    scores=json.loads((DEST/"scores.json").read_text());policy=json.loads((DEST/"selection.json").read_text())
    receipt=json.loads((folder/"receipt.json").read_text())
    if sha(DEST/"selection.json")!=scores["source_selection_sha256"] or sha(folder/"gate-weights.npz")!=receipt["weights_sha256"]:
        raise ValueError("Checked source analysis or gate weights differ")
    forms=[name for name in ("native","surface") if policy["methods"][name+"_product"]["selected"]]
    criterion=Path("research/reading_validation/transition-pointer-serving-criteria.md")
    if not forms:
        dump(DEST/"serving-decision.json",{"qualified":False,"reason":"No independently source-selected rule","criteria_sha256":sha(criterion)})
        return
    method=forms[0];record=scores["selection"][method+"_product"]["source_selected"]["articles"]
    precision=record["precision"]
    if precision is None or precision["whole_group_95_interval"][0]<.70:
        dump(DEST/"serving-decision.json",{"qualified":False,"reason":"The designated source rule failed the all-action external precision criterion",
             "method":method,"source_threshold":record["threshold"],"heldout_precision":precision,"criteria_sha256":sha(criterion)})
        return
    with np.load(folder/"gate-weights.npz") as z:
        vals={k:z[method+"__"+k].tolist() for k in ("mean","sd","beta","bias")}
        d=(len(z["native__beta"])-len(NAMES))//2
    packet={"schema":"cogllm-english-applicable-pronoun-v1","revision":REVISION,"pointer_bundle_sha256":BUNDLE_SHA,
         "method":method,"hidden_width":d,"threshold":record["threshold"],"external_release_criterion_passed":True,
         "criteria_sha256":sha(criterion),"selection_sha256":sha(DEST/"selection.json"),"external_scores_sha256":sha(DEST/"scores.json"),
         "source_weights_sha256":receipt["weights_sha256"],"surface_columns":list(NAMES),"weights":vals}
    directory.mkdir(parents=True,exist_ok=True);p=directory/"applicable-pronoun.json";dump(p,packet)
    dump(DEST/"serving-decision.json",{"qualified":True,"method":method,"threshold":record["threshold"],
         "expected_precision":precision,"coverage":record["coverage"],"bundle_sha256":sha(p),"criteria_sha256":sha(criterion)})


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--features",type=Path,required=True);p.add_argument("--output",type=Path,default=BUNDLE)
    a=p.parse_args();export(a.features,a.output)
