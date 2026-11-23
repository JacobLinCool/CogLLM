"""Rebuild the English conditional word-link bundle from its public frozen source fit."""

from pathlib import Path
import json
from hashlib import sha256

import numpy as np

from research.reading_validation.antecedent_inference import BUNDLE,BUNDLE_SHA,POS_SHA,REVISION
from research.reading_validation.transition_pointer_data import OUT
from research.reading_validation.transition_score import sha


def export(destination=BUNDLE):
    source=OUT/"source-weights.npz"; selection=OUT/"selective/source-selection.json"
    r=json.loads((OUT/"receipt.json").read_text())
    if sha(source)!=r["weights_sha256"] or r["revision"]!=REVISION:raise ValueError("The source-trained pointer changed")
    policy=json.loads(selection.read_text())["methods"]["headwise"]["selected"]
    with np.load(source) as weights:
        vector={k:weights["headwise__"+k].tolist() for k in ("mean","sd","beta")}
    packet={"schema":"cogllm-english-pronoun-link-v1","revision":REVISION,"native_attention_heads":48,
            "training_weights_sha256":sha(source),"selection_sha256":sha(selection),"threshold":policy["threshold"],
            "pos_sha256":POS_SHA,"weights":vector}
    encoded=(json.dumps(packet,indent=2,allow_nan=False)+"\n").encode()
    if sha256(encoded).hexdigest()!=BUNDLE_SHA:raise ValueError("Rebuilt public source bundle differs from the validated artifact")
    Path(destination).mkdir(parents=True,exist_ok=True)
    (Path(destination)/"word-pointer.json").write_bytes(encoded)


if __name__=="__main__":
    export()
