"""Stage the ordinary story text only; the human question wordings remain local to training."""

import argparse
from pathlib import Path

from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import read_json,digest
from research.reading_validation.identity_colab import DATA,CHECKPOINT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PLAN=Path(__file__).with_name("narrative-survival-protocol.md")


def prepare():
    source=DATA/"narrative-course/packet.jsonl"
    receipt=read_json(Path("research/results/reading-validation-narrative-course/input-seal.json"))
    if digest(source)!=receipt["packet"]:
        raise ValueError("The four unmarked stories changed after initial verification")
    pack_job(DATA/"l4",PLAN,"reading-plot-survival","reading-validation-plot-survival",
             dependencies("research/reading_validation/narrative_semantic_l4.py"),
             {"stimuli.jsonl":source,"curve.pt":CHECKPOINT},{},1800,
             {"purpose":"Single-pass question-lifetime features from full stories","curve_sha256":digest(CHECKPOINT)})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    actions=p.add_subparsers(dest="action",required=True)
    actions.add_parser("prepare");actions.add_parser("run").add_argument("job",type=Path)
    a=p.parse_args()
    if a.action=="prepare":
        prepare()
    else:
        run(a.job,protocol=PLAN)
