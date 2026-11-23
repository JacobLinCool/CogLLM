"""Package or run the factorial clause-order experiment on the research accelerator."""

import argparse
from pathlib import Path

from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import DATA,CHECKPOINT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PROTOCOL=Path(__file__).with_name("order-protocol.md")


def prepare():
    source=DATA/"discourse-order/stimuli.jsonl"
    seal=read_json(Path("research/results/reading-validation-discourse-order/inputs.json"))
    if seal["registration"]!=digest(PROTOCOL) or seal["text"]!=digest(source):
        raise ValueError("Published clause-order materials changed")
    pack_job(DATA/"l4",PROTOCOL,"reading-discourse-order","reading-validation-discourse-order",
             dependencies("research/reading_validation/expectation_l4.py"),
             {"stimuli.jsonl":source,"curve.pt":CHECKPOINT},{},2400,
             {"curve_sha256":digest(CHECKPOINT),"purpose":"How given information, event order and clause structure affect human reading"})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    q=p.add_subparsers(dest="action",required=True)
    q.add_parser("prepare");q.add_parser("run").add_argument("job",type=Path)
    a=p.parse_args()
    if a.action=="prepare":
        prepare()
    else:
        run(a.job,protocol=PROTOCOL)
