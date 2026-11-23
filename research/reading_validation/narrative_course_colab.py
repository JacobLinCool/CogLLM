"""Seal or run the complete-story evidence-only job on the research L4."""

import argparse
from pathlib import Path

from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import ROOT,DATA,CHECKPOINT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job


PROTOCOL=Path(__file__).with_name("narrative-course-protocol.md")


def prepare(writer=False):
    seal=read_json(ROOT/"research/results/reading-validation-narrative-course/input-seal.json")
    packet=DATA/"narrative-course/packet.jsonl"
    if digest(packet)!=seal["packet"] or digest(PROTOCOL)!=seal["protocol"]:
        raise ValueError("Narrative text or registered comparison changed")
    inputs={"stimuli.jsonl":packet,"curve.pt":CHECKPOINT}
    suffix="-writer" if writer else ""
    entry="narrative_writer_l4" if writer else "narrative_course_l4"
    if writer:
        for leaf in ("reading-validation-local/serving/manifest.json","reading-validation-local/serving/lexicon.json.gz",
                     "reading-validation-local/serving/weights.npz","reading-validation-transition-external/readouts.json",
                     "reading-validation-definitions/weights.json","reading-validation-definitions/stock-thresholds.json",
                     "reading-validation-definitions/stock-probability-weights.json","reading-validation-definitions/stock-count-scales.json"):
            path=Path("research/results")/leaf
            inputs[str(path)]=ROOT/path
    pack_job(DATA/"l4",PROTOCOL,"reading-narrative"+suffix,"reading-validation-narrative"+suffix,
             dependencies("research/reading_validation/"+entry+".py"),
             inputs,{},1800,
             {"curve_sha256":digest(CHECKPOINT),"purpose":"Four complete original stories, causal source trace"})


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    actions=p.add_subparsers(dest="action",required=True)
    actions.add_parser("prepare").add_argument("--writer",action="store_true")
    actions.add_parser("run").add_argument("job",type=Path)
    args=p.parse_args()
    if args.action=="prepare":
        prepare(args.writer)
    else:
        run(args.job,protocol=PROTOCOL)
