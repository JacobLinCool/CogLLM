"""Prepare or run the sealed counterfactual retrieval job."""

from argparse import ArgumentParser
from pathlib import Path

from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest, read_json
from research.reading_validation.identity_colab import CHECKPOINT, DATA, ROOT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job


PROTOCOL = Path(__file__).with_name("retrieval-protocol.md")


def prepare(head):
    seal = read_json(ROOT / "research/results/reading-validation-retrieval/input-seal.json")
    packet = DATA / "retrieval/packet.jsonl"
    if seal["protocol"] != digest(PROTOCOL) or seal["files"]["packet.jsonl"] != digest(packet):
        raise ValueError("Registered retrieval inputs changed")
    if digest(head) != "463d38528f0d63eb4b7a382eea3022079f701341365f42710adb3f91a765bf1e":
        raise ValueError("Unexpected prior conflict model")
    inputs = {"stimuli.jsonl":packet,"frozen.pt":head,"curve.pt":CHECKPOINT}
    pack_job(DATA/"l4",PROTOCOL,"reading-retrieval","reading-validation-retrieval",
             dependencies("research/reading_validation/retrieval_l4.py"),inputs,{},5400,
             {"purpose":"Single-pass word evidence for earlier-statement retrieval","curve_sha256":digest(CHECKPOINT)})


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action",required=True)
    actions.add_parser("prepare").add_argument("head",type=Path)
    actions.add_parser("run").add_argument("job",type=Path)
    args = parser.parse_args()
    if args.action=="prepare":
        prepare(args.head)
    else:
        run(args.job,protocol=PROTOCOL)
