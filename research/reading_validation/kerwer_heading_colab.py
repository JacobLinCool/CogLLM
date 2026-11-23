"""Run the frozen PLABA sentence head on the original heading manipulation."""
from argparse import ArgumentParser
from pathlib import Path

from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest
from research.reading_validation.identity_colab import CHECKPOINT, DATA, ROOT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PLAN = Path(__file__).with_name("native-heading-protocol.md")
SOURCE = DATA / "native-heading-stimuli.jsonl"


def prepare(stimuli_sha256, protocol_sha256, worker_sha256):
    worker = Path(__file__).with_name("kerwer_heading_l4.py")
    for path, expected in ((SOURCE, stimuli_sha256), (PLAN, protocol_sha256), (worker, worker_sha256)):
        if digest(path) != expected:
            raise ValueError(f"Scientific input changed: {path.name}")
    resources = ROOT / "research/results/reading-validation-local/serving"
    members = {"stimuli.jsonl": SOURCE, "curve.pt": CHECKPOINT}
    for name in ("manifest.json", "weights.npz", "lexicon.json.gz"):
        path = resources / name
        members[str(path.relative_to(ROOT))] = path
    entry = "research/reading_validation/kerwer_replay_bundle_l4.py"
    pack_job(
        DATA / "l4", PLAN, "reading-kerwer-headings", "reading-validation-kerwer-headings",
        dependencies(entry), members, {}, 900,
        {"purpose": "Transport of the deployed human-trained sentence observer across a scientific heading manipulation",
         "source_sha256": stimuli_sha256, "study_worker_sha256": worker_sha256},
    )


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    setup = sub.add_parser("prepare")
    for field in ("stimuli-sha256", "protocol-sha256", "worker-sha256"):
        setup.add_argument("--" + field, required=True)
    sub.add_parser("run").add_argument("job", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.stimuli_sha256, args.protocol_sha256, args.worker_sha256)
    else:
        run(args.job, protocol=PLAN)
