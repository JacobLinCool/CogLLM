"""Run the native premise-retrieval extractor on an owned L4."""

from argparse import ArgumentParser
from pathlib import Path

from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation.retrieval_forecast import run


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for name in ("model", "stimuli", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("candidate", "protocol-sha256"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    run_phases(args.model, args.candidate, args.output, args.protocol_sha256, [("retrieval", run, args.stimuli)])
