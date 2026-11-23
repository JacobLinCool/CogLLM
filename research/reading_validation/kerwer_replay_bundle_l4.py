"""Score each structured/unstructured abstract once with the frozen sentence observer."""
from argparse import ArgumentParser
from pathlib import Path

from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation.kerwer_heading_l4 import run


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    for field in ("model", "stimuli", "output"):
        parser.add_argument("--" + field, required=True, type=Path)
    for field in ("candidate", "protocol-sha256"):
        parser.add_argument("--" + field, required=True)
    args = parser.parse_args()
    run_phases(
        args.model, args.candidate, args.output, args.protocol_sha256,
        [("kerwer-headings", run, args.stimuli)],
    )
