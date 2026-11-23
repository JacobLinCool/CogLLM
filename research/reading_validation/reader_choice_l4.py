"""Acquire two separately registered human-reading interventions on one frozen encoder."""
from argparse import ArgumentParser
from pathlib import Path
from research.reading_validation import nominal_external_l4, transition_essay_choice_l4
from research.reading_validation.identity_run_l4 import run_phases


def run(model, candidate, stimuli, output, protocol_sha256):
    phases = [
        ("nominal-reports", nominal_external_l4.run, stimuli),
        ("essay-choices", transition_essay_choice_l4.run, Path("essay-choices.jsonl")),
    ]
    run_phases(model, candidate, output, protocol_sha256, phases)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    for key in ("model", "stimuli", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("candidate", "protocol-sha256"):
        p.add_argument("--" + key, required=True)
    run(**vars(p.parse_args()))
