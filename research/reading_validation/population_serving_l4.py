"""Verify actual writer inference and a separately registered narrative reading intervention."""
from argparse import ArgumentParser
from pathlib import Path
from research.reading_validation import population_verify_l4, writer_expectation_l4, transition_goal_access_l4
from research.reading_validation.population_inference import PopulationWordInference
from research.reading_validation.identity_run_l4 import run_phases


def run(model,candidate,stimuli,output,protocol_sha256):
    jobs=[
        ("population-writer",writer_expectation_l4.run,stimuli),
        ("population-parity",population_verify_l4.run,stimuli),
        ("goal-access",transition_goal_access_l4.run,Path("goal-access.jsonl")),
    ]
    run_phases(model,candidate,output,protocol_sha256,jobs)


if __name__=="__main__":
    p=ArgumentParser()
    for n in ("model","stimuli","output"):p.add_argument("--"+n,type=Path,required=True)
    for n in ("candidate","protocol-sha256"):p.add_argument("--"+n,required=True)
    run(**vars(p.parse_args()))
