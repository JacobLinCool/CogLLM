"""Run two independently audited reading studies in one Qwen initialization."""

from argparse import ArgumentParser
from pathlib import Path

from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation.narrative_course_l4 import run as complete_stories
from research.reading_validation.writer_term_verify import run as verify_writer


if __name__=="__main__":
    p=ArgumentParser(description=__doc__)
    for n in ("model","stimuli","output"):
        p.add_argument("--"+n,type=Path,required=True)
    for n in ("candidate","protocol-sha256"):
        p.add_argument("--"+n,required=True)
    a=p.parse_args()
    run_phases(a.model,a.candidate,a.output,a.protocol_sha256,
               [("terminology-writer",verify_writer,a.stimuli),("narrative",complete_stories,a.stimuli)])
