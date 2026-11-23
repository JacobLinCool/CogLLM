"""Seal the independently registered, grade-free recall and editing packets."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest
from research.reading_validation.identity_colab import DATA, CHECKPOINT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PLAN = Path(__file__).with_name("reader_choice_runtime.md")


def prepare():
    paths = {
        "stimuli.jsonl": (DATA/"nominal-report-texts.jsonl","e7a7e96258e0a71c23c66a488be521c895fe60c2f0f920167a0ef04e0309adcf"),
        "essay-choices.jsonl": (DATA/"essay-expert-choices-stimuli.jsonl","610bfffe4a39c6c7f4db6bafc7d95f0746e672cde16d1da656420725bc6f9dff"),
        "curve.pt": (CHECKPOINT,"c1704d4436d767fa43bdb59e9216698514e1f6bd1d45cad4b60d0f018d84abb5"),
    }
    for key,(source,expected) in paths.items():
        if digest(source)!=expected:
            raise ValueError("A previously registered source changed: "+key)
    code=dependencies("research/reading_validation/reader_choice_l4.py")
    pack_job(DATA/"l4", PLAN, "reading-expert-choices", "reading-validation-expert-choices", code,
             {key:source for key,(source,_) in paths.items()}, {}, 1700, 
             {"purpose":"Text-only reader recall and independent author choice"})


if __name__ == "__main__":
    p=ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="action",required=True)
    sub.add_parser("prepare")
    sub.add_parser("run").add_argument("job",type=Path)
    a=p.parse_args()
    if a.action=="prepare":
        prepare()
    else:
        run(a.job,protocol=PLAN)
