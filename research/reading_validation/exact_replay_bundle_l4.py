"""Verify the final single-pass source-fitted population spelling observer."""
from argparse import ArgumentParser
from pathlib import Path
from research.reading_validation.exact_word_l4 import run
from research.reading_validation.identity_run_l4 import run_phases

if __name__=='__main__':
    p=ArgumentParser()
    for k in ('model','stimuli','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('candidate','protocol-sha256'):p.add_argument('--'+k,required=True)
    a=p.parse_args();run_phases(a.model,a.candidate,a.output,a.protocol_sha256,[('exact-word',run,a.stimuli)])
