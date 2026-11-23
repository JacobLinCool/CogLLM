"""Independent German word-response and Spanish summary observations taken with the frozen model."""
from argparse import ArgumentParser
from pathlib import Path
from research.reading_validation import muse_l4,summeval_l4
from research.reading_validation.identity_run_l4 import run_phases

if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for k in ('model','stimuli','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('candidate','protocol-sha256'):p.add_argument('--'+k,required=True)
    a=p.parse_args()
    run_phases(a.model,a.candidate,a.output,a.protocol_sha256,
               [('muse',muse_l4.run,a.stimuli),('seahorse-spanish',summeval_l4.run,Path('spanish-stimuli.jsonl'))])
