"""Run unmarked argumentative paragraphs and the normal manual reference application."""
import argparse
from pathlib import Path
from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation import state_transport_l4,writer_reference_l4

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('model','stimuli','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('candidate','protocol-sha256'):p.add_argument('--'+k,required=True)
    a=p.parse_args()
    run_phases(a.model,a.candidate,a.output,a.protocol_sha256,[('manual-writing',writer_reference_l4.run,Path('manual.json')),('reasoning',state_transport_l4.run,a.stimuli)])
