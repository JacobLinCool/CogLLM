"""Capture the original public Q1 materials in six independently recoverable shards."""
from argparse import ArgumentParser
from pathlib import Path
from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation.summeval_l4 import run as extract

PARTS=('train0','validation','test','train1','train2','train3')


def run(model,candidate,stimuli,output,protocol_sha256):
    jobs=[('seahorse-'+k,extract,stimuli if i==0 else Path('seahorse-'+k+'.jsonl')) for i,k in enumerate(PARTS)]
    run_phases(model,candidate,output,protocol_sha256,jobs)


if __name__=='__main__':
    p=ArgumentParser()
    for n in ('model','stimuli','output'):p.add_argument('--'+n,required=True,type=Path)
    for n in ('candidate','protocol-sha256'):p.add_argument('--'+n,required=True)
    run(**vars(p.parse_args()))
