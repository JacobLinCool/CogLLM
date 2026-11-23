"""Independent public word-difficulty and scientific-symbol assays on one pinned Qwen allocation."""
from argparse import ArgumentParser
from pathlib import Path
from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation.nns_l4 import run as nns
from research.reading_validation.symbol_forecast import run as symbols

if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for k in ('model','output','stimuli'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('candidate','protocol-sha256'):p.add_argument('--'+k,required=True)
    a=p.parse_args();run_phases(a.model,a.candidate,a.output,a.protocol_sha256,[('nns-dense',nns,Path('dense.jsonl')),('nns-sparse',nns,Path('sparse.jsonl')),('symbols',symbols,a.stimuli)])
