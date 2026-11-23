"""Package the unmarked PERSUADE and Darmstadt essay bank for the owned L4."""
import argparse
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import CHECKPOINT,DATA
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PROTOCOL=Path(__file__).with_name('argument-protocol.md')


def prepare():
    source=DATA/'argument/stimuli.jsonl'
    seal=read_json(Path('research/results/reading-validation-argument/inputs.json'))
    if seal['protocol']!=digest(PROTOCOL) or seal['stimuli']!=digest(source):raise ValueError('Original essay selection changed')
    pack_job(DATA/'l4',PROTOCOL,'reading-argument','reading-validation-argument',
             dependencies('research/reading_validation/argument_l4.py'),{'curve.pt':CHECKPOINT,'stimuli.jsonl':source},
             {},4200,{'purpose':'Causal identification of which earlier statement an argument supports','curve_sha256':digest(CHECKPOINT)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    q=p.add_subparsers(dest='action',required=True);q.add_parser('prepare');q.add_parser('run').add_argument('job',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.job,protocol=PROTOCOL)
