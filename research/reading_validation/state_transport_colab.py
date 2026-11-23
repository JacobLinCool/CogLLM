"""Package the human-authored physical-state and explanatory ordering material without outcomes."""
import argparse
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import CHECKPOINT,DATA
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PROTOCOL=Path(__file__).with_name('state-transport-protocol.md')


def prepare():
    source=DATA/'world-states/stimuli.jsonl';seal=read_json(Path('research/results/reading-validation-state-transport/inputs.json'))
    if seal['protocol']!=digest(PROTOCOL) or seal['source']!=digest(source):raise ValueError('Original article collection changed')
    pack_job(DATA/'l4',PROTOCOL,'reading-state-transport','reading-validation-state-transport',
             dependencies('research/reading_validation/state_transport_l4.py'),{'curve.pt':CHECKPOINT,'stimuli.jsonl':source},
             {},5400,{'purpose':'Causal entity-state changes and the reading consequences of information order','curve_sha256':digest(CHECKPOINT)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);q=p.add_subparsers(dest='action',required=True)
    q.add_parser('prepare');q.add_parser('run').add_argument('job',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.job,protocol=PROTOCOL)
