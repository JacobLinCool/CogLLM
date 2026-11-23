"""Package the author's production word-expectation observer with outcome-free test passages."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest
from research.reading_validation.identity_colab import DATA,ROOT,CHECKPOINT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job
PLAN=Path(__file__).with_name('population-serving-protocol.md')

def prepare():
    paths={'stimuli.jsonl':DATA/'cloze-population/serving-materials.jsonl','curve.pt':CHECKPOINT}
    if digest(paths['stimuli.jsonl'])!='f62e5d5cadf4836757932015bed292025dd5486a49c9d143e0cb7162c40cb8e7':raise ValueError('Original exposure changed')
    for sub,s in [('reading-validation-word-dispersion/forecast-lock.json','b9e49f6789e54994475cd56760144837439876e42e9653a6f5111e9157ffaad1'),('reading-validation-spelling-agreement/fit-lock.json','44d7cbf5b8d4dc6dde725b6519ba30195294300adbb4c9b83e5e77f5f167491a')]:
        p=ROOT/'research/results'/sub
        if digest(p)!=s:raise ValueError('Locked trained population head changed')
        paths[str(p.relative_to(ROOT))]=p
    for f in (ROOT/'research/results/reading-validation-cloze-population/serving').iterdir():paths[str(f.relative_to(ROOT))]=f
    pack_job(DATA/'l4',PLAN,'reading-exact-word','reading-validation-exact-word',dependencies('research/reading_validation/exact_replay_bundle_l4.py'),
             paths,{},1200,{'purpose':'One-pass exact-spelling disagreement observer verification against original held passages'})

if __name__=='__main__':
    p=ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    s.add_parser('prepare');s.add_parser('run').add_argument('job',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.job,protocol=PLAN)
