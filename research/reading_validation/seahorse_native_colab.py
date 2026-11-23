"""Seal the original summary-only Q1 material from six official English shards."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import ROOT,DATA,CHECKPOINT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PLAN=Path(__file__).with_name('transition-seahorse-protocol.md')

def prepare():
    meta=read_json(ROOT/'research/results/reading-validation-transition-external/seahorse/exposure-seal.json')
    if meta['protocol']!=digest(PLAN) or meta['total_distinct']!=21247:raise ValueError('Different external study materials')
    files={'curve.pt':CHECKPOINT}
    for k,v in meta['packets'].items():
        s=ROOT/v['file']
        if digest(s)!=v['sha256']:raise ValueError('Unsealed source '+k)
        files['stimuli.jsonl' if k=='train0' else 'seahorse-'+k+'.jsonl']=s
    n='research/results/reading-validation-transition-external/essay-cross-topic/serving.json'
    if digest(ROOT/n)!='77a0634d12f400d1987d425a56f89c27e92d48025210f38c8e9cf7d0034cff13':raise ValueError('Changed observer')
    files[n]=ROOT/n
    pack_job(DATA/'l4',PLAN,'reading-summary-choices','reading-validation-summary-choices',
        dependencies('research/reading_validation/seahorse_native_l4.py'),files,{},3300,
        {'purpose':'Independent summary understandability, one frozen forward per original passage'})

if __name__=='__main__':
    p=ArgumentParser();s=p.add_subparsers(dest='action',required=True);s.add_parser('prepare');s.add_parser('run').add_argument('job',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.job,protocol=PLAN)
