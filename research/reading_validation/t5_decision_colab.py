"""Replay validated T5 source weights on original ungraded specialist sentences."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest
from research.reading_validation.identity_colab import DATA
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PLAN=Path(__file__).with_name('transition-t5-decision-protocol.md')
SOURCE='cc7c979d6ad7ecb176e6440a80c8c415441766fb91c6503a299ba49c394682e3'
PROTOCOL='a5fe408cd893f997b6de15c45f8fabcf7b411503faa11ef8069fbc358c079d6c'
ADAPTER='a82d523cd771dca02cbaf55cc67d75f6da454be05276da0d2d670d6e70113c7c'


def prepare():
    source=DATA/'snac-t5-calibration-stimuli.jsonl';adapter=DATA/'l4/cogllm-reading-t5-adapter-20261122T204033Z-d0d3a7/extraction/t5-adapter-forecast/selected-adapter.pt'
    if digest(PLAN)!=PROTOCOL or digest(source)!=SOURCE or digest(adapter)!=ADAPTER:raise ValueError('Specialist source or registration changed')
    pack_job(DATA/'l4',PLAN,'reading-t5-decision','reading-validation-t5-decision',dependencies('research/reading_validation/t5_decision_comparator.py'),{'stimuli.jsonl':source,'selected-adapter.pt':adapter},{},3600,{'purpose':'Independent fixed specialist scoring under previously trained category likelihoods','adapter':ADAPTER},candidate='t5',model_spec={'repo':'google-t5/t5-large','revision':'150ebc2c4b72291e770f58e6057481c8d2ed331a'})


if __name__=='__main__':
    p=ArgumentParser();q=p.add_subparsers(dest='action',required=True);q.add_parser('prepare');q.add_parser('run').add_argument('job',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.job,protocol=PLAN)
