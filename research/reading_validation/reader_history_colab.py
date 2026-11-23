"""Seal original English completion contexts and investigator-authored matched passages."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import read_json,digest
from research.reading_validation.identity_colab import DATA,ROOT,CHECKPOINT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PLAN=Path(__file__).with_name('reader-history-protocol.md')


def prepare(objects):
    parent=ROOT/'research/results/reading-validation-reader-history'
    source=DATA/'reader-history/verified-trial-stimuli.jsonl'
    if digest(source)!=read_json(parent/'verified-input-seal.json')['packet'] or digest(PLAN)!=read_json(parent/'registration.json')['protocol']:
        raise ValueError('Different original reader-history materials')
    files={'curve.pt':CHECKPOINT,'stimuli.jsonl':source}
    banks=['character-names','character-objects'] if objects else ['character-names']
    dest=ROOT/'research/results/reading-validation-character-access'
    for name in banks:
        seal=dest/({'character-names':'name-input.json','character-objects':'object-input.json'}[name])
        f=DATA/(name+'-presentations.jsonl')
        if digest(f)!=read_json(seal)['input']:raise ValueError("Narrative experiment changed")
        files[name+'.jsonl']=f
    served=ROOT/'research/results/reading-validation-cloze-population/serving'
    for f in served.iterdir():files[str(f.relative_to(ROOT))]=f
    pack_job(DATA/'l4',PLAN,'reading-reader-history','reading-validation-reader-history',
        dependencies('research/reading_validation/reader_history_bundle_l4.py'),files,{'study-list.json':banks},1800,
        {'purpose':'Population and individual prediction of genuine future choices after earlier reader history'})


if __name__=='__main__':
    p=ArgumentParser();s=p.add_subparsers(dest='action',required=True);s.add_parser('prepare').add_argument('--include-objects',action='store_true')
    s.add_parser('run').add_argument('job',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.include_objects)
    else:run(a.job,protocol=PLAN)
