"""Package original German completion stems and Spanish reader-rated summary displays, with no outcomes."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import DATA,ROOT,CHECKPOINT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PLAN=Path(__file__).with_name('muse-protocol.md')


def prepare():
    parent=ROOT/'research/results/reading-validation-muse'
    s=read_json(parent/'exposure-seal.json')
    files={'curve.pt':CHECKPOINT,'stimuli.jsonl':DATA/'muse/stimuli.jsonl',
           'german-word-inventory.json':DATA/'muse/dictionary.json','spanish-stimuli.jsonl':DATA/'seahorse-spanish-stimuli.jsonl'}
    required={'stimuli.jsonl':s['materials'],'german-word-inventory.json':s['lexicon'],
              'spanish-stimuli.jsonl':'84b87dfea07991c964b3a72996a5872ae90cec75ca9e19275429ffdad47911b6'}
    if s['protocol']!=digest(PLAN) or any(digest(files[k])!=v for k,v in required.items()):
        raise ValueError('Source exposure or protocol changed')
    serving='research/results/reading-validation-transition-external/essay-cross-topic/serving.json'
    files[serving]=ROOT/serving
    pack_job(DATA/'l4',PLAN,'reading-muse','reading-validation-muse',dependencies('research/reading_validation/muse_bundle_l4.py'),
             files,{},1600,{'purpose':'Original German sentence-completion human replication and independent Spanish Q1/Q2/Q3 transfer'})

if __name__=='__main__':
    p=ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True)
    s.add_parser('prepare');s.add_parser('run').add_argument('job',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.job,protocol=PLAN)
