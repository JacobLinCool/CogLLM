"""Stage a dedicated non-Qwen L4 for the official T5-Large full-sentence comparison."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest
from research.reading_validation.identity_colab import DATA
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PROTOCOL=Path(__file__).with_name('transition-t5-protocol.md')


def prepare():
    source=DATA/'snac-t5-stimuli.jsonl';model={'repo':'google-t5/t5-large','revision':'150ebc2c4b72291e770f58e6057481c8d2ed331a'}
    if digest(source)!='6acf4a262716687a591fbd0522b743bd820c4fa99bd7b3ea1c3be00309f759bb' or digest(PROTOCOL)!='7aa3d0c1dc825755862ed91097c13bc7d9d7b069b2ed6a4ab61ffbb3f2a4d51f':
        raise ValueError('T5 comparator input changed')
    pack_job(DATA/'l4',PROTOCOL,'reading-t5','reading-validation-t5',dependencies('research/reading_validation/t5_comparator.py'),
        {'stimuli.jsonl':source},{},8100,{'purpose':'Original published full-sentence specialist architecture trained on books only',
        'external_supervised_training':True,'heldout_movie_outcomes_in_worker':False},candidate='t5',model_spec=model)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run'));p.add_argument('job',nargs='?',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.job,protocol=PROTOCOL)
