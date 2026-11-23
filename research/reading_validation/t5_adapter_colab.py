"""Stage the bounded memory and exact-recovery specialist baseline on a dedicated T5 L4."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import DATA
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PROTOCOL=Path(__file__).with_name('transition-t5-adapter-protocol.md')


def prepare(resume):
    src=DATA/'snac-t5-stimuli.jsonl';model={'repo':'google-t5/t5-large','revision':'150ebc2c4b72291e770f58e6057481c8d2ed331a'};inp={'stimuli.jsonl':src}
    if digest(src)!='6acf4a262716687a591fbd0522b743bd820c4fa99bd7b3ea1c3be00309f759bb' or digest(PROTOCOL)!='4d212bcfc6bd0039c7d82601df0230deca700a6080e714ca69e221df4fbc0eb3':raise ValueError('T5 adapter source changed')
    meta=None
    if resume:
        meta=read_json(resume/'manifest.json');ckpt=resume/'snapshot.pt'
        if meta['protocol_sha256']!=digest(PROTOCOL) or digest(ckpt)!=meta['sha256'] or meta['bytes']!=ckpt.stat().st_size:raise ValueError('Uncertified T5 restart')
        inp.update({'prior-snapshot.pt':ckpt,'prior-manifest.json':resume/'manifest.json'})
    pack_job(DATA/'l4',PROTOCOL,'reading-t5-adapter','reading-validation-t5-adapter',dependencies('research/reading_validation/t5_adapter_comparator.py'),inp,{},10000,{'purpose':'Supervised full-sentence T5 with rank-eight query and value adaptation and exact restart','initial_snapshot':meta},candidate='t5',model_spec=model)

if __name__=='__main__':
    p=ArgumentParser(description=__doc__);s=p.add_subparsers(dest='action',required=True);s.add_parser('prepare').add_argument('--resume',type=Path);s.add_parser('run').add_argument('job',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.resume)
    else:run(a.job,protocol=PROTOCOL)
