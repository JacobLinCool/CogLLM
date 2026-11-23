"""Stage held-paper, label-free scientific-language and grade-free prospective lexical data."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest, read_json
from research.reading_validation.identity_colab import CHECKPOINT, DATA, ROOT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PROTOCOL=Path(__file__).with_name('symbol-protocol.md')


def prepare(resume=False):
    seal=read_json(ROOT/'research/results/reading-validation-symbols/input-seal.json');resource='research/results/reading-validation-local/serving/weights.npz'
    if seal['protocol']!=digest(PROTOCOL):raise ValueError('Changed symbol paper-holdout protocol')
    extra=read_json(ROOT/'research/results/reading-validation-symbols/equation-key-addition.json')
    if extra['previous_packet']!=seal['stimuli'] or read_json(ROOT/'research/results/reading-validation-symbols/prefix-repair-coverage.json')['stimuli']!=extra['stimuli']:raise ValueError('Visible notation dictionary differs from public seal')
    inputs={'curve.pt':CHECKPOINT}
    for name,path,sha in [('stimuli.jsonl',DATA/'symbols/stimuli-native.jsonl',extra['stimuli']),('dense.jsonl',DATA/'nns/dense.jsonl','b7b72e89e4c0249e44e0202e92166657edbaffbd25e646c0d27126e1aa443a87'),('sparse.jsonl',DATA/'nns/sparse.jsonl','fc6affa5c216f1ac8f3fc949699a5bc2dea5fcfd8a46aad054b76e0d738b7d63'),(resource,ROOT/resource,'2f72b3b28ec5bc4f47bfe084ae76ccf2f9fdc01e4f4d61bfd4a7a38bda99bfa3')]:
        if digest(path)!=sha:raise ValueError(f'Changed public stimulus or model {name}')
        inputs[name]=path
    if resume:
        from research.reading_validation.t5_checkpoint_transport import replace_json
        store=DATA/'l4/symbol-progress';members={}
        for file in sorted(store.glob('shard-*.json')):
            data=read_json(file);source=store/data['name']
            if digest(source)!=data['sha256'] or data['stimuli']!=extra['stimuli']:raise ValueError('Unverified checkpoint on previous allocation')
            members[data['name']]=data
        if not members:raise ValueError('No saved scientific features to continue')
        target=DATA/'symbols/inherited.json';replace_json(target,members);inputs['symbol-inherited.json']=target
    pack_job(DATA/'l4',PROTOCOL,'reading-symbols','reading-validation-symbols',dependencies('research/reading_validation/symbol_l4.py'),inputs,{},5400,{'purpose':'Independent held scientific papers with two outstanding symbols and lay reader word assessments','published_annotations_in_worker':False})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run'));p.add_argument('job',nargs='?',type=Path);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='prepare':prepare(a.resume)
    else:run(a.job,protocol=PROTOCOL)
