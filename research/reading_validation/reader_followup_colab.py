"""Verify individual public reading-study registrations and stage a shared Qwen allocation."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import DATA,ROOT,CHECKPOINT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PROTOCOL=Path(__file__).with_name('reader-followup-protocol.md')


def prepare():
    results=ROOT/'research/results';members={};studies={};extra=[]
    def mount(path,name,expected):
        if digest(path)!=expected:raise ValueError(f'Signed packet mismatch: {name}')
        if name in members and members[name]!=path:raise ValueError(f'Conflicting input mappings: {name}')
        members[name]=path
    def study(k,path,expected,protocol=None):
        name=k+'.jsonl';mount(path,name,expected);studies[k]=dict(source=name,sha256=expected,protocol_sha256=protocol)
    def prepared(k,job,source,implementation):
        plan=read_json(DATA/'l4'/job/'plan.json');tokens=plan['payload_members_sha256'];key='research/reading_validation/'+implementation+'.py'
        if digest(ROOT/key)!=tokens[key]:raise ValueError('An official extraction worker has changed')
        study(k,source,tokens['stimuli.jsonl'],plan['protocol_sha256'])
    prepared('argument','cogllm-reading-argument-20261122T194354Z-3853a0',DATA/'argument/stimuli.jsonl','argument_l4')
    prepared('discourse-order','cogllm-reading-discourse-order-20261122T192405Z-74d8bf',DATA/'discourse-order/stimuli.jsonl','expectation_l4')
    seal=read_json(results/'reading-validation-convergent/erp-exposure-seal.json');study('convergent-original',DATA/'convergent/erp-stimuli.jsonl',seal['stimuli'],seal['registration'])
    study('semlex',DATA/'semeval-lex/stimuli.jsonl','64de65fa875e4c33661a9c6a032fb9c9fbf3a249125ffc5d8c626773dbeb090a');extra+=dependencies('research/reading_validation/nns_rewrite_l4.py')
    learner=read_json(results/'reading-validation-nns/writer/input-packet.json');study('learner-writer',ROOT/learner['source'],learner['source_sha256'])
    for rel,h in learner['extra_files'].items():mount(ROOT/rel,rel,h)
    for rel,alias in learner['stage_to_literal_root_name'].items():mount(ROOT/rel,alias,learner['offline_wheels_sha256'])
    study('semantic-pronoun',DATA/'dpr-pointer-stimuli.jsonl','8efcfbb0884c36adf240503b9bd05fa27d1b236dcaaa67791d183f788a095ccc','ef8809e8c5391c0feefce61d8cd2f5a2cace788cb6fbb040a7eb56fb4469e678')
    study('gate-pointer',DATA/'gate-pointer-stimuli.jsonl','c034d8a68a8414be1bd13e25123bcdc0614a8d9cc795db631e6dab9c2b1f2538','bbe820cd0d143c54f4e95f4ff6709472abf345e7fd5f9b5460f86e7b260dbc2d')
    study('jones',DATA/'jones-pointer-stimuli.jsonl','d0439250a93b77e4d83da445ccfd2313e3f0c9038952b1e1219e441aec8999aa','c3c4c200c9ae07342ba86c3403f3c6f6f9b6be18d6ef90e02fcbbc1a80f7f6b8')
    study('pointer-writer',DATA/'pointer-serving-replay.jsonl','2d7c1b9c759524cb605008338b77f769c05f9701396a871de5ae42d1f253ed61');ref='e2d10e82405c6625b21711ae5a5183ae27c18956eb9d72d1cb6589b2759690ca';mount(DATA/'pointer-serving-reference.npz','pointer-serving-reference.npz',ref);studies['pointer-writer']['reference_sha256']=ref
    pointer=DATA/'l4/cogllm-reading-scientific-clues-20261122T185857Z-0d9589/extraction/reader-pointer-forecast';weights=read_json(pointer/'receipt.json')['weights_sha256'];mount(pointer/'source-weights.npz','source-pointer/source-weights.npz',weights);mount(pointer/'receipt.json','source-pointer/receipt.json',digest(pointer/'receipt.json'))
    for k in ('averaged_perceptron_tagger_eng.zip','word-pointer.json'):
        leaf='research/results/reading-validation-transition-pointer/serving/'+k;mount(ROOT/leaf,leaf,digest(ROOT/leaf))
    mount(CHECKPOINT,'curve.pt','c1704d4436d767fa43bdb59e9216698514e1f6bd1d45cad4b60d0f018d84abb5');members['stimuli.jsonl']=DATA/'convergent/erp-stimuli.jsonl'
    files=sorted(set(dependencies('research/reading_validation/reader_followup_l4.py')+extra))
    pack_job(DATA/'l4',PROTOCOL,'reading-reader-followup','reading-validation-reader-followup',files,members,{'studies.json':studies},5000,{'purpose':'Nine predeclared endpoint-specific single-pass human reading and serving analyses','source_supervision_on_device':['source-only historical pronoun actions for the reserved gate fit','declared training split of the paired semantic pronoun experiment']})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run'));p.add_argument('job',nargs='?',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.job,protocol=PROTOCOL)
