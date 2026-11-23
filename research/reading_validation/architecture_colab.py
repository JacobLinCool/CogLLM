"""Stage one named backbone at a time with independently locked public data and weights."""
from argparse import ArgumentParser
from pathlib import Path

from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import CHECKPOINT,DATA,ROOT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PROTOCOL=Path(__file__).with_name('transition-architecture-protocol.md')
SOURCE='3643a827d22636eba092274865003ee4739d77b86baacc5681142362f5f895ac'
SPAN={'qwen':'qwen-20261122T073707Z-c53e7d','lfm3':'lfm3-20261122T073707Z-d3fdb6','gemma4':'gemma4-20261122T074159Z-e865f1'}


def prepare(candidate):
    src=DATA/'cross-snac-stimuli.jsonl'
    base=ROOT/'.cache/cogllm/token-readout/transfer-l4'/('cogllm-portable-'+SPAN[candidate])
    if digest(src)!=SOURCE or digest(PROTOCOL)!='c53f7d74223456454b0352089dea54621049db7c4d53e88fa5193fb8794d044e':
        raise ValueError('Registered common architecture sources changed')
    inputs={'stimuli.jsonl':src,'span.pt':base/'extraction/span.pt'}
    if candidate=='qwen':
        d=DATA/'definitions';lock=read_json(ROOT/'research/results/reading-validation-definitions/stock-lock.json')
        if digest(d/'stock-native-stimuli.jsonl')!=lock['native_stimuli']:raise ValueError('Sealed textbook source mismatch')
        inputs.update({'curve.pt':CHECKPOINT,'stock-native-stimuli.jsonl':d/'stock-native-stimuli.jsonl','stock-serving.json':d/'stock-serving.json'})
        w00=read_json(ROOT/'research/results/reading-validation-definitions/stock-external-lock.json')
        if w00['stimuli']!=digest(d/'w00-stimuli.jsonl'):raise ValueError('W00 input no longer sealed')
        inputs['w00-stimuli.jsonl']=d/'w00-stimuli.jsonl'
        pkg=ROOT/'research/results/reading-validation-local/help-choice/l4-worker-package.json'
        if digest(pkg)!='5dc297ff94a31511059dc57b3d7138518cf0235aa209f46fec450a449bcf2453':raise ValueError('Help export changed')
        contract=read_json(pkg)
        for f,sha in contract['resource_sha256'].items():
            if digest(ROOT/f)!=sha:raise ValueError(f'Frozen clinical model input changed: {f}')
            inputs[f]=ROOT/f
        controls=DATA/'guo-help-controls.jsonl'
        if digest(controls)!=contract['controls_sha256']:raise ValueError('Clinical control articles changed')
        inputs['guo-help-controls.jsonl']=controls;inputs['relay/local-stimuli.jsonl']=DATA/'relay/local-stimuli.jsonl'
        noun=read_json(ROOT/'research/results/reading-validation-noun/input-seal.json')
        for field,f in [('packet','local-stimuli.jsonl'),('lexical','lexical.npz')]:
            ref=DATA/'noun'/f
            if digest(ref)!=noun[field]:raise ValueError('Changed noun input')
            inputs['noun/'+f]=ref
        exp=read_json(ROOT/'research/results/reading-validation-exposure/input-seal.json')
        for field,leaf in [('stimuli','stimuli.jsonl'),('vocabulary','vocabulary.json')]:
            src=DATA/'exposure'/leaf
            if digest(src)!=exp[field]:raise ValueError('Practice study source mismatch')
            inputs['exposure/'+leaf]=src
        for name in ('weights.json','joint-weights.json','stock-thresholds.json'):
            f='research/results/reading-validation-definitions/'+name;inputs[f]=ROOT/f
    pack_job(DATA/'l4',PROTOCOL,'reading-architecture-'+candidate,'reading-validation-architecture',
             dependencies('research/reading_validation/architecture_l4.py'),inputs,{},5400,
             {'purpose':'Three-backbone replication at common narrative word coordinates','span_sha256':digest(inputs['span.pt'])},candidate=candidate)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='action',required=True)
    sp.add_parser('prepare').add_argument('candidate',choices=tuple(SPAN));sp.add_parser('run').add_argument('job',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.candidate)
    else:run(a.job,protocol=PROTOCOL)
