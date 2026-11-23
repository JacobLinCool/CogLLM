"""Stage independently registered native reading experiments on one Qwen reservation."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.extract import digest,read_json
from research.backbone_transfer.colab import run
from research.reading_validation.identity_colab import ROOT,DATA,CHECKPOINT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PROTOCOL=Path(__file__).with_name('state-transport-protocol.md')


def prepare():
    l4=DATA/'l4'
    primary=l4/'cogllm-reading-state-transport-20261122T195907Z-fc66be'
    reason=l4/'cogllm-reading-reason-quality-20261122T202059Z-f0ee25'
    older=l4/'cogllm-reading-reader-followup-20261122T195311Z-9081f5'
    past=read_json(older/'plan.json')['payload_members_sha256']
    first=read_json(primary/'plan.json'); second=read_json(reason/'plan.json'); inputs={}
    def add(alias,source,checksum):
        if digest(source)!=checksum:raise ValueError(f'Changed signed input {alias}')
        inputs[alias]=source
    def check_implementation(filename,checksum):
        if digest(Path(__file__).with_name(filename))!=checksum:raise ValueError(f'Signed worker changed: {filename}')
    add('stimuli.jsonl',DATA/'world-states/stimuli.jsonl',first['payload_members_sha256']['stimuli.jsonl'])
    if first['protocol_sha256']!=digest(PROTOCOL):raise ValueError('Changed physical-state protocol')
    add('reasoning.jsonl',DATA/'reason-quality/stimuli.jsonl',second['payload_members_sha256']['stimuli.jsonl'])
    add('manual.json',reason/'manual.json',second['payload_members_sha256']['manual.json'])
    for alias,checksum in second['payload_members_sha256'].items():
        if alias.startswith('research/results/'):add(alias,ROOT/alias,checksum)
    for filename in ('state_transport_l4.py','writer_reference_l4.py'):
        key='research/reading_validation/'+filename
        check_implementation(filename,second['payload_members_sha256'][key])
        if key in first['payload_members_sha256']:check_implementation(filename,first['payload_members_sha256'][key])
    add('mandarin.jsonl',DATA/'mandarin-reading-stimuli.jsonl','b41e5c9cf82fbc3d513ff9ea2fef642c1268312b592c2c2800059d84fe1a9a3c')
    check_implementation('transition_mandarin_l4.py','cd9ffaf524dfcbf118394a1119b037a493ba5f31a72b452f2babb7883e01a776')
    add('pointer-writer.jsonl',DATA/'pointer-serving-replay.jsonl',past['pointer-writer.jsonl'])
    add('pointer-serving-reference.npz',DATA/'pointer-serving-reference.npz',past['pointer-serving-reference.npz'])
    check_implementation('transition_pointer_serving_verify.py','7ad14635ddb38362c88b5e9b3ded0abf1f3c36562beb4ff4af7c70f93f338fe2')
    add('null-choice.jsonl',DATA/'null-choice-pointer-stimuli.jsonl','d57549e3452409c31e4368d5f10cbeed06a40d08b468ade0675feae272878f20')
    check_implementation('transition_null_l4.py','eecc32b53525883d8b9f7d454984a2a558cf8ee739fa94b74ea9d8a8a52acceb')
    source=l4/'cogllm-reading-scientific-clues-20261122T185857Z-0d9589/extraction/reader-pointer-forecast'
    gated=older/'extraction/gate-pointer-forecast'
    for key,orig in [('source-pointer',source),('source-gate',gated)]:
        add(key+'/receipt.json',orig/'receipt.json',digest(orig/'receipt.json'))
        weights='source-weights.npz' if key=='source-pointer' else 'gate-weights.npz'
        add(key+'/'+weights,orig/weights,read_json(orig/'receipt.json')['weights_sha256'])
    info=read_json(ROOT/'research/results/reading-validation-multilab/input-seal.json')
    add('multilab.jsonl',DATA/'multilab/stimuli.jsonl',info['stimuli'])
    folder='research/results/reading-validation-convergent/serving';spec=read_json(ROOT/folder/'spec.json')
    add(folder+'/weights.npz',ROOT/folder/'weights.npz',spec['weights_sha256'])
    add(folder+'/spec.json',ROOT/folder/'spec.json',digest(ROOT/folder/'spec.json'))
    add('word-stimuli.jsonl',DATA/'convergent/word-serving-stimuli.jsonl',spec['verification_packet'])
    add('curve.pt',CHECKPOINT,'c1704d4436d767fa43bdb59e9216698514e1f6bd1d45cad4b60d0f018d84abb5')
    files=dependencies('research/reading_validation/transport_mandarin_l4.py')
    pack_job(l4,PROTOCOL,'reading-transport-mandarin','reading-validation-transport-mandarin',files,inputs,{},6000,dict(purpose='Article-only state predictions, multilingual human exposures, sufficiency and reference assessment',registered_primary=digest(primary/'plan.json'),registered_reasoning=digest(reason/'plan.json')))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('job',nargs='?',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.job,protocol=PROTOCOL)
