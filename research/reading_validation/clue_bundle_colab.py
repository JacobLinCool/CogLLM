"""Freeze the independent model and input hashes while sharing one accelerator across small studies."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import CHECKPOINT,DATA,ROOT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PROTOCOL=Path(__file__).with_name('spanish-clue-protocol.md')


def prepare():
    seal=read_json(ROOT/'research/results/reading-validation-spanish-clues/input-seal.json');res='research/results/reading-validation-local/serving/weights.npz';spec='research/results/reading-validation-transition-partial/serving/character-word.json';inputs={'curve.pt':CHECKPOINT}
    if digest(PROTOCOL)!=seal['protocol']:raise ValueError('Spanish word-region plan changed')
    entries=[('stimuli.jsonl',DATA/'semantic-clues/stimuli.jsonl',seal['stimuli']),('vocabulary.json',DATA/'semantic-clues/vocabulary.json',seal['vocabulary']),('rewrite-stimuli.jsonl',DATA/'nns-rewrite/stimuli.jsonl','9350ac9cdc944c2a4eac9b2c8a47f3f7abed8bc75d9c6f2678bebcd69b9c0974'),('marginal-stimuli.jsonl',DATA/'narrative-marginal-serving-stimuli.jsonl','b915ebd74440c810817ba7e9de90e247531fae73b04bc3270b08863a9f3e2d25'),('pointer-stimuli.jsonl',DATA/'pointer-stimuli.jsonl','2dee71b79d698976c0ab2bb6fa100f3fe085f45fa47f08e01aac6770711d94b0'),(res,ROOT/res,'2f72b3b28ec5bc4f47bfe084ae76ccf2f9fdc01e4f4d61bfd4a7a38bda99bfa3'),(spec,ROOT/spec,'571895011d9c618e13f2bff0a5f172e33fa5a1e7f825bd79f1ffebb18b99d22c')]
    for remote,path,sha in entries:
        if digest(path)!=sha:raise ValueError('Different certified input '+remote)
        inputs[remote]=path
    papers=[('expectation','20261122T184830Z-7cf90b','expectation','expectation_l4'),('plot-survival','20261122T185452Z-9b8cef','narrative-course','narrative_semantic_l4')]
    protocols={}
    for name,date,local,module in papers:
        package=read_json(DATA/'l4'/('cogllm-reading-'+name+'-'+date)/'plan.json');members=package['payload_members_sha256'];inp=DATA/local/('stimuli.jsonl' if local=='expectation' else 'packet.jsonl')
        for file in dependencies('research/reading_validation/'+module+'.py'):
            if digest(ROOT/file)!=members[file]:raise ValueError(f'{name}: frozen source changed {file}')
        if digest(inp)!=members['stimuli.jsonl'] or digest(CHECKPOINT)!=members['curve.pt']:raise ValueError(f'{name} changed stimuli')
        inputs[('expectation' if name=='expectation' else 'narrative')+'-stimuli.jsonl']=inp;protocols[name]=package['protocol_sha256']
    for module,sha in [('transition_pointer_l4','0afb73b20ba0e850638d4956dd75b5de98effc6743b814c68f86023c669dcafa'),('transition_pointer_features','888e97557b896ebd1ad97461fec2becfa6db940c81b3f630e737474d871d617b'),('nns_rewrite_l4','f829ed2b6d238161aa3a28bccae8925eea21324f8b5d8259577c96d196478743'),('transition_partial_verify','477b717be9a5499d1073ac1c9059f5fe5f0ff5fbff9a75dab807fd4113fff1ff')]:
        if digest(Path(__file__).with_name(module+'.py'))!=sha:raise ValueError('Changed independently signed assay '+module)
    pack_job(DATA/'l4',PROTOCOL,'reading-scientific-clues','reading-validation-scientific-clues',dependencies('research/reading_validation/clue_bundle_l4.py'),inputs,{},5400,{'purpose':'Same-issue Spanish human learning; separate English prediction, joint readability, narrative deployment and selective prior referent controls','original_paper_protocols':protocols,'spanish_additional_baseline':digest(Path(__file__).with_name('spanish-clue-controls.md'))})

if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run'));p.add_argument('job',nargs='?',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare()
    else:run(a.job,protocol=PROTOCOL)
