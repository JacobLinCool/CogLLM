"""Package one complete architecture and its exact independent reading exposures."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import CHECKPOINT,DATA,ROOT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PROTOCOL=Path(__file__).with_name('physiology-portability-protocol.md')


def prepare(candidate):
    source=DATA/'physiology-portability/stimuli.jsonl';out=ROOT/'research/results/reading-validation-physiology-portability'
    seal=read_json(out/'input-seal.json');inputs={}
    def stage(name,p,sha):
        if digest(p)!=sha:raise ValueError('Changed registered scientific asset: '+name)
        inputs[name]=p
    if seal['protocol']!=digest(PROTOCOL):raise ValueError('Changed registration')
    stage('stimuli.jsonl',source,seal['stimuli'])
    if candidate=='qwen':
        stage('narrative-parity.jsonl',DATA/'narrative-marginal-serving-stimuli.jsonl','b915ebd74440c810817ba7e9de90e247531fae73b04bc3270b08863a9f3e2d25')
        for package,files in [('reading-validation-transition-partial',('character-word.json','character-operating-points.json')),('reading-validation-transition-snac',('character-word.json',))]:
            for name in files:
                rel='research/results/'+package+'/serving/'+name
                stage(rel,ROOT/rel,digest(ROOT/rel))
        stage('danish.jsonl',DATA/'danish-order/stimuli.jsonl','735bcfe5d5ca768e8da532896609a7debcfca4f66baa81c44d285199f4c0826f')
        stage('italian-source.jsonl',DATA/'italian-order/source-counterfactuals.jsonl','98f581bea470d452921d2570d5991b16173d0f558d5c64e2a0777896cfb7a9e6')
        stage('italian.jsonl',DATA/'italian-order/counterfactuals.jsonl','bfe4127c175304d66787f4206eff4305d20fb9cde314372692dc4456082c164d')
        stage('ambicoref.jsonl',DATA/'ambicoref-reading-stimuli.jsonl','73da8cb603efacafd8b7516a87def1e33219d6398c9ebf2a47566c4b7aec9d58')
        source_model=DATA/'l4/cogllm-reading-scientific-clues-20261122T185857Z-0d9589/extraction/reader-pointer-forecast'
        stage('source-pointer/source-weights.npz',source_model/'source-weights.npz','9162700762a7574f45532bac88f8c018b1f2922118ac033bd1206a1edaecfaf5')
        stage('source-pointer/receipt.json',source_model/'receipt.json',digest(source_model/'receipt.json'))
        semantic=DATA/'l4/cogllm-reading-reader-followup-20261122T195311Z-9081f5/extraction/semantic-pronoun-forecast'
        stage('source-meaning/source-semantic-weights.npz',semantic/'source-semantic-weights.npz',read_json(semantic/'receipt.json')['weights_sha256'])
        stage('source-meaning/receipt.json',semantic/'receipt.json',digest(semantic/'receipt.json'))
        stage('curve.pt',CHECKPOINT,'c1704d4436d767fa43bdb59e9216698514e1f6bd1d45cad4b60d0f018d84abb5')
    pack_job(DATA/'l4',PROTOCOL,'reading-physiology-'+candidate,'reading-validation-physiology',dependencies('research/reading_validation/physiology_portability_l4.py'),inputs,{},5400,{'purpose':'Compare three human readouts at identical printed word ends','common_input':seal['stimuli']},candidate=candidate)


if __name__=='__main__':
    p=ArgumentParser();sub=p.add_subparsers(dest='action',required=True);sub.add_parser('prepare').add_argument('candidate',choices=('qwen','lfm3','gemma4'));sub.add_parser('run').add_argument('job',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.candidate)
    else:run(a.job,protocol=PROTOCOL)
