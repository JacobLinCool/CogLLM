"""Prepare one solver-recovery game study and, with sealed source weights, external GAP transfer."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_colab import DATA,CHECKPOINT,ROOT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PROTOCOL=Path(__file__).with_name('transition-game-supervision-protocol.md')


def prepare(pointer):
    tasks=[('stimuli.jsonl',DATA/'game-native-stimuli.jsonl','3b711a247e6b7372ad8772aa2e877efc20f35d5ee38f3cac9186e6cf200b410a')]
    tasks.extend([('emtec-stimuli.jsonl',DATA/'emtec/stimuli.jsonl','ce982ffd7f176652f5c201d7387974e97b475ce665fbd24eff2c094d55cc2a04'),('curve.pt',CHECKPOINT,'c1704d4436d767fa43bdb59e9216698514e1f6bd1d45cad4b60d0f018d84abb5')])
    for name in ['manifest.json','weights.npz','lexicon.json.gz']:
        relative='research/results/reading-validation-local/serving/'+name;tasks.append((relative,ROOT/relative,digest(ROOT/relative)))
    if digest(Path(__file__).with_name('emtec_l4.py'))!='a9429a160b7ff6d41140880dab4d966610bfc94043233ae6f4b4306b5e7b7fb6':raise ValueError('External reading assay changed')
    for leaf in ('reading-validation-transition-external/readouts.json','reading-validation-transition-partial/serving/character-word.json','reading-validation-transition-snac/serving/character-word.json','reading-validation-definitions/weights.json','reading-validation-definitions/stock-thresholds.json','reading-validation-definitions/stock-probability-weights.json','reading-validation-definitions/stock-count-scales.json'):
        rel='research/results/'+leaf;tasks.append((rel,ROOT/rel,digest(ROOT/rel)))
    cl=read_json(ROOT/'research/results/reading-validation-spanish-clues/card-expansion-lock.json')
    for n in ('stimuli','vocabulary'):
        filename='stimuli.jsonl' if n=='stimuli' else 'vocabulary.json';tasks.append(('spanish/'+filename,DATA/'semantic-clues/expanded'/filename,cl[n]))
    source=read_json(ROOT/'research/results/reading-validation-convergent/input-seal.json');tasks.append(('convergent.jsonl',DATA/'convergent/stimuli.jsonl',source['stimuli']))
    if pointer is not None:
        rec=read_json(pointer/'receipt.json')
        tasks.extend([('gap-stimuli.jsonl',DATA/'gap-pointer-stimuli.jsonl','5a96815dd0dadf10ef992973c10f63a77b6f259477eb779c3776e8c7b871e1fe'),
            ('source-pointer/receipt.json',pointer/'receipt.json',digest(pointer/'receipt.json')),
            ('source-pointer/source-weights.npz',pointer/'source-weights.npz',rec['weights_sha256'])])
    if digest(PROTOCOL)!='373bb5cf8cd007f08613d1b7846bd3e9fe47bde2977a815d366bf3086bcb89b6':raise ValueError('Scientific likelihood protocol changed')
    for name,ck in [('transition_game_l4','d140c2e2eee3b6674bff1c913cafdf9b9067de7e7829bc4aecaa8e5c2ec33b8e'),('transition_game_optimizer','22f4162ce5de04df690c262bb877083ecce36ad6f1b6f3f0fbdeb58b20251c4a'),('transition_pointer_gap_l4','4653a6a6267f127f37df94a2f77a6a044c9fb8c897f1350e151c780f60a9e24d')]:
        if digest(Path(__file__).with_name(name+'.py'))!=ck:raise ValueError('Signed worker changed')
    if any(digest(p)!=s for _,p,s in tasks):raise ValueError('Stimulus or source receipt mismatch')
    pack_job(DATA/'l4',PROTOCOL,'reading-game-recovery','reading-validation-game-recovery',dependencies('research/reading_validation/game_recovery_l4.py'),{n:p for n,p,_ in tasks},{},5000,{'purpose':'Preserve fixed multitask likelihood and optionally evaluate untouched external articles using pre-trained click probabilities'})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('action',choices=('prepare','run'));p.add_argument('job',nargs='?',type=Path);p.add_argument('--pointer',type=Path);a=p.parse_args()
    if a.action=='prepare':prepare(a.pointer)
    else:run(a.job,protocol=PROTOCOL)
