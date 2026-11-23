"""Run two separate, sealed reading-game studies with independent numerical receipts."""
from argparse import ArgumentParser
from pathlib import Path
import traceback
from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.transition_game_l4 import run as game
from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation.emtec_l4 import run as reading
from research.reading_validation.spanish_clue_l4 import run as clues
from research.reading_validation.convergent_word_l4 import run as convergent
import gc
import tempfile
import torch
import json
from research.reading_validation.writer_narrative_verify import run as narrative_writer
from research.reading_validation.writer_term_verify import run as term_writer
from research.reading_validation.transition_pointer_gap_l4 import run as gap

GAME='3b711a247e6b7372ad8772aa2e877efc20f35d5ee38f3cac9186e6cf200b410a'
GAP='5a96815dd0dadf10ef992973c10f63a77b6f259477eb779c3776e8c7b871e1fe'
GAP_PLAN='ad99fff48d468ec2dd0d138fcd949bff7ee67f29992448b30eaa86714225b418'


def run(model,candidate,stimuli,output,protocol_sha256):
    if candidate!='qwen':raise ValueError('Source protocol is specific to the certified Qwen input states')
    output.mkdir();studies={}
    with tempfile.TemporaryDirectory(dir=output) as scratch:
        exp=Path(scratch)/'extraction'
        run_phases(model,candidate,exp,protocol_sha256,[(name,worker,Path('emtec-stimuli.jsonl')) for name,worker in [('narrative-writer',narrative_writer),('term-writer',term_writer),('emtec',reading)]] + [('spanish-expanded',clues,Path('spanish/stimuli.jsonl')),('convergent-words',convergent,Path('convergent.jsonl'))])
        files=json.loads((exp/'causality.json').read_text());studies.update(files['studies'])
        for lab in exp.glob('*-forecast'):
            lab.rename(output/lab.name)
    gc.collect();torch.cuda.empty_cache()
    jobs={'game':(game,(model,stimuli,output/'game-forecast',protocol_sha256,GAME))}
    if Path('gap-stimuli.jsonl').exists():
        jobs['gap-pointer']=(gap,(model,Path('gap-stimuli.jsonl'),Path('source-pointer'),output/'gap-pointer-forecast',GAP_PLAN,GAP))
    for name,(op,args) in jobs.items():
        try:
            op(*args);studies[name]={'completed':True}
        except Exception as e:
            traceback.print_exc();studies[name]={'completed':False,'exception':type(e).__name__,'message':str(e)}
    write_json(output/'causality.json',{'candidate':candidate,'studies':studies})
    write_json(output/'seal.json',{'protocol_sha256':protocol_sha256,'all_completed':all(r['completed'] for r in studies.values()),
        'files_sha256':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for k in ('model','stimuli','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('candidate','protocol-sha256'):p.add_argument('--'+k,required=True)
    run(**vars(p.parse_args()))
