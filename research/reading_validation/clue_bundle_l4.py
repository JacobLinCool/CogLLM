"""Run six separately registered language experiments in one bounded Colab allocation."""
from argparse import ArgumentParser
import gc
import json
from pathlib import Path
import torch
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation.spanish_clue_l4 import run as clues
from research.reading_validation.expectation_l4 import run as english
from research.reading_validation.narrative_semantic_l4 import run as narrative
from research.reading_validation.transition_partial_verify import run as serving
from research.reading_validation.nns_rewrite_l4 import run as rewrite
from research.reading_validation.transition_pointer_l4 import run as pointer
from research.reading_validation.t5_checkpoint_transport import replace_json


def run(model,candidate,stimuli,output,protocol_sha256):
    jobs=[('marginal-narrative',serving,Path('marginal-stimuli.jsonl')),('spanish-clues',clues,stimuli),('nns-rewrite',rewrite,Path('rewrite-stimuli.jsonl')),('expectation',english,Path('expectation-stimuli.jsonl')),('plot-survival',narrative,Path('narrative-stimuli.jsonl'))]
    run_phases(model,candidate,output,protocol_sha256,jobs);gc.collect();torch.cuda.empty_cache()
    p=Path('pointer-stimuli.jsonl');print('Begin independent click attribution',flush=True)
    try:
        pointer(model,p,output/'reader-pointer-forecast','2e6d39f45053dfcf336061d3f07bbebe2bba4f59fbef42d61dfee9c7af9f88af','2dee71b79d698976c0ab2bb6fa100f3fe085f45fa47f08e01aac6770711d94b0')
        phase={'completed':True,'input_sha256':digest(p)}
    except Exception as error:
        import traceback
        traceback.print_exc();phase={'completed':False,'exception':type(error).__name__,'message':str(error)}
    details=read_json(output/'causality.json');details['studies']['reader-pointer']=phase;replace_json(output/'causality.json',details)
    old=read_json(output/'seal.json');old['study_status']['reader-pointer']=phase;old['all_completed']=all(r['completed'] for r in old['study_status'].values());old['files_sha256']={str(k.relative_to(output)):digest(k) for k in sorted(output.rglob('*')) if k.is_file() and k.name!='seal.json'};replace_json(output/'seal.json',old)

if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for k in ('model','output','stimuli'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('candidate','protocol-sha256'):p.add_argument('--'+k,required=True)
    run(**vars(p.parse_args()))
