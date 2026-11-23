"""Export only the frozen word-level character detector, its weights and source-calibrated threshold."""
import json

from research.reading_validation.transition_score import dump, sha
from research.reading_validation.transition_snac_prepare import RESULTS


def export():
    locked=json.loads((RESULTS/'prediction-lock.json').read_text()); models=RESULTS/'readouts.json'
    if sha(models)!=locked['model_sha256']:raise ValueError('Model differs from the SNaC prediction seal')
    readout=json.loads(models.read_text())['readouts']['CharE']['antecedents'];data=readout['weights']
    package={'schema':'cogllm-narrative-character-en-v1','backbone':'Qwen/Qwen3.5-2B',
             'revision':'15852e8c16360a2fea060d615a32b45270f8a8fc',
             'reader_checkpoint_sha256':'c1704d4436d767fa43bdb59e9216698514e1f6bd1d45cad4b60d0f018d84abb5',
             'source_model_sha256':locked['model_sha256'],'source_protocol_sha256':locked['protocol_sha256'],
             'intended_observation':'A word overlaps an English narrative character reference that at least one SNaC reader marked unclear.',
             'threshold':readout['threshold'],'weights':{k:data[k] for k in ('intercept','weights','mean','std')}}
    out=RESULTS/'serving'/'character-word.json';out.parent.mkdir(exist_ok=True,parents=True);dump(out,package);print(sha(out),readout['threshold'])


if __name__=='__main__':export()
