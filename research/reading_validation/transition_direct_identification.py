"""Audit native-coordinate word forecasts under every permitted movie annotation placement."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_direct_data import OUT as NATIVE
from research.reading_validation.transition_full_movie_bounds import construct
from research.reading_validation.transition_paired_ap_bounds import factor_envelope
from research.reading_validation.transition_partial_loss import OUT as MARGINAL
from research.reading_validation.transition_score import dump,sha


def run(candidate):
    src=NATIVE/candidate;r=json.loads((src/'receipt.json').read_text());read=src/'word-probabilities.npz'
    if sha(read)!=r['predictions_sha256']:raise ValueError('Native forecast changed')
    folds,_,docs=construct();p=np.load(read);T=r['tasks'].index('CharE')
    methods={k:np.concatenate([p[d][:,r['arms'].index(k),T] for d in docs]) for k in ('direct','historic','ner','shared')}
    if candidate=='qwen':
        if sha(MARGINAL/'locked-forecasts.npz')!=json.loads((MARGINAL/'scores.json').read_text())['forecasts_sha256']:
            raise ValueError('Production source-observer forecast changed')
        methods['marginal_production']=np.load(MARGINAL/'locked-forecasts.npz')['set_marginal']
    out={}
    for k,control in methods.items():
        if k=='direct':continue
        estim=factor_envelope(folds,methods['direct'],control,0,True)['exact_extrema']
        assert all(np.isfinite(s['estimate']) for s in estim.values())
        out['direct_minus_'+k]={'range':[estim['False']['estimate'],estim['True']['estimate']],
                               'admissible_assignment_for_each_endpoint':estim}
    dump(src/'full-word-identification.json',{'candidate':candidate,'native_forecast_sha256':sha(read),'movie_words':len(methods['direct']),'contrasts':out})
    print(candidate,{k:y['range'] for k,y in out.items()})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--candidate',required=True,choices=('qwen','gemma4','lfm3'))
    run(p.parse_args().candidate)
