"""Compare live forecasts with offline one-forward features, weights and original offsets."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from scipy.special import expit
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.convergent_word_prepare import ROOT,OUT


def compare(live,source):
    spec=read_json(OUT/'serving/spec.json');report=read_json(live/'receipt.json');responses=read_json(live/'responses.json');fit=read_json(OUT/'adaptive/context-fold-5.json');p=source/'word-states.npz'
    if (digest(p)!=spec['source_reference_features'] or report['weights']!=spec['weights_sha256'] or digest(live/'responses.json')!=report['output'] or report['ordinary_forwards']!=9):raise ValueError('One or both independent forecast paths changed')
    rows={r['id']:r for r in stream_rows(ROOT/'stimuli.jsonl')};m=fit['models']['shared'];prep=fit['projection'];errors=np.zeros(3);n=0
    with np.load(p) as z:
        for key,w in responses.items():
            x=z[key].astype(float);D=np.c_[np.ones(len(x)),(x[:,:6]-prep['mean'])/prep['sd'],((x[:,6:]-prep['context_mean'])/prep['context_scale'])@np.asarray(prep['basis'])];H=D@np.asarray(m['B']);H[:,1]=expit(H[:,1]);H=H*m['scale']+m['mu'];expected=H[1:].copy();expected[:,1]=1-expected[:,1];expected[:,2]*=-1
            measured=np.asarray([[q['expected_N400_negativity'],q['cloze_target_probability'],q['raw_fitted_rating']] for q in w]);G=rows[key]['words'][1:]
            if [tuple(q[a] for a in ('start','end','commit_token')) for q in w]!=[(a,b,hi) for a,b,lo,hi in G]:raise ValueError('Native writer and original scientific target disagree about exact position')
            errors=np.maximum(errors,np.max(abs(expected-measured),axis=0));n+=len(w)
    if errors.max()>5e-5 or report['total_committed_human_units']!=n:raise ArithmeticError(f'Serving output differs from sealed held evaluation: {errors}')
    write_json(OUT/'serving/verification.json',dict(extraction=digest(live/'receipt.json'),original_features=digest(p),matched_word_positions=n,numerical_comparisons=3*n,max_abs_by_endpoint=errors.tolist(),native=report));print(n,errors)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('live',type=Path);p.add_argument('original',type=Path);a=p.parse_args();compare(a.live,a.original)
