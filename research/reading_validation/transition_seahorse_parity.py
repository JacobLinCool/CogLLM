"""Replay both frozen news utilities on every previous news draft before evaluating SEAHORSE ratings."""
import argparse
from pathlib import Path
import json
import numpy as np
from research.reading_validation.summeval_ballots import forecast
from research.reading_validation.summeval_forecast import open_old
from research.reading_validation.transition_seahorse_features import document_features, save, load
from research.reading_validation.transition_seahorse_data import DEST, WORK, sha


def run(folder):
    model=Path("research/results/reading-validation-summeval")
    p,_=open_old(); inputs={v['id']:v['text'] for v in map(json.loads,(WORK/'summeval/local-stimuli.jsonl').read_text().splitlines())}
    rec=load(folder/'receipt.json');ws=model/'ballot-weights.npz';gold=model/'ballot-forecasts.npz'
    lock=load(model/'ballot-forecast-seal.json')
    if sha(ws)!=lock['weights_sha256'] or sha(gold)!=lock['forecast_sha256'] or sha(folder/'features.npz')!=rec['file']:
        raise ValueError("Earlier original reference changed")
    with np.load(folder/'features.npz',allow_pickle=False) as a, np.load(gold,allow_pickle=False) as ref, np.load(ws,allow_pickle=False) as w:
        keys=list(ref['keys'])
        units=[document_features(a,k,rec['articles'][k],inputs[k],p)[0] for k in keys]
        if not all(v is not None for v in units):raise ValueError("A certified source article cannot be replayed")
        basis=np.stack(units); errors={}
        for k,features in [('form',basis[:,:15]),('native',basis)]:
            weights={a:w[k+'.'+a] for a in ('center','scale','slope')}
            gap=np.abs(forecast(weights,features)-ref[k+'_choice'])
            errors[k]={'maximum_absolute':float(gap.max()),'mean_absolute':float(gap.mean())}
            if gap.max()>1e-6:raise ArithmeticError(("News observer source mismatch",k,float(gap.max())))
    save(DEST/'news-observer-replay.json',{'n':len(keys),'source_forecast_sha':sha(gold),'source_native_sha':rec['file'],
            'source_weights_sha':sha(ws),'errors':errors})
    print(len(keys),errors)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);run(p.parse_args().folder)
