"""Fit one common scale for all unresolved, introduced and resolved probability in a passage."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.definition_data import OUT,ROOT
from research.reading_validation.definition_features import Corpus
from research.reading_validation.definition_stock_fit import gold,load_models
from research.reading_validation.stock_probability import ARMS,predict
from research.reading_validation.stock_probability_score import bounds


def calibrate(bank_path):
    bank=Corpus(bank_path);fits=read_json(OUT/'stock-probability-weights.json')['fits'];sources=load_models()
    counts=Counter();tables={a:[] for a in (*ARMS,'constant')};labels=[];groups=[]
    for r in stream_rows(ROOT/'calibration.jsonl'):
        pg=bank.page(r['id']);lo,hi,_=bounds(pg.n,gold(pg,r));labels.append(np.column_stack((lo,hi)));groups.extend([r['source']]*pg.n);counts[r['source']]+=pg.n
        for a in ARMS:tables[a].append(predict(pg,a,fits[a],sources[a]).unresolved.sum(1))
        tables['constant'].append(np.ones(pg.n))
    lo,hi=np.vstack(labels).T;w=np.asarray([1./len(counts)/counts[g] for g in groups]);result={}
    for a,pieces in tables.items():
        x=np.concatenate(pieces)
        def mse(scale):
            y=x*scale;v=np.maximum(lo-y,0.)+np.maximum(y-hi,0.)
            return float(w@(v*v))
        opt=minimize_scalar(mse,bounds=(0,1),method='bounded',options={'xatol':1e-12})
        best=min((0.,float(opt.x),1.),key=lambda v:(mse(v),v));result[a]={'scale':best,'source_balanced_mse':mse(best),'unscaled_mse':mse(1.)}
    write_json(OUT/'stock-count-scales.json',{'scales':result,'source_count':len(counts),'word_count':len(groups),
        'weights':digest(OUT/'stock-probability-weights.json'),'development_data':digest(ROOT/'calibration.jsonl'),
        'protocol':digest(Path(__file__).with_name('stock-calibration-protocol.md'))})
    print({a:round(v['scale'],5) for a,v in result.items()},flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('bank',type=Path);calibrate(p.parse_args().bank)
