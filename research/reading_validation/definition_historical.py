"""Trace predicted textbook concept spans through the original matched scientific controls."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.reading_validation.definition_data import OUT, ROOT
from research.reading_validation.definition_features import Corpus
from research.reading_validation.definition_fit import all_spans
from research.reading_validation.definition_predict import feature_series
from research.reading_validation.identity_causal_fit import OUT as CAUSAL, trace
from research.reading_validation.identity_causal_score import historical
from research.reading_validation.identity_prepare import PREVIOUS, TOKENIZER


def hazard(levels):
    result = {}
    for key, values in levels.items():
        previous = np.r_[0., values[:-1]]
        result[key] = (np.maximum(0., values - previous) / np.maximum(1.-previous, 1e-9)).tolist()
    return {'hazard': result}


def word_curve(page, span, size, model, start=0):
    v = np.zeros(size)
    if span is not None:
        _, probabilities = feature_series(page, span, model, 'semantic')
        v[page.words[:, 3]] = probabilities
    v[:start] = 0.
    return np.maximum.accumulate(v)


def run(root, causal_root):
    corpus = Corpus(root)
    keys = read_json(ROOT/'external-keys.json')['historical']
    layout = read_json(PREVIOUS/'qwen.json')['rows']
    source_geometry = read_json(causal_root/'geometry.json')
    spans = dict(zip(source_geometry['ids'], source_geometry['sentence_tokens'],strict=True))
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    keys = {key: val for key, val in keys.items() if layout[key]['kind'] in ('ab','controls')}
    texts = {key: corpus.rows[val]['text'] for key,val in keys.items()}
    offs = {k: tokenizer.encode(t, add_special_tokens=False).offsets for k,t in texts.items()}
    bank = SimpleNamespace(spans=spans, text=texts, offsets=offs)
    causal = read_json(CAUSAL/'weights.json')
    separate = read_json(OUT/'weights.json')
    joint = read_json(OUT/'joint-weights.json')
    models = {'separate': (separate['term']['semantic'],separate['link']['semantic']),
              'joint': (joint['term'],joint['link'])}
    result = {method: {} for method in ('separate','joint','joint_span_ecare')}
    decisions = []; relative=[]; worst=[]
    original = read_json(CAUSAL/'authored-forecasts.json')
    with np.load(causal_root/'token-states.npz',allow_pickle=False) as st, np.load(causal_root/'token-attention.npz',allow_pickle=False) as at:
        for key,ident in sorted(keys.items()):
            pg = corpus.page(ident); group = layout[key]; token_spans = spans[key]; n = len(offs[key])
            aux=st[key].astype('float32'); width=pg.tokens.shape[1]//2; half=aux.shape[1]//2
            projected=np.column_stack((aux[:,:width],aux[:,half:half+width])); d=projected-pg.tokens
            relative.append(float(np.linalg.norm(d)/np.linalg.norm(pg.tokens)));worst.append(float(np.abs(d).max()))
            source = group['anchor_tokens']; manual = {}
            for name,(a,b) in source.items():
                indexed=np.flatnonzero((pg.words[:,2]>=a)&(pg.words[:,3]<b))
                manual[name]=(int(indexed[0]),int(indexed[-1])) if len(indexed) else None
            locations={k:next(j for j,(a,b) in enumerate(token_spans) if a<=s[0]<b) for k,s in source.items()}
            for method,(proposal,relation) in models.items():
                l,r,v=all_spans(pg,proposal,'semantic'); selected={};all_curves={};all_tokens={}
                for j,(a,b) in enumerate(token_spans):
                    ok=np.flatnonzero((pg.words[l,2]>=a)&(pg.words[r,3]<b))
                    if not len(ok):raise ValueError('Historical sentence lacks any concept candidates')
                    best=ok[np.argmax(v[ok])]; span=(int(l[best]),int(r[best]));selected[str(j)]=span
                    all_tokens[str(j)]=[int(pg.words[span[0],2]),int(pg.words[span[1],3])+1]
                    # A one-per-sentence candidate becomes available when that sentence ends.
                    all_curves[str(j)]=word_curve(pg,span,n,relation,start=b)
                given={name:word_curve(pg,anchor,n,relation) for name,anchor in manual.items()}
                guess={name:all_curves[str(j)] for name,j in locations.items()}
                result[method][key]={'candidate_sentences':token_spans,'automatic':hazard(guess),'supplied':hazard(given),'all_candidates':hazard(all_curves)}
                for name,j in locations.items():
                    a,b=source[name]; p,q=all_tokens[str(j)]; intersection=max(0,min(b,q)-max(a,p))
                    decisions.append({'document':key,'kind':group['kind'],'antecedent':name,'fit':method,'gold':[a,b],'predicted':[p,q],
                                      'exact':bool(a==p and b==q),'token_iou':intersection/(b-a+q-p-intersection)})
                if method=='joint':
                    auto,_=trace(bank,causal,key,aux,at[key],all_tokens)
                    chosen={name:auto[str(j)] for name,j in locations.items()}
                    result['joint_span_ecare'][key]={'candidate_sentences':token_spans,'automatic':hazard(chosen),'supplied':original[key]['supplied'],'all_candidates':hazard(auto)}
    if max(relative)>.002:raise ValueError('Independently recorded overlapping projected dimensions disagree')
    write_json(OUT/'historical-replay.json',{'prior_gates':{method:historical(data) for method,data in result.items()},
              'span_decisions':decisions,'max_relative_representation_error':max(relative),'max_absolute_representation_error':max(worst),
              'reused_historical_controls':True,'causal_weights':digest(CAUSAL/'weights.json'),'joint_weights':digest(OUT/'joint-weights.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('causal_root',type=Path)
    args=p.parse_args();run(args.root,args.causal_root)
