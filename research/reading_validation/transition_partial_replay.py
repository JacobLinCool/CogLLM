"""Replay the source-only marginal head on certified original coordinates and cached states."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from research.reading_validation.narrative_inference import MARGINAL_SHA,MarginalNarrativeInference,certified_words
from research.reading_validation.transition_partial_loss import OUT
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_snac_fit import inputs


def run(features,tokenizer_json):
    meta=json.loads((OUT/'scores.json').read_text());fpath=features/'word-features.npz'
    frozen=OUT/'locked-forecasts.npz';model=MarginalNarrativeInference.load()
    if sha(frozen)!=meta['forecasts_sha256'] or sha(fpath)!=meta['feature_sha256']:raise ValueError('Expected original word readout and features')
    scores=np.exp(-np.logaddexp(0.,-np.load(frozen)['set_marginal']));x=np.load(fpath)
    docs,materials=inputs();tok=Tokenizer.from_file(str(tokenizer_json));index=0;flags=0;games=0;worst=0.
    for mat in materials['documents']:
        if mat['split']!='test':continue
        d=mat['doc'];text=docs[d]['text'];raw=tok.encode(text,add_special_tokens=False)
        words=certified_words(text,tok,tuple(raw.ids),tuple(map(tuple,raw.offsets)),[(s['start'],s['end']) for s in mat['segments']])
        if words!=docs[d]['words']:raise ValueError('Changed visible word coordinates')
        probs=model.probabilities(x[d]);gold=scores[index:index+len(words)];index+=len(words)
        if not np.array_equal(probs>model.threshold,gold>model.threshold):raise ValueError('Changed writer flags')
        worst=max(worst,float(np.max(abs(gold-probs))));flags+=int((probs>model.threshold).sum());games+=1
    if index!=len(scores) or index!=26417 or games!=39 or worst>1e-10:raise ValueError('Incomplete replay')
    result={'bundle_sha256':MARGINAL_SHA,'feature_source_sha256':sha(fpath),'forecast_sha256':sha(frozen),'movies':games,
            'words':index,'flags':flags,'max_absolute_score_gap':worst,'threshold':model.threshold}
    dump(OUT/'serving'/'cached-replay.json',result)
    print(result)


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__);cli.add_argument('--features',type=Path,required=True);cli.add_argument('--tokenizer',type=Path,required=True)
    args=cli.parse_args();run(args.features,args.tokenizer)
