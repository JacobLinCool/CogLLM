"""Verify the serving bundle on every test word, including ambiguous-label segments."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from research.reading_validation.narrative_inference import NarrativeInference, certified_words, BUNDLE
from research.reading_validation.transition_score import dump, sha
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_prepare import RESULTS


def verify(directory,snapshot):
    docs,meta=inputs();sealed=RESULTS/'heldout-probabilities.npz';base=directory/'word-features.npz'
    if sha(base)!=json.loads((directory/'receipt.json').read_text())['feature_sha256'] or sha(sealed)!=json.loads((RESULTS/'prediction-lock.json').read_text())['forecast_sha256']:
        raise ValueError('Expected original feature matrix and frozen holdout forecasts')
    features=np.load(base);p0=np.load(sealed);tokenizer=Tokenizer.from_file(str(snapshot/'tokenizer.json'))
    model=NarrativeInference.load();differences=[];n=flags=0
    for m in meta['documents']:
        if m['split']!='test':continue
        d=m['doc'];doc=docs[d];text=doc['text'];enc=tokenizer.encode(text,add_special_tokens=False)
        spans=[(b['start'],b['end']) for b in m['segments']]
        words=certified_words(text,tokenizer,tuple(enc.ids),tuple(map(tuple,enc.offsets)),spans)
        if words!=doc['words']:raise ValueError('Saved study coordinates and serving word offsets disagree')
        p=model.probabilities(features[d]);sealed_p=p0['CharE__'+d][:,3]
        if not np.array_equal(p>model.threshold,sealed_p>model.threshold):raise ValueError('A serving threshold decision changed')
        differences.append(float(np.max(np.abs(p-sealed_p))));n+=len(p);flags+=int((p>model.threshold).sum())
    if n!=26417 or len(differences)!=39 or max(differences)>1e-10:raise ValueError('Incomplete or inaccurate movie replay')
    result={'model_sha256':sha(BUNDLE),'movie_forecast_sha256':sha(sealed),'movies':len(differences),'word_offsets_verified':n,
            'all_flags_reproduced':True,'word_flags':flags,'max_abs_risk_difference':max(differences)}
    dump(RESULTS/'serving-replay.json',result);return result


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--features',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True)
    args=p.parse_args()
    print(verify(args.features,args.snapshot))
