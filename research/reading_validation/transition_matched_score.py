"""Score the directed edit comparison using equal prior turn histories per pair."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_external_fit import predict
from research.reading_validation.transition_external_prepare import OUT, ROOT
from research.reading_validation.transition_external_score import grouped_distribution, edit_quality
from research.reading_validation.transition_score import dump, sha, with_interval


def score(directory):
    path=directory / 'matched-logits.json'
    receipt=json.loads((directory / 'receipt.json').read_text())
    if (receipt['forecast_sha256']!=sha(path) or
            receipt['input_sha256']!=sha(ROOT/'cohesion-matched-stimuli.jsonl') or
            receipt['documents']!=424 or receipt['directed_forward_calls']!=424):
        raise ValueError('Matched edit packet or predictions changed')
    models=OUT/'readouts.json'
    if sha(models)!=json.loads((OUT/'prediction-lock.json').read_text())['model_sha256']:
        raise ValueError('The CoheSentia train-only model changed after forecast seal')
    trained=json.loads(models.read_text())
    model=trained['models']['directed'];cut=trained['thresholds']['directed']
    logits=json.loads(path.read_text())
    pairs=[r for r in json.loads((OUT/'stimuli.json').read_text())['material'] if r['source']=='decor_rewrites']
    quality=edit_quality([r['id'] for r in pairs])
    result={'receipt_sha256':sha(directory/'receipt.json')}
    for key, subset in [('all', pairs), ('changed', [r for r,q in zip(pairs,quality,strict=True) if q=='changed'])]:
        groups, indices, draws=grouped_distribution([r['group'] for r in subset])
        weights=draws[:,indices]
        old,new=[predict(np.array([logits[r[k]] for r in subset])[:,None],model) for k in ('counterpart','id')]
        identical = edit_quality([r['id'] for r in subset])=='identical'
        raw_diff = np.abs(new[identical]-old[identical])
        new=np.where(identical,old,new)
        improvements=(new<old);ties=(new==old)
        rows={}
        for name,array in [('new_minus_old',new-old),('strict_improvement',improvements),('half_tie_preference',improvements+.5*ties)]:
            rows[name]=with_interval(array.mean(),weights@array/weights.sum(axis=1))
        rows.update(lower_count=int(improvements.sum()),tie_count=int(ties.sum()),old_alarms=int((old>cut).sum()),new_alarms=int((new>cut).sum()),
                    identical_input_max_abs_difference=float(raw_diff.max()) if len(raw_diff) else None)
        result[key]={'pairs':len(subset),'essays':len(groups),'models':{'directed':rows}}
    dump(OUT/'matched-scores.json',result)
    return result


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--features',type=Path,required=True)
    print(score(parser.parse_args().features))
