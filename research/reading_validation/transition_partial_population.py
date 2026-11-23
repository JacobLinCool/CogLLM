"""Propagate both movie variation and source alignment ambiguity for macro average precision."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_full_movie_bounds import construct
from research.reading_validation.transition_paired_ap_bounds import factor_envelope
from research.reading_validation.transition_partial_loss import OUT
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_prepare import RESULTS


PROTOCOL=Path('research/reading_validation/transition-partial-population-protocol.md')


def main():
    config,paper,names=construct();docs,_=inputs()
    src=OUT/'locked-forecasts.npz';ner=RESULTS/'entity-control-probabilities.npz'
    if (sha(src)!=json.loads((OUT/'scores.json').read_text())['forecasts_sha256'] or
            sha(ner)!=json.loads((RESULTS/'entity-control-lock.json').read_text())['forecast_sha256']):
        raise ValueError('External model estimates changed')
    fit=np.load(src);nerfile=np.load(ner)
    candidate=fit['set_marginal'];controls={'ner':np.concatenate([nerfile[d][:,0] for d in names]),
            'segment_unique':fit['segment_unique'],'word_identified':fit['word_identified']}
    membership=[];start=0;scope={}
    for d in names:
        end=start+len(docs[d]['words']);membership.extend([d]*(end-start));scope[d]=(start,end);start=end
    groups=defaultdict(list)
    for idx,z in config:
        d=membership[idx[0]];s,e=scope[d]
        if idx.min()<s or idx.max()>=e:raise ValueError('A segment crosses two original movies')
        groups[d].append((idx-s,z))
    result={};omitted=[]
    for d,(a,b) in scope.items():
        segments=groups[d]
        nmin=sum(t.sum(1).min() for _,t in segments)
        if nmin==0:
            if sum(t.sum(1).max() for _,t in segments)!=0:raise ValueError('A movie has ambiguously defined positive denominator')
            omitted.append(d);continue
        res={}
        for name,ctrl in controls.items():
            val=factor_envelope(segments,candidate[a:b],ctrl[a:b],0,True)['exact_extrema']
            res[name]=[val['False']['estimate'],val['True']['estimate']]
        result[d]=res
    ids,_,draws=grouped_distribution(sorted(result));summary={}
    for name in controls:
        vals=np.array([result[d][name] for d in ids]);boot=draws@vals/len(ids)
        summary[name]={'eligible_movies':len(vals),'sharp_macro_ap_difference':vals.mean(0).tolist(),
                       'movie_bootstrap_and_alignment_envelope':[float(np.quantile(boot[:,0],.025)),float(np.quantile(boot[:,1],.975))],
                       'always_positive':int((vals[:,0]>1e-10).sum()),'always_negative':int((vals[:,1]<-1e-10).sum()),
                       'possible_tie_or_reversal':int(((vals[:,0]<=1e-10)&(vals[:,1]>=-1e-10)).sum())}
    dump(OUT/'movie-variation-and-location.json',{'protocol_sha256':sha(PROTOCOL),'forecast_sha256':sha(src),
         'movie_groups':len(scope),'movies_without_any_selected_character':omitted,'summary':summary,'movie_ranges':result})
    print(summary)


if __name__=='__main__':main()
