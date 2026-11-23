"""Check whether the partial-observation readout's gain depends on hidden span locations."""
from argparse import ArgumentParser
import json
import numpy as np

from research.reading_validation.transition_full_movie_bounds import construct
from research.reading_validation.transition_paired_ap_bounds import factor_envelope
from research.reading_validation.transition_partial_loss import OUT
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_snac_prepare import RESULTS


def main(candidate=None):
    destination=OUT/candidate if candidate else OUT
    fits=json.loads((destination/'scores.json').read_text());packet=destination/'locked-forecasts.npz'
    ner=RESULTS/'entity-control-probabilities.npz'
    ner_lock=json.loads((RESULTS/'entity-control-lock.json').read_text());column=0
    if candidate:
        from research.reading_validation.transition_architecture_score import RESULT
        ner=RESULT/(candidate+'-movie-forecasts.npz')
        ner_lock=json.loads((RESULT/(candidate+'-lock.json')).read_text());column=ner_lock['arm_order'].index('ner')
    if sha(packet)!=fits['forecasts_sha256'] or sha(ner)!=ner_lock['forecast_sha256']:
        raise ValueError('A policy differs from the one fitted on independent books')
    segments,_,docs=construct();f=np.load(packet);tags=np.load(ner);m=f['set_marginal']
    results={}
    for name,ctl in (('versus_strong_ner',np.concatenate([tags[d][:,column] for d in docs])),
                     ('versus_segment_unique',f['segment_unique']),('versus_word_identified',f['word_identified'])):
        solved=factor_envelope(segments,m,ctl,0,True)
        extrema=solved['exact_extrema']
        for record in extrema.values():
            states=iter(record['admissible_state_indices']);y=np.zeros(len(m),dtype=bool)
            for ids,options in segments:y[ids]=options[next(states) if len(options)>1 else 0]
            if next(states,None) is not None:raise ValueError('State list differed from the 39 studied movies')
            vals=[]
            for scored in (m,ctl):
                ranks=np.argsort(-scored,kind='stable');boundary=np.r_[np.flatnonzero(np.diff(scored[ranks])!=0),len(y)-1]
                cum=y[ranks].cumsum()[boundary];vals.append(float(np.diff(np.r_[0,cum])@(cum/(boundary+1))/y.sum()))
            if abs(vals[0]-vals[1]-record['estimate'])>1e-9:raise ValueError('Sharp ranking witness failed')
            record['realized_ap']=vals
        results[name]={'sharp_all_word_difference':[extrema['False']['estimate'],extrema['True']['estimate']],'witnesses':extrema}
    dump(destination/'identification.json',{'forecast_sha256':sha(packet),'all_words':len(m),'paired_extrema':results})
    print({k:v['sharp_all_word_difference'] for k,v in results.items()})


if __name__=='__main__':
    cli=ArgumentParser(description=__doc__);cli.add_argument('--candidate',choices=('qwen','lfm3','gemma4'))
    main(cli.parse_args().candidate)
