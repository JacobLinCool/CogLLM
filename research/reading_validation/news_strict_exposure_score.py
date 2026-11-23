"""Repeat prespecified news contrasts after excluding any historically supervised source story."""
import json
from pathlib import Path
import numpy as np
from research.reading_validation.coherence_order_data import read
from research.reading_validation.local_forecast import digest
from research.reading_validation.news_preference_crossover import compute,table
from research.reading_validation.summeval_ballots import distributions,risk
from research.reading_validation.summeval_data import OUT
from research.reading_validation.summeval_forecast import judgments
from research.reading_validation.summeval_score import PRIMARY,paired,result


def main():
    cfg=read(OUT/'exposure-sensitivity-lock.json');here=Path(__file__)
    if any(digest(p)!=cfg[k] for p,k in [(here,'code'),(here.with_name('summeval-exposure-sensitivity.md'),'protocol'),(OUT/'historical-exposure.json','audit')]):raise ValueError('Strict cohort changed')
    exposures=read(OUT/'historical-exposure.json')['any_historical_supervised_match']
    omit={x[1] for q in exposures.values() for x in q};geo=read(OUT/'story-geometry.json');votes=judgments('confirm')
    with np.load(OUT/'frozen-predictions.npz',allow_pickle=False) as a,np.load(OUT/'calibrated-predictions.npz',allow_pickle=False) as b,np.load(OUT/'ballot-forecasts.npz',allow_pickle=False) as c:
        ok=(a['roles']=='confirm')&~np.isin(a['stories'],list(omit));keys=a['keys'][ok];g=a['stories'][ok];arms={k:s[k][ok] for s,ns in [(a,['adaptive','surprisal','essay_hazard','essay_lexical','geval']),(b,['independent_native','shared_native','system_mean']),(c,['mixture_choice','form_choice','native_choice','native_absolute'])] for k in ns}
    dist=distributions(keys,g,geo,votes);y=dist@np.arange(1,6);groups=np.unique(g);core={a+'_minus_'+b:paired(y,arms[a],arms[b],groups,g,3) for a,b in PRIMARY};ball={'ballot_native_minus_absolute_grade_native_order':paired(y,arms['native_choice'],arms['native_absolute'],groups,g,2),'adaptive_over_form_ballot_logloss':result(risk(g,dist,arms['form_choice'])-risk(g,dist,arms['mixture_choice']),2)}
    gr,systems,ids,v,raw=table(keys,g,geo,votes,arms);cross=compute(ids,v,raw);out={'stories':len(gr),'omitted_stories':sorted(omit),'direct_expert_rated_drafts':len(keys),'primary_frozen_and_adapted':core,'ballot':ball,'crossover':cross['registered_tests'],'crossover_target_reversals':cross['observed_preference_reversals'],'audited_artifact':cfg}
    (OUT/'exposure-sensitivity-scores.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print('strict complete',len(gr))

if __name__=='__main__':main()
