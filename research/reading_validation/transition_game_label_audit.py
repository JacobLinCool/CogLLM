"""Check the effect of silver vs expert entity identities on the same human votes."""
from collections import Counter
import json
from zipfile import ZipFile

import numpy as np
import pandas as pd

from research.reading_validation.transition_direct_data import display,gold_docs
from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_game_prepare import OUT,PD,ROOT,votes_for
from research.reading_validation.transition_score import dump,with_interval


def main():
    inputs={r['id']:r for r in map(json.loads,(ROOT/'snac-direct-stimuli.jsonl').read_text().splitlines()) if r['split']=='independent'}
    data=[];scope=Counter()
    with ZipFile(PD) as archive:
        for d,g,root,crac,els,tokens in gold_docs():
            if d not in inputs:continue
            row=inputs[d];_,co,_=display(root,els,tokens);name=d.split('-',2)[2]
            path=f'gold/test/{g}/silver_CRAC/{name}.CRAC'
            shadow=[l.split('\t') for l in archive.read(path).decode().splitlines()[1:] if l.strip()]
            if [l[0] for l in shadow]!=tokens:raise ValueError('Gold/silver visible-text streams differ')
            expert,_=votes_for(root,crac,els,tokens,co,row['words'])
            crowd,_=votes_for(root,shadow,els,tokens,co,row['words'])
            scope['gold_eligible']+=len(expert);scope['silver_eligible']+=len(crowd)
            for k in expert.keys()&crowd.keys():
                if expert[k]['entity_readers']!=crowd[k]['entity_readers']:raise ValueError('Different reader population on aligned comparison')
                data.append((d,g,expert[k]['p'],crowd[k]['p']))
    out={}
    for lab,records in [('all',data)]+[(g,[r for r in data if r[1]==g]) for g in sorted({r[1] for r in data})]:
        docs,idx,draw=grouped_distribution([r[0] for r in records])
        e=np.array([r[2] for r in records]);s=np.array([r[3] for r in records]);delta=s-e
        cnt=np.bincount(idx);bootstrap=(draw@np.bincount(idx,weights=delta))/(draw@cnt)
        rho=float(pd.Series(e).rank().corr(pd.Series(s).rank())) if e.std() and s.std() else None
        out[lab]={'mentions':len(records),'articles':len(docs),'gold_pair_disagreement':float(e.mean()),
                  'silver_pair_disagreement':float(s.mean()),'silver_minus_gold':with_interval(delta.mean(),bootstrap),
                  'same_exact_response_fraction':float(np.isclose(s,e).mean()),'pooled_rank_correlation':rho}
    dump(OUT/'weak-label-audit.json',{'eligibility':dict(scope),'paired':out})
    print(out['all'])


if __name__=='__main__':main()
