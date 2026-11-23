"""Post-primary sensitivity to each expert-coded cause of an incoherent essay sentence."""
import json

import numpy as np
import pandas as pd

from research.reading_validation.transition_external_prepare import OUT
from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_score import dump, with_interval


REASONS={1:'semantic connection',2:'entity reference',3:'discourse relation',4:'consistency',
         5:'contextual relevance',6:'tangential relevance',7:'other'}


def run():
    model=json.loads((OUT/'readouts.json').read_text());frame=pd.read_csv(OUT/'heldout-scored.csv');frame=frame[frame.source.eq('decor')]
    groups,idx,draws=grouped_distribution(frame.group.to_numpy());coded={}
    for i,name in REASONS.items():
        event=frame[f'reason_{i}'].fillna(0).to_numpy(dtype=int)
        denominator=np.bincount(idx,weights=event,minlength=len(groups));bden=draws@denominator
        reason={'positive_examples':int(event.sum()),'eligible_essays':int((denominator>0).sum())}
        for arm in ('integration','student','lexical_surprisal','directed','display'):
            hits=(frame[arm].to_numpy()>model['thresholds'][arm])*event
            counts=np.bincount(idx,weights=hits,minlength=len(groups))
            with np.errstate(invalid='ignore',divide='ignore'):
                rate=with_interval(hits.sum()/event.sum(),draws@counts/bden)
            reason[arm]={'hits':int(hits.sum()),'sensitivity':rate}
        coded[name]=reason
    dump(OUT/'reason-sensitivity.json',{'design':'descriptive after primary test; source reason types can overlap; fixed thresholds; whole-essay bootstrap',
      'clean_negative_denominator':int((1-frame.y).sum()),'reasons':coded})
    return coded


if __name__=='__main__':
    for k,m in run().items():print(k,m['positive_examples'],m['integration'])
