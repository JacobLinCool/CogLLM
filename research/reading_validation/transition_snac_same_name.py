"""Compare different occurrences of exactly the same word within the same movie summary."""
from collections import defaultdict
import json

import numpy as np

from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_score import dump, with_interval
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import RESULTS,source


def analyze():
    docs,meta=inputs();raw=source();pred=np.load(RESULTS/'heldout-probabilities.npz');ner=np.load(RESULTS/'entity-control-probabilities.npz');results={}
    for restriction in ('same_form','same_form_repeated_only','same_form_recognized_person'):
        nums=[];pairs=[];group_ids=[];form_count=0
        for m in meta['documents']:
            if m['split']!='test':continue
            d=m['doc'];doc=docs[d];ok,votes,_=outcomes(doc,m,raw[d],'CharE');by_form=defaultdict(list);seen=set()
            for i,(a,b,*_) in enumerate(doc['words']):
                form=doc['text'][a:b].casefold();eligible=ok[i]
                if restriction=='same_form_repeated_only':eligible &= form in seen
                if restriction=='same_form_recognized_person':eligible &= bool(ner[d][i,1])
                if eligible:by_form[form].append(i)
                seen.add(form)
            table=np.column_stack((pred['CharE__'+d][:,3],ner[d][:,0],pred['CharE__'+d][:,1]));counts=np.zeros(table.shape[1]);n=0
            for indices in by_form.values():
                ii=np.array(indices);y=votes[ii]>0
                if y.all() or not y.any():continue
                a=table[ii[y]];b=table[ii[~y]];delta=a[:,None,:]-b[None,:,:]
                counts+=((delta>0)+.5*(delta==0)).sum(axis=(0,1));n+=a.shape[0]*b.shape[0];form_count+=1
            if n:nums.append(counts);pairs.append(n);group_ids.append(d)
        if len(group_ids)<2:
            results[restriction]={'movies':len(group_ids),'pairs':sum(pairs),'groups':form_count};continue
        groups,idx,draws=grouped_distribution(group_ids);counts=np.array(nums);den=np.array(pairs)
        # The sampling matrix columns follow sorted group IDs.
        order=np.argsort(idx);counts=counts[order];den=den[order];per_movie=counts/den[:,None];samples=draws@per_movie/draws.sum(axis=1)[:,None]
        stats={k:with_interval(per_movie[:,i].mean(),samples[:,i]) for i,k in enumerate(('contextual','ner','surprisal'))}
        delta=per_movie[:,0]-per_movie[:,1];stats['contextual_minus_ner']=with_interval(delta.mean(),samples[:,0]-samples[:,1]);
        results[restriction]={'movies':len(groups),'pairs':int(den.sum()),'distinct_movie_word_groups':form_count,'models':stats}
    dump(RESULTS/'same-form-context.json',{'design':'post-primary comparison of marked and unmarked different occurrences of the same casefolded word within a movie; movies equally weighted','results':results})
    return results


if __name__=='__main__':
    for k,v in analyze().items():print(k,v)
