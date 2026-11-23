"""Partially identify movie-word performance for every admissible human span placement."""
from collections import Counter
from itertools import product
import json
from pathlib import Path
import re

import numpy as np

from research.reading_validation.transition_score import dump, sha
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import RESULTS, source


PROTOCOL = Path('research/reading_validation/transition-full-movie-bounds-protocol.md')


def segment_solutions(bounds, text, errors):
    options=[]
    for error in errors:
        phrase=' '.join(error['span'].split())
        if not phrase:raise ValueError('Empty recorded span')
        placements=[]
        for match in re.finditer(f'(?={re.escape(phrase)})',text):
            start=match.start()
            mask=(bounds[:,0]<start+len(phrase))&(bounds[:,1]>start)
            placements.append(mask)
        if not placements:raise ValueError('No literal occurrence of the recorded human span')
        options.append(placements)
    solutions=np.zeros((1,len(bounds)),dtype=bool)
    for placements in options:
        solutions=np.unique(np.concatenate([solutions|mask for mask in placements],axis=0),axis=0)
    return solutions,[len(p) for p in options]


def construct():
    docs,meta=inputs();raw=source();items=[];audit=Counter();n=0;known=[]
    for m in meta['documents']:
        if m['split']!='test':continue
        d=m['doc'];doc=docs[d];words=np.array(doc['words'])
        valid,votes,_=outcomes(doc,m,raw[d],'CharE')
        for seg in m['segments']:
            k=seg['index'];offset=seg['start']
            block=raw[d][str(k)];indices=np.flatnonzero(words[:,4]==k)
            text=doc['text'][offset:seg['end']]
            if text!=' '.join(block['text'].split()):raise ValueError('Released source coordinates changed')
            records=[e for e in block['errors'] if e['error_type']=='CharE']
            possibilities,multiplicity=segment_solutions(words[indices,:2]-offset,text,records)
            if valid[indices].all() and (len(possibilities)!=1 or not np.array_equal(possibilities[0],votes[indices]>0)):
                raise ValueError('Gold mask differs from previously identifiable word labels')
            for t in multiplicity:audit['source_spans_with_'+str(t)+'_literal_occurrences']+=1
            audit['released_records']+=len(records);audit['segments']+=1
            audit['ambiguous_segments']+=len(possibilities)>1
            audit['words_in_ambiguous_segments']+=len(indices) if len(possibilities)>1 else 0
            audit['unresolved_word_label']+=int((possibilities.max(0)!=possibilities.min(0)).sum())
            audit['guaranteed_positive_words']+=int(possibilities.min(0).sum())
            audit['possible_positive_words']+=int(possibilities.max(0).sum())
            items.append((indices+n,possibilities));known.extend(valid[indices])
        n+=len(words)
    if n!=sum(len(ids) for ids,_ in items) or sum(known)!=22133 or n!=26417:
        raise ValueError('Evaluation universe differs from released movie readout')
    return items,audit,[m['doc'] for m in meta['documents'] if m['split']=='test']


def ratio_extreme(choices,cols,high):
    n_col,d_col=cols
    nums=[c@n_col for c in choices];dens=[c@d_col for c in choices]
    pick=np.argmax if high else np.argmin
    ratio=.5
    for _ in range(100):
        indices=[pick(a-ratio*b) for a,b in zip(nums,dens,strict=True)]
        num=sum(a[i] for a,i in zip(nums,indices,strict=True))
        den=sum(a[i] for a,i in zip(dens,indices,strict=True))
        if not den:raise ValueError('Performance denominator vanishes in the annotated movies')
        new=num/den
        if abs(ratio-new)<1e-13:return {'estimate':float(new),'numerator':int(num),'denominator':int(den)}
        ratio=new
    raise RuntimeError('Extremal performance ratio did not converge')


def threshold_bounds(solutions,flags,relaxed):
    matrices=[]
    for indices,z in solutions:
        f=flags[indices];possible=z.max(axis=0);fixed=z.min(axis=0)
        if relaxed:
            uncertain=possible&~fixed
            z=np.stack([fixed|(uncertain&f&use_flag)|(uncertain&~f&use_nonflag)
                        for use_flag,use_nonflag in product((False,True),repeat=2)])
        matrices.append(np.stack([(z&f).sum(1),(~z&f).sum(1),(z&~f).sum(1),(~z&~f).sum(1)],axis=1))
    metrics={'precision':([1,0,0,0],[1,1,0,0]),'recall':([1,0,0,0],[1,0,1,0]),
             'f1':([2,0,0,0],[2,1,1,0]),'false_alert_fraction':([0,1,0,0],[0,1,0,1])}
    return {key:[ratio_extreme(matrices,pair,v) for v in (False,True)] for key,pair in metrics.items()}


def ap_bounds(solutions,score,relaxed=False):
    # All ranks use the end of a score-tie group. This reproduces non-interpolated, tie-grouped AP.
    _,reverse,occ=np.unique(-score,return_inverse=True,return_counts=True)
    rank=np.cumsum(occ);ld=np.zeros(len(rank),dtype=int);ud=ld.copy()
    for indices,choices in solutions:
        if relaxed:choices=np.stack([choices.min(0),choices.max(0)])
        seen=choices.any(0);places=reverse[indices[seen]];choices=choices[:,seen]
        if not len(places):continue
        cols=np.unique(places)
        cnt=np.stack([choices[:,places==k].sum(1) for k in cols],axis=1).cumsum(axis=1)
        ld[cols]+=np.diff(np.r_[0,cnt.min(0)]);ud[cols]+=np.diff(np.r_[0,cnt.max(0)])
    lower=ld.cumsum();upper=ud.cumsum();lo=int(lower[-1]);hi=int(upper[-1])
    if not 0<lo<=hi:raise ValueError('No guaranteed positive location')
    bound_low=[];bound_high=[]
    for j in range(1,hi+1):
        first=np.searchsorted(upper,j)
        last=min(np.searchsorted(lower,j),len(rank)-1)
        support=slice(first,last+1)
        bound_low.append(j/rank[last])
        # At the end of its score tie, the jth positive has at most tie_size-1 further positives.
        ceiling=np.minimum(upper[support],j+occ[support]-1)/rank[support]
        bound_high.append(float(ceiling.max()))
    denom=np.arange(lo,hi+1)
    return {'positive_word_count':[lo,hi],
            'ap_bounds':[float((np.cumsum(bound_low)[lo-1:]/denom).min()),
                         float((np.cumsum(bound_high)[lo-1:]/denom).max())]}


def test():
    rng=np.random.default_rng(823)
    for _ in range(45):
        options=[(np.arange(0,3),np.array([[1,0,1],[0,1,1]],bool)),
                 (np.arange(3,6),rng.binomial(1,.4,size=(3,3)).astype(bool)),
                 (np.arange(6,9),rng.binomial(1,.5,size=(3,3)).astype(bool))]
        s=rng.integers(0,5,size=9);flags=s>1
        p=ap_bounds(options,s)['ap_bounds']
        vals=[]
        for x in product(*(z for _,z in options)):
            y=np.concatenate(x)
            order=np.argsort(-s,kind='stable')
            stop=np.r_[np.flatnonzero(s[order][1:]!=s[order][:-1]),len(s)-1]
            cum=np.cumsum(y[order])[stop]
            vals.append(np.diff(np.r_[0,cum])@(cum/(stop+1))/y.sum())
        if min(vals)<p[0]-1e-12 or max(vals)>p[1]+1e-12:raise AssertionError((vals,p))
        if not flags.any() or flags.all():continue
        resp=threshold_bounds(options,flags,False)
        for metric,anchors in resp.items():
            ratios=[]
            for x in product(*(z for _,z in options)):
                y=np.concatenate(x);tp=np.sum(y&flags);fp=np.sum(~y&flags);n=np.sum(y)
                if metric=='precision':ratios.append(tp/(tp+fp))
                if metric=='recall':ratios.append(tp/n)
                if metric=='f1':ratios.append(2*tp/(n+tp+fp))
                if metric=='false_alert_fraction':ratios.append(fp/(len(y)-n))
            np.testing.assert_allclose([a['estimate'] for a in anchors],[min(ratios),max(ratios)])


def main():
    test();solutions,audit,docs=construct()
    readouts=json.loads((RESULTS/'readouts.json').read_text())['readouts']['CharE']['antecedents']
    qfile=RESULTS/'heldout-probabilities.npz';nerfile=RESULTS/'entity-control-probabilities.npz'
    nlock=json.loads((RESULTS/'entity-control-lock.json').read_text())
    if (sha(qfile)!=json.loads((RESULTS/'prediction-lock.json').read_text())['forecast_sha256']
            or sha(nerfile)!=nlock['forecast_sha256']):raise ValueError('A policy changed')
    q=np.load(qfile);ner=np.load(nerfile)
    methods={'qwen':(np.concatenate([q['CharE__'+d][:,3] for d in docs]),readouts['threshold']),
             'ner_surprisal':(np.concatenate([ner[d][:,0] for d in docs]),nlock['cutoff'])}
    result={}
    for policy,(score,cut) in methods.items():
        result[policy]={}
        for relaxed in (False,True):
            config={'cut':cut,'alerts':int((score>cut).sum()),**ap_bounds(solutions,score,relaxed),
                    'operating_bounds':threshold_bounds(solutions,score>cut,relaxed)}
            result[policy]['arbitrary_matching_word' if relaxed else 'one_placement_per_record']=config
    diff={}
    for model in result['qwen']:
        a=result['qwen'][model]['ap_bounds'];b=result['ner_surprisal'][model]['ap_bounds']
        diff[model]=[a[0]-b[1],a[1]-b[0]]
    dump(RESULTS/'full-movie-identification.json',{'protocol_sha256':sha(PROTOCOL),'movie_titles':len(docs),
         'all_words':26417,'audit':dict(audit),'methods':result,'qwen_minus_ner_conservative_ap_bounds':diff})
    print(dict(audit))
    for policy in result:
        v=result[policy]['one_placement_per_record']
        print(policy,v['positive_word_count'],v['ap_bounds'], {k:[t['estimate'] for t in r] for k,r in v['operating_bounds'].items()})
    print('paired AP bounds',diff)


if __name__=='__main__':main()
