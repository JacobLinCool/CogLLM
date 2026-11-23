"""Certify a paired ranking comparison over every compatible movie-span placement."""
from itertools import product
import json

import numpy as np

from research.reading_validation.transition_full_movie_bounds import construct
from research.reading_validation.transition_score import dump, sha
from research.reading_validation.transition_snac_prepare import RESULTS


def ap_quadratic(scores, possible):
    _,tie,count=np.unique(-scores,return_inverse=True,return_counts=True)
    denom=np.cumsum(count)[tie][possible]
    order=tie[possible]
    # y_i y_j contributes whenever j scores at least as highly as i, including score ties.
    coeff=(order[:,None]>=order[None,:])/denom[:,None]
    return (coeff+coeff.T)*.5


def exact_ratio(base, mean, local, pairs, groups, high):
    sign=-1 if high else 1
    count0=mean.sum();mass=[d.sum(axis=1) for _,d in groups]
    graph=[[] for _ in local]
    for i,j,e in pairs:
        graph[i].append((j,sign*e));graph[j].append((i,sign*e.T))
    calls=0;logs=[]
    def eliminate(allowed,linear):
        nonlocal calls
        calls+=1
        if calls>10000:raise RuntimeError('Exact ambiguity enumeration exhausted its certified branch budget')
        work=True
        while work:
            work=False
            for i,ix in enumerate(allowed):
                if len(ix)<2:continue
                # If one configuration costs more for every possible setting elsewhere, remove it.
                dif=linear[i][ix,None]-linear[i][ix][None,:]
                for j,coeff in graph[i]:
                    terms=coeff[np.ix_(ix,allowed[j])]
                    dif+=(terms[:,None,:]-terms[None,:,:]).min(axis=2)
                keep=~(dif>1e-10).any(axis=1)
                if not keep.all():
                    if not keep.any():raise ArithmeticError('Strict dominance retired every placement')
                    allowed[i]=ix[keep];work=True
        return allowed
    def solve(r):
        linear=[sign*(v-r*m) for v,m in zip(local,mass,strict=True)]
        constant=sign*(base-r*count0)
        best=float('inf');choice=None;leaves=0
        def visit(allowed):
            nonlocal best,choice,leaves
            allowed=eliminate(allowed,linear)
            sizes=[len(s) for s in allowed]
            if max(sizes,default=1)==1:
                z=np.array([s[0] for s in allowed]);leaves+=1
                value=constant+sum(c[i] for c,i in zip(linear,z,strict=True))+sum(sign*e[z[i],z[j]] for i,j,e in pairs)
                if value<best:best,choice=float(value),z
                return
            bound=constant+sum(c[g].min() for c,g in zip(linear,allowed,strict=True))
            bound+=sum((sign*e[np.ix_(allowed[i],allowed[j])]).min() for i,j,e in pairs)
            if bound>best+1e-9:return
            k=int(np.argmax(sizes))
            for option in allowed[k]:
                nxt=allowed.copy();nxt[k]=np.array([option]);visit(nxt)
        visit([np.arange(len(c)) for c in local])
        if choice is None:raise ValueError('No admissible human-span placement remained')
        numerator=base+sum(z[i] for z,i in zip(local,choice,strict=True))+sum(v[choice[i],choice[j]] for i,j,v in pairs)
        count=count0+sum(z[i] for z,i in zip(mass,choice,strict=True))
        logs.append({'ratio':float(r),'objective':best,'leaves':leaves,'positive_words':int(round(count))})
        return float(numerator/count),choice
    r=0.
    for step in range(12):
        updated,state=solve(r)
        if abs(updated-r)<1e-12:return {'estimate':updated,'evaluated_nodes':calls,'trace':logs,
                                       'admissible_state_indices':state.tolist()}
        r=updated
    raise ArithmeticError('The identified performance ratio did not converge')


def factor_envelope(solutions, model, control, dual_sweeps=0, certify=False):
    possible=np.zeros(len(model),dtype=bool)
    for ix,z in solutions:possible[ix]=z.any(0)
    n=int(possible.sum());reduced=np.cumsum(possible)-1
    A=ap_quadratic(model,possible)
    if control is not None:A-=ap_quadratic(control,possible)
    positive_min=0;positive_max=0;mean=np.zeros(n);groups=[]
    for ix,z in solutions:
        positive_min+=z.sum(1).min();positive_max+=z.sum(1).max()
        kept=possible[ix];loc=reduced[ix[kept]]
        m=z[:,kept].mean(0);mean[loc]=m
        varies=(0<m)&(m<1)
        if varies.any():groups.append((loc[varies],z[:,kept][:,varies]-m[varies]))
    if positive_min<=0:raise ValueError('AP denominator is unidentified')
    centered_linear=2*A@mean;base=float(mean@A@mean)
    local=[];pairs=[];low=high=base
    for i,(loc,diff) in enumerate(groups):
        terms=diff@centered_linear[loc]+np.sum((diff@A[np.ix_(loc,loc)])*diff,axis=1)
        low+=terms.min();high+=terms.max();local.append(terms)
        for j,(old,previous) in enumerate(groups[:i]):
            coupling=2*(diff@A[np.ix_(loc,old)])@previous.T
            low+=coupling.min();high+=coupling.max()
            pairs.append((i,j,coupling))
    # Redistribute edge row/column means to the unary factors exactly. Different gauges
    # describe the same objective, so their strongest envelopes can all be retained.
    for _ in range(4):
        for i,j,edge in pairs:
            for side,k in ((1,i),(0,j)):
                shift=(edge.max(side)+edge.min(side))*.5
                edge-=shift[:,None] if side==1 else shift[None,:]
                local[k]+=shift
        low=max(low,base+sum(t.min() for t in local)+sum(e.min() for _,_,e in pairs))
        high=min(high,base+sum(t.max() for t in local)+sum(e.max() for _,_,e in pairs))
    certificates={}
    for direction,minimum in [('lower',True),('upper',False)]:
        extremum=np.min if minimum else np.max
        loc=[v.copy() for v in local];edges=[(i,j,v.copy()) for i,j,v in pairs]
        choose=np.argmin if minimum else np.argmax
        weights={}
        for j,z in enumerate(loc):weights[j]=[]
        for i,j,data in edges:
            weights[i].append((j,data));weights[j].append((i,data.T))
        def feasible():
            state=[choose(z) for z in loc]
            for iteration in range(100):
                updates=0
                for i,z in enumerate(loc):
                    c=z.copy()
                    for j,block in weights[i]:c+=block[:,state[j]]
                    new=choose(c)
                    if c[new]!=c[state[i]]:
                        state[i]=new;updates+=1
                if not updates:break
            return float(base+sum(z[state[i]] for i,z in enumerate(loc))+sum(e[state[i],state[j]] for i,j,e in edges))
        records=[];best=float('-inf') if minimum else float('inf')
        for sweep in range(dual_sweeps+1):
            total=float(base+sum(extremum(t) for t in loc)+sum(extremum(e) for _,_,e in edges))
            best=max(best,total) if minimum else min(best,total)
            if sweep in (0,1,2,5,10,20,40,80,160,320,640,dual_sweeps):
                records.append([sweep,total])
            if sweep==dual_sweeps:break
            if sweep%2:edges.reverse()
            for i,j,edge in edges:
                for axis,k in ((1,i),(0,j)):
                    shift=(extremum(edge,axis=axis)-loc[k])*.5
                    edge-=shift[:,None] if axis==1 else shift[None,:]
                    loc[k]+=shift
            if sweep%2:edges.reverse()
        certificates[direction]={'dual_sweeps':dual_sweeps,'certified_numerator':best,
                                 'feasible_numerator':feasible(),'trace':records}
        if minimum:low=max(low,best)
        else:high=min(high,best)
    ratios=[v/d for v in (low,high) for d in (positive_min,positive_max)]
    exact={str(k):exact_ratio(base,mean,local,pairs,groups,k) for k in (False,True)} if certify else {}
    return {'exact_extrema':exact,'possible_words':n,'nondeterministic_segments':len(groups),'positive_word_count':[int(positive_min),int(positive_max)],
            'numerator_certificates':certificates,'paired_ap_numerator_bounds':[float(low),float(high)],'paired_ap_bounds':[float(min(ratios)),float(max(ratios))]}


def test():
    rng=np.random.default_rng(951)
    for _ in range(35):
        s=rng.integers(0,5,9);t=rng.integers(0,5,9)
        src=[(np.arange(3),np.array([[1,0,1],[0,1,1]],bool)),
             (np.arange(3,6),rng.binomial(1,.5,(3,3)).astype(bool)),
             (np.arange(6,9),rng.binomial(1,.5,(3,3)).astype(bool))]
        result=factor_envelope(src,s,t,0,True)
        lo,hi=result['paired_ap_bounds'];actual=[]
        for config in product(*(z for _,z in src)):
            y=np.concatenate(config).astype(float)
            ap=[]
            for values in (s,t):
                # Independent, direct tie-grouped rank calculation.
                rank=np.argsort(-values,kind='stable')
                stops=np.r_[np.flatnonzero(np.diff(values[rank])!=0),len(y)-1]
                acc=y[rank].cumsum()[stops]
                ap.append(float(np.diff(np.r_[0,acc])@(acc/(stops+1))/y.sum()))
            actual.append(ap[0]-ap[1])
            if not lo-1e-11<=ap[0]-ap[1]<=hi+1e-11:raise AssertionError((lo,ap,hi))
        np.testing.assert_allclose([c['estimate'] for c in result['exact_extrema'].values()],[min(actual),max(actual)],atol=1e-11)


def main():
    test()
    solutions,audit,docs=construct()
    meta=json.loads((RESULTS/'full-movie-identification.json').read_text())
    qfile=RESULTS/'heldout-probabilities.npz';nerfile=RESULTS/'entity-control-probabilities.npz'
    q=np.load(qfile);ner=np.load(nerfile)
    model=np.concatenate([q['CharE__'+d][:,3] for d in docs]);control=np.concatenate([ner[d][:,0] for d in docs])
    ans=factor_envelope(solutions,model,control,5,True)
    reports={}
    for name,scores in [('qwen_minus_ner',None),('qwen',model),('ner_surprisal',control)]:
        data=ans if scores is None else factor_envelope(solutions,scores,None,0,True)
        for side,record in data['exact_extrema'].items():
            selected=iter(record['admissible_state_indices']);y=np.zeros(len(model),dtype=bool)
            for indices,states in solutions:
                ix=next(selected) if len(states)>1 else 0
                y[indices]=states[ix]
            if next(selected,None) is not None:raise ValueError('Unconsumed annotation placement')
            measures=[]
            for design in (model,control) if scores is None else (scores,):
                idx=np.argsort(-design,kind='stable');end=np.r_[np.flatnonzero(np.diff(design[idx])!=0),len(y)-1]
                cum=y[idx].cumsum()[end];v=float(np.diff(np.r_[0,cum])@(cum/(end+1))/y.sum());measures.append(v)
            witnessed=measures[0]-measures[1] if scores is None else measures[0]
            if abs(witnessed-record['estimate'])>1e-9:raise ArithmeticError('Human assignment does not reproduce exact AP')
            record['realized_on_all_words']=witnessed;record['positive_words']=int(y.sum())
        reports[name]=data['exact_extrema']
    ans['sharp_ranges']={n:[d['False']['estimate'],d['True']['estimate']] for n,d in reports.items()}
    ans['separate_extrema']=dict(qwen=reports['qwen'],ner_surprisal=reports['ner_surprisal'])
    dump(RESULTS/'full-movie-paired-ranking.json',{'source_envelope_sha256':sha(RESULTS/'full-movie-identification.json'),
            'policy_sha256':sha(qfile),'control_sha256':sha(nerfile),**ans})
    print(ans['sharp_ranges'])


if __name__=='__main__':main()
