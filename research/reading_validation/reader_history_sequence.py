"""Score later assignments with same-person, matched and peer-averaged Bayesian response profiles."""
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import softmax
from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.reader_history_external import base, get_geometry, records, answer_normalizer, DIR, OUT
from research.reading_validation.reader_history_peer import histories, conditionals, binary, BETA
from research.reading_validation.reader_history_model import report
from research.reading_validation.reader_history_decompose import decompose

PLAN=Path(__file__).with_name('reader-history-sequential.md')


def timeline(idx):
    initial,other=histories(idx)
    geom,_=get_geometry()
    wanted=defaultdict(list)
    for r,f,k,col,q,i in read_json(DIR/'independent-future-items.json')['chronological']:
        wanted[f,k].append((r,col,q,i))
    available=defaultdict(list)
    for (f,k),(r,accept,submit,items) in geom.items():
        available[r].append((submit,accept,f,k))
    first={r:min(block)[0] for r,block in available.items()}
    priors=defaultdict(list)
    for r,i,w in initial:
        priors[r].append((first[r],i,w))
    normalize=answer_normalizer()
    events=[]
    for f,k,row,_ in records():
        _,accept,submit,_=geom[f,k]
        for r,col,q,i in wanted.get((f,k),()):
            w,_=normalize(row[col],q)
            if w is None:continue
            priors[r].append((submit,i,w))
            events.append((r,i,w,accept,submit))
    for r,rows in priors.items():
        rows.sort(key=lambda v:(v[0],v[1]))
        if len(rows)!=len({i for _,i,_ in rows}):
            raise ValueError('Repeated viewed stimulus in personal history')
    return np.asarray(events),priors,other


def history_sums(reader_rows, sf, b, yes, no, lp, zn, outside):
    people=sorted(reader_rows)
    times=[];class_sums=[];name_sums=[]
    for r in people:
        t,i,w=np.array(reader_rows[r]).T
        i=i.astype(int);w=w.astype(int)
        ok=sf[i]!=b
        t,i,w=t[ok],i[ok],w[ok]
        flag=w==outside
        cl=np.where(flag[:,None],yes[i],no[i])
        named=lp[i,w,None].astype(float)*BETA-zn[i]
        named[flag,:]=0
        times.append(t)
        class_sums.append(np.vstack((np.zeros(cl.shape[1]), np.cumsum(cl,axis=0))))
        name_sums.append(np.vstack((np.zeros(len(BETA)),np.cumsum(named,axis=0))))
    return people,times,class_sums,name_sums


def probabilities(plus,minus,conditional,anchors):
    values=[];residuals=[]
    for class_post, joint in anchors:
        out=plus @ class_post
        joint_named=np.sum((minus @ joint)*conditional,axis=1)
        values.append(joint_named);residuals.append(out)
    return np.column_stack(values),np.column_stack(residuals)


def mixture(pc,pn):
    return pc,np.outer(pc,pn)


def run(native):
    check=read_json(OUT/'sequence-registration.json')
    if (check['protocol']!=digest(PLAN) or check['frozen_prior']!=digest(OUT/'peer-prior-lock.json')
            or check['previous']!=digest(OUT/'peer-scores.json')):
        raise ValueError('Sequential source changed')
    lp,idx,_=base(native);odds,zn=conditionals(lp);outside=lp.shape[1]-1
    ev,reader_rows,other=timeline(idx)
    r,i,w=ev[:,:3].astype(int).T;flag=w==outside
    accept=ev[:,3];submit=ev[:,4]
    lck=read_json(OUT/'peer-prior-lock.json')
    src=DIR/'peer-fit-parameters.json';aux=DIR/'peer-future-geometry.npz'
    if digest(src)!=lck['weights'] or digest(aux)!=lck['geometry']:
        raise ValueError('Sequential fit terms changed')
    with np.load(aux) as f:
        sf=f['stem'];rf=dict(zip(f['subjects'],f['reader'],strict=True))
    rs=np.array([rf[q] for q in r])
    loss=np.full((len(ev),4),np.nan);qout=np.full_like(loss,np.nan)
    support=np.zeros(len(ev),int);available_peer_count=np.zeros(len(ev),int)
    future_time_violations=0
    for m in read_json(src):
        a,b=m['reader_fold'],m['stem_fold']
        yes,no=binary(odds,np.array(m['class_coeff']))
        pc=np.array(m['class_prior']);pn=np.array(m['named_prior'])
        people,times,cls,nam=history_sums(reader_rows,sf,b,yes,no,lp,zn,outside)
        readermap={v:j for j,v in enumerate(people)}
        lengths=np.array([len(t) for t in times])
        folded=np.flatnonzero((rs==a)&(sf[i]==b))
        groups=defaultdict(list)
        for q in folded:groups[(r[q],accept[q])].append(q)
        # Uniform averaging across eligible peers is computed at precisely the matched exposure count.
        cache={}
        for (subject,start),indices in groups.items():
            receiving=readermap[subject];donor=readermap[other[subject]]
            actual=np.searchsorted(times[receiving],start,side='right')
            n=min(int(actual),int(lengths[donor]))
            if n<20:continue
            if times[receiving][n-1]>start or np.any(submit[indices]<start):
                future_time_violations+=1;continue
            if n not in cache:
                use=np.flatnonzero(lengths>=n)
                cc=softmax(np.stack([cls[k][n] for k in use])+np.log(pc),axis=1)
                nn=softmax(np.stack([nam[k][n] for k in use])+np.log(pn),axis=1)
                # Map from the reusable absolute reader index to this qualifying cohort row.
                ix={k:p for p,k in enumerate(use)}
                cache[n]=ix,cc,nn,cc.sum(0),cc.T@nn
            ix,cc,nn,cs,joint=cache[n];u=ix[receiving];d=ix[donor];K=len(ix)-1
            if K<1:raise ValueError('No independently matched reader')
            own=mixture(cc[u],nn[u]);yoke=mixture(cc[d],nn[d]);unknown=mixture(pc,pn)
            mean=((cs-cc[u])/K,(joint-own[1])/K)
            indices=np.array(indices);sel=i[indices];words=w[indices]
            word_cond=np.exp(lp[sel,words,None].astype(float)*BETA-zn[sel])
            named,out=probabilities(np.exp(yes[sel]),np.exp(no[sel]),word_cond,[own,mean,yoke,unknown])
            loss[indices]=-np.log(np.where(flag[indices,None],out,named))
            support[indices]=n;qout[indices]=out;available_peer_count[indices]=K
    use=support>=20
    if future_time_violations or not np.all(np.isfinite(loss[use])) or np.any((qout[use]<=0)|(qout[use]>=1)):
        raise ValueError('Timing or probability broken')
    e=ev[use,:3].astype(int);L=loss[use];O=qout[use];N=support[use]
    result=report(e,L,O)
    result['choice_content']=decompose(e,L,O)
    by_group={}
    for lo,hi in ((20,49),(50,199),(200,799),(800,10**6)):
        sel=(N>=lo)&(N<=hi)
        if not np.any(sel):continue
        sub=report(e[sel],L[sel],O[sel],draws=1000)
        by_group[str(lo)]=dict(n=sub['scored_responses'],people=sub['readers'],new_over_experience_matched=sub['paired']['same_over_unknown_reader'])
    save=DIR/'sequential-forecast-scores.npz'
    np.savez_compressed(save,events=e,loss=L,other=O,support=N)
    write_json(OUT/'sequential-scores.json',dict(registration=digest(OUT/'sequence-registration.json'),scored_arrays=digest(save),
        studies={'all':result,'bands_by_minimum':by_group},comparison_keys={
            'same_history':'Own earlier non-evaluation-fold responses from submitted questionnaires.',
            'unknown_reader':'Distribution marginalized over every other eligible peer at the same history length.',
            'other_history':'One matched stranger, same number of responses.',
            'single_head':'Original fitted peer-population prior with no individual responses.'},
        maximum_time_violations=future_time_violations,offered_nonempty=int(len(ev)),
        excluded_under20_matching=int((~use).sum()),history_size_percentiles=np.quantile(N,[0,.1,.25,.5,.75,.9,1]).tolist(),
        other_peer_counts=np.quantile(available_peer_count[use],[0,.5,1]).tolist(),mode='time-respecting observational reanalysis'))
    print({k:v for k,v in result['paired'].items()},flush=True)
    print('history-support',np.quantile(N,[0,.5,1]),by_group,flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('native',type=Path);run(p.parse_args().native)
