"""Estimate peer-population priors on earlier pages and evaluate identified future word choices."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.reader_history_external import (
    DIR, OUT, answer_normalizer, base, exposures, get_geometry, records, sources
)
from research.reading_validation.reader_history_model import SHARPNESS, report
from research.reading_validation.reader_history_decompose import decompose

OFFSETS = np.linspace(-3, 3, 9)
BETA = SHARPNESS[::7]
PROTOCOL = Path(__file__).with_name('reader-history-target-prior.md')


def fold(tag, value):
    return int.from_bytes(sha256(f'peer {tag} {value}'.encode()).digest()[:8], 'big') % 5


def conditionals(lp):
    outside = np.exp(lp[:, -1].astype(float))
    log_odds = lp[:, -1].astype(float) - np.log1p(-outside)
    normalizers = np.empty((len(lp), len(BETA)))
    for start in range(0, len(lp), 128):
        block = lp[start:start+128, :-1].astype(float)
        for j, b in enumerate(BETA):
            normalizers[start:start+len(block), j] = logsumexp(b * block, axis=1)
    if not np.allclose(normalizers[:,3], np.log1p(-outside), atol=8e-7):
        raise ValueError('Named and residual probabilities disagree')
    return log_odds, normalizers


def histories(idx):
    lock = read_json(OUT/'external-lock.json')
    path = DIR / 'independent-reader-posteriors.npz'
    if digest(path) != lock['prior_posteriors'] or sources() != lock['inputs']:
        raise ValueError('Observation geometry or public inputs changed')
    with np.load(path) as saved:
        people = saved['chronological_readers'].tolist()
        other = dict(zip(people, saved['chronological_null'].tolist(), strict=True))
    groups, _ = exposures(get_geometry()[0], idx)
    requests = defaultdict(list)
    for r in people:
        past, _, _ = groups['chronological'][r]
        for order, (key, field, question, item) in enumerate(past):
            requests[key].append((r, order, field, question, item))
    normalize = answer_normalizer()
    decoded = defaultdict(list)
    for f, k, row, _ in records():
        for r, order, field, question, item in requests.get((f,k), ()):
            choice, _ = normalize(row[field], question)
            if choice is not None:
                decoded[r].append((order, item, choice))
    out = np.array([(r, item, choice) for r in people for _,item,choice in sorted(decoded[r])[:20]])
    assert len(out) == 20*len(people)
    return out, other


def calibrate(log_odds, flag):
    x = np.column_stack((np.ones(len(flag)),log_odds))
    center = np.array([0.,1.])
    def loss(theta):
        v = x @ theta
        residual = expit(v)-flag
        diff = theta-center
        return float(np.sum(np.logaddexp(0., v)-flag*v) + diff @ diff / 32), x.T @ residual + diff / 16
    ans = minimize(loss, center, jac=True, method='L-BFGS-B',
                   options={'maxiter':150, 'gtol':1e-8, 'ftol':1e-14})
    grad = np.max(abs(loss(ans.x)[1])) / len(flag)
    if not np.isfinite(ans.fun) or grad > 1e-7:
        raise ValueError(f'Peer class calibration failed, residual {grad}')
    return ans.x, float(grad)


def population_weights(observations, readers):
    """Solve the convex marginal-likelihood problem on the open simplex."""
    groups, inverse = np.unique(readers, return_inverse=True)
    k = observations.shape[1]
    evidence = np.zeros((len(groups), k))
    np.add.at(evidence, inverse, observations)
    evidence = np.exp(evidence-evidence.max(axis=1,keepdims=True))
    eta = 1/k
    def loss(p):
        return -np.log(evidence @ p).sum() - eta*np.log(p).sum()
    p = np.full(k,1/k)
    # Columns span the tangent space of the simplex exactly.
    tangent = np.vstack((np.eye(k-1), -np.ones(k-1)))
    for iteration in range(100):
        rate = evidence / (evidence @ p)[:,None]
        grad = -rate.sum(axis=0) - eta/p
        kkt = np.max(np.abs(grad-grad.mean()))/(len(groups)+1)
        if kkt < 1e-7:
            return p, dict(stationarity=float(kkt), steps=iteration)
        hessian = rate.T @ rate + np.diag(eta/p**2)
        direction = -tangent @ np.linalg.solve(tangent.T @ hessian @ tangent, tangent.T @ grad)
        scale = min(1., 0.99*np.min(-p[direction<0]/direction[direction<0]))
        objective = loss(p)
        for _ in range(50):
            candidate = p+scale*direction
            if np.all(candidate>0) and loss(candidate) <= objective + min(0.,1e-4*scale*(grad @ direction)) + 1e-12:
                p = candidate
                break
            scale /= 2
        else:
            raise ValueError('Peer prior failed sufficient descent')
    raise ValueError(f'Peer prior did not satisfy KKT, residual {kkt}')


def binary(logodds, coeff):
    v = coeff[0] + coeff[1]*logodds[:,None] + OFFSETS
    return -np.logaddexp(0.,-v), -np.logaddexp(0.,v)


def fit(native):
    prior_lock = read_json(OUT/'peer-registration.json')
    if prior_lock['protocol'] != digest(PROTOCOL):
        raise ValueError('Changed peers protocol')
    lp, idx, _ = base(native)
    logodds, zn = conditionals(lp)
    hist, other = histories(idx)
    subjects = sorted(other)
    byid = {v:k for k,v in idx.items()}
    sf = np.array([fold('sentence',byid[i]) for i in range(len(lp))])
    rf = {r:fold('reader',r) for r in subjects}
    r, item, w = hist.T
    flag = w == lp.shape[1] - 1
    is_named = ~flag
    named = lp[item,w,None].astype(float)*BETA - zn[item]
    records_fit = []
    for a in range(5):
        available = np.array([rf[i] != a for i in r])
        for b in range(5):
            use = available & (sf[item] != b)
            theta, grad = calibrate(logodds[item[use]],flag[use])
            yes, no = binary(logodds,theta)
            z = np.where(flag[:,None],yes[item],no[item])
            pc, mc = population_weights(z[use],r[use])
            use_name = use & is_named
            pn, mn = population_weights(named[use_name],r[use_name])
            post = {}
            for s in subjects:
                h = r == s
                post[s] = dict(cls=softmax(np.log(pc)+z[h].sum(0)).tolist(),
                               named=softmax(np.log(pn)+named[h & is_named].sum(0)).tolist())
            records_fit.append(dict(reader_fold=a, stem_fold=b, people_fit=len(np.unique(r[use])),
                                past_choices_fit=int(use.sum()), named_choices_fit=int(use_name.sum()),
                                class_kkt=grad, class_prior_kkt=mc['stationarity'],
                                named_prior_kkt=mn['stationarity'], class_coeff=theta.tolist(),
                                class_prior=pc.tolist(), named_prior=pn.tolist(), post=post))
    file=DIR/'peer-fit-parameters.json'
    write_json(file, records_fit)
    auxiliary = DIR/'peer-future-geometry.npz'
    np.savez_compressed(auxiliary, stem=sf, subjects=subjects, reader=[rf[r] for r in subjects],
                        null=[other[s] for s in subjects])
    write_json(OUT/'peer-prior-lock.json', dict(source=prior_lock, hist=digest(DIR/'independent-reader-posteriors.npz'),
               weights=digest(file), geometry=digest(auxiliary), models=len(records_fit),
               first_page_personal_choices=len(hist), held_future_fit_choices=0,
               max_gradient=max(max(s['class_kkt'],s['class_prior_kkt'],s['named_prior_kkt']) for s in records_fit),
               min_peer_people=min(s['people_fit'] for s in records_fit), max_peer_people=max(s['people_fit'] for s in records_fit)))


def score(native):
    lock=read_json(OUT/'peer-prior-lock.json')
    params=DIR/'peer-fit-parameters.json'
    auxiliary=DIR/'peer-future-geometry.npz'
    future=DIR/'independent-future-items.json'
    if (digest(params)!=lock['weights'] or digest(auxiliary)!=lock['geometry'] or
        digest(future)!=read_json(OUT/'external-lock.json')['future_item_assignments']):
        raise ValueError('Changed peer forecasts or future field geometry')
    model=read_json(params)
    lp, idx, _=base(native)
    odds, zn=conditionals(lp)
    future_fields=defaultdict(list)
    for r,f,k,col,q,i in read_json(future)['chronological']:
        future_fields[f,k].append((r,col,q,i))
    normalize=answer_normalizer()
    ev=[]
    for f,k,row,_ in records():
        for r,col,q,i in future_fields.get((f,k),()):
            w,_=normalize(row[col],q)
            if w is not None:ev.append((r,i,w))
    ev=np.asarray(ev)
    r,i,w=ev.T
    with np.load(auxiliary) as f:
        sf=f['stem'][i]
        rf=dict(zip(f['subjects'],f['reader'],strict=True))
        other=dict(zip(f['subjects'].tolist(),f['null'].tolist(),strict=True))
    rs=np.array([rf[q] for q in r]); outside=w==lp.shape[1]-1
    scores=np.full((len(ev),4),np.nan);pouts=np.full_like(scores,np.nan)
    for m in model:
        yes, no=binary(odds,np.asarray(m['class_coeff']))
        for person in set(r[rs==m['reader_fold']]):
            sel=np.flatnonzero((r==person)&(sf==m['stem_fold']))
            if not len(sel):continue
            post=m['post'][str(person)];ctrl=m['post'][str(other[person])]
            c=np.stack((post['cls'],m['class_prior'],ctrl['cls']))
            n=np.stack((post['named'],m['named_prior'],ctrl['named']))
            x=i[sel];y=w[sel]
            q=np.exp(yes[x])@c.T
            named=lp[x,y,None].astype(float)*BETA-zn[x]
            conditional=logsumexp(named[:,:,None]+np.log(n.T)[None,:,:],axis=1)
            scores[sel,:3]=-np.where(outside[sel,None],np.log(q),np.log1p(-q)+conditional)
            pouts[sel,:3]=q
    scores[:,3]=-lp[i,w];pouts[:,3]=np.exp(lp[i,-1])
    if not np.isfinite(scores).all() or np.any((pouts<=0)|(pouts>=1)):
        raise ValueError('Incomplete or invalid peer prediction')
    raw=read_json(OUT/'external-scores.json')['study']['chronological']
    result=report(ev,scores,pouts)
    if result['scored_responses']!=raw['scored_responses']:
        raise ValueError('Independent chronology changed')
    result['choice_content']=decompose(ev,scores,pouts)
    result['median_person_nats_gain_over_other']=float(np.median([np.mean((scores[:,2]-scores[:,0])[r==v]) for v in set(r)]))
    result['zero_shot_NLL']=raw['NLL']
    write_json(OUT/'peer-scores.json',dict(results=result,locked=digest(OUT/'peer-prior-lock.json')))
    print(result['paired'],flush=True)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('action',choices=['fit','score']);p.add_argument('native',type=Path)
    args=p.parse_args();(fit if args.action=='fit' else score)(args.native)
