"""Adapt a shared/private density pool to the support of each measured reading opportunity."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit, logsumexp
from scipy.spatial import cKDTree
import torch

from research.reading_validation.borrowing_data import OUTPUT, CALIBRATE, EVALUATE, FIT, data
from research.reading_validation.borrowing_score import GeometricPool, material_weights
from research.reading_validation.memory_data import sha
from research.union_training.fit_observations import COUNT_MODELS, weights

OUT = Path("research/results/reading-validation-support-borrowing")
PLAN = Path(__file__).with_name("support-borrowing-protocol.md")


def unit(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm>0)


def centroids(cache, tasks, split):
    exposure = defaultdict(set)
    permissions = defaultdict(set)
    for task in tasks:
        take = np.flatnonzero(split[task.name] == FIT)
        permissions[task.source].update(task.states)
        for family, stops in zip(task.family[take], task.coordinates[take].numpy(), strict=True):
            exposure[task.source, family].update(stops[stops>=0].tolist())
    banks = defaultdict(list)
    for (source, _), ids in sorted(exposure.items()):
        banks[source].append(cache[np.asarray(sorted(ids))].mean(axis=0))
    return {k:unit(np.stack(x)) for k,x in banks.items()}, permissions


def opportunity(cache, locations):
    vectors = []
    for block in np.array_split(locations, max(1,(len(locations)+255)//256)):
        present = block>=0
        if np.any(present.sum(axis=1)==0):
            raise ValueError("A response lacks all eligible original coordinates.")
        vectors.append(unit((cache[block.clip(min=0)] * present[...,None]).sum(axis=1) / present.sum(axis=1)[:,None]))
    return np.vstack(vectors)


def distance(query, tree):
    k = min(tree.n, 3)
    d, _ = tree.query(query, k=list(range(1,k+1)), workers=1)
    return .5*np.square(d).mean(axis=1)


def geometry():
    cache, tasks, parts = data()
    matrix = unit(cache.numpy())
    banks, permissions = centroids(matrix, tasks, parts)
    own = {k:cKDTree(v) for k,v in banks.items()}
    others = {}
    outputs = {}; evidence = {}
    for task in tasks:
        compatible = tuple(sorted(s for s in banks if s!=task.source and permissions[s].intersection(task.states)))
        if compatible and compatible not in others:
            others[compatible] = cKDTree(np.vstack([banks[s] for s in compatible]))
        regions = {}
        for split, label in ((CALIBRATE,"cal"),(EVALUATE,"test")):
            ids = np.flatnonzero(parts[task.name] == split)
            if len(ids):
                if compatible:
                    state = opportunity(matrix,task.coordinates[ids].numpy())
                    regions[label] = distance(state,own[task.source])-distance(state,others[compatible])
                else:
                    regions[label] = np.zeros(len(ids))
        if "cal" not in regions:
            if "test" in regions:
                raise ValueError("A scored endpoint lacks source support calibration.")
            continue
        center = np.mean(regions["cal"])
        spread = np.std(regions["cal"])
        for label, row in regions.items():
            outputs[task.name+"."+label] = ((row-center)/spread).clip(-4,4) if spread>1e-9 else np.zeros_like(row)
        evidence[task.name] = dict(source=task.source, permitted_other_sources=list(compatible),
                                   own_families=len(banks[task.source]), center=float(center), spread=float(spread))
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/"support.npz", **outputs)
    (OUT/"geometry.json").write_text(json.dumps(dict(protocol=sha(PLAN),support=sha(OUT/"support.npz"),
                                      sources={k:len(v) for k,v in banks.items()}, endpoints=evidence),indent=2)+"\n")


class VaryingPool(GeometricPool):
    def per_row(self, a):
        if a.shape!=(len(self.y),):
            raise ValueError("The gate must produce exactly one weight per reading opportunity.")
        if self.kind in COUNT_MODELS:
            logp = self.left + a[:,None] * self.delta
            logp -= logsumexp(logp,axis=1,keepdims=True)
            return -(self.y*np.where(self.y>0,logp,0)).sum(axis=1)
        if self.kind=="paired_gaussian":
            p = self.p0 + a[:,None,None]*(self.p1-self.p0)
            eta = self.eta0 + a[:,None]*(self.eta1-self.eta0)
            mu = np.linalg.solve(p,eta[:,:,None]).squeeze(2)
            delta = self.y-mu
            return np.log(2*np.pi)-.5*np.linalg.slogdet(p)[1]+.5*np.einsum("bi,bij,bj->b",delta,p,delta)
        p = self.p0 + a*(self.p1-self.p0)
        eta = self.eta0 + a*(self.eta1-self.eta0)
        if self.kind=="gaussian":
            n,y,within=self.y.T
        else:
            n,y,within=1.,self.y[:,0],0.
        return .5*n*(np.log(2*np.pi)-np.log(p)+p*((y-eta/p)**2+within))


def collect(seed, partition):
    _,tasks,parts=data()
    path=OUTPUT/f"seed-{seed}"
    fit=json.loads((path/"fit.json").read_bytes())
    lock=json.loads((path/"borrowing-lock.json").read_bytes())
    tag,region=("calibration","cal") if partition==CALIBRATE else ("development","test")
    if sha(OUT/"support.npz")!=json.loads((OUT/"geometry.json").read_bytes())["support"]:
        raise ValueError("Geometry changed after support extraction.")
    files=[f"{arm}-{tag}.npz" for arm in ("shared","private")]
    if any(sha(path/f)!=fit["files"][f] for f in files):
        raise ValueError("The comparison experts differ from the original forecasts.")
    result=defaultdict(list)
    with np.load(OUT/"support.npz") as support,np.load(path/files[0]) as shared,np.load(path/files[1]) as private:
        for t in tasks:
            ids=np.flatnonzero(parts[t.name]==partition)
            if not len(ids):continue
            pool=VaryingPool(t,ids,shared[t.name],private[t.name])
            a=lock["coefficients"][t.name]["weight"]
            const=pool.per_row(np.full(len(ids),a))
            if not np.allclose(const,pool.loss(a),rtol=1e-9,atol=1e-9):
                raise ValueError(f"Scalar and per-opportunity likelihoods differ: {t.name}")
            z=support[t.name+"."+region]
            if partition==CALIBRATE:
                w, nf=material_weights(t,ids)
                w/=nf
            else:
                w=weights(t,ids)
                w=w/np.dot(w,t.trials[ids])
                nf=len(set(t.family[ids]))
            result[t.source].append((t.name,pool,logit(a),z,w,nf))
    return result


def risk(measures, slope):
    return np.mean([w@p.per_row(expit(intercept+slope*z)) for _,p,intercept,z,w,_ in measures])


def estimate(seed):
    models=collect(seed,CALIBRATE)
    folder=OUT/f"seed-{seed}";folder.mkdir(exist_ok=True)
    # Cache by slope because all target exclusions share the same source contributions.
    cache={}
    def values(b):
        if b not in cache:
            cache[b]={s:min(32,np.mean([nf for *_,nf in tasks]))*risk(tasks,b) for s,tasks in models.items()}
        return cache[b]
    coef={}
    for excluded in models:
        def objective(b):
            return .5*b*b+sum(v for s,v in values(float(b)).items() if s!=excluded)
        opt=minimize_scalar(objective,method="bounded",bounds=(0,8),options={"xatol":2e-8})
        if not opt.success or opt.x>7.99:raise ArithmeticError("Support gate failed its finite optimum check")
        b=float(opt.x) if opt.fun < objective(0.)-1e-11 else 0.
        coef[excluded]=dict(slope=b,source_calibration_studies=len(models)-1,source_objective=objective(b))
    receipt=dict(seed=seed,protocol=sha(PLAN),geometry=sha(OUT/"geometry.json"),
                 original=sha(OUTPUT/f"seed-{seed}"/"borrowing-lock.json"),models=coef)
    (folder/"forecast.json").write_text(json.dumps(receipt,indent=2)+"\n")


def score(seed):
    folder=OUT/f"seed-{seed}"
    receipt=json.loads((folder/"forecast.json").read_bytes())
    if receipt["protocol"]!=sha(PLAN) or receipt["geometry"]!=sha(OUT/"geometry.json"):
        raise ValueError("Support gate changed after locking the held predictions.")
    original=json.loads((OUTPUT/f"seed-{seed}"/"scores.json").read_bytes())
    models=collect(seed,EVALUATE);scores={}
    for s,tasks in models.items():
        expected=original["sources"][s]["adaptive"]
        fixed=risk(tasks,0.)
        if abs(fixed-expected)>1e-7:raise ValueError("Reference endpoint coefficient does not reproduce.")
        b=receipt["models"][s]["slope"]
        scores[s]={**original["sources"][s],"support":float(risk(tasks,b)),"slope":b}
    generators=np.random.default_rng(4173).integers(0,len(scores),(4000,len(scores)))
    averages={k:np.mean([r[k] for r in scores.values()]) for k in ("private","shared","midpoint","adaptive","support")}
    contrasts={}
    for key in ("private","shared","midpoint","adaptive"):
        d=np.array([r["support"]-r[key] for r in scores.values()])
        contrasts[key]=dict(change=float(d.mean()),interval95=np.quantile(d[generators].mean(1),[.025,.975]).tolist(),
                            improved=int((d < -1e-10).sum()),unchanged=int((abs(d)<=1e-10).sum()))
    (folder/"scores.json").write_text(json.dumps(dict(forecast=sha(folder/"forecast.json"),studies=scores,
                                  mean=averages,contrasts=contrasts),indent=2,allow_nan=False)+"\n")
    print("Support-dependent minus endpoint-constant loss",contrasts["adaptive"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("geometry","fit","score"))
    parser.add_argument("--seed",type=int,default=1729)
    args=parser.parse_args(); torch.set_num_threads(1)
    if args.action=="geometry":geometry()
    elif args.action=="fit":estimate(args.seed)
    else:score(args.seed)
