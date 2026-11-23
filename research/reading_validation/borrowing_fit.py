"""Train sharing and source-private experts on identical material with disjoint early stopping."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from research.backbone_transfer.extract import digest
from research.reading_validation.borrowing_data import (OUTPUT, PACKET, CALIBRATE, EVALUATE, FIT, TUNE, data)
from research.union_training.fit_observations import Readouts, row_nll, weights


class Private(nn.Module):
    def __init__(self, shared, tasks):
        super().__init__()
        self.measurements = deepcopy(shared.measurements)
        self.sources = nn.ModuleDict({f'source_{source}': nn.ModuleDict({name: deepcopy(shared.shared[name])
                                     for name in sorted({s for t in tasks if t.source == source for s in t.states})})
                                     for source in sorted({t.source for t in tasks})})

    def forward(self, task, cache, rows):
        positions = task.coordinates[rows]
        states = cache[positions.clamp_min(0)]
        modules = self.sources[f'source_{task.source}']
        latent = torch.cat([modules[key](states).tanh() for key in task.states], dim=-1)
        return self.measurements[task.name](latent, positions >= 0)


@torch.no_grad()
def assessment(expert, tasks, cache, parts, partition, *, score=True):
    values, blocks = {}, {}
    for t in tasks:
        rows = np.flatnonzero(parts[t.name] == partition)
        if not len(rows):
            continue
        y = torch.cat([expert(t, cache, torch.from_numpy(chunk)) for chunk in np.array_split(rows, max(1, (len(rows)+511)//512))])
        if score:
            mass = weights(t, rows)
            loss = row_nll(t, y, t.response[rows]).numpy()
            values.setdefault(t.source, []).append(float(np.dot(loss, mass) / np.dot(t.trials[rows], mass)))
        blocks[t.name] = y.numpy()
    if not score:
        return None, blocks
    if not values:
        raise ValueError('No independent observations for evaluation')
    return float(np.mean([np.mean(v) for v in values.values()])), blocks


def train(expert, label, tasks, cache, parts, seed, out, *, penalty=None):
    batch_size, cap = 256, 1500
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(expert.parameters(), lr=.001, weight_decay=.01)
    sources = Counter(t.source for t in tasks)
    rows = [np.flatnonzero(parts[t.name] == FIT) for t in tasks]
    mass = [weights(t, r) for t,r in zip(tasks,rows,strict=True)]
    total = [np.dot(t.trials[r], w) for t,r,w in zip(tasks,rows,mass,strict=True)]
    best = deepcopy(expert.state_dict()); best_epoch = 0
    best_score, _ = assessment(expert, tasks, cache, parts, TUNE)
    reference = best_score
    history = [dict(epoch=0, inner_tuning=best_score)]
    start, stale, reductions = perf_counter(), 0, 0
    for epoch in range(1,cap+1):
        batches = []
        for i, r in enumerate(rows):
            order = rng.permutation(len(r))
            batches.extend((i, order[a:a+batch_size]) for a in range(0,len(r),batch_size))
        schedule = rng.permutation(len(batches))
        steps = (len(batches)+3)//4
        clips = 0
        for index in range(0,len(schedule),4):
            optimizer.zero_grad(set_to_none=True)
            loss = cache.new_zeros(())
            for j in schedule[index:index+4]:
                i, selection = batches[j]
                t, ids = tasks[i], torch.from_numpy(rows[i][selection])
                raw = expert(t,cache,ids)
                scale = steps / (len(sources)*sources[t.source]*total[i])
                loss = loss + scale*row_nll(t,raw,t.response[ids]).dot(torch.as_tensor(mass[i][selection],dtype=raw.dtype))
            if penalty is not None:
                loss = loss + penalty()
            loss.backward()
            grad = nn.utils.clip_grad_norm_(expert.parameters(),max_norm=5.,error_if_nonfinite=True)
            clips += int(grad>5.)
            optimizer.step()
        score,_ = assessment(expert,tasks,cache,parts,TUNE)
        history.append(dict(epoch=epoch,inner_tuning=score,lr=optimizer.param_groups[0]['lr'],clips=clips,updates=steps))
        if score < best_score:
            best_score,best_epoch,best = score,epoch,deepcopy(expert.state_dict())
        if score <= reference-.0001:
            reference,stale=score,0
        else:
            stale+=1
        if epoch%50==0 or stale==5:
            print(f'{label} epoch {epoch}, inner NLL {score:.6f}, best {best_epoch}',flush=True)
        if stale==5:
            if reductions==2:
                break
            optimizer.param_groups[0]['lr'] *= .5
            reductions+=1
            stale=0
    expert.load_state_dict(best)
    np.savez_compressed(out/f'{label}-parameters.npz',**{k:v.numpy() for k,v in best.items()})
    for split,name in ((CALIBRATE,'calibration'),(EVALUATE,'development')):
        _, forecasts = assessment(expert,tasks,cache,parts,split,score=False)
        np.savez_compressed(out/f'{label}-{name}.npz',**forecasts)
    return dict(history=history,selected_epoch=best_epoch,stop='inner_likelihood_plateau' if epoch<cap else 'budget',
                seconds=perf_counter()-start,trainable_parameters=sum(x.numel() for x in expert.parameters()))


def run(seed=1729):
    torch.set_num_threads(1);torch.manual_seed(seed)
    out=OUTPUT/f'seed-{seed}';out.mkdir(parents=True,exist_ok=True)
    cache,tasks,parts=data()
    shared=Readouts(cache.shape[1],tasks);private=Private(shared,tasks)
    with torch.no_grad():
        for t in tasks:
            ids=torch.as_tensor(np.flatnonzero(parts[t.name]==FIT)[:3]);a=shared(t,cache,ids);b=private(t,cache,ids)
            if not torch.equal(a,b):
                raise ValueError('Experts must start at identical predictions')
    results={label:train(expert,label,tasks,cache,parts,seed,out) for label,expert in [('shared',shared),('private',private)]}
    files={p.name:digest(p) for p in sorted(out.glob('*.npz'))}
    summary=dict(seed=seed,cache_shape=list(cache.shape),packet_sha256=digest(PACKET/'receipt.json'),
                 state_sha256=digest(PACKET/'states.npy'),results=results,files=files,
                 roles={t.name:np.bincount(parts[t.name],minlength=4).tolist() for t in tasks})
    (out/'fit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,default=1729)
    run(**vars(parser.parse_args()))
