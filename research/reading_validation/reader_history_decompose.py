"""Decompose personal information about response class and exact named words."""
import numpy as np
from scipy.sparse import coo_matrix
from research.reading_validation.reader_history_model import NAMES


def decompose(ev,loss,qother,groups=None,draws=1800):
    outside=(ev[:,2]==8192)
    mask=~outside
    p=np.clip(qother,1e-12,1-1e-12)
    binary=-np.where(outside[:,None],np.log(p),np.log1p(-p))
    named=np.where(mask[:,None],loss+np.log1p(-p),0.)
    if abs(binary+named-loss).max()>2e-5:
        raise ValueError("Distribution does not factorize into class and named choice")
    r,ri=np.unique(ev[:,0],return_inverse=True)
    g,gi=np.unique(ev[:,1] if groups is None else groups[ev[:,1]],return_inverse=True)
    R,G=len(r),len(g)
    rng=np.random.default_rng(76391);sr=rng.multinomial(R,np.full(R,1/R),size=draws);sw=rng.multinomial(G,np.full(G,1/G),size=draws)
    def agg(x):
        a=coo_matrix((x,(ri,gi)),shape=(R,G)).tocsr();v=[]
        for i in range(0,len(sr),50):
            v.extend(np.sum(sr[i:i+50]*(a@sw[i:i+50].T).T,axis=1))
        return np.asarray(v)
    n=agg(np.ones(len(ev)));nname=agg(mask.astype(float));out={}
    for j in [1,2,3]:
        nb=binary[:,j]-binary[:,0];nw=named[:,j]-named[:,0]
        if abs(nb.mean()+nw.mean()-(loss[:,j]-loss[:,0]).mean())>1e-7:raise ValueError("Invalid score decomposition")
        b=agg(nb);w=agg(nw)
        out['same_over_'+NAMES[j]]={
             'class_nats_per_answer':float(nb.mean()),'class_interval':np.quantile(b/n,[.025,.975]).tolist(),
             'named_identity_nats_per_named_choice':float(nw.sum()/mask.sum()),'named_interval':np.quantile(w/nname,[.025,.975]).tolist(),
             'named_identity_contribution_per_answer':float(nw.mean())}
    return dict(named_responses=int(mask.sum()),other_responses=int(outside.sum()),paired=out)


if __name__=='__main__':
    from research.backbone_transfer.extract import write_json,digest
    from research.reading_validation.reader_history_materials import OUT,DIR
    from research.reading_validation.cloze_population_fit import data
    from argparse import ArgumentParser
    from pathlib import Path
    q=ArgumentParser();q.add_argument('native',type=Path)
    *_,meta=data(q.parse_args().native);g=meta['articles'][meta['source']]
    tables=[]
    for k in range(5):
        with np.load(DIR/f'fold-{k}-readouts.npz') as f:tables.append([f[x] for x in ['event','NLL','other']])
    E,L,O=[np.vstack(x) for x in zip(*tables)]
    write_json(OUT/'source-choice-content.json',dict(source=digest(OUT/'source-lock.json'),analysis=decompose(E,L,O,g)))
