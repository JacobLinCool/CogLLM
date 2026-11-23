"""Compare lexical probability and alternative-meaning distance against observed N400."""
import numpy as np
from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.reading_validation.convergent_word_score import corr
from research.reading_validation.multilab_materials import ROOT as LABS
from research.reading_validation.multilab_score import unpack, human, summary, ratio
from research.reading_validation.physiology_portability_score import fast_ranks, R
from research.reading_validation.semantic_expectation_fit import RESULT, PLAN, NAMES


def estimate(value, bootstrap):
    return {"estimate": float(value), "interval": summary(bootstrap)}


def english(P, y, sent):
    groups, group = np.unique(sent, return_inverse=True)
    rng = np.random.default_rng(42062)
    rep = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=2500)
    ranks = [fast_ranks(x) for x in P.T]
    gold = fast_ranks(y)
    full = np.ones(len(y))
    sq = (P - y[:,None]) ** 2
    rho = np.zeros((len(rep)+1, len(NAMES)))
    mse = np.zeros_like(rho)
    for i in range(len(rho)):
        weight = full if i == 0 else rep[i-1][group]
        ry = gold(weight)
        rho[i] = [R(r(weight), ry, weight) for r in ranks]
        mse[i] = (weight @ sq) / weight.sum()
    def summarise(arr):
        return {name:estimate(arr[0,j], arr[1:,j]) for j,name in enumerate(NAMES)}
    compare = {}
    for later in range(1,len(NAMES)):
        for earlier in range(later):
            gain = rho[:,later]-rho[:,earlier]
            reduction = mse[:,earlier]-mse[:,later]
            compare[NAMES[later]+"_over_"+NAMES[earlier]] = {
                "rho_gain":estimate(gain[0],gain[1:]),
                "MSE_reduction":estimate(reduction[0],reduction[1:])}
    return {"words": len(y), "sentences": len(groups), "rho": summarise(rho),
            "MSE": summarise(mse), "paired":compare}


def nine(arc, ordered):
    _, _, obs, _, sl, items, coverage, detail = unpack()
    mapping = list(stream_rows(LABS/"conditions.jsonl"))
    if [r["id"] for r in mapping] != ordered:
        raise ValueError("Source and native forecast orders differ")
    human_mean = human(obs,np.ones((1,len(sl))))[0]
    observed = human_mean[:,1] - human_mean[:,0]
    N = len(items)
    if N != 80:
        raise ValueError("Missing original item families")
    rng=np.random.default_rng(91402)
    I = rng.multinomial(N,np.full(N, 1/N),size=3000)
    labs = np.unique(sl)
    site = rng.multinomial(len(labs),np.full(len(labs),1/len(labs)),size=len(I))
    W = np.zeros((len(I), len(sl)))
    for k in labs:
        ids = np.flatnonzero(sl == k)
        W[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=len(I))*site[:,k,None]
    hc=human(obs,W)
    D=hc[:,:,1]-hc[:,:,0]
    valid=np.isfinite(D).all(axis=(1,2))
    if valid.mean()<.98:
        raise ValueError("Insufficient measurement overlap across resampled sites")
    D,I=D[valid],I[valid]
    def summarize(name):
        p=np.full((N,2,2),np.nan)
        for r,row in zip(mapping,arc[name], strict=True):
            p[items.index(r['item']),int(r['condition']=='unexpected')]=row
        if not np.isfinite(p).all():
            raise ValueError("Missing forecast at an original printed word")
        contrasts=p[:,1]-p[:,0]
        avg=contrasts.mean(0)
        bs=I@contrasts/N
        point=np.asarray([corr(contrasts[:,j],observed[:,j]) for j in range(2)])
        rs=np.asarray([[corr(contrasts[:,j],d[:,j],iw) for j in range(2)]
                       for d,iw in zip(D[::3],I[::3])])
        return {"unexpected_minus_expected":{"article":estimate(avg[0],bs[:,0]),"noun":estimate(avg[1],bs[:,1])},
                "noun_minus_article":estimate(avg[1]-avg[0],bs[:,1]-bs[:,0]),
                "article_over_noun":estimate(ratio(avg),ratio(bs)),
                "item_rho":{"article":estimate(point[0],rs[:,0]), "noun":estimate(point[1],rs[:,1])}},rs,point
    stats={};boot={};point={}
    for name in NAMES:
        stats[name],boot[name],point[name]=summarize(name)
    paired={}
    for j,new in enumerate(NAMES):
        for old in NAMES[:j]:
            dd=boot[new]-boot[old];diff=point[new]-point[old]
            paired[new+"_over_"+old]={k:estimate(diff[t],dd[:,t]) for t,k in enumerate(("article","noun"))}
    return {"families":N,"labs":len(labs),"readers":detail['subjects'],"trials":coverage,
            "crossed_design_resamples":len(I),"model":stats,"paired_item_rho":paired,
            "observed_mean_signed_microvolts":observed.mean(0).tolist()}


def main():
    lock=read_json(RESULT/'forecast-lock.json')
    path=RESULT/'forecasts.npz'
    if lock['protocol']!=digest(PLAN) or digest(path)!=lock['files'][path.name]:
        raise ValueError("Physiological forecasts changed after registration")
    if [digest(RESULT/f'fold-{k}.json') for k in range(6)]!=lock['folds']:
        raise ValueError("Source models changed after sealing")
    with np.load(path) as arc:
        eng=english(arc['english'],arc['human'],arc['sent'])
        independent=nine(arc,lock['labs_order'])
    result={"forecast_lock":digest(RESULT/'forecast-lock.json'),
            "english":eng,"nine_laboratory":independent}
    write_json(RESULT/'scores.json',result)
    print("English held N400 correlations",eng['rho'])
    print("Independent noun item association", {n:independent['model'][n]['item_rho']['noun'] for n in NAMES})


if __name__ == '__main__':
    main()
