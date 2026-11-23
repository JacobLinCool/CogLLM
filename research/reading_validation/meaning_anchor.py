"""Cross-fit target-specific lexical support against delayed human meaning judgments."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.meaning_score import assemble,crossfit,designs,outcomes,rank_corr
from research.reading_validation.meaning_data import DIR,OUT
from research.reading_validation.identity_causal_score import confidence


def main(folder):
    cases=read_json(DIR/'cases.json');rows,_=outcomes(cases);surface,sem,prior,_=assemble(folder,cases)
    vocab=read_json(DIR/'vocabulary.json')['onsets'];z=np.where(np.isfinite(prior),prior,-np.log(len(vocab)));miss=~np.isfinite(prior)
    des=designs(cases,surface,sem);repeat=lambda x:np.repeat(np.asarray(x),2,axis=0)
    support=repeat(np.c_[z,miss]);aug=repeat(np.c_[z,miss,z*surface[:,3]])
    X={'support':support,'locus':np.c_[des['surface'],aug],'offline':np.c_[des['design_surface'],aug], 'oracle_support':np.c_[des['oracle'],aug]}
    pred,checks=crossfit(cases,rows,X);original=DIR/'human-crossfit-predictions.npz'
    primary=read_json(OUT/'forecast-receipt.json')
    if digest(original)!=primary['forecasts']:raise ValueError('Original forecasts have changed')
    with np.load(original) as archived:pred.update({k:archived[k] for k in ('surface','raw','known','native','design_surface','oracle','constant')})
    ix,people,practice,y=rows.T;truth=y>=4;g=np.asarray([c['word'] for c in cases])[ix];fit=np.asarray([c['fit'] for c in cases]);ids=np.unique(ix)
    estimates={};loss={};obs=y[:,None]>np.arange(1,6)
    for arm,P in pred.items():
        q=P[:,2].clip(1e-9,1-1e-9);loss[arm]=np.c_[-(truth*np.log(q)+~truth*np.log1p(-q)),((P-obs)**2).mean(1)]
        e=np.array([truth[ix==k].mean() for k in ids]);pr=np.array([q[ix==k].mean() for k in ids]);isfit=fit[ids]
        estimates[arm]=dict(nll=float(loss[arm][:,0].mean()),rps=float(loss[arm][:,1].mean()),accuracy=float(np.mean((q>=.5)==truth)),item_rho=rank_corr(pr,e),within_fit_rho=rank_corr(pr[isfit],e[isfit]),within_nofit_rho=rank_corr(pr[~isfit],e[~isfit]))
    comparisons={}
    for a,b in [('support','surface'),('support','constant'),('support','native'),('locus','surface'),('locus','raw'),('offline','design_surface'),('oracle_support','oracle')]:
        d=loss[b]-loss[a];comparisons[a+'__over__'+b]={metric:{axis:confidence([v[groups==gid].mean() for gid in np.unique(groups)]) for axis,groups in [('word',g),('reader',people)]} for v,metric in zip(d.T,('NLL','RPS'),strict=True)}
    path=DIR/'anchor-crossfit.npz';np.savez_compressed(path,**{a:pred[a] for a in X});write_json(OUT/'anchor-forecasts.json',dict(protocol=digest(Path(__file__).with_name('meaning-anchor-protocol.md')),model=digest(Path(__file__)),data=digest(DIR/'cases.json'),reader_word_folds=checks,sha256=digest(path)))
    write_json(OUT/'anchor-scores.json',dict(estimates=estimates,comparisons=comparisons,matched_trials=len(rows),words=len(set(g)),readers=len(set(people)),missing_prefixes=int(miss.sum())))

if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);main(p.parse_args().folder)
