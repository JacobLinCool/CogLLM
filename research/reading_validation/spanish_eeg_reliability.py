"""Examine independent-reader reproducibility of the measured target-family contrasts."""
import numpy as np
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.spanish_clue_data import OUT
from research.reading_validation.spanish_eeg_score import buffer,rho
from pathlib import Path


def main():
    protocol=Path(__file__).with_name('spanish-eeg-reliability.md');fam=read_json(OUT/'cue-scores.json')['families'];Y=buffer(fam,'valid');n=len(Y);rng=np.random.default_rng(9022);splits=np.array([rng.permutation(n) for _ in range(2000)]);results={}
    for current,window,name in [(17,0,'category_N400'),(6,0,'before_clue_N400'),(17,1,'early_category')]:
        X=Y[:,:,:,current,window];ok=np.isfinite(X);Q=np.nan_to_num(X)
        corr=[];counts=[]
        for s in splits:
            diff=[]
            for group in (s[:n//2],s[n//2:]):
                cnt=ok[group].sum(0);sums=np.sum(Q[group],axis=0);v=np.divide(sums,cnt,out=np.full_like(sums,np.nan),where=cnt>=3);diff.append(v[:,1]-v[:,0])
            a,b=diff;good=np.isfinite(a)&np.isfinite(b);corr.append(rho(a[good],b[good]));counts.append(int(good.sum()))
        m=float(np.nanmedian(corr));results[name]={'median_split_half_rho':m,'split_sensitivity_2.5_97.5':np.nanquantile(corr,[.025,.975]).tolist(),'median_retained_families':float(np.median(counts)),'fraction_positive':float(np.mean(np.asarray(corr)>0)),'spearman_brown_of_median':2*m/(1+m)}
    write_json(OUT/'eeg-item-reliability.json',dict(primary=digest(OUT/'eeg-scores.json'),protocol=digest(protocol),splits=2000,estimates=results))

if __name__=='__main__':main()
