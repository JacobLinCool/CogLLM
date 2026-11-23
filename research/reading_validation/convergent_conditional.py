"""Estimate held-sentence conditional association after lexical controls and published surprisal."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import digest,read_json,write_json
from research.reading_validation.convergent_word_score import OUT,assemble,fit,predict,correlations
from research.reading_validation.meaning_score import folds


def cdf(train,x):
    ref=np.sort(train);return (np.searchsorted(ref,x,'left')+np.searchsorted(ref,x,'right'))/(2*len(ref))


def main(folder):
    (S,Y,w,E,H,c,pub,freq),_=assemble(folder);j=read_json(OUT/'scores.json');sents=c[:,0];split=folds(sents,'UCL 2026 reading sentence:');fold=np.array([split[v] for v in sents]);src=j['external_spanish_no_updates']['spanish_coefficients']
    source=predict(E,src['context'])[:,0];surface=np.c_[E[:,:5],freq,np.isnan(freq)];designs={'surface_frequency':surface,'plus_native_surprisal':np.c_[surface,E[:,5]],'plus_native_and_published_surprisal':np.c_[surface,E[:,5],pub[:,3]]}
    corr={}
    for name,Z in designs.items():
        scores=np.zeros(len(E));truth=np.zeros((len(E),3))
        for f in range(5):
            tr=fold!=f;te=fold==f;D=Z.copy();D[:,5]=np.nan_to_num(D[:,5],nan=np.nanmedian(D[tr,5]))
            u=np.column_stack([cdf(D[tr,i],D[:,i]) for i in range(D.shape[1])]);R=np.column_stack([source,H]);r=np.column_stack([cdf(R[tr,i],R[:,i]) for i in range(4)])
            A=np.c_[np.ones(len(u)),u];beta=np.linalg.lstsq(A[tr],r[tr],rcond=1e-10)[0];res=r[te]-A[te]@beta;scores[te]=res[:,0];truth[te]=res[:,1:]
        out=correlations({'source_N400_residual':scores},truth,sents)['source_N400_residual'][0]
        corr[name]=out
    write_json(OUT/'conditional-associations.json',dict(scores=digest(OUT/'scores.json'),directions=j['outcomes'],residuals='Training-fold empirical midrank transforms; nuisance coefficients estimated only on other sentences.',correlations=corr))

if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);main(p.parse_args().folder)
