"""Check prediction causality, reference recovery and the analytical joint training gradient."""

from pathlib import Path

import numpy as np
from scipy.optimize._numdiff import approx_derivative

from research.backbone_transfer.extract import write_json
from research.reading_validation.definition_data import OUT
from research.reading_validation.definition_features import Page
from research.reading_validation.definition_fit import ARMS
from research.reading_validation.definition_joint import full_loss
from research.reading_validation.definition_predict import get_forecasts


def main():
    rng=np.random.default_rng(194819);nt,nl=5,7
    a=rng.normal(size=(70,nt)).astype('float32');b=rng.normal(size=(90,nl)).astype('float32')
    pairs=[rng.normal(size=(45,d)).astype('float32') for d in [nt,nt,nl,nl]]
    data=a,np.ones(len(a),dtype='float32')/len(a),b,np.ones(len(b),dtype='float32')/len(b),pairs,np.ones(45,dtype='float32')/45
    v=rng.normal(size=nt+nl)*.2;old=rng.normal(size=nt+nl)*.3
    finite=approx_derivative(lambda x:np.array([full_loss(x,old,None,data,.15)[0]]),v,method='3-point',abs_step=.002).ravel()
    gradient=float(np.abs(full_loss(v,old,None,data,.15)[1]-finite).max())
    if gradient>1e-4:raise ArithmeticError('Joint objective gradient is inconsistent')
    words='One novel term describes how hot water moves in winter'.split();T=len(words);off=[];cursor=0
    for i,word in enumerate(words):off.append([cursor,cursor+len(word),i,i]);cursor+=len(word)+1
    states=rng.normal(size=(T,6)).astype('float32');attn=rng.integers(0,77,size=(T,T,2),dtype=np.uint8)
    p=Page(' '.join(words),off,states,attn);stop=6;q=Page(' '.join(words[:stop]),off[:stop],states[:stop],attn[:stop,:stop])
    models={'term':{},'link':{}}
    for m in (*ARMS,'joint'):
        arm='semantic' if m=='joint' else m
        l,r=p.grid();term=p.span_features(l[:1],r[:1],'surface' if arm=='position' else arm)
        shapes=[4 if arm=='position' else term.shape[1],p.link_features((1,2),[4,8],arm).shape[1]]
        for name,d in zip(('term','link'),shapes,strict=True):
            models[name][m]={'bias':-.3,'weights':(rng.normal(size=d)*.2).tolist(),'threshold':0.0}
    full=get_forecasts(p,models,{'term':[off[1][0],off[2][1]]});prefix=get_forecasts(q,models,{'term':[off[1][0],off[2][1]]})
    errors={};stable={}
    for m in (*ARMS,'joint'):
        errors[m]=float(np.max(np.abs(full[m+'_qprob'][:,:stop]-prefix[m+'_qprob'])))
        stable[m]=np.array_equal(full[m+'_top'][:stop],prefix[m+'_top'])
    if max(errors.values())>1e-6 or not all(stable.values()):raise ValueError('Changing the continuation changed an existing prefix forecast')
    x={'joint_objective_max_abs_gradient_difference':gradient,'prefix_probability_max_error':errors,'invariant_article_only_source_ids':{k:bool(v) for k,v in stable.items()}}
    write_json(OUT/'simulation.json',x);print(x)


if __name__=='__main__':main()
