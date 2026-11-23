"""Read three population-level observations at each committed printed word in one encoded sentence."""
from dataclasses import replace
from pathlib import Path
import re
import numpy as np
from scipy.special import expit
from research.backbone_transfer.extract import digest,read_json
from research.reading_validation.transition_external_forecast import token_surprisals
from research.token_readout.runner import trace_rows

BUNDLE=Path(__file__).resolve().parents[1]/'results/reading-validation-convergent/serving'


class WordResponseInference:
    def __init__(self,params,meta):
        self.params=params;self.meta=meta

    @classmethod
    def load(cls,directory=BUNDLE):
        directory=Path(directory);meta=read_json(directory/'spec.json');weights=directory/'weights.npz'
        if digest(weights)!=meta['weights_sha256']:raise ValueError('Word response coefficients do not match the certified release')
        with np.load(weights) as z:p={k:z[k] for k in z.files}
        if p['B'].shape!=(31,3) or p['basis'].shape[1]!=24:raise ValueError('Unexpected word response feature geometry')
        return cls(p,meta)

    def from_features(self,text,runtime,reader,checkpoint,features):
        """Consume an existing native-causal backbone call; an observation is attached only at the last subword."""
        if runtime.revision!=self.meta['revision'] or len(features.offsets)!=len(features.token_ids):raise ValueError('Different model revision or token offsets')
        spans=list(re.finditer(r'\S+',text));positions=[]
        for w in spans:
            a,b=w.span();idx=[i for i,(l,h) in enumerate(features.offsets) if l<b and h>a and text[max(l,a):min(h,b)].strip()]
            if not idx or features.offsets[idx[-1]][1]!=b or (positions and idx[0]<=positions[-1][3]):return {'available':False,'reason':'Printed word lacks its own committed last token','words':[]}
            positions.append((a,b,idx[0],idx[-1]))
        if len(positions)<2:return {'available':True,'words':[],'measured_noninitial_words':0}
        A,B=reader.project(features);p=reader.score_projected(A,B,features.relations);sc=checkpoint['scale']
        p=replace(p,opening=p.opening*sc['opening_gain'],resolution=p.resolution*sc['resolution_gain']+sc['resolution_bias'])
        tr=trace_rows(text,features.offsets,p,checkpoint['priors'],reader.anchor_tokens);sur=token_surprisals(runtime,features);states=np.c_[A.detach().float().cpu().numpy(),B.detach().float().cpu().numpy()];previous=np.zeros(states.shape[1],dtype='float32');obs=[]
        for i,(a,b,lo,hi) in enumerate(positions):
            st=states[hi];delta=st-previous;word=text[a:b];stock=[tr[hi][k] for k in ('unresolved','opened','resolved','transition')]
            obs.append(np.r_[np.log1p(len(word)),np.log1p(hi-lo+1),np.log1p(i),float(word[:1].isupper()),float(word[-1:] in ',.;:!?'),np.sum(sur[lo:hi+1]) if lo else 0.,stock,np.linalg.norm(delta),st,delta]);previous=st
        x=np.asarray(obs,dtype=np.float32).astype(np.float64);q=self.params;design=np.column_stack([np.ones(len(x)),(x[:,:6]-q['mean'])/q['sd'],((x[:,6:]-q['context_mean'])/q['context_scale'])@q['basis']]);eta=design@q['B'];eta[:,1]=expit(eta[:,1]);fore=eta*q['scale']+q['mu'];out=[]
        for (a,b,lo,hi),(n400,miss,neg_rating) in zip(positions[1:],fore[1:]):
            rating=-float(neg_rating)
            out.append(dict(start=a,end=b,commit_token=hi,word=text[a:b],expected_N400_negativity=float(n400),cloze_target_probability=float(1.-miss),expected_predictability_rating=rating if 1<=rating<=5 else None,raw_fitted_rating=rating))
        return {'available':True,'words':out,'measured_noninitial_words':len(out),'length_outside_original_sentences':not 5<=len(positions)<=15}
