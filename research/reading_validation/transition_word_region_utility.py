"""Estimate the expected acceptance of a specific earlier word under observed reader intervals."""
from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp

from research.reading_validation.transition_gate_features import features
from research.reading_validation.transition_pointer_features import FORMS


def word_features(row,idx,u):
    q=row["queries"][idx];score=np.array(u,dtype=float)
    if score.shape!=(q,):raise ValueError("Visible scores do not match the pronoun")
    tags=list(row["pos"][:q]);tags[-1]=row["pos_last"][idx]
    words=[row["text"][a:b] for a,b in row["words"][:q+1]];forms=[w.casefold() for w in words]
    logp=score-logsumexp(score);p=np.exp(logp);top=int(score.argmax());iw=np.arange(q)
    entropy=float(-p@logp);query=np.r_[features(row,q,tags),entropy,entropy/np.log(max(2,q)),p[top],np.sort(p)[-3:].sum(),np.log1p(q-top)]
    competitor=max(np.partition(score,-2)[-2],score.min()) if q>1 else score[0]
    pos=np.array(tags);sents=row["sentences"][q]-np.asarray(row["sentences"][:q])
    first=lambda w:w[0].isupper()
    word=np.column_stack([logp,score-score[top],np.where(iw==top,score-competitor,score-score[top]),iw==top,
                  np.log1p(np.abs(iw-top)),np.r_[0.,p[:-1]],np.r_[p[1:],0.],np.log1p(q-iw),1/(q-iw),
                  sents==0,np.log1p(sents),np.isin(pos,["NNP","NNPS"]),np.isin(pos,["NN","NNS"]),
                  np.isin(pos,["PRP","PRP$"]),[FORMS.get(s)==FORMS[forms[q]] for s in forms[:q]],
                  [s==forms[q] for s in forms[:q]],list(map(first,words[:q])),np.r_[False,pos[:-1]=="DT"]])
    return query,word,logp,.5*(p+1/q)


def observed(actions,q):
    y=np.zeros(q);n=0
    for response,a,b,count in actions:
        if response=="DO":
            if not 0<=a<b<=q:raise ValueError("Reader click escapes current text")
            y[a:b]+=count
        n+=count
    if not n:raise ValueError("A target has no independent reading")
    return y/n,n


def learn(rows,source_logits,arm,training):
    indexed={r["id"]:r for r in rows};positions={r["id"]:{q:i for i,q in enumerate(r["queries"])} for r in rows}
    groups=[indexed[key.rsplit("/",1)[0]]["group"] for key in training];groups_count=Counter(groups)
    n=sum(int(k.rsplit("/",1)[1]) for k in training);nq=len(training)
    X=Q=w=y=logp=owner=None;left=0
    for h,(key,actions) in enumerate(training.items()):
        doc,s=key.rsplit("/",1);q=int(s);row=indexed[doc]
        a,b,z,m=word_features(row,positions[doc][q],source_logits[key][arm])
        if X is None:
            Q=np.empty((nq,len(a))); X=np.empty((n,b.shape[1]));w=np.empty(n);y=np.empty(n);logp=np.empty(n);owner=np.empty(n,np.int32)
        r=slice(left,left+q);left+=q
        point,_=observed(actions,q);Q[h]=a;X[r]=b;logp[r]=z;owner[r]=h;y[r]=point
        w[r]=m/(len(groups_count)*groups_count[row["group"]])
    if left!=n or not np.isclose(w.sum(),1):raise ValueError("Source work or earlier-word weighting changed")
    target_w=np.bincount(owner,weights=w,minlength=len(Q))
    def standardize(Z,v):
        mu=v@Z;Z-=mu
        # One column at a time keeps the peak memory independent of a second dense source design.
        sd=np.array([np.sqrt(np.dot(v,np.square(col))) for col in Z.T]);sd[sd<1e-8]=1.;Z/=sd
        return mu,sd
    mx,sx=standardize(X,w);mq,sq=standardize(Q,target_w);cut=Q.shape[1]
    def objective(theta):
        intercept=theta[0];bq=theta[1:1+cut];bc=theta[1+cut:];l=logp+intercept+(Q@bq)[owner]+X@bc
        residual=w*(expit(l)-y)
        loss=w@(np.logaddexp(0,l)-y*l)+.005*(theta[1:]@theta[1:])
        grad=np.r_[residual.sum(),Q.T@np.bincount(owner,weights=residual,minlength=len(Q)),X.T@residual]
        grad[1:]+=.01*theta[1:]
        return float(loss),grad
    initial=np.zeros(1+Q.shape[1]+X.shape[1]);first=objective(initial)[0]
    result=minimize(objective,initial,method="L-BFGS-B",jac=True,options={"gtol":1e-9,"ftol":1e-15,"maxiter":600,"maxls":40})
    maxgrad=float(abs(result.jac).max())
    if not np.isfinite(result.fun) or maxgrad>1e-6:raise RuntimeError(f"Word-region utility did not converge {result.fun} {maxgrad}")
    return {"q_mean":mq,"q_sd":sq,"word_mean":mx,"word_sd":sx,"beta":result.x}, {"training_pronouns":nq,"candidate_words":n,
            "source_works":len(groups_count),"observed_acceptance_at_sampled_word":float(w@y),"start":first,"objective":float(result.fun),
            "maximum_gradient":maxgrad,"iterations":int(result.nit),"free_parameters":len(result.x)}


def infer(row,index,u,model):
    q,x,prior,weight=word_features(row,index,u);n=len(q);v=model["beta"]
    logits=prior+v[0]+((q-model["q_mean"])/model["q_sd"])@v[1:n+1]+((x-model["word_mean"])/model["word_sd"])@v[n+1:]
    return logits,weight
