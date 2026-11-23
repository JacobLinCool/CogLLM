"""Estimate an unknown reader's lexical response distribution from a finite random-effects law."""
import numpy as np
from scipy.special import logsumexp,softmax
from scipy.sparse import coo_matrix

SHARPNESS=np.repeat(2.**(np.arange(-3.,4.)/3),7)
OUTSIDE=np.tile(np.arange(-3.,4.)/2,7)
NAMES=('same_history','unknown_reader','other_history','single_head')


def response_law(lp):
    n=len(lp);z=np.zeros((n,len(SHARPNESS)))
    for lo in range(0,n,64):
        b=lp[lo:lo+64].astype(float)
        for j,beta in enumerate(SHARPNESS[::7]):
            named=logsumexp(beta*b[:,:-1],axis=1)
            cols=slice(j*7,(j+1)*7)
            z[lo:lo+len(b),cols]=np.logaddexp(named[:,None],beta*b[:,-1,None]+OUTSIDE[cols])
    other=np.exp(lp[:,-1,None]*SHARPNESS+OUTSIDE-z)
    return z,other


def scores(lp,z,ev):
    i,w=ev[:,1],ev[:,2]
    return lp[i,w,None]*SHARPNESS + (w==lp.shape[1]-1)[:,None]*OUTSIDE-z[i]


def fit_prior(likelihood,readers):
    groups,ix=np.unique(readers,return_inverse=True)
    J=likelihood.shape[1]
    logl=np.zeros((len(groups),J))
    np.add.at(logl,ix,likelihood)
    X=np.exp(logl-logl.max(1)[:,None])
    prior=np.ones(J)/J;eta=1/J
    def objective(w):return -np.log(X@w).sum()-eta*np.log(w).sum()
    for it in range(150):
        relative=X/(X@prior)[:,None]
        grad=-relative.sum(0)-eta/prior
        station=float(np.max(abs(grad-grad.mean()))/(len(groups)+1))
        if station<1e-7:break
        H=relative.T@relative + np.diag(eta/prior**2)
        two=np.linalg.solve(H,np.column_stack((grad,np.ones(J))))
        direction= -two[:,0] + two[:,1]*(two[:,0].sum()/two[:,1].sum())
        old=objective(prior);delta=grad@direction
        scale=min(1.,.99*np.min(-prior[direction<0]/direction[direction<0]))
        for _ in range(65):
            step=prior+scale*direction
            if np.all(step>0) and objective(step)<=old+min(1e-4*scale*delta,0)+1e-12:break
            scale/=2
        else:raise ValueError("Population mixing weights failed descent")
        prior=step
    else:raise ValueError("Random-effects prior failed convergence: "+str(station))
    if abs(prior.sum()-1)>1e-9:raise ValueError("Invalid mixing distribution")
    return prior,dict(people=len(groups),iterations=it,stationarity=station,mode_mass=float(prior.max()),
          mean_precision=float(prior@SHARPNESS),mean_other_log_odds=float(prior@OUTSIDE))


def match(eligible,available):
    """Deterministic alternative history maximizes overlap of offered materials."""
    ordered=sorted(eligible);result={};ratios=[]
    for i,r in enumerate(ordered):
        candidates=ordered[i+1:]+ordered[:i]
        source=available[r]
        n=max(len(source&available[q]) for q in candidates)
        result[r]=next(q for q in candidates if len(source&available[q])==n)
        ratios.append(n/len(source))
    return result,ratios


def predict(lp, prior, histories, queries, available):
    norm,other=response_law(lp)
    logpri=np.log(prior)
    people=sorted(histories)
    if set(queries)!=set(people) or len(people)<2:raise ValueError("Unusable reader control")
    post={r:softmax(logpri+scores(lp,norm,histories[r]).sum(0)) for r in people}
    null,overlap=match(people,available)
    e=[];loss=[];other_probs=[]
    for r in people:
        q=queries[r]
        L=scores(lp,norm,q)
        prior_types=np.stack((post[r],prior,post[null[r]]))
        lp_choice=logsumexp(L[:,:,None]+np.log(prior_types.T[None,:,:]),axis=1)
        direct=lp[q[:,1],q[:,2]]
        values=np.column_stack((lp_choice,direct))
        qother=np.column_stack((other[q[:,1]]@prior_types.T,np.exp(lp[q[:,1],-1])))
        if not np.isfinite(values).all() or np.any(qother>1+1e-7):raise ValueError("Invalid forecast")
        e.append(q);loss.append(-values);other_probs.append(qother)
    return np.vstack(e),np.vstack(loss),np.vstack(other_probs),dict(null_material_overlap_min=float(min(overlap)),null_material_overlap_mean=float(np.mean(overlap)))


def report(ev,loss,other,draws=1800,materials=None):
    subject,ir=np.unique(ev[:,0],return_inverse=True);words,iw=np.unique(ev[:,1] if materials is None else materials[ev[:,1]],return_inverse=True)
    R,W=len(subject),len(words);rng=np.random.default_rng(76391)
    sr=rng.multinomial(R,np.full(R,1/R),size=draws)
    sw=rng.multinomial(W,np.full(W,1/W),size=draws)
    def total(value):
        mat=coo_matrix((value,(ir,iw)),shape=(R,W)).tocsr()
        sums=[]
        for k in range(0,draws,50):
            sums.extend(np.sum(sr[k:k+50]*(mat@sw[k:k+50].T).T,axis=1))
        return np.asarray(sums)
    n=total(np.ones(len(ev)));cmp={}
    flag=(ev[:,2]==8192)
    for j in (1,2,3):
        delta=loss[:,j]-loss[:,0]
        boot=total(delta)/n
        cmp['same_over_'+NAMES[j]]=dict(nats_per_response=float(delta.mean()),reader_sentence_interval=np.quantile(boot,[.025,.975]).tolist(),
            fraction_readers_better=float(np.mean(np.bincount(ir,weights=delta)/np.bincount(ir)>0)))
    return dict(readers=R,positions=len(set(ev[:,1])),source_sentence_groups=W,scored_responses=len(ev),crossed_bootstraps=draws,
        NLL={m:float(loss[:,j].mean()) for j,m in enumerate(NAMES)},
        outside_Brier={m:float(((other[:,j]-flag)**2).mean()) for j,m in enumerate(NAMES)},
        outside_accuracy={m:float(np.mean((other[:,j]>=.5)==flag)) for j,m in enumerate(NAMES)},observed_outside=float(flag.mean()),paired=cmp)
