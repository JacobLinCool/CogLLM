"""Fit different observation laws to one referent/expression distribution across Mandarin cohorts."""

from argparse import ArgumentParser
from collections import Counter,defaultdict
import csv
from hashlib import blake2b
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logsumexp

from research.backbone_transfer.extract import digest as sha,write_json as dump


ROOT=Path(".cache/cogllm/reading-validation")
OUT=Path("research/results/reading-validation-transition-mandarin")
PROTOCOL=Path("research/reading_validation/transition-mandarin-protocol.md")
CASES=np.array([[-1.,-1.,1.],[-1.,1.,-1.],[1.,-1.,-1.],[1.,1.,1.]])*.5
FAMILIES=36
LABELS=("np2_nonovert","np2_overt","np1_nonovert","np1_overt")


def cosine(X):
    norm=np.linalg.norm(X,axis=1,keepdims=True)
    z=np.divide(X,norm,out=np.zeros_like(X,dtype=float),where=norm>0)
    return z@z.T


def characters(texts):
    big=np.zeros((len(texts),512),dtype=float); surface=[]
    for i,s in enumerate(texts):
        for n in (1,2):
            for j in range(len(s)-n+1):
                z=int.from_bytes(blake2b(s[j:j+n].encode(),digest_size=8).digest(),"big")
                big[i,z%512]+=1 if z>>63 else -1
        surface.append(["被" in s,"把" in s,len(s)/10,s.count("，"),s.count("。"),s.count("他"),s.count("她"),s.rfind("了")/len(s)])
    form=np.asarray(surface,float)
    return form@form.T,cosine(big)


def center_matrix(gram,idx):
    means=gram[:,idx].mean(1);total=means[idx].mean()
    centered=gram-means[:,None]-means[None,:]+total
    var=centered.diagonal()[idx].mean()
    return centered/var if var>1e-10 else np.zeros_like(centered)


def representation(kernels,idx):
    k=np.mean([center_matrix(v,idx) for v in kernels],axis=0)
    source=k[np.ix_(idx,idx)]
    eig,U=np.linalg.eigh((source+source.T)/2)
    good=eig>1e-9
    Z=k[:,idx]@(U[:,good]/np.sqrt(eig[good]))
    return np.column_stack([np.ones(len(Z)),Z])


def counts_for(observations):
    by=defaultdict(lambda:np.zeros((2,4),int));info={}
    for r in observations:
        key=(r["exp"],r["family"],r["structure"],r["bare"])
        j=int(r["condition"]=="pronoun");c=2*r["first"]+r["overt"]
        by[key][j,c]+=1
        info[key]=(r["class"],r["source_verb"])
    cells=sorted(by)
    return cells,np.stack([by[c] for c in cells]),info


def task_mass(raw,families):
    mass=np.zeros_like(raw,dtype=float)
    for group in sorted(set(families)):
        sel=np.array([g==group for g in families])
        part=raw[sel].sum(axis=(0,2));mass[sel]=raw[sel]/part[None,:,None]/len(set(families))
    if not np.allclose(mass.sum(axis=(0,2)),1.):raise ValueError("Original observation types or source balancing differ")
    return mass


def fit(x,counts,mode):
    dim=x.shape[1]
    if mode=="coupled":
        f=counts[:,0];p=counts[:,1][:,[1,3]]
        scales=np.array([2.,1.,2.])
        def objective(v):
            w=v.reshape(dim,3);logits=x@w@CASES.T
            l0=logits-logsumexp(logits,axis=1,keepdims=True)
            selected=logits[:,[1,3]];l1=selected-logsumexp(selected,axis=1,keepdims=True)
            residual=.5*(np.exp(l0)*f.sum(1,keepdims=True)-f)
            residual[:,[1,3]]+=.5*(np.exp(l1)*p.sum(1,keepdims=True)-p)
            reg=w.copy()*scales;reg[0]=0
            loss=-.5*((l0*f).sum()+(l1*p).sum())+.005*(reg*w).sum()
            grad=x.T@residual@CASES+.01*reg
            return float(loss),grad.flatten()
        initializer=np.zeros(dim*3)
    else:
        tab=counts[:,1] if mode=="overt_only" else counts.mean(1)
        n=tab.sum(1);yes=tab[:,2:].sum(1)
        def objective(w):
            eta=x@w;p=expit(eta);reg=w.copy();reg[0]=0
            return float(n@np.logaddexp(0,eta)-yes@eta+.005*(reg@w)),x.T@(n*p-yes)+.01*reg
        initializer=np.zeros(dim)
    opt=minimize(objective,initializer,jac=True,method="L-BFGS-B",options={"ftol":1e-15,"gtol":2e-9,"maxiter":500,"maxls":50})
    grad=float(np.abs(opt.jac).max())
    if not np.isfinite(opt.fun+grad) or grad>1e-6:raise ValueError(f"Mandarin {mode} stationarity failed: {opt.fun} {grad}")
    return opt.x,{"objective":float(opt.fun),"max_gradient":grad,"steps":int(opt.nit)}


def predict(x,w,mode):
    if mode=="coupled":
        logits=x@w.reshape(x.shape[1],3)@CASES.T
        probabilities=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
        return probabilities[:,3]/(probabilities[:,3]+probabilities[:,1]),probabilities
    return expit(x@w),None


def paper_controls(cells,info):
    records=list(csv.DictReader((ROOT/"mandarin-ba-counts.csv").open()))
    models=sorted({r["model"] for r in records})
    indexed={(r["verb"],r["model"]):r for r in records}
    calc={}; published={}
    for model in models:
        subset=[r for r in records if r["model"]==model]
        pred=np.array([float(r["predictNP1smoothed"]) for r in subset]);obs=np.array([float(r["observedNP1"]) for r in subset])
        published[model]={"rows":len(subset),"item_mse":float(np.mean((obs-pred)**2)),"squared_pearson":float(np.corrcoef(obs,pred)[0,1]**2)}
        for c in cells:
            if c[0]!="ba":continue
            spelling=info[c][1];key=("ba_" if c[2]=="ba" else "can_")+spelling
            field=indexed[key,model];calc.setdefault(c,{})["same_item_human_"+model]=float(field["predictNP1smoothed"])
    for model,val in {"Bayesian":.04611493,"Expectancy":.09211334,"Mirror":.03179736}.items():
        if abs(published[model]["item_mse"]-val)>1e-7:raise ValueError("Published Mandarin controls failed exact replication")
    return calc,published


def generate(source,obs,stats,surface_only=False,*,lexical_priors=None):
    cells,counts,info=counts_for(obs);texts={r["id"]:r["text"] for r in source}
    if len({c[1] for c in cells})!=FAMILIES:raise ValueError("The original agent/patient pairs changed")
    strings=[texts[c[-1]] for c in cells]
    forms,grams=characters(strings)
    annotated=np.array([[info[c][0]=="IC-1","被" in strings[i],"把" in strings[i],
                         (info[c][0]=="IC-1") and "被" in strings[i],(info[c][0]=="IC-1") and "把" in strings[i]] for i,c in enumerate(cells)],float)
    paper,published=paper_controls(cells,info)
    blocks={"surface":[forms,grams],"rated_semantic_class":[annotated@annotated.T]}
    if not surface_only:
        native=np.stack([stats[c[-1]] for c in cells]).astype(float)
        if native.shape[1]!=4:raise ValueError("Mandarin contexts lack two frozen layers and two pooling positions")
        ks=[cosine(native[:,j]) for j in range(4)]
        # Four contextual blocks carry half the combined weight; the two surface groups share the other half.
        blocks["native"]=[forms,forms,grams,grams,*ks]
    if lexical_priors:
        for name, by_verb in lexical_priors.items():
            prior = np.array([by_verb[info[c][1]] for c in cells])
            if not np.all((prior > 0) & (prior < 1)):
                raise ValueError("Expected a finite source-trained referent probability for every verb")
            odds = np.log(prior) - np.log1p(-prior)
            base = blocks["native"]
            blocks[name] = [*base, *([np.outer(odds, odds)] * len(base))]
    train=np.array([c[0]!="ba" for c in cells]);evals=np.array([c[0]=="ba" for c in cells])
    yhat=defaultdict(dict);qhat=defaultdict(dict);health=defaultdict(list)
    for family in sorted({c[1] for c in cells}):
        same=np.array([c[1]==family for c in cells]);idx=np.flatnonzero(train&~same);test=np.flatnonzero(evals&same)
        if len(test)!=2 or set(cells[j][-1] for j in idx)&set(cells[j][-1] for j in test):raise ValueError("Held cohort or predicate leaked")
        mass=task_mass(counts[idx],[cells[j][1] for j in idx])
        for model,kernels in blocks.items():
            x=representation(kernels,idx)
            methods=(("coupled",) if lexical_priors and model in lexical_priors else
                     ("overt_only",) if model=="rated_semantic_class" else ("coupled","overt_only","pooled"))
            for mode in methods:
                name=model+"_"+mode;w,h=fit(x[idx],mass,mode);health[name].append(h)
                p,joint=predict(x[test],w,mode)
                for j,i in enumerate(test):
                    key=cells[i];yhat[key][name]=float(p[j])
                    if joint is not None:qhat[key][name]=joint[j].tolist()
    for c,p in paper.items():yhat[c].update(p)
    return cells,info,yhat,qhat,health,published


def summarise(obs,info,yhat,qhat,*,additional_comparisons=None):
    ext=[r for r in obs if r["exp"]=="ba"];cats=sorted(yhat);families=sorted({c[1] for c in cats});people=sorted({r["reader"] for r in ext})
    C=np.zeros((len(people),len(cats),2,4),int)
    for r in ext:
        c=(r["exp"],r["family"],r["structure"],r["bare"])
        C[people.index(r["reader"]),cats.index(c),r["condition"]=="pronoun" and 1 or 0,2*r["first"]+r["overt"]]+=1
    order=np.array([families.index(c[1]) for c in cats])
    rng=np.random.default_rng(37497);pw=rng.exponential(size=(4000,len(people)));fw=rng.exponential(size=(4000,len(families)))
    cw=fw[:,order];cw/=cw.sum(1,keepdims=True)
    counts=C.sum(0); bcounts=(pw@C.reshape(len(people),-1)).reshape(len(pw),*counts.shape)
    y=counts[:,1,3]/counts[:,1].sum(-1);yb=bcounts[:,:,1,3]/bcounts[:,:,1,:].sum(-1)
    free=counts[:,0]/counts[:,0].sum(-1,keepdims=True)
    fb=bcounts[:,:,0]/bcounts[:,:,0].sum(-1,keepdims=True)
    ci=lambda p,b:[float(p),*np.quantile(b,[.025,.975]).tolist()]
    metrics={};matrix={}
    for name in yhat[cats[0]]:
        p=np.array([yhat[c][name] for c in cats])
        if not np.all((p>0)&(p<1)):raise ValueError("A Mandarin probability is degenerate")
        hard=(np.sign(p-.5)+1)/2
        def utility(r):
            return np.stack([-r*np.log(p)-(1-r)*np.log1p(-p),r*(1-p)**2+(1-r)*p**2,
                             r*hard+(1-r)*(1-hard),p-r],axis=-1)
        obsval=utility(y).mean(0);boot=(cw[:,:,None]*utility(yb)).sum(1)
        fields=("choice_logloss","reader_brier","choice_accuracy","signed_calibration_error")
        metrics[name]={k:ci(obsval[i],boot[:,i]) for i,k in enumerate(fields)};matrix[name]=(obsval,boot)
        if name in qhat[cats[0]]:
            j=np.array([qhat[c][name] for c in cats])
            losses=-(free*np.log(j)).sum(-1);bl=-(fb*np.log(j)).sum(-1)
            metrics[name]["free_joint_logloss"]=ci(losses.mean(),(cw*bl).sum(1))
            predicted=j[:,2:].sum(1);observed=free[:,2:].sum(1);observed_boot=fb[:,:,2:].sum(2)
            score=observed*(1-predicted)**2+(1-observed)*predicted**2
            boot_score=observed_boot*(1-predicted)**2+(1-observed_boot)*predicted**2
            metrics[name]["free_first_referent_brier"]=ci(score.mean(),(cw*boot_score).sum(1))
            metrics[name]["free_first_referent_item_mse"]=ci(np.square(predicted-observed).mean(),
                                                            (cw*np.square(predicted-observed_boot)).sum(1))
    deltas={}
    for x,a in matrix.items():
        if additional_comparisons and x in additional_comparisons:
            comps=additional_comparisons[x]
        elif x in ("native_coupled","surface_coupled"):
            comps=[x.replace("coupled","overt_only"),x.replace("coupled","pooled"),"surface_overt_only","same_item_human_Bayesian","rated_semantic_class_overt_only"]
            if "native" in x:comps.append("surface_coupled")
        else:
            continue
        for comp in dict.fromkeys(comps):
            v0,b0=matrix[comp];delta=a[0]-v0;db=a[1]-b0
            deltas[x+"_minus_"+comp]={k:ci(delta[i],db[:,i]) for i,k in enumerate(fields)}
    effects={}
    modes={"human_observed":(y,yb),"human_free":(free[:,2:].sum(1),fb[:,:,2:].sum(2))}
    for nm in yhat[cats[0]]:
        s=np.array([yhat[c][nm] for c in cats]);modes[nm]=(s,np.broadcast_to(s,(len(fw),len(cats))))
    for nm in qhat[cats[0]]:
        s=np.array([sum(qhat[c][nm][2:]) for c in cats]);modes["free_"+nm]=(s,np.broadcast_to(s,(len(fw),len(cats))))
    for nm,(s,sb) in modes.items():
        subsets={}
        for structure in ("active","ba"):
            for cl in ("IC-1","IC-2"):
                ix=np.array([c[2]==structure and info[c][0]==cl for c in cats])
                m=(cw[:,ix]*sb[:,ix]).sum(1)/cw[:,ix].sum(1)
                subsets[structure,cl]=(s[ix].mean(),m)
        def difference(a,b):return ci(subsets[a][0]-subsets[b][0],subsets[a][1]-subsets[b][1])
        effects[nm]={"IC1_minus_IC2_"+s:difference((s,"IC-1"),(s,"IC-2")) for s in ("active","ba")}
        effects[nm].update({"ba_minus_active_"+c:difference(("ba",c),("active",c)) for c in ("IC-1","IC-2")})
    return {"external_persons":len(people),"families":len(families),"observed_target_types":LABELS,
            "original_condition_cells":len(cats),"actual_overt_choices":int(C[:,:,1,:].sum()),"actual_free_continuations":int(C[:,:,0,:].sum()),
            "metrics":metrics,"paired":deltas,"interventions":effects,
            "source_evaluation_support":[{"family":c[1],"structure":c[2],"external_verb":info[c][1],"overt_choices":int(counts[i,1].sum()),
                                         "observed_first":float(y[i]),"forecasts":yhat[c]} for i,c in enumerate(cats)]}


def main(folder):
    seal=json.loads((OUT/"source-seal.json").read_text());pack=ROOT/"mandarin-reading-stimuli.jsonl";human=ROOT/"mandarin-anonymous-choices.json"
    if (sha(PROTOCOL)!=seal["protocol_sha256"] or sha(pack)!=seal["input_sha256"] or sha(human)!=seal["external_measurements_sha256"]):
        raise ValueError("Original Mandarin protocol or human responses changed")
    src=list(map(json.loads,pack.read_text().splitlines()));obs=json.loads(human.read_text())
    receipt=None;states=None
    if folder:
        receipt=json.loads((folder/"receipt.json").read_text());p=folder/"unprompted-states.npz"
        if (sha(p)!=receipt["features_sha256"] or receipt["protocol_sha256"]!=sha(PROTOCOL) or receipt["input_sha256"]!=sha(pack)
                or receipt["encoder_forwards"]!=len(src)):raise ValueError("Original single-pass Mandarin forecast changed")
        states=np.load(p)
    cells,info,yhat,qhat,health,published=generate(src,obs,states,folder is None)
    result=summarise(obs,info,yhat,qhat)
    dump(OUT/("scores.json" if folder else "surface-and-human-controls-with-free.json"),{"protocol_sha256":sha(PROTOCOL),"accelerator_receipt":receipt,
         "published_replication":published,"source_fit_health":{k:{"fits":len(v),"max_gradient":max(t["max_gradient"] for t in v),
         "max_steps":max(t["steps"] for t in v)} for k,v in health.items()},**result})
    print(result["actual_overt_choices"],{k:round(v["choice_logloss"][0],4) for k,v in result["metrics"].items()},flush=True)


if __name__=="__main__":
    p=ArgumentParser(description=__doc__);p.add_argument("--features",type=Path)
    main(p.parse_args().features)
