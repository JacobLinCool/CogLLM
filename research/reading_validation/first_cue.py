"""Conserve the mass of not-yet-reported readers before and after a single experimental cue."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import argparse,json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit,logit
import torch

from research.reading_validation.intervention_moment import measurements
from research.reading_validation.memory_data import BASE,load_rows,sha
from research.reading_validation.memory_fit import counts
from research.reading_validation.memory_score import reveal,summarize
from research.reading_validation.premise_emission import emissions
from research.reading_validation.referent_features import setup
from research.reading_validation.taxonomic_monitor import Route
from research.reading_validation.taxonomic_pairs import Ontology,form,native_words,opportunities,source_links

OUT=Path("research/results/reading-validation-first-cue")
PLAN=Path("research/reading_validation/first-cue-protocol.md")
OLD=Path("research/results/reading-validation-intervention-moment/forecast")

def source_schedule(rows,length):
    families={}
    for r in rows:
        families.setdefault(r["story"],[]).append(r["target"]-1)
    pos=np.array([np.median(x) for x in families.values()])
    if not np.all(pos==np.round(pos)):raise ValueError("Ambiguous protocol cue coordinate in a family")
    return np.array([float(logit(((pos==j).sum()+.25)/((pos>=j).sum()+.5))) for j in range(length)])

def design(x):
    # Separate cue salience from evidence about what the retrieved noun means.
    role=np.clip(x[:,4:5],1e-7,1-1e-7)
    gate=np.column_stack([x[:,:4],logit(role),x[:,5:9]])
    sign=np.column_stack([x[:,9:17],x[:,5:9]])
    return gate,sign

def curves(g,c,h):
    """Exact finite-state marginal first-response mass for all reading paths."""
    n,L=g.shape
    fresh=torch.ones(n,dtype=g.dtype);passed=torch.zeros_like(fresh)
    first=[];haz=[]
    for t in range(L):
        if t==0:
            first.append(torch.zeros_like(fresh));haz.append(torch.zeros_like(fresh));continue
        gate=g[:,t];ordinary=h[:,t,0];entry=(1-c[:,t])*h[:,t,1]+c[:,t]*h[:,t,2]
        event=(fresh*(1-gate)+passed)*ordinary+fresh*gate*entry
        haz.append(event/(fresh+passed).clamp_min(1e-12));first.append(event)
        fresh,passed=fresh*(1-gate)*(1-ordinary),passed*(1-ordinary)+fresh*gate*(1-entry)
    return torch.stack(haz,1),torch.stack(first,1),fresh+passed

def stage(source,outer,x,prior,observations,emission,moment):
    groups={}
    for i,r in enumerate(source):groups.setdefault(r["story"],{})[r["arm"]]=i
    pair=[(v["inconsistent"],v["neutral"]) for v in groups.values()]
    maxlen=max(len(x[r["id"]]) for r in source+outer);ns=len(source)
    rows=source+outer
    def pad(v):
        return np.pad(v,[(0,maxlen-len(v))]+[(0,0)]*(v.ndim-1),mode="constant")
    def tensor(v):return torch.as_tensor(np.asarray(v),dtype=torch.float64)
    gate,sign=[],[]
    for r in rows:
        a,b=design(x[r["id"]]);gate.append(a);sign.append(b)
    def scale(block):
        # Source-only standardization. The readout may see anywhere in a held prefix.
        allx=np.vstack(block[:ns]);mean=allx.mean(0);sd=allx.std(0);sd[sd<1e-6]=1
        return tensor(np.stack([pad(np.c_[np.ones(len(a)),(a-mean)/sd]) for a in block])),mean,sd
    X,mx,sx=scale(gate);Z,mz,sz=scale(sign);D=X.shape[-1]
    baseline=tensor(np.stack([pad(prior[r["id"]]) for r in rows]));h=tensor(np.stack([pad(emission[r["id"]]) for r in rows]))
    k=torch.arange(ns)
    t=torch.tensor([r["target"]-1 for r in source])
    y=tensor([r["arm"]=="inconsistent" for r in source])
    pair=np.array(pair)
    at=[];evt=[];se=[]
    for r in source:
        risk,events,n=observations[r["id"]]
        at.append(pad(risk/n));evt.append(pad(events/n))
        prop=(events[r["target"]-1]+.5)/(n+1);se.append(prop*(1-prop)/(n+1))
    at=tensor(at);evt=tensor(evt)
    diff=evt[k,t][pair[:,0]]-evt[k,t][pair[:,1]]
    var=np.array(se)[pair].sum(1);precision=tensor(1/np.maximum(np.median(var),var))
    def rates(v):
        g=torch.sigmoid(baseline+X@v[:D])
        c=torch.sigmoid(Z@v[D:])
        risk,f,end=curves(g,c,h)
        return g,c,risk,f,end
    mask=torch.arange(maxlen)[None,:]<t[:,None];mask[:,0]=False
    def objective(v0):
        v=tensor(v0).requires_grad_(True);g,c,risk,f,_=rates(v)
        source_g=g[:ns];clamp=lambda a:a.clamp(1e-12,1-1e-12)
        cue=(-torch.log(clamp(source_g[k,t]))-torch.sum(torch.log1p(-clamp(source_g))*mask,axis=1)).mean()
        yes=clamp(c[k,t]);which=-(y*yes.log()+(1-y)*torch.log1p(-yes)).mean()
        rh=clamp(risk[:ns]);path=-(evt*rh.log()+(at-evt)*torch.log1p(-rh)).sum(1).mean()
        loss=path+.5*(cue+which)+.04*(v[1:D]@v[1:D]+v[D+1:]@v[D+1:])+.005*(v[0]**2+v[D]**2)
        if moment:
            critical=f[k,t];resid=critical[pair[:,0]]-critical[pair[:,1]]-diff
            loss+=.5*(resid.square()*precision).mean()
        loss.backward()
        return float(loss.detach()),v.grad.numpy()
    start=np.zeros(X.shape[-1]+Z.shape[-1]);unit=np.sin(np.arange(len(start))+1);unit/=np.linalg.norm(unit)
    numeric=(objective(start+1e-5*unit)[0]-objective(start-1e-5*unit)[0])/2e-5
    err=abs(numeric-objective(start)[1]@unit)
    opt=minimize(objective,start,method="L-BFGS-B",jac=True,options={"gtol":2e-7,"ftol":1e-14,"maxiter":350})
    if not opt.success or np.max(abs(opt.jac))>3e-6 or err>2e-7:raise ArithmeticError((opt.message,opt.jac.max(),err))
    with torch.no_grad():
        g,c,risk,first,end=rates(tensor(opt.x));rg,fg,eg=curves(torch.sigmoid(baseline),torch.full_like(c,.5),h)
        conservation=max(float((first.sum(1)+end-1).abs().max()),float((fg.sum(1)+eg-1).abs().max()))
        if conservation>1e-9:raise ArithmeticError("Reader risk is not conserved")
        estimates={r["id"]:(risk[ns+j,:len(x[r["id"]])].numpy(),rg[ns+j,:len(x[r["id"]])].numpy()) for j,r in enumerate(outer)}
    return estimates,dict(beta=opt.x.tolist(),objective=float(opt.fun),gradient=float(np.abs(opt.jac).max()),iteration=int(opt.nit),
                         scale=[mx.tolist(),sx.tolist(),mz.tolist(),sz.tolist()],gradient_check=float(err),conservation=conservation)

def run(archive):
    torch.set_num_threads(1);setup();folder=OUT/"forecast";folder.mkdir(exist_ok=True,parents=True)
    rows=json.loads((BASE/"memory/study-map.json").read_text());raw={r["id"]:r["sentences"] for r in load_rows(BASE/"memory/stories.jsonl")}
    docs={k:list(map(form,v)) for k,v in raw.items()}
    native=native_words(archive,raw);ontology=Ontology();bags={k:opportunities(v,native[k],ontology) for k,v in raw.items()}
    source=[r for r in rows if r["study"] in (1,2)];e2=[r for r in source if r["study"]==2]
    observations=counts(reveal(2,rows),e2);emobs={(2,k):v for k,v in observations.items()};links=source_links(source,raw)
    @lru_cache(None)
    def router(exclusions):
        return Route([r for r in source if r["story"] not in exclusions],docs,bags,links)
    pred={};history={}
    for held in sorted({r["story"] for r in e2}):
        train=[r for r in e2 if r["story"]!=held];fitrows=[r for r in source if r["story"]!=held];outer=[r for r in rows if r["story"]==held and r["study"]==3]
        em,ed=emissions(fitrows,emobs,docs);X={};prior={};hv={}
        for r in train+outer:
            excl=tuple(sorted({r["story"],held}));k=r["id"];s=docs[k];rt=router(excl)
            X[k]=measurements(rt,s,bags[k]);prior[k]=source_schedule([r for r in source if r["story"] not in excl],len(s));hv[k]=em(s)
        history[held]={"source_emissions":ed}
        for arm in ("first_cue","first_cue_moment"):
            results,details=stage(train,outer,X,prior,observations,hv,arm.endswith("moment"))
            history[held][arm]=details
            for key,(z,position) in results.items():
                pred["strict."+arm+"."+key]=z;pred["strict.first_cue_position."+key]=position
        print("Conserved source and held pathways",held,flush=True)
    seal=json.loads((OLD/"seal.json").read_text())
    if sha(OLD/"hazards.npz")!=seal["files"]["hazards.npz"]:raise ValueError("Comparison forecast changed")
    with np.load(OLD/"hazards.npz") as older:
        for k in older.files:pred[k]=older[k]
    np.savez_compressed(folder/"hazards.npz",**pred)
    (folder/"fits.json").write_text(json.dumps(history,indent=2)+"\n")
    (folder/"seal.json").write_text(json.dumps({"protocol":sha(PLAN),"prior":sha(OLD/"seal.json"),"native":sha(archive/"receipt.json"),
        "files":{k:sha(folder/k) for k in ("hazards.npz","fits.json")},"third_cohort_responses_in_fit":False},indent=2)+"\n")

def grade():
    folder=OUT/"forecast";lock=json.loads((folder/"seal.json").read_text())
    if any(sha(folder/k)!=v for k,v in lock["files"].items()):raise ValueError("Source forecast changed")
    rows=json.loads((BASE/"memory/study-map.json").read_text());observed=reveal(3,rows)
    arms=("original","directional","trial","moment","oracle","first_cue_position","first_cue","first_cue_moment")
    with np.load(folder/"hazards.npz") as f:
        scores=summarize(f,observed,376,cohorts=("strict",),methods=arms,focus="first_cue_moment",controls=arms[:-1])
        details={}
        for g in sorted({r["story"] for r in observed}):
            per={}
            for a in ("inconsistent","neutral"):
                group=[r for r in observed if r["story"]==g and r["arm"]==a];k=group[0]["id"];t=group[0]["target"]-1
                human=counts(group,[{"id":k}])[k][1][t]/len(group)
                per[a]={"human":float(human),**{m:float(f["strict."+m+"."+k][t]*np.prod(1-f["strict."+m+"."+k][:t])) for m in arms}}
            details[g]={m:v-per["neutral"][m] for m,v in per["inconsistent"].items()}
        scores["family_effects"]=details
    (OUT/"scores.json").write_text(json.dumps(scores,indent=2,allow_nan=False)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("fit","score"));p.add_argument("archive",nargs="?",type=Path)
    args=p.parse_args()
    if args.action=="fit":run(args.archive)
    else:grade()
