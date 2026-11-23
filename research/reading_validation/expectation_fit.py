"""Fit human response functions separately on groups of unexposed source stimuli."""

import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from wordfreq import zipf_frequency

from research.reading_validation.expectation_data import OUT,WORK,PUBLIC
from research.reading_validation.expectation_measure import Materials,Observations
from research.reading_validation.memory_data import h,sha

OUTCOMES={"michaelov":("plausibility","eeg"),"brothers":("timing",)}
REFERENCE={"pythia":"EleutherAI__pythia-2_8b","rwkv":"RWKV__rwkv-4-3b-pile","mamba":"state-spaces__mamba-2_8b-hf"}
METHODS=("constant","form","qwen_surprisal","expectation",*REFERENCE)


def design(archive,mat):
    native=json.loads((archive/"receipt.json").read_bytes())
    if sha(archive/"features.npz")!=native["file"] or native["source"]!=mat.seal["text"]:
        raise ValueError("Native target features differ from their single-pass lock")
    independent={};files={}
    for data,(stim,*_) in PUBLIC.items():
        for model,repo in REFERENCE.items():
            path=WORK/(stim.removesuffix(".stims")+".surprisal."+repo+".causal.output")
            with path.open() as f:
                independent[data,model]={h(r["FullSentence"]):float(r["Surprisal"]) for r in csv.DictReader(f,delimiter="\t")}
            files[path.name]=sha(path)
    out=defaultdict(dict);missing=Counter()
    with np.load(archive/"features.npz",allow_pickle=False) as saved:
        for study,rows in mat.rows.items():
            for k,row in rows.items():
                a,b=row["span"];coords=saved[k+".word"]
                found=np.flatnonzero((coords[:,0]==a)&(coords[:,1]==b)&(coords[:,4]==1))
                if len(found)!=1:
                    missing[study]+=1
                    continue
                x=saved[k+".features"][found[0]].astype(float)
                lex=np.r_[x[:4],zipf_frequency(mat.text[k][a:b],"en")]
                extra={}
                for m in REFERENCE:
                    table=independent[row["collection"],m]
                    val=table.get(k,table.get(row["source_id"]))
                    if val is None or not np.isfinite(val):
                        break
                    extra[m]=np.r_[lex,val]
                if len(extra)!=len(REFERENCE):
                    missing[study]+=1
                    continue
                out[study][k]=dict(form=lex,constant=np.empty(0),qwen_surprisal=np.r_[lex,x[4]],
                                   expectation=np.r_[lex,x[4:15]],**extra)
    return out,dict(uncertified_common=missing,comparators=files,native=sha(archive/"receipt.json"))


def fit(items,method):
    # Equal mass per source study, material family, and presentation within family.
    lab_n=Counter({lab:len({g for g,y,x in vals}) for lab,vals in items.items()})
    counts={lab:Counter(g for g,y,x in vals) for lab,vals in items.items()}
    xs=[];ys=[];weights=[]
    for lab,records in items.items():
        for group,y,features in records:
            xs.append(features[method]);ys.append(y)
            weights.append(1/(len(items)*lab_n[lab]*counts[lab][group]))
    x=np.stack(xs);y=np.array(ys);w=np.array(weights)
    if not np.isclose(w.sum(),1) or not np.isfinite(y).all():
        raise ValueError("Invalid source observation weights")
    mu=w@x;sd=np.sqrt(w@((x-mu)**2));sd[sd<1e-7]=1
    mean=w@y;scale=np.sqrt(w@((y-mean)**2)) or 1.
    z=(x-mu)/sd
    # The intercept is the unpenalized, weighted source mean.
    lhs=(z.T*w)@z+.02*np.eye(z.shape[1]);rhs=(z.T*w)@((y-mean)/scale)
    b=np.linalg.solve(lhs,rhs)
    err=float(max(np.abs(lhs@b-rhs),default=0))
    if err>1e-8:
        raise ArithmeticError("The regularized optimum failed its normal equations")
    return dict(mean=float(mean),scale=float(scale),center=mu.tolist(),sd=sd.tolist(),
                slopes=b.tolist(),error=err,source_rows=len(y))


def predict(spec,x):
    v=(np.asarray(x)-spec["center"])/spec["sd"]
    return float(spec["mean"]+spec["scale"]*(v@spec["slopes"]))


def run(archive,dest):
    if dest.exists():
        raise FileExistsError(dest)
    mat=Materials();features,check=design(archive,mat)
    sets={s:{k for k in rows if mat.assignment[s,k][1]=="train"} for s,rows in features.items()}
    training=Observations(mat,sets);betas=training.slopes()
    forecasts={};models={}
    for study,rows in features.items():
        for task in OUTCOMES.get(study,("eeg",)):
            modes=["within"] if task!="eeg" else ["within","other_studies"]
            for mode in modes:
                prior={study:sets[study]} if mode=="within" else mat.other_materials(study)
                materials={}
                for lab,ks in prior.items():
                    y=training.responses(lab,task,betas.get(lab,0.))
                    items=[(mat.assignment[lab,k][0],y[k],features[lab][k]) for k in sorted(ks&set(y)&set(features[lab]))]
                    if items:
                        materials[lab]=items
                if not materials:
                    raise ValueError("No unexposed source study material")
                key=study+"|"+task+"|"+mode
                fitted={m:fit(materials,m) for m in METHODS};models[key]=fitted
                # A source EEG corpus without an available baseline cannot furnish a universal
                # scalp-voltage intercept. Keep the target as actually recorded for external transfer.
                beta=betas.get(study,0.) if mode=="within" else (
                    float(np.mean([betas[lab] for lab in materials if betas.get(lab,0.)!=0]))
                    if study!="michaelov" and any(betas.get(lab,0.)!=0 for lab in materials) else 0.)
                forecasts[key]=dict(beta=beta,methods=list(METHODS),source={k:len(v) for k,v in materials.items()},
                    held={k:[predict(fitted[m],rows[k][m]) for m in METHODS] for k in sorted(rows) if mat.assignment[study,k][1]=="test"})
    dest.mkdir(parents=True)
    for name,obj in (("predictions.json",forecasts),("source-fits.json",models)):
        (dest/name).write_text(json.dumps(obj,allow_nan=False)+"\n")
    stamp=dict(inventory=sha(OUT/"input-seal.json"),estimator=sha(Path(__file__).with_name("expectation-estimator.md")),
               extraction=check,rows={s:len(v) for s,v in features.items()},splits={
               s:dict(Counter(mat.assignment[s,k][1] for k in v)) for s,v in features.items()},
               files={n:sha(dest/n) for n in ("source-fits.json","predictions.json")})
    (dest/"seal.json").write_text(json.dumps(stamp,indent=2)+"\n")
    print("Sealed",len(forecasts),"response and transfer comparisons across",len(features),"study datasets")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path);p.add_argument("--dest",type=Path,default=OUT/"forecast")
    run(**vars(p.parse_args()))
