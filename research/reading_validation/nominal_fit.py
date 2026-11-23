"""Fit source noun decisions and seal all candidate forecasts before opening gold votes."""
from argparse import ArgumentParser
import json
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.nominal_data import OUT,PLAN,ROOT,REJECT
from research.reading_validation.nominal_estimator import Words,action_model,action_predict,range_inputs,range_model
from research.reading_validation.referent_fit import projects
from research.reading_validation.referent_sketch_prepare import material

def pool(raw,rows,center,basis):
    return [{**r,"features":Words(r,(raw[r["id"]+"__h"].astype(float)-center)@basis)} for r in rows]

def run(native):
    intake=json.loads((OUT/"observations.json").read_text());rec=json.loads((native/"receipt.json").read_text())
    path=native/"word-states.npz";src=ROOT/"nominal-source-observations.json"
    if digest(PLAN)!=intake["protocol_sha256"] or digest(src)!=intake["files"]["source"] or digest(path)!=rec["archive_sha256"] or rec["external_outcomes"]!=0:
        raise ValueError("Study, source votes, or sealed encoder outputs changed.")
    obs=json.loads(src.read_text());rows=material()["pointer"];tr=[r for r in rows if r["fold"]=="train"]
    with np.load(path) as raw:
        mu,basis,words=projects(tr,raw);source=pool(raw,tr,mu,basis)
        remaining=pool(raw,[r for r in rows if r["fold"]!="train"],mu,basis)
    fit_rows=[(r,obs[r["id"]]) for r in source]
    data=range_inputs(fit_rows)
    models={};diag={}
    for m,ctx in (("form",False),("native",True)):
        ac,fa=action_model(fit_rows,ctx)
        rg,fr=range_model(data,ctx,models["form"][1] if ctx else None)
        models[m]=(ac,rg);diag[m]={"action":fa,"range":fr}
        print(m,diag[m],flush=True)
    nquery=len(data[3]);nwords=len(data[0]);del data
    weights={"mean":mu,"basis":basis};outputs={};locations={}
    for m,(ac,rg) in models.items():
        for name,v in zip(("action_mu","action_sd","action_coef"),ac,strict=True):weights[m+"."+name]=v
        weights[m+".range_mu"]=rg.form_mu;weights[m+".range_sd"]=rg.form_sd;weights[m+".range_coef"]=rg.coef
    for r in remaining:
        ix=[];actions=[];refs=[];f=r["features"]
        for q,w in enumerate(f.forms):
            if not q or w in REJECT or not w[0].isalpha() or not w[-1].isalpha():continue
            ix.append(q)
            actions.append(np.stack([action_predict(ac,f.action(q,rg.context)) for ac,rg in models.values()]))
            refs.append(np.stack([rg.predict(f,q) for ac,rg in models.values()]).astype("float32"))
        if ix:
            outputs[r["id"]+".a"]=np.stack(actions);outputs[r["id"]+".r"]=np.concatenate(refs,axis=1)
        locations[r["id"]]=ix
    np.savez_compressed(OUT/"source-models.npz",**weights);np.savez_compressed(OUT/"predictions.npz",**outputs)
    write_json(OUT/"queried-positions.json",locations)
    write_json(OUT/"fit.json",{"plan":digest(PLAN),"sketch":rec["archive_sha256"],"source_basis_words":words,
         "source_click_queries":nquery,"source_candidate_words":nwords,"optimization":diag,"models":list(models),
         "unmarked_forecast_positions":sum(map(len,locations.values())),"files":{k:digest(OUT/k) for k in ("source-models.npz","predictions.npz","queried-positions.json")}})

if __name__=="__main__":
    p=ArgumentParser();p.add_argument("native",type=Path);run(p.parse_args().native)
