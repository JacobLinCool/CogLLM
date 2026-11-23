"""Transfer a source-fitted first-report model to end-of-passage school judgments."""

import argparse
import json
from pathlib import Path

import numpy as np

from research.reading_validation.memory_data import load_rows, sha
from research.reading_validation.memory_fit import design
from research.reading_validation.retrieval_calibration import predict
from research.reading_validation.retrieval_data import RESULT, TRACK
from research.semantic_tracking import larrc_consistency_evaluate as evaluation


SCHOOL = TRACK/"larrc-consistency"


def forecast(calibration, readout, dest):
    source=json.loads((calibration/"seal.json").read_bytes())
    binary=json.loads((readout/"seal.json").read_bytes())
    if (sha(calibration/"models.json")!=source["files"]["models.json"] or
        sha(readout/"currie.json")!=binary["files"]["currie.json"] or
        sha(readout/"frick.json")!=source["extra"]):
        raise ValueError("Cross-study predictors differ from the original locked calibration")
    models=json.loads((calibration/"models.json").read_bytes())
    evidence=json.loads((readout/"currie.json").read_bytes())
    historical={r["id"]:r["methods"] for r in load_rows(SCHOOL/"frozen-forecasts/forecasts.jsonl")}
    forecasts=[]
    for article in load_rows(SCHOOL/"prepared/stimuli.jsonl"):
        key=article["id"]
        surface=design(article["text"].split("\n\n"))
        surprisal=np.column_stack((surface,evidence[key]["surprisal"]))
        transfer=np.column_stack((surprisal,evidence[key]["retrieval"]))
        row={"historical_"+k:v for k,v in historical[key].items()}
        for name,features in (("surface",surface),("surprisal",surprisal),("transfer",transfer)):
            for objective in ("path","paired"):
                label=f"{name}.{objective}"
                selected=[m for k,m in models.items() if k.endswith("."+label)]
                if len(selected)!=20:
                    raise ValueError("Every source calibration must contribute once")
                estimates=[-np.expm1(np.log1p(-predict(m,features)).sum()) for m in selected]
                row[label]=float(np.mean(estimates))
        forecasts.append(dict(id=key,methods=row))
    if len(forecasts)!=48:
        raise ValueError("Expected 24 complete contrasts")
    dest.write_text(json.dumps(dict(calibration=sha(calibration/"seal.json"),readout=sha(readout/"seal.json"),
                                    articles=forecasts),allow_nan=False)+"\n")


def score(file,dest):
    source=json.loads(file.read_bytes())
    layout=load_rows(SCHOOL/"prepared/layout.jsonl")
    positions,genres,matched=evaluation.index_layout(layout)
    present,report=evaluation.index_outcomes(load_rows(SCHOOL/"prepared/outcomes.jsonl"),positions,len(genres))
    names=list(source["articles"][0]["methods"])
    z=np.full((len(genres),2,len(names)),np.nan)
    for row in source["articles"]:
        z[positions[row["id"]]]=[row["methods"][m] for m in names]
    if not ((z>0)&(z<1)).all():
        raise ValueError("Missing school forecast probability")
    primary=[np.flatnonzero(genres==g) for g in range(2)]
    exact=[np.flatnonzero((genres==g)&matched) for g in range(2)]
    sampled=evaluation.bootstrap(present,report,z,primary,exact)
    estim=evaluation.estimate
    i=names.index("transfer.paired")
    results={}
    for effect,groups in (("narrative_minus_expository",primary),("far_minus_near",exact)):
        h,p,loss=evaluation.contrasts(evaluation.point(present,report,z,groups),effect)
        hb,pb,lb=sampled[effect]
        results[effect]=dict(families=list(map(len,groups)),human=estim(h[0],hb),methods={
            m:dict(model_effect=estim(p[0,j],pb[:,j]), model_minus_human=estim(p[0,j]-h[0],pb[:,j]-hb),
                   path_to_final_nll=estim(loss[0,j],lb[:,j]),
                   gain_of_paired_retrieval=estim(loss[0,j]-loss[0,i],lb[:,j]-lb[:,i]))
            for j,m in enumerate(names)})
    output=dict(forecast=sha(file),readers=present.shape[-1],judgments=int(present.sum()),results=results)
    dest.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print("Transferred calibrated hazards to school readers",output["readers"],output["judgments"])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("forecast","score"))
    parser.add_argument("--calibration",type=Path,default=RESULT/"context-calibration")
    parser.add_argument("--readout",type=Path,default=RESULT/"readout")
    parser.add_argument("--file",type=Path,default=RESULT/"school-forecast.json")
    parser.add_argument("--dest",type=Path,default=RESULT/"school-scores.json")
    args=parser.parse_args()
    if args.mode=="forecast":
        forecast(args.calibration,args.readout,args.file)
    else:
        score(args.file,args.dest)
