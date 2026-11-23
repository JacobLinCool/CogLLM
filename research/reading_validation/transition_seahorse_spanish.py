"""Freeze English-to-Spanish forecasts before testing their transport to untouched professional judgments."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from research.reading_validation.memory_data import h
from research.reading_validation.summary_understanding import SummaryUnderstanding
from research.reading_validation.summeval_forecast import open_old
from research.reading_validation.transition_seahorse_data import DEST, sha
from research.reading_validation.transition_seahorse_features import document_features, load, save
from research.reading_validation.transition_seahorse_fit import corpus, key
from research.reading_validation.transition_seahorse_gated import TASKS, answers_report, compare_predictions, labels
from research.reading_validation.transition_seahorse_rank import grouped_auc
from research.reading_validation.transition_seahorse_score import group_rows, stats, summary, within
from research.reading_validation.transition_seahorse_spanish_data import INPUT, PLAN, spanish

FORECAST=DEST/"spanish-forecasts-before-grades.npz"


def project(folder):
    locked=load(DEST/"spanish-exposure-seal.json")
    receipt=load(folder/"receipt.json");features=folder/"features.npz"
    weights=DEST/"gated-weights.npz"
    if (receipt["file"] != sha(features) or receipt["source"] != locked["input_sha256"] or
            sha(INPUT) != locked["input_sha256"] or sha(weights) != locked["native_weights"] or
            sha(PLAN) != locked["protocol_sha256"]):
        raise ValueError("Spanish source text, native features, preregistration or English weights changed")
    panels=list(spanish())
    unique={h(r["summary"]):r["summary"] for r in panels}
    if set(receipt["articles"])|set(receipt["skipped"])!=set(unique):
        raise ValueError("Unexpected external case coverage")
    output={};none=dict(receipt["skipped"])
    production=SummaryUnderstanding.load();p0={}
    frozen,_=open_old()
    with np.load(features,allow_pickle=False) as archive:
        for uid,detail in receipt["articles"].items():
            text=unique[uid]
            x,reason=document_features(archive,uid,detail,text,frozen)
            if reason:
                none[uid]=reason
                continue
            output[uid]=x
            parts={k:archive[uid+"."+k] for k in ("char","surprisal","rel","words")}
            original=production.from_arrays(text,parts)
            if not original["available"]:
                raise ValueError("Native Q1 and research projection disagree on availability")
            p0[uid]=original["probability_difficult"]
    scored=[r for r in panels if h(r["summary"]) in output]
    x=np.stack([output[h(r["summary"])] for r in scored])
    ids=[key(r) for r in scored]
    if len(ids)!=len(set(ids)):
        raise ValueError("Duplicate Spanish row key")
    predictions={"keys":np.asarray(ids)}
    with np.load(weights,allow_pickle=False) as bundle:
        for typ,width in (("form",15),("native",187)):
            z=(x[:,:width]-bundle[typ+".center"])/bundle[typ+".scale"]
            for fit in ("independent","adaptive"):
                name=typ+"_"+fit;c=bundle[name+".theta"];cal=bundle[name+".cal"]
                predictions[name]=expit(cal[:,0]+(c[0]+z@c[1:])*cal[:,1])
    expected=np.array([p0[h(r["summary"])] for r in scored])
    diff=float(np.max(abs(expected-predictions["native_independent"][:,0])))
    if diff>1e-9:
        raise ArithmeticError("English production observer loses parity on the Spanish word archive")
    np.savez_compressed(FORECAST,**predictions)
    save(DEST/"spanish-forecast-seal.json",{
        "protocol":sha(PLAN),"weights":sha(weights),"held_forecasts":sha(FORECAST),
        "originals":len(panels),"scored":len(ids),"unique_scored_texts":len(output),
        "unscored_original_rows":{key(r):none[h(r["summary"])] for r in panels if h(r["summary"]) in none},
        "readout_parity":diff,"native_input_sha256":sha(INPUT),"native_receipt":sha(folder/"receipt.json"),
        "seconds":receipt["seconds"],"batches":receipt["encoder_batches"],"spanish_grades_seen":False})
    print("Sealed Spanish",len(ids),len(output),diff,sha(FORECAST),flush=True)


def stratified(rows,pred):
    y=labels(rows);task={}
    all_corpora=np.array([corpus(r["gem_id"]) for r in rows]);system=np.array([r["model"] for r in rows])
    g=np.array([r["gem_id"] for r in rows])
    for j,name in enumerate(TASKS):
        ok=np.isfinite(y[:,j]);yy=y[ok,j];gg=g[ok];typ=all_corpora[ok];model=system[ok]
        groups,gid=group_rows(gg);G=len(groups)
        C=np.random.default_rng(6027+j).multinomial(G,np.full(G,1/G),size=1800)
        C=np.vstack((np.ones(G),C))
        keypairs=np.char.add(np.char.add(typ,"/"),model)
        strata=[np.flatnonzero(keypairs==v) for v in np.unique(keypairs)]
        methods={};datasets={}
        for method in ("native_independent","native_adaptive","form_adaptive"):
            p=pred[method][ok,j]
            local,n=within(groups,yy,p);take=np.isfinite(local)
            z=(C[:,take]@local[take])/C[:,take].sum(1)
            rank=grouped_auc(yy,p,gid,strata,C)
            overall=grouped_auc(yy,p,gid,[np.arange(len(yy))],C)
            methods[method]={"within_article":summary(z[0],z[1:]),"opposite_response_pairs":int(n.sum()),
                "within_article_clusters":int(take.sum()),"within_system_and_corpus":summary(rank[0],rank[1:]),
                "AUROC":summary(overall[0],overall[1:])}
        for dataset in np.unique(typ):
            selection=typ==dataset;actual=yy[selection];p=pred["native_adaptive"][ok,j][selection]
            data={"rows":len(actual),"no":int(actual.sum()),"mean_predicted":float(p.mean())}
            if len(np.unique(actual))==2:
                data.update(AUROC=float(roc_auc_score(actual,p)),average_precision=float(average_precision_score(actual,p)))
            datasets[dataset]=data
        task[name]={"comparisons":methods,"corpus_native":datasets}
    return task


def evaluate():
    seal=load(DEST/"spanish-forecast-seal.json")
    if sha(FORECAST)!=seal["held_forecasts"] or sha(PLAN)!=seal["protocol"]:
        raise ValueError("Frozen Spanish predictions or plan changed")
    original=list(spanish(grades=True));all_rows={key(r):r for r in original}
    with np.load(FORECAST,allow_pickle=False) as z:
        rows=[all_rows[k] for k in z["keys"]];pred={k:z[k] for k in z.files if k!="keys"}
    summary_stats=compare_predictions(rows,pred)
    contamination=set(load(DEST/"spanish-exposure-seal.json")["affected_spanish_articles"])
    clean=np.array([r["gem_id"] not in contamination for r in rows])
    sensitivity=compare_predictions([r for i,r in enumerate(rows) if clean[i]],{k:v[clean] for k,v in pred.items()})
    yy=labels(rows)[:,0]; valid=np.isfinite(yy)
    threshold=SummaryUnderstanding.load().threshold
    threshold_result=stats(yy[valid],pred["native_independent"][valid,0],threshold)[0]
    save(DEST/"spanish-results.json",{
        "protocol":sha(PLAN),"forecast_sha256":sha(FORECAST),"original_rows":len(original),"english_validation_threshold":threshold,
        "official_answer_counts":answers_report(original),"excluded_grades":dict(Counter(all_rows[k]["question1"] for k in seal["unscored_original_rows"])),
        "threshold_transfer":threshold_result,"conditioned_ranks":stratified(rows,pred),"held":summary_stats,
        "no_matched_articles_sensitivity":sensitivity})
    print(summary_stats["metrics"]["native_adaptive"],summary_stats["paired"]["native_adaptive minus form_adaptive"],flush=True)


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("operation",choices=("project","evaluate"));parser.add_argument("folder",type=Path,nargs="?")
    args=parser.parse_args()
    if args.operation=="project":project(args.folder)
    else:evaluate()
