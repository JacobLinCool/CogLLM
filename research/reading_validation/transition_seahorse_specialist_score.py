"""Compare the published source-aware specialist and the summary-only readout on identical official English judgments."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from research.reading_validation.transition_seahorse_data import DEST,WORK,sha,source_rows
from research.reading_validation.transition_seahorse_features import load,save
from research.reading_validation.transition_seahorse_fit import LABEL,key
from research.reading_validation.transition_seahorse_rank import grouped_auc
from research.reading_validation.transition_seahorse_score import group_rows,stats,summary,within


def evaluate(y, methods, story):
    groups,gid=group_rows(story);n=len(groups);rng=np.random.default_rng(73135)
    C=rng.multinomial(n,np.full(n,1/n),size=3000).astype(np.int16)
    C=np.vstack((np.ones(n,dtype=np.int16),C));d=C@np.bincount(gid)
    scores={}; reps={}
    for k,p in methods.items():
        prob=expit(p);s,loss,brier=stats(y,prob,.5)
        scores[k]={};reps[k]={}
        for name,val in (("nll",loss),("brier",brier)):
            packed=(C.astype(float)@np.bincount(gid,weights=val))/d
            reps[k][name]=packed;scores[k][name]=summary(packed[0],packed[1:])
        roc=grouped_auc(y,p,gid,[np.arange(len(y))],C)
        reps[k]["AUROC"]=roc;scores[k]["AUROC"]=summary(roc[0],roc[1:])
        score,pairs=within(groups,y,p);ok=np.isfinite(score);macro=C[:,ok].astype(float)@score[ok]/C[:,ok].sum(1)
        reps[k]["within_article"]=macro;scores[k]["within_article"]=summary(macro[0],macro[1:])
        scores[k]["average_precision_no"]=s["average_precision_no"]
    contrast={}
    for k in methods:
        if k=="published_q1":continue
        contrast[k+" minus published_q1"]={}
        for name in ("nll","AUROC","within_article"):
            delta=reps[k][name]-reps["published_q1"][name]
            contrast[k+" minus published_q1"][name]=summary(delta[0],delta[1:])
    return {"rows":len(y),"no":int(y.sum()),"articles":n,"informative_articles":int(ok.sum()),
            "within_article_pairs":int(pairs.sum()),"scores":scores,"paired":contrast}


def main(folder):
    from research.reading_validation.transition_seahorse_specialist_data import TABLES,PLAN,INPUT
    import pyarrow.parquet as pq
    from hashlib import sha256
    receipt=load(folder/"receipt.json"); f=folder/"specialist-forecasts.jsonl"
    if receipt["source_sha256"]!=sha(INPUT) or receipt["protocol_sha256"]!=sha(PLAN) or receipt["forecasts_sha256"]!=sha(f):
        raise ValueError("Changed specialist or input identities")
    output={q["id"]:q for q in map(json.loads,f.read_text().splitlines())}
    rows=list(source_rows("test",grades=True));wanted={r["gem_id"] for r in rows}
    text={}
    for name,(col,expected) in TABLES.items():
        path=WORK/f"seahorse-specialist-{name}.parquet"
        if sha(path)!=expected:raise ValueError("Article archive changed")
        for batch in pq.ParquetFile(path).iter_batches(columns=["gem_id",col],batch_size=512):
            text.update({r["gem_id"]:r[col] for r in batch.to_pylist() if r["gem_id"] in wanted})
    src=DEST/"forecasts-before-held-responses.npz"
    if sha(src)!=load(DEST/"fit-seal.json")["forecast_sha256"]:raise ValueError("Source prediction lock failed")
    with np.load(src,allow_pickle=False) as z:
        ix={k:i for i,k in enumerate(z["keys"])}
        wanted_models=("native","form","character_native","metadata","metadata_native")
        compact={k:z[k] for k in wanted_models}
    forecasts=[];all_official=[];vis=Counter()
    for r in rows:
        article=text[r["gem_id"]];identity=sha256(json.dumps([article,r["summary"]],ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
        row=output[identity]
        visible="full" if row["visible_summary_pieces"]==row["all_summary_pieces"] else "absent" if row["visible_summary_pieces"]==0 else "partial"
        vis[visible]+=1
        if r["question1"] not in LABEL:continue
        y=LABEL[r["question1"]];all_official.append((y,row["log_odds_no"]))
        if key(r) not in ix:continue
        odds={k:float(np.log(np.clip(p[ix[key(r)]],1e-15,1)/np.clip(1-p[ix[key(r)]],1e-15,1))) for k,p in compact.items()}
        odds["published_q1"]=row["log_odds_no"]
        forecasts.append((y,r["gem_id"],visible,odds))
    matrix=np.asarray(all_official); paired={}
    for label,selected in (("all_shared",forecasts),("full_summary_visible",[q for q in forecasts if q[2]=="full"])):
        yy=np.array([q[0] for q in selected]);groups=np.array([q[1] for q in selected])
        p={k:np.array([q[3][k] for q in selected]) for k in selected[0][3]}
        paired[label]=evaluate(yy,p,groups)
    s,_loss,_brier=stats(matrix[:,0],expit(matrix[:,1]),.5)
    dest={"receipt":receipt,"source":sha(INPUT),"protocol":sha(PLAN),"full_eligible_specialist":s,
          "classification_token_visibility_counts":dict(vis),"shared":paired,"input_privileges":"mT5 uses original article plus summary. Rater and CogLLM see only the summary."}
    save(DEST/"specialist-comparison.json",dest)
    print("English specialist",s["AUROC"],s["average_precision_no"],"visibility",vis,
          "paired",paired["all_shared"]["paired"]["native minus published_q1"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("folder",type=Path);main(p.parse_args().folder)
