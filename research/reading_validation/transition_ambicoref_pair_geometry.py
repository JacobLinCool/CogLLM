"""Verify letter-for-letter minimal pairs and their common-prefix uncertainty trajectories."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_ambicoref_prepare import OUT,ROOT
from research.reading_validation.transition_score import dump,sha


def run():
    src=OUT/"ambicoref-confirmation.json";study=json.loads(src.read_text());votes=json.loads((ROOT/"ambicoref-observed-choices.json").read_text())
    packet=ROOT/"ambicoref-reading-stimuli.jsonl";texts={r["id"]:r for r in map(json.loads,packet.read_text().splitlines())}
    reports={}
    for design in ("exact_matched_word_changes","documented_verb_class_word_changes"):
        rows=study[design]["rows"];keys=sorted({r[k] for r in rows for k in ("ambiguous","unambiguous")});lookup={k:j for j,k in enumerate(keys)}
        parent=list(range(len(keys)))
        def find(x):
            while x!=parent[x]:
                parent[x]=parent[parent[x]];x=parent[x]
            return x
        edges=[]
        for r in rows:
            a=texts[r["ambiguous"]];b=texts[r["unambiguous"]];position=r["changed_word"]
            x,y=a["words"][position];i,j=b["words"][position]
            if (a["text"][:x]!=b["text"][:i] or a["text"][y:]!=b["text"][j:] or a["text"][x:y]==b["text"][i:j]):
                raise ValueError("A purported exact pair changed another source character")
            ai,bi=lookup[a["id"]],lookup[b["id"]];parent[find(ai)]=find(bi);edges.append((ai,bi))
        a,b=np.array(edges).T
        origin,block=np.unique([find(i) for i in range(len(keys))],return_inverse=True);g=block[a];n=np.bincount(g)
        if not np.array_equal(g,block[b]):raise ValueError("A pair crosses independent sentence sets")
        trial=[(i,s,c) for i,k in enumerate(keys) for s,c in votes[k]["questionnaire_choices"]]
        item,sheet,response=np.asarray(trial).T;ids,sheet=np.unique(sheet,return_inverse=True);beg=np.r_[0,np.flatnonzero(np.diff(item))+1]
        weights=np.ones(len(rows))/n[g]/len(n)
        def human(w):
            per=w[:,sheet];u=np.add.reduceat(per*(response==2),beg,axis=1)/np.add.reduceat(per,beg,axis=1)
            return u[:,a]-u[:,b]
        h=human(np.ones((1,len(ids))))[0];methods=list(rows[0]["models"]);totals=[];bounds={}
        for m in methods:
            p=np.array([r["models"][m] for r in rows])
            totals.append(p[:,1:]);bounds[m]=float(abs(p[:,0]).max())
        Z=np.stack(totals,axis=1);pt=np.einsum("e,emj->mj",weights,Z);hp=float(h@weights)
        rng=np.random.default_rng(2219);curves=[];obs=[]
        for t in range(0,4000,32):
            size=min(32,4000-t);s=rng.exponential(size=(size,len(ids)));c=rng.exponential(size=(size,len(n)))
            w=c[:,g]/n[g];w/=w.sum(1,keepdims=True)
            curves.append(np.einsum("be,emj->bmj",w,Z));obs.append((w*human(s)).sum(1))
        w=np.concatenate(curves);u=np.concatenate(obs)
        out={"exact_raw_text_pairs":len(rows),"distinct_sentences":len(keys),"independent_text_components":len(n),
             "largest_component_edges":int(n.max()),"questionnaire_assignments":len(ids),"largest_prechange_probability_difference":bounds,
             "human_explicit_ambiguity_difference":[hp,*np.quantile(u,[.025,.975]).tolist()],"ambiguity_condition_minus_control":{}}
        for i,m in enumerate(methods):
            out["ambiguity_condition_minus_control"][m]={key:[float(pt[i,j]),*np.quantile(w[:,i,j],[.025,.975]).tolist()]
                    for j,key in enumerate(("at_changed_word","after_final_word"))}
        reports[design]=out
    dump(OUT/"letter-exact-pair-confirmation.json",{"forecast_result_sha256":sha(src),"original_sentence_file_sha256":sha(packet),
         "designs":reports,"scope":"Final human judgments; timing identifies observer movement only"})
    print(reports["documented_verb_class_word_changes"],flush=True)


if __name__=="__main__":run()
