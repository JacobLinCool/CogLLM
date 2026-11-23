"""Restore each independent initial noun-reference decision from source XML."""
from collections import Counter,defaultdict
from itertools import chain
import json
from pathlib import Path
import re
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.transition_pointer_features import FORMS
from research.reading_validation.transition_direct_data import PD,PDFINGER,display,gold_docs
from research.reading_validation.transition_game_prepare import silver_docs

ROOT=Path(".cache/cogllm/reading-validation")
OUT=Path("research/results/reading-validation-nominal-reference")
PLAN=Path("research/reading_validation/nominal-reference-protocol.md")
DETERMINERS=frozenset("a an the this these that those some any each every most both another our your their his her its my".split())
REJECT=frozenset(FORMS)|DETERMINERS|frozenset("who whom whose one ones i me you we us mine ours yours myself yourself ourselves which what there".split())
ACTIONS=("DO","DN","PR","NR")

def restore(row,tree,elements,raw,exclude_users=frozenset(),minimum_readers=4):
    text,coords,_=display(tree,elements,raw)
    if text!=row["text"]:raise ValueError("Game text and sealed input differ")
    word={tuple(span):i for i,span in enumerate(row["words"])}
    rect={el:i for i,el in enumerate(elements)}
    spans={e.attrib["id"]:tuple(rect[w] for w in e.iter("W")) for e in tree.iter("ne")}
    groups=defaultdict(list);diag=Counter()
    for mark in tree.iter("PDante"):
        src=spans.get(mark.get("id"),())
        if not src or len(src)>2:continue
        if any(k in mark.attrib for k in ("start_chr","end_chr","visibility")):continue
        if len(src)==2 and (src[0]+1!=src[1] or raw[src[0]].lower() not in DETERMINERS):continue
        final=raw[src[-1]].casefold()
        if not re.fullmatch(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*",final) or final in REJECT:continue
        q=word.get(coords[src[-1]])
        if q and all(coords[i] in word for i in src):groups[q].append((mark,src))
    kept={}
    for q,choices in groups.items():
        if len(choices)!=1:diag["colliding_mention_ends"]+=1;continue
        mark,src=choices[0];votes=defaultdict(set);bad=False
        for a in mark.iter("anchor"):
            if a.get("mode")!="a":continue
            uid,action=a.get("user_id"),a.get("type")
            if uid in exclude_users:continue
            if not uid or action not in ACTIONS:raise ValueError("Unknown independent choice")
            rng=(action,-1,-1)
            if action=="DO":
                before=spans.get(a.get("ante"),())
                if not before or max(before)>=min(src):
                    bad=True;break
                ids=[word[coords[i]] for i in before if coords[i] in word]
                if not ids or ids[-1]>=q or ids!=list(range(ids[0],ids[-1]+1)):
                    bad=True;break
                rng=(action,ids[0],ids[-1]+1)
            votes[uid].add(rng)
        if bad:diag["unusable_earlier_click_at_queued_mention"]+=1;continue
        actual=[next(iter(k)) for k in votes.values() if len(k)==1]
        diag["inconsistent_repeated_person_action"]+=sum(len(k)>1 for k in votes.values())
        if len(actual)<minimum_readers:diag["under_minimum_independent_people" if minimum_readers!=4 else "under_four_independent_people"]+=1;continue
        counts=Counter(a[0] for a in actual)
        linked=Counter((a,b) for s,a,b in actual if s=="DO")
        kept[str(q)]={"action":[counts[k] for k in ACTIONS],"clicks":[[a,b,n] for (a,b),n in sorted(linked.items())],
                      "markable_start":word[coords[src[0]]]}
        diag["initial_reader_choices"]+=len(actual);diag["queries"]+=1;diag["literal_clicks"]+=counts["DO"]
    return kept,diag

def main():
    from research.reading_validation.referent_sketch_prepare import material
    if digest(PD)!=PDFINGER:raise ValueError("Official game source archive changed")
    rows={r["id"]:r for r in material()["pointer"]}
    rec={};cnt=defaultdict(Counter)
    src=chain(silver_docs(),((d,g,"test",tree,cr,els,tokens) for d,g,tree,cr,els,tokens in gold_docs()))
    for d,g,fold,tree,cr,els,tokens in src:
        if d not in rows:continue
        r=rows[d]
        if fold!=r["fold"]:raise ValueError("Published work moved partitions")
        v,counts=restore(r,tree,els,tokens);rec[d]=v;cnt[fold].update(counts)
    if set(rec)!=set(rows):raise ValueError("Source material coverage changed")
    OUT.mkdir(exist_ok=True,parents=True)
    files={}
    for subset in ("source","external"):
        keep={d:v for d,v in rec.items() if (rows[d]["fold"]=="test")==(subset=="external")}
        path=ROOT/("nominal-"+subset+"-observations.json");write_json(path,keep);files[subset]=digest(path)
    write_json(OUT/"observations.json",{"protocol_sha256":digest(PLAN),"raw_sha256":digest(PD),"split_counts":{k:dict(s) for k,s in cnt.items()},
             "files":files,"action_order":ACTIONS,"documents":len(rows)})
    print({k:dict(s) for k,s in cnt.items()},flush=True)

if __name__=="__main__":main()
