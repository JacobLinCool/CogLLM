"""Prepare outcome-blind whole stories, essays, and expert-revised passages."""
import csv
from collections import defaultdict
import hashlib
from io import StringIO
import json
from pathlib import Path
import re

from research.reading_validation.transition_external_source import open_verified

ROOT=Path(".cache/cogllm/reading-validation")
OUT=Path("research/results/reading-validation-transition-external")


def norm(x):
    return " ".join(x.split())


def key(x):
    return hashlib.sha256(x.encode()).hexdigest()


def csv_rows(name):
    return list(csv.DictReader(StringIO(open_verified(name).decode("utf-8-sig"))))


def read_articles():
    articles = {}
    stops = []
    cohorts = defaultdict(set)

    def article(text):
        doc=key(text)
        info=articles.setdefault(doc,{"id":doc,"text":text,"stops":{}})
        if info["text"] != text:
            raise ValueError("Duplicate article digest")
        return doc,info

    def register(stim,doc,group,split,source,target_start,target_end,index,source_ref,*,goldstate=False):
        s,e=target_start,target_end
        target=stim["text"][s:e]
        if not target or not target.strip() or s<1:
            raise ValueError("Boundary lacks context or words")
        bid=key(doc+":"+str(s)+":"+str(e))
        stop=stim["stops"].setdefault(bid,{"id":bid,"start":s,"end":e,"index":index,"archive_final":goldstate})
        if stop["index"]!=index:
            raise ValueError("Original stop index mismatch")
        stops.append({"id":bid,"doc":doc,"group":group,"split":split,"source":source,"index":index,
                      "source_ref":source_ref,"prefix_hash":key(norm(stim["text"][:e]).casefold()),
                      "sentence_hash":key(norm(target).casefold())})
        cohorts[source].add(group)

    for file,split in (("TrainData.json","source"),("TestData.json","holdout")):
        items=json.loads(open_verified(file))
        for _,rec in items.items():
            sents=[norm(s) for s in rec["IncrementalData"]["sentences"]]
            if not sents or not all(sents):
                raise ValueError("Raw story has unusable sentence coordinates")
            title=norm(rec["Title"])
            text=title+"\n"+" ".join(sents)
            doc,info=article(text)
            g="cohesentia-"+str(rec["StoryID"])
            p=len(title)+1
            for i,s in enumerate(sents):
                start=p;end=p+len(s);p=end+1
                if text[start:end]!=s:
                    raise ValueError("Original sentence offset mismatch")
                if i:
                    register(info,doc,g,split,"cohesentia",start,end,i+1,[file,str(rec["StoryID"]),i],goldstate=split=="holdout")
    files=["test_1355_clean.csv","test_rewrite_213_no_delete.csv"]
    originals=csv_rows(files[0])
    rewrites=csv_rows(files[1])
    groups=defaultdict(list)
    for i,r in enumerate(originals):
        groups[r["essay_id"]].append((i,r))
    pair_keys={}
    for essay,group in groups.items():
        group.sort(key=lambda v:len(norm(v[1]["context"])))
        title=norm(group[0][1]["topic"])
        if not all(norm(r["topic"])==title for _,r in group):
            raise ValueError("Essay prompt changed within one source essay")
        body=max([norm(r["context"]+" "+r["sentence"]) for _,r in group],key=len)
        text=title+"\n"+body
        doc,info=article(text)
        previous=""
        # Where filtering omitted earlier boundaries, count visible sentence-final punctuation in that gap.
        position=1
        for n,(row_id,r) in enumerate(group):
            context=norm(r["context"]);target=norm(r["sentence"]);shown=norm(context+" "+target)
            if not body.startswith(shown) or (previous and not context.startswith(previous)):
                raise ValueError("A public original does not align as a prefix of the complete essay")
            gap=context[len(previous):].strip()
            if gap:
                position+=max(1,len(re.findall(r'[.!?]+(?:["”\x27)]*)\s+|[.!?]+(?:["”\x27)]*)$',gap)))
            start=len(title)+1+len(context)+1;end=start+len(target)
            if text[start:end]!=target:
                raise ValueError("DECOR sentence alignment changed")
            if not context:
                continue
            register(info,doc,essay,"external","decor",start,end,position,[files[0],row_id])
            k=(essay,context,target)
            if k in pair_keys:
                raise ValueError("Ambiguous source for an expert rewrite")
            pair_keys[k]=(stops[-1]["id"],position)
            position+=1; previous=shown
    for i,r in enumerate(rewrites):
        context,original,revised=map(norm,[r["context"],r["sentence"],r["Rewrite"]])
        source_key=(r["essay_id"],context,original)
        if source_key in pair_keys:
            counterpart,index=pair_keys[source_key]
        else:
            # The rewrite workbook contains two source rows absent from the detection release.
            # Their original text and context are explicit, so score that exact pair independently.
            title=norm(r["topic"])
            source_text=title+"\n"+context+" "+original
            source_doc,source_info=article(source_text)
            index=1+max(1,len(re.findall(r'[.!?]+(?:["”\x27)]*)\s+|[.!?]+(?:["”\x27)]*)$',context)))
            register(source_info,source_doc,r["essay_id"],"external","decor_rewrite_original",
                     len(source_text)-len(original),len(source_text),index,[files[1],i])
            counterpart=stops[-1]["id"]
        if not revised:
            raise ValueError("The no-deletion file should supply rewritten sentences")
        text=norm(r["topic"])+"\n"+context+" "+revised
        doc,info=article(text)
        start=len(text)-len(revised)
        register(info,doc,r["essay_id"],"external","decor_rewrites",start,len(text),index,[files[1],i])
        stops[-1]["counterpart"]=counterpart
    for info in articles.values():
        info["stops"]=sorted(info["stops"].values(),key=lambda b:b["end"])
    groups=sorted({s["group"] for s in stops if s["source"]=="cohesentia" and s["split"]=="source"},
                  key=lambda s:key("cogllm-incremental-calibration-v1:"+s))
    train=set(groups[:int(.8*len(groups))])
    for s in stops:
        if s["split"]=="source":
            s["split"]="train" if s["group"] in train else "dev"
    return list(articles.values()),stops


def prepare():
    docs,stops=read_articles()
    training={x["prefix_hash"] for x in stops if x["split"] in ("train","dev")}
    overlap=[x for x in stops if x["split"] in ("holdout","external") and x["prefix_hash"] in training]
    if overlap:
        raise ValueError(f"Exact train-test prefix overlap: {len(overlap)}")
    usable=[x for x in docs if x["stops"]]
    output=ROOT/"cohesion-stimuli.jsonl"
    output.write_text("".join(json.dumps(q,sort_keys=True)+"\n" for q in usable))
    OUT.mkdir(exist_ok=True,parents=True)
    meta={"protocol_sha256":key(Path("research/reading_validation/transition-external-protocol.md").read_text()),
          "packet_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"docs":len(usable),"boundaries":len(stops),
          "exact_fit_to_test_prefix_overlaps":len(overlap),"material":stops}
    (OUT/"stimuli.json").write_text(json.dumps(meta,indent=2)+"\n")
    print("documents",len(usable),"boundary rows",len(stops),"per source")
    from collections import Counter
    print(Counter((x["source"],x["split"]) for x in stops))

if __name__=="__main__":
    prepare()
