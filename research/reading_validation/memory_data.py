"""Build counterfactual training triplets and stimulus-only held-cohort packets."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import re
from xml.etree.ElementTree import iterparse
from zipfile import ZipFile

BASE = Path(".cache/cogllm/reading-validation")
OUT = Path("research/results/reading-validation-memory")
E1 = Path(".cache/cogllm/semantic-tracking/frick-consistency/prepared")
PLAN = Path(__file__).with_name("memory-protocol.md")
SS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
DISPLAY = re.compile(r"(\d\d)(\d\d)(\d\d)([ts]?)")
WORDS = re.compile(r"[^\W\d_]{3,}")
STOP = set("the and was were with from that this had who for she his her they has their have been some are man men woman women people wearing person two next group one there are while".split())


def h(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sha(file):
    with file.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n")


def load_rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def rows_xlsx(path, fields, require="Satz"):
    """Stream just selected cells. Never decode participant responses in stimulus mode."""
    with ZipFile(path) as z:
        strings=[]
        with z.open("xl/sharedStrings.xml") as f:
            for _,elem in iterparse(f,events=("end",)):
                if elem.tag==SS+"si":
                    strings.append("".join(t.text or "" for t in elem.iter(SS+"t")))
                    elem.clear()
        field_cols={}
        with z.open("xl/worksheets/sheet1.xml") as f:
            for _,row in iterparse(f,events=("end",)):
                if row.tag!=SS+"row":
                    continue
                header=row.get("r")=="1"
                result={}
                for cell in row:
                    ref=cell.get("r","").rstrip("0123456789")
                    if not header and ref not in field_cols:
                        continue
                    value=cell.findtext(SS+"v")
                    if cell.get("t")=="s" and value is not None:
                        value=strings[int(value)]
                    if header:
                        if value in fields:
                            if value in field_cols.values():
                                raise ValueError("Duplicate spreadsheet column")
                            field_cols[ref]=value
                    else:
                        result[field_cols[ref]]=value
                if header and set(field_cols.values())!=set(fields):
                    raise ValueError(f"Missing columns in {path.name}")
                row.clear()
                if not header and (require is None or result.get(require) not in ("NA",None,"")):
                    yield result


def story_versions(path,e1_map):
    variants=defaultdict(dict)
    fields={"Satz","Bild","ID","LoopSentences.thisN"}
    current=None
    previous=0
    for r in rows_xlsx(path,fields):
        raw_id=str(r["ID"])
        m=DISPLAY.fullmatch(raw_id.zfill(6) if not raw_id[-1:].isalpha() else raw_id[:-1].zfill(6)+raw_id[-1])
        ix=int(r["LoopSentences.thisN"])+1
        if ix==1:
            if m is None or int(m[3])!=1:
                raise ValueError("Unknown first sentence of a displayed story")
            current=(int(m[1]),m[2])
            previous=0
        if current is None or ix!=previous+1:
            raise ValueError("Nonconsecutive actual presentations")
        previous=ix
        # Preserve the programmed presentation order. Some filler ID strings omit a digit.
        role=raw_id[-1] if raw_id[-1] in ("t","s") else ""
        point=(r["Satz"],role,r["Bild"])
        variant=variants[current]
        if variant.setdefault(ix,point)!=point:
            raise ValueError("A source condition specifies different material at one position")
    records=[]
    for (source_story,condition), points in sorted(variants.items()):
        if sorted(points)!=list(range(1,len(points)+1)):
            raise ValueError("Story has an unaccounted display position")
        if any(picture!="trans.png" for _,_,picture in points.values()):
            continue
        targets=[i for i,(_,s,_) in points.items() if s=="t"]
        if len(targets)!=1:
            # Filler stories have their own IDs and were never experimental interventions.
            continue
        target,=targets
        sentences=[points[i][0] for i in sorted(points)]
        family=e1_map.get(sentences[target-1])
        if family is None or source_story!=family:
            # E2/E3 keep the E1 experimental story number. One separately numbered
            # filler (25) reuses the target of E1 family 15 with filler conditions.
            continue
        text="\n\n".join(sentences)
        records.append(dict(source_story=source_story,condition=condition,story=family,target=target,
                            text=text,sentences=sentences,id=h(text)))
    return records


def materials():
    src={r["id"]:r for r in load_rows(E1/"stimuli.jsonl")}
    first=[]
    family_by_target={}
    inconsistent={}
    for r in load_rows(E1/"metadata.jsonl"):
        doc=src[r["id"]]
        text=doc["text"]
        sentences=[text[s["start"]:s["stop"]] for s in r["sentences"]]
        t=r["target_index"]+1
        if family_by_target.setdefault(sentences[t-1],r["story"])!=r["story"]:
            raise ValueError("Ambiguous material identity by fixed target")
        if r["condition"]=="02":
            inconsistent[r["story"]]=sentences
        first.append(dict(study=1,id=r["id"],story=r["story"],condition=r["condition"],
                          arm="inconsistent" if r["condition"]=="02" else "consistent",
                          target=t,sentences=sentences,text=text))
    packet_dir=BASE/"memory"
    packet_dir.mkdir(exist_ok=True)
    retained=list(first)
    for study,required_sha in ((2,"ba899dc12a3295c74f669c69f08d88a24e9cf4309d36451ee18f7f8b628cf822"),
                               (3,"38b7531bfad4934a25a43fa9e6e5e3e62ac0fcf068079051a978d22808820ada")):
        if sha(BASE/f"frick{study}.xlsx")!=required_sha:
            raise ValueError(f"Workbook {study} differs from the OSF archive")
        reports=story_versions(BASE/f"frick{study}.xlsx",family_by_target)
        group=defaultdict(list)
        for r in reports:
            group[r["story"]].append(r)
        if len(group)!=12:
            raise ValueError(f"Expected 12 distinct text-only families, found {len(group)}")
        for story,versions in group.items():
            if len(versions)!=2:
                raise ValueError(f"Need precisely two story versions, got {len(versions)} in {story}")
            a,b=versions
            def same(r):
                # The source describes the critical elaboration in E2/3 as identical to E1.
                # Compare pre-target positions only. A few peripheral sentences can be edited.
                old=inconsistent[story]
                align=lambda s: tuple(re.findall(r"[^\W_]+", s.casefold()))
                return sum(align(x)==align(y) for x,y in zip(old[:r["target"]-1],r["sentences"][:r["target"]-1]))
            if same(a)==same(b):
                raise ValueError(f"Ambiguous manipulation in family {story}")
            matched=a if same(a)>same(b) else b
            for r in versions:
                r["arm"]="inconsistent" if r is matched else "neutral"
                r["study"]=study
                retained.append(r)
            diffs=[i for i,(x,y) in enumerate(zip(a["sentences"],b["sentences"],strict=True),1) if x!=y]
            if not diffs or max(diffs)>=a["target"] or a["target"]!=b["target"]:
                raise ValueError("Counterfactual pairs must share the target and continuation")
    unique={r["id"]:r["sentences"] for r in retained}
    if len(unique)!=len(set(r["text"] for r in retained)):
        raise ValueError("A passage ID denotes multiple sentence streams")
    (packet_dir/"stories.jsonl").write_text("".join(json.dumps({"id":k,"sentences":ss},ensure_ascii=False)+"\n"
                                                     for k,ss in sorted(unique.items())))
    for r in retained:
        del r["text"],r["sentences"]
    dump(packet_dir/"study-map.json",retained)
    return {"stories":len(unique),"study_arms":dict(Counter(str(r["study"])+"."+r["arm"] for r in retained)),
            "workbooks":{str(k):sha(BASE/f"frick{k}.xlsx") for k in (2,3)}}


def canonical(text):
    return " ".join(text.casefold().split())


def keywords(text):
    return set(WORDS.findall(text.lower()))-STOP


def triplets():
    all_groups={}
    counts={}
    for split in ("train","dev","test"):
        by_hyp=defaultdict(list)
        for role in ("original","edits"):
            with (BASE/f"cad-{role}-{split}.tsv").open() as f:
                for r in csv.DictReader(f,delimiter="\t"):
                    by_hyp[canonical(r["sentence2"])].append((role,r))
        use=[]
        for items in by_hyp.values():
            family=[r for _,r in items]
            if len(family)!=3 or sum(x[0]=="original" for x in items)!=1:
                continue
            if {r["gold_label"] for r in family}!={"neutral","entailment","contradiction"}:
                continue
            hyp=family[0]["sentence2"].strip()
            if any(canonical(r["sentence2"])!=canonical(hyp) for r in family):
                raise ValueError("The focal hypothesis must be shared")
            premises=[r["sentence1"].strip() for r in family]
            if len(set(canonical(p) for p in premises))!=3:
                continue
            uid=h(canonical(hyp))
            use.append(dict(id=uid,hyp=hyp,premises=premises,y=[int(r["gold_label"]=="contradiction") for r in family]))
        all_groups[split]=use
        counts[split+".complete"]=len(use)
    def strings(r):
        return {canonical(s) for s in [*r["premises"],r["hyp"]]}
    held={s for split in ("dev","test") for r in all_groups[split] for s in strings(r)}
    test={s for r in all_groups["test"] for s in strings(r)}
    all_groups["train"]=[r for r in all_groups["train"] if not (held&strings(r))]
    all_groups["dev"]=[r for r in all_groups["dev"] if not (test&strings(r))]
    # Set aside a deterministic text-only donor pool; its labels never enter training.
    donors=sorted(all_groups["train"],key=lambda r:h("filler.20261122."+r["id"]))[:160]
    donor_ids={r["id"] for r in donors}
    all_groups["train"]=[r for r in all_groups["train"] if r["id"] not in donor_ids]
    paragraphs=[r["premises"][0] for r in donors]
    keyword_sets=[keywords(s) for s in paragraphs]
    bank=[]
    for split,rows in all_groups.items():
        counts[split+".used"]=len(rows)
        for r in sorted(rows,key=lambda r:r["id"]):
            rng=random.Random(int(h("cad.20261122."+r["id"]),16))
            n=rng.choice((0,2,4,8))
            terms=keywords(" ".join(r["premises"]+[r["hyp"]]))
            eligible=[i for i,w in enumerate(keyword_sets) if not (terms&w)]
            chosen=rng.sample(eligible,n)
            example=dict(id=r["id"],split=split,y=r["y"],n=n)
            example["texts"]=["\n\n".join([s,*[paragraphs[i] for i in chosen],r["hyp"]]) for s in r["premises"]]
            example["adjacent"]=["\n\n".join((s,r["hyp"])) for s in r["premises"]]
            bank.append(example)
    dest=BASE/"memory/cad.jsonl"
    dest.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in bank))
    for tag in ("train", "test"):
        public=[]
        for r in bank:
            if (r["split"]=="test")!=(tag=="test"):
                continue
            public.append({k:v for k,v in r.items() if tag=="train" or k!="y"})
        (dest.parent/f"cad-{tag}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in public))
    return counts


if __name__=="__main__":
    OUT.mkdir(parents=True,exist_ok=True)
    (BASE/"memory").mkdir(exist_ok=True)
    controls=triplets()
    studies=materials()
    files={k:sha(BASE/"memory"/k) for k in ("stories.jsonl","cad.jsonl","cad-train.jsonl","cad-test.jsonl","study-map.json")}
    info=dict(protocol=sha(PLAN),training=controls,reading=studies,files=files)
    dump(OUT/"input-seal.json",info)
    print(json.dumps(info))
