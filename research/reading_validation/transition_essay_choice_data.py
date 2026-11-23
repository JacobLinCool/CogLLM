"""Seal new human-evaluator passages before exposing any individual preference."""
import csv
from hashlib import sha1
import json
from pathlib import Path

from research.reading_validation.transition_external_prepare import ROOT, OUT, csv_rows, key, norm
from research.reading_validation.transition_score import dump, sha

PLAN=Path(__file__).with_name("transition-essay-human-choice-protocol.md")
DEST=OUT/"essay-expert-choice"
FILES={"human":"human_gpt_100","without_reasons":"llam2_nr_gpt_100","with_reasons":"llama2r_gpt_100"}
BLOBS={"human":"52625bf7600fb64e1c3a2c742436e5b857d567ba","without_reasons":"f2bf2bf2dda10bbee044c6f83b16808d184e7764",
       "with_reasons":"cff16be4c6fb998f01cbedbea651e2b96da0946c"}
INPUT=ROOT/"essay-expert-choices-stimuli.jsonl"


def read(mode):
    f=ROOT/"cohesion"/(FILES[mode]+".csv");data=f.read_bytes()
    if sha1(f"blob {len(data)}\0".encode()+data).hexdigest()!=BLOBS[mode]:
        raise ValueError("Unexpected human comparison source release")
    with f.open(encoding="utf-8-sig",newline="") as fp:return list(csv.DictReader(fp))


def prepare():
    existing=json.loads((OUT/"stimuli.json").read_text())["material"]
    original=csv_rows("test_rewrite_213_no_delete.csv")
    def ident(r):return tuple(norm(r[n]) for n in ("topic","context","sentence"))
    known={ident(original[r["source_ref"][1]]):(original[r["source_ref"][1]],r["index"]) for r in existing if r["source"]=="decor_rewrites"}
    rows={m:read(m) for m in FILES};pairs={};packet={}
    for m,records in rows.items():
        if len(records)!=100 or len(set(ident(r) for r in records))!=100:raise ValueError("Incomplete original choice block")
        for j,r in enumerate(records):
            src=ident(r);ref,pos=known[src];topic,context,_=src;pre=topic+"\n"+context+" "
            candidates=[norm(r[k]) for k in ("model1","model2")]
            if not all(candidates):raise ValueError("Original judge saw missing alternative")
            variants=[]
            for alt in candidates:
                text=pre+alt;uid=key(text)
                source={"id":uid,"text":text,"start":len(pre),"end":len(text),"index":pos}
                if packet.setdefault(uid,source)!=source:raise ValueError("Alternative sentence index changed")
                variants.append(uid)
            u=key("\x00".join(src));info=pairs.setdefault(u,{"topic":topic,"essay":ref["essay_id"],"human_reference":key(pre+norm(ref["Rewrite"])),"sources":{}})
            info["sources"][m]={"variants":variants,"row":j,"instruction":any(v=="Delete second half of sentence" for v in candidates)}
    for rec in pairs.values():
        sets=[set(v["variants"]) for v in rec["sources"].values()]
        if len(sets)!=3:raise ValueError("Shared material is absent from an expert comparison")
        common=set.intersection(*sets)
        human=rec["human_reference"]
        gpt=(common-{human}) if len(common)>1 else common
        rec["gpt_candidate"]=next(iter(gpt)) if len(gpt)==1 else None
    if len(pairs)!=100:raise ValueError("Expert choices cannot be aligned across the three material arms")
    INPUT.write_text("".join(json.dumps(packet[k],ensure_ascii=False)+"\n" for k in sorted(packet)))
    DEST.mkdir(exist_ok=True,parents=True)
    dump(DEST/"exposure-seal.json",{"protocol_sha256":sha(PLAN),"packet_sha256":sha(INPUT),"source_git_blobs":BLOBS,
         "candidate_passages":len(packet),"distinct_materials":len(pairs),"judged_pairs":sum(len(x["sources"]) for x in pairs.values()),
         "conditionally_identifiable_gpt":sum(x["gpt_candidate"] is not None for x in pairs.values()),"items":pairs})
    print(sha(INPUT),len(packet),len(pairs),flush=True)


if __name__=="__main__":prepare()
