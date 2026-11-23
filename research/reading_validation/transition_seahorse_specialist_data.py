"""Verify all original source documents by GEM ID and every available released reference summary."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import pyarrow.parquet as pq

from research.reading_validation.transition_seahorse_data import DEST, WORK, sha, source_rows

PLAN=Path(__file__).with_name("transition-seahorse-specialist.md")
INPUT=WORK/"seahorse-specialist-stimuli.jsonl"
TABLES={"wiki":("source","df0994daaf324ba0027886cbcad3e65d2e1f8011c5c7372e5e0578e27d94fa52"),
       "xsum":("document","deb87ec1684ce56afd12b852337df0036db8e24e80f14aef1ebca19f6537b66e"),
       "xlsum":("text","f7adb759cc8f0dd549343c14aca27c897cc4bd9fcc6d02f364e19a0129fbd1db")}


def main():
    material=list(source_rows("test"));needed={r["gem_id"] for r in material};pairs={}
    data={}
    for k,(field,expected) in TABLES.items():
        p=WORK/f"seahorse-specialist-{k}.parquet"
        if sha(p)!=expected:raise ValueError("The authoritative archived GEM source changed")
        for batch in pq.ParquetFile(p).iter_batches(columns=["gem_id",field,"target"],batch_size=512):
            for r in batch.to_pylist():
                if r["gem_id"] in needed:
                    if r["gem_id"] in data:raise ValueError("Duplicate article record")
                    data[r["gem_id"]]=(r[field],r["target"])
    if set(data)!=needed:raise ValueError("Missing original source article")
    refs=Counter()
    for r in material:
        src,target=data[r["gem_id"]]
        if r["model"]=="reference":
            if " ".join(target.split())!=" ".join(r["summary"].split()):raise ValueError("Reference/source identity mismatch")
            refs[r["gem_id"].split("-")[0]]+=1
        part=json.dumps([src,r["summary"]],ensure_ascii=False,separators=(",",":"))
        k=sha256(part.encode()).hexdigest()
        pairs[k]={"id":k,"article":src,"summary":r["summary"]}
    INPUT.write_text("".join(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\n" for _,r in sorted(pairs.items())))
    (DEST/"specialist-input.json").write_text(json.dumps({"protocol":sha(PLAN),"source":sha(INPUT),"article_tables":TABLES,
       "all_test_rows":len(material),"distinct_source_summary_inputs":len(pairs),"original_articles":len(needed),
       "matched_official_references":dict(refs),"missing_references":0},indent=2)+"\n")
    print(len(pairs),len(needed),refs,sha(INPUT),flush=True)


if __name__=="__main__":
    main()
