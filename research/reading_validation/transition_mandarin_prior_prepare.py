"""Extract complete published lexical-choice tables and seal an outcome-free verb dictionary."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import re
import subprocess

from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_mandarin_fit import ROOT,OUT


DEST=OUT/"independent-prior"
PROTOCOL=Path("research/reading_validation/transition-mandarin-lexical-prior-protocol.md")
FILES={"mandarin-independent-norms":"d914140bdc029c982a352959c7d268d4f2ab020787f27f0e6c43fc4229dba262",
       "mandarin-xia-data":"a745e36ec65380a7eb7336748fc9bc8c833bd10a6116a394d0ea43ef13322ef2"}


def read_pdf(stem):
    path=ROOT/(stem+".pdf")
    if sha(path)!=FILES[stem]:raise ValueError("The registered original PDF has changed")
    return subprocess.run(["pdftotext","-layout",str(path),"-"],check=True,capture_output=True,encoding="utf8").stdout


def prepare():
    norms=read_pdf("mandarin-independent-norms").split("Table B1. Database of Chinese Implicit Causality Verbs")[1].split("a.   Semantic Category.")[0]
    record=re.compile(r"^\s*(\d+)\s+(\S+)\s+.*?\s+(Positive|Negative|Neutral)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$",re.MULTILINE)
    found=record.findall(norms)
    if sorted(int(r[0]) for r in found)!=list(range(1,151)):raise ValueError("Incomplete 150-verb source table")
    vals=defaultdict(list)
    for i,v,_,bs,_ in found:
        if "…" in v:continue
        vals[v].append((1+float(bs))/2)
    xia=read_pdf("mandarin-xia-data");area=xia.split("5.5 Appendix")[-1].split("Table 5.7")[0]
    record=re.compile(r"^\s*(Ba|Dui|Xiang)\s+([\u3400-\u9fff]+)\s+.+?\s+(\d+[,\.]\d+)\s+(\d+[,\.]\d+)\s*$",re.MULTILINE)
    observed={v:{"construction":c,"svo_subject":1-float(a.replace(",","."))/100,"scov_subject":1-float(b.replace(",","."))/100}
              for c,v,a,b in record.findall(area)}
    if len(observed)!=49:raise ValueError(f"Expected all 49 released Mandarin verbs, got {len(observed)}")
    original={r["id"]:r["text"] for r in map(json.loads,(ROOT/"mandarin-reading-stimuli.jsonl").read_text().splitlines())}
    data=json.loads((ROOT/"mandarin-anonymous-choices.json").read_text());spelling={}
    for r in data:
        if r["structure"]!="active":continue
        text=original[r["bare"]]
        if text[4]!="了":raise ValueError("Source active clause does not expose a unique original two-character verb")
        verb=text[2:4];z=spelling.setdefault(r["source_verb"],verb)
        if z!=verb:raise ValueError("Romanization maps to inconsistent source predicates")
    if len(spelling)!=38:raise ValueError("Incomplete Zhan lexicon")
    held=set(observed)|set(spelling.values());training=sorted(set(vals)-held)
    vocabulary=sorted(set(vals)|held);DEST.mkdir(parents=True,exist_ok=True)
    packet=ROOT/"mandarin-lexical-prior-stimuli.jsonl"
    packet.write_text("".join(json.dumps({"verb":v},ensure_ascii=False,separators=(",",":"))+"\n" for v in vocabulary))
    src=ROOT/"mandarin-original-norm-outcomes.json";dump(src,{"zhang":dict(vals),"xia":observed,"zhan_romanzi":spelling})
    result={"protocol_sha256":sha(PROTOCOL),"input_sha256":sha(packet),"outcomes_sha256":sha(src),"original_pdfs":FILES,
            "source_rows":len(found),"source_uninterrupted_rows":sum(map(len,vals.values())),"source_duplicate_verbs":[v for v,x in vals.items() if len(x)>1],
            "source_eligible_distinct":len(vals),"source_fitted_verbs":training,"held_norms_excluded":sorted(set(vals)&held),
            "xia_verbs":len(observed),"zhan_spellings":len(spelling),"all_unique_predicates":len(vocabulary)}
    dump(DEST/"seal.json",result);print(result["all_unique_predicates"],len(training),len(set(vals)&held),sha(packet))


if __name__=="__main__":prepare()
