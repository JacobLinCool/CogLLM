"""Seal the original AmbiCoref sentence, two earlier regions and five human response categories."""
from collections import Counter,defaultdict
import csv
from hashlib import sha256
import json
from pathlib import Path
import re
import tarfile

from research.reading_validation.antecedent_inference import BUNDLE,load_tagger
from research.reading_validation.transition_architecture_prepare import load_tokenizer
from research.reading_validation.transition_jones_prepare import article
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_snac_prepare import ROOT,WORD


OUT=Path("research/results/reading-validation-transition-anchored-reference")
PROTOCOL=Path("research/reading_validation/transition-anchored-reference-protocol.md")
CATEGORIES=("definite_first","probable_first","ambiguous","probable_second","definite_second")


def tokens(s):
    return tuple(m.group().casefold().replace("’","'") for m in WORD.finditer(s))


def eight(s):
    v=tokens(s)
    return {v[i:i+8] for i in range(len(v)-7)}


def key(s):
    return " ".join(tokens(s))


def source_conditions(archive):
    conditions=defaultdict(set)
    with tarfile.open(archive) as t:
        for f in t:
            if "/Data/Sentences/" not in f.name or not f.name.endswith(".txt"):continue
            name=Path(f.name).stem
            with t.extractfile(f) as src:
                for s in src.read().decode("utf-8-sig").splitlines():conditions[key(s)].add(name)
    return conditions


def response(answer,slot):
    s=answer.lstrip("$")
    if s=="Ambiguous":return 2
    m=re.fullmatch(r"(def|prob)-Q(\d+)-Name([12])",s)
    if not m or int(m[2])!=slot:raise ValueError("Unexpected original ordinal-response code")
    return {("def","1"):0,("prob","1"):1,("prob","2"):3,("def","2"):4}[m[1],m[3]]


def geometry(row,opts):
    forms=[row["text"][a:b].casefold() for a,b in row["words"]]
    matches=[]
    for name in opts:
        bits=tokens(name)
        starts=[i for i in range(len(forms)-len(bits)+1) if tuple(forms[i:i+len(bits)])==bits]
        if len(starts)!=1 or not bits:return "unmapped_or_nonunique_answer_phrase"
        matches.append(list(range(starts[0],starts[0]+len(bits))))
    if set(matches[0])&set(matches[1]):return "overlapping_opposing_choices"
    qs=[q for q in row["queries"] if q>max(max(v) for v in matches)]
    if len(qs)!=1:return "unresolved_target_pronoun"
    q=qs[0]
    # Use only outcome-independent form and noun regions to derive a repeatable resampling block.
    v=list(forms)
    for j,z in enumerate(matches):
        for i in z:v[i]="" if i!=z[0] else "NP"+str(j+1)
    v=[("pronoun" if i==q else x) for i,x in enumerate(v) if x]
    return q,matches," ".join(v)


def main():
    packet=ROOT/"dpr-pointer-stimuli.jsonl";click=ROOT/"pointer-stimuli.jsonl"
    prior=[r["text"] for p in (packet,click) for r in map(json.loads,p.read_text().splitlines()) if r["fold"]=="train"]
    supervised=set().union(*(eight(t) for t in prior));exact={tokens(t) for t in prior}
    archive=ROOT/"ambicoref-source.tgz";results=ROOT/"ambicoref.csv"
    with tarfile.open(archive) as t:
        member=next(x for x in t if x.name.endswith("/human_judgements_data.csv"))
        if t.extractfile(member).read()!=results.read_bytes():raise ValueError("Original study workbook differs from upstream")
    cat=source_conditions(archive)
    tok=load_tokenizer("qwen",json.loads(Path("research/backbone_transfer/models.json").read_text())["models"]["qwen"])
    tag=load_tagger(BUNDLE/"averaged_perceptron_tagger_eng.zip")
    materials={};seen=defaultdict(set);answers=defaultdict(list);nraw=0;rows=0
    with results.open(newline="") as f:
        for sheet,entry in enumerate(csv.DictReader(f)):
            if entry["Answer.native"]!="yes":raise ValueError("Non-native or missing questionnaire differs from public method")
            rows+=1
            for i in range(10):
                text=entry[f"Input.sentence{i}"].strip("\r\n");opts=(entry[f"Input.Q{i}-Name1"],entry[f"Input.Q{i}-Name2"])
                q=entry[f"Input.question{i}"];ident=sha256(text.encode()).hexdigest()[:18]
                seen[ident].add((q,opts));materials[ident]=text
                answers[ident].append((sheet,response(entry[f"Answer.Q{i}"],i)));nraw+=1
    annotations={};rows_for_gpu=[];audit=Counter();types=Counter()
    for ident,text in materials.items():
        if len(seen[ident])!=1:audit["conflicting_question_or_option_mapping"]+=1;continue
        question,opts=next(iter(seen[ident]))
        if tokens(text) in exact or eight(text)&supervised:audit["eight_word_or_full_source_overlap"]+=1;continue
        row=article(ident,text,tok,tag);geo=geometry(row,opts)
        if isinstance(geo,str):audit[geo]+=1;continue
        q,regions,family=geo;row["measurement"]={"q":q,"k":len(row["words"])-1}
        conditional=sorted(cat[key(text)])
        if len(conditional)>1:raise ValueError("The public complete corpus gives conflicting design conditions")
        label=conditional[0] if conditional else "unmatched"
        types[label]+=1
        annotations[ident]={"names":regions,"q":q,"k":len(row["words"])-1,"condition":label,
            "family":family,"visible_words":[text[a:b] for a,b in row["words"]],
            "geometry_source_question":question,"questionnaire_choices":answers[ident]}
        rows_for_gpu.append(row)
    if len(rows_for_gpu)+sum(audit.values())!=len(materials):raise ValueError("Missing original cases")
    OUT.mkdir(exist_ok=True,parents=True);dest=ROOT/"ambicoref-reading-stimuli.jsonl"
    dest.write_text("".join(json.dumps(r,separators=(",",":"))+"\n" for r in rows_for_gpu))
    gold=ROOT/"ambicoref-observed-choices.json";dump(gold,annotations)
    result={"protocol_sha256":sha(PROTOCOL),"input_sha256":sha(dest),"outcomes_sha256":sha(gold),"questionnaire_sha256":sha(results),
            "upstream_commit":"69a6639cdaaa2b7bcaadd173da960b20c0335742","upstream_csv_git_blob":"7638331e0768ca3710e63fbd320788503da72ba1",
            "response_categories":CATEGORIES,"source_documents_for_overlap":len(prior),"original_questionnaires":rows,"original_choices":nraw,
            "unique_original_sentences":len(materials),"exclusions":dict(audit),"included_sentences":len(rows_for_gpu),
            "included_questionnaires":len({p for a in annotations.values() for p,c in a["questionnaire_choices"]}),
            "included_decisions":sum(len(v["questionnaire_choices"]) for v in annotations.values()),"matched_design_labels":dict(types),
            "lexical_skeletons":len({v["family"] for v in annotations.values()}),"max_tokens":max(r["token_count"] for r in rows_for_gpu)}
    dump(OUT/"external-seal.json",result);print(result,flush=True)


if __name__=="__main__":main()
