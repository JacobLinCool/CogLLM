"""Prepare only original words and their model coordinates for out-of-domain cue monitoring."""
import json
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.nominal_data import OUT,ROOT
from research.reading_validation.memory_data import load_rows
from research.reading_validation.referent_features import WORD
from research.reading_validation.transition_architecture_prepare import load_tokenizer,word_indices
from research.reading_validation.transition_pointer_features import sentence_ids
from pathlib import Path

def main():
    raw=ROOT/"memory/stories.jsonl";spec=json.loads(Path("research/backbone_transfer/models.json").read_text())["models"]["qwen"]
    tokenizer=load_tokenizer("qwen",spec);rows=[];all_n=0
    for r in load_rows(raw):
        text="\n\n".join(r["sentences"]);e=tokenizer.encode(text,add_special_tokens=False)
        words=[m.span() for m in WORD.finditer(text)];all_n+=len(words);maps=word_indices(text,words,tokenizer,e)
        certified=[(s,t) for s,t in zip(words,maps,strict=True) if t]
        rows.append(dict(id=r["id"],text=text,words=[a for a,b in certified],tokens=[b for a,b in certified]))
    dest=ROOT/"nominal-report-texts.jsonl"
    with dest.open("w") as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    summary=dict(source=digest(dest),original=digest(raw),protocol=digest(Path("research/reading_validation/nominal-report-transfer-protocol.md")),
                 texts=len(rows),words=sum(len(r["words"]) for r in rows),original_words=all_n)
    write_json(OUT/"report-exposure.json",summary);print(summary)

if __name__=="__main__":main()
