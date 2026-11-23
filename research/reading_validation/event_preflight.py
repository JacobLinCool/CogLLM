"""Verify every original sentence stop against the pinned Qwen tokenizer before GPU use."""

import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer


def verify():
    root=Path(".cache/cogllm/reading-validation")
    home=Path.home()/".cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots"
    path=home/"15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer.json"
    with path.open("rb") as f:
        digest=hashlib.file_digest(f,"sha256").hexdigest()
    if digest!="5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42":
        raise ValueError("Different tokenizer")
    tokenizer=Tokenizer.from_file(str(path))
    rows=stops=longest=0
    failed=[]
    for line in (root/"event-stimuli.jsonl").open():
        row=json.loads(line)
        text=row["text"]
        encoding=tokenizer.encode(text,add_special_tokens=False)
        longest=max(longest,len(encoding.ids))
        rows+=1
        for a,b in row["spans"]:
            ids=tokenizer.encode(text[:b],add_special_tokens=False).ids
            indexes=[i for i,(l,h) in enumerate(encoding.offsets) if l<b and h>a]
            if not indexes or ids!=encoding.ids[:indexes[-1]+1]:
                failed.append((row["id"],a,b))
            stops+=1
    if failed or longest>2048:
        raise ValueError(f"Uncertified story geometry: {failed[:5]}; longest {longest}")
    result=dict(stories=rows,certified_sentence_ends=stops,largest_document_tokens=longest,tokenizer_sha256=digest)
    Path("research/results/reading-validation-events/geometry.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))


if __name__=="__main__":
    verify()
