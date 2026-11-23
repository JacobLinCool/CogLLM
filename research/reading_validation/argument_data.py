"""Recover unmarked whole essays and certify the provenance of their human annotations."""
from collections import Counter, defaultdict
import csv
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

from research.reading_validation.memory_data import BASE, h, sha

WORK = BASE / 'argument'
OUT = Path('research/results/reading-validation-argument')
PLAN = Path(__file__).with_name('argument-protocol.md')
SPLIT = re.compile(r'(?<=[.!?])\s+(?=\S)|\s*\n+\s*(?=\S)')
WORD = re.compile(r"\w+", re.UNICODE)


def sentences(text):
    boundaries = [0] + [m.end() for m in SPLIT.finditer(text)] + [len(text)]
    spans = []
    for a,b in zip(boundaries,boundaries[1:]):
        while a < b and text[a].isspace(): a += 1
        while a < b and text[b-1].isspace(): b -= 1
        if a < b: spans.append((a,b))
    return spans


def author_rows():
    with (WORK/'author-training').open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def independent_essays():
    with ZipFile(WORK/'relations.zip') as outer, ZipFile(BytesIO(outer.read('ArgumentAnnotatedEssays-2.0/brat-project-final.zip'))) as inner:
        for name in sorted(inner.namelist()):
            m=re.fullmatch(r'brat-project-final/essay(\d+)\.txt',name)
            if m: yield int(m[1]),inner.read(name).decode('utf-8')


def chunks(text):
    words=WORD.findall(text.casefold())
    return {' '.join(words[i:i+20]) for i in range(len(words)-19)}


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    meta={};topics=defaultdict(list);seen=set()
    for r in author_rows():
        key=r['essay_id_comp']
        if not key or key in seen:continue
        seen.add(key);t=r['full_text'];topic=r['prompt_name'];uid=h(t)
        if 70<=len(t.split())<=550 and t and topic and uid not in meta:
            meta[uid]=(topic,key);topics[topic].append(uid)
    order=sorted(topics,key=lambda k:h('argument-topic-20261122.'+k))
    role={g:'train' if i<int(.6*len(order)) else 'dev' if i<int(.75*len(order)) else 'held' for i,g in enumerate(order)}
    selected={uid for group in topics.values() for uid in sorted(group,key=lambda k:h('argument-essay-20261122.'+k))[:120]}
    selected_keys={meta[k][1] for k in selected};texts={};seen.clear()
    for r in author_rows():
        k=r['essay_id_comp']
        if k in selected_keys and k not in seen:
            seen.add(k);t=r['full_text'];texts[h(t)]=t
    if set(texts)!=selected:raise ValueError('Essay ID did not resolve to the certified article')
    archive={k:dict(study='persuade',topic=meta[k][0],id=meta[k][1],role=role[meta[k][0]]) for k in selected}
    # Only selected PERSUADE materials can influence these models.
    window=set()
    for t in texts.values():window.update(chunks(t))
    removed=Counter();external=0
    for i,t in independent_essays():
        external+=1
        if not 70<=len(t.split())<=550:removed['external_length']+=1;continue
        k=h(t)
        if k in texts or not window.isdisjoint(chunks(t)):removed['external_overlap']+=1;continue
        texts[k]=t;archive[k]=dict(study='darmstadt',id=i,role='external',topic=None)
    with (WORK/'stimuli.jsonl').open('w') as f:
        for k,t in sorted(texts.items()):f.write(json.dumps(dict(id=k,text=t),ensure_ascii=False)+'\n')
    summary=dict(protocol=sha(PLAN),raw={p:sha(WORK/p) for p in ('author-training','relations.zip','sufficiency.zip','votes.zip')},
          source_essays=len(meta),source_ids=len(seen),eligible_topics={k:len(v) for k,v in topics.items()},roles=role,
          independent_original=external,exclusions=dict(removed),articles=archive,stimuli=sha(WORK/'stimuli.jsonl'))
    (OUT/'inputs.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(len(texts),Counter(a['role'] for a in archive.values()),dict(removed))


if __name__=='__main__':prepare()
