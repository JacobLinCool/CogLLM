"""Seal original research-text inputs and disjoint paper splits for symbol attribution."""
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import re
from tokenizers import Tokenizer
from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.symbol_geometry import OUT, ROOT, scan_symlink
from research.reading_validation.identity_prepare import TOKENIZER, sealed_rows

MATH = re.compile(r'(?<!\\)\$(?!\$)(.+?)(?<!\\)\$|\[\[(.+?)\]\]', re.DOTALL)


def external_rows():
    groups=defaultdict(list); stats=Counter()
    for name in ('train','test'):
        for r in read_json(ROOT/(name+'.json')):
            tokens=r['tokens'];symbols=r['symbols'];labels=r['labels']; paper=r['arxiv_id']
            if not re.fullmatch(r'\d{4}\.\d{4,5}',paper) or not symbols:continue
            if len(tokens)!=len(labels) or tokens.count('SYMBOL')!=len(symbols) or any('SYMBOL' in w and w!='SYMBOL' for w in tokens):
                stats['failed_placeholder_rows']+=1;continue
            items=[];terms=[];defs=[];mark=[];length=0;pos=0
            for word,tag in zip(tokens,labels,strict=True):
                if word=='</s>':continue
                if word=='SYMBOL':word=symbols[pos];pos+=1
                if not word:continue
                a=length+bool(items);b=a+len(word);items.append(word);length=b
                if tag=='B-TERM':
                    m=MATH.fullmatch(word)
                    if not m:break
                    capture=1 if m.group(1) is not None else 2
                    left,right=m.span(capture);s=word[left:right]
                    terms.append((a+left+len(s)-len(s.lstrip()),a+right-len(s)+len(s.rstrip())))
                if tag in ('B-DEF','I-DEF'):
                    if tag=='B-DEF' and mark:defs.append((mark[0],mark[-1]));mark=[]
                    mark=(mark[:1] or [a])+[b]
                elif mark:defs.append((mark[0],mark[-1]));mark=[]
            else:
                if mark:defs.append((mark[0],mark[-1]))
                if len(terms)>1:continue
                text=' '.join(items);groups[(paper,str(r['sentence_id']))].append((text,terms,defs))
    rows=[]
    for (paper,sentence),records in sorted(groups.items()):
        texts={r[0] for r in records}
        if len(texts)!=1:stats['inconsistent_sentence']+=1;continue
        text=texts.pop();entries=defaultdict(set)
        for _,terms,ds in records:
            if len(terms)==1:entries[terms[0]].update(ds)
        issues=[]
        for anchor,definitions in entries.items():
            ds=sorted(definitions);state='forward' if ds and ds[0][0]>=anchor[1] else 'already_or_concurrently_explained' if ds else 'no_direct_explanation_in_paragraph'
            issues.append({'sources':[list(anchor)],'definitions':list(ds),'kind':state})
        rows.append({'id':sha256(text.encode()).hexdigest(),'source':paper+'_'+sentence,'paper':paper,'published':'symdef','text':text,'issues':issues})
    return rows,dict(stats)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);sy,source,files=scan_symlink();ex,ext=external_rows();tokenizer=Tokenizer.from_file(str(TOKENIZER));stats=Counter()
    train={r['paper'] for r in sy if r['published']=='train'};dev={r['paper'] for r in sy if r['published']=='dev'};extpapers={r['paper'] for r in ex}
    order=sorted(train-dev,key=lambda s:sha256(('symlink-held-papers-v1 '+s).encode()).digest());held=set(order[::4]);rest=[x for x in order if x not in held];cal=set(rest[::5]);lookup=defaultdict(list)
    for r in sy+ex:lookup[r['id']].append(r)
    packet=[];maps=[]
    for key,group in sorted(lookup.items()):
        if len({(r['paper'],r['source'],r['published']) for r in group})>1:stats['duplicate_ambiguous_texts']+=1;continue
        r=group[0];p=r['paper'];source_name=r['published']
        split='external' if source_name=='symdef' else 'published_dev' if source_name=='dev' else 'held' if p in held else 'calibration' if p in cal else 'fit'
        if (source_name=='symdef' and p in train|dev) or (source_name=='train' and p in dev):stats['paper_cross_source']+=1;continue
        enc=tokenizer.encode(r['text'],add_special_tokens=False);n=len(enc.ids)
        if not 1<=n<=1024:stats['length']+=1;continue
        spans=[m.span() for m in MATH.finditer(r['text'])];keys=[i for i,(a,b) in enumerate(enc.offsets) if any(b>l and a<u for l,u in spans)]
        if not keys:stats['no_markup']+=1;continue
        packet.append({'id':key,'text':r['text'],'tokens':n,'keys':keys})
        maps.append({**{a:v for a,v in r.items() if a!='text'},'split':split})
    sealed_rows(ROOT/'stimuli.jsonl',packet);sealed_rows(ROOT/'mapping.jsonl',maps)
    from research.reading_validation.symbol_geometry import competing
    summary={s:{'papers':len({r['paper'] for r in maps if r['split']==s}),'paragraphs':sum(r['split']==s for r in maps),'dual':sum(len(competing(r['issues'])) for r in maps if r['split']==s),'forward':sum(sum(j['kind']=='forward' for j in r['issues']) for r in maps if r['split']==s)} for s in sorted({r['split'] for r in maps})}
    write_json(OUT/'input-seal.json',{'protocol':digest(Path(__file__).with_name('symbol-protocol.md')),'stimuli':digest(ROOT/'stimuli.jsonl'),'mapping':digest(ROOT/'mapping.jsonl'),'sources':files|{s+'.json':digest(ROOT/(s+'.json')) for s in ('train','test')},'symlink_archive':source,'symdef_archive':ext,'exclusions':dict(stats),'splits':summary,'tokens':sum(r['tokens'] for r in packet)})
    print(summary)


if __name__=='__main__':prepare()
