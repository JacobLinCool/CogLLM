"""Convert DEFT term–definition links to causal English-text windows with original character provenance."""

from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import re
import zipfile

from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.identity_prepare import DATA,PREVIOUS,TOKENIZER,sealed_rows


ROOT=DATA/'definitions'
OUT=Path('research/results/reading-validation-definitions')
PROTOCOL=Path(__file__).with_name('definition-protocol.md')
REVISION='db8c95565c2e58d861537cb8cb4621c50b75cd13'
# Matches a printed word with internal hyphens/apostrophes, with all reading stops after visible tokens.
WORD=re.compile(r"[^\W_]+(?:[\-’'][^\W_]+)*",re.UNICODE)
FILES=('data/deft_files/train/','data/deft_files/dev/','data/test_files/labeled/subtask_3/')


def parse_chunk(chunk,counts):
    lines=[[s.strip() for s in row.split('\t')] for row in chunk.splitlines() if row.strip()]
    if len(lines)<3 or any(len(row)!=8 for row in lines) or not lines[0][0].isdigit() or lines[1][0]!='.':
        counts['excluded_numbering_or_schema']+=1;return None
    number=lines[0][0];lines=lines[2:];current=0;blocks=[];positions=[]; tags=defaultdict(list)
    prev=None
    for i,(word,file,start,end,bio,entity,root,relation) in enumerate(lines):
        start,end=int(start),int(end)
        if not word or end<=start or prev is not None and start<prev:
            counts['excluded_overlapping_or_empty_source_token']+=1;return None
        if prev is not None and start>prev:blocks.append(' ');current+=1
        a=current;blocks.append(word);current+=len(word);positions.append((a,current))
        if entity!='-1' and bio!='O':tags[entity].append((i,bio,root,relation))
        prev=end
    text=''.join(blocks);spans={};starts=[];cs=Counter()
    for ident,values in tags.items():
        ids=[x[0] for x in values]; labels={x[1][2:] for x in values}
        if len(labels)!=1 or ids!=list(range(ids[0],ids[-1]+1)):
            cs['noncontiguous_or_mixed_entity']+=1;continue
        kind=labels.pop();root={x[2] for x in values};rel={x[3] for x in values}
        if len(root)!=1 or len(rel)!=1:cs['ambiguous_relation']+=1;continue
        spans[ident]={'span':[positions[ids[0]][0],positions[ids[-1]][1]],'kind':kind,'root':root.pop(),'relation':rel.pop()}
    terms={i:v['span'] for i,v in spans.items() if v['kind']=='Term'}
    links=[]
    for i,v in spans.items():
        if v['kind']!='Definition' or v['relation']!='Direct-Defines':continue
        if v['root'] not in terms:cs['missing_local_primary_term']+=1;continue
        a,b=v['span'];term=terms[v['root']]
        if term[1]>a:cs['definition_precedes_or_overlaps_term']+=1;continue
        links.append({'term':v['root'],'definition':[a,b]})
    counts.update(cs)
    return text,{'number':number,'terms':terms,'links':links}


def word_geometry(text,tokenizer):
    e=tokenizer.encode(text,add_special_tokens=False);off=e.offsets;result=[]
    for word in WORD.finditer(text):
        a,b=word.span();sel=[i for i,(l,r) in enumerate(off) if r>a and l<b and l<r]
        if not sel:raise ValueError('Word has no model token coverage')
        z=sel[-1]
        # A source word can lie inside a multiword model token. Consume the token before making a prediction.
        if off[z][1]>b or result and z<=result[-1][3]:continue
        result.append([a,b,sel[0],z])
    return result,len(e.ids)


def prepare():
    OUT.mkdir(exist_ok=True,parents=True)
    tok=Tokenizer.from_file(str(TOKENIZER));warnings=Counter();truth=defaultdict(list);docs={};source_id={}
    # Restrict training/test to nonoverlapping textbook domains. Preserve deterministic precedence for exact duplicates.
    with zipfile.ZipFile(ROOT/'deft.zip') as archive:
        names=[n for n in archive.namelist() if n.endswith('.deft') and any(s in n for s in FILES)]
        def rank(n):
            suffix=Path(n).name.replace('task_3_','');subject=suffix.split('_')[1]
            return (0 if subject in ('biology','physics') else 1 if subject=='psychology' else 2),suffix,n
        for name in sorted(names,key=rank):
            _,filename,_=rank(name);subject=filename.split('_')[1];part='test' if subject in ('biology','physics') else 'calibration' if subject=='psychology' else 'train'
            contents=archive.read(name).decode('utf-8-sig').replace('\r\n','\n').strip();chapters=re.split(r'\n[ \t]*\n[ \t]*\n',contents)
            for chapter in chapters:
                parsed=parse_chunk(chapter,warnings)
                if parsed is None:continue
                text,lab=parsed;ident=sha256(text.encode()).hexdigest()
                if ident in source_id:warnings['excluded_duplicate_text']+=1;continue
                source_id[ident]=part
                ws,n=word_geometry(text,tok)
                if n>1024 or not ws:warnings['excluded_empty_or_overlong']+=1;continue
                docs[ident]={'id':ident,'text':text,'words':ws}
                truth[part].append({'id':ident,'subject':subject,'source':filename,**lab})
    # Do not read participant ratings or corpus-specific evaluation outcomes here.
    guo=DATA/'jargon/local-stimuli.jsonl'
    guo_ids={}
    for row in stream_rows(guo):
        text=row['text'];ident=sha256(text.encode()).hexdigest();ws,n=word_geometry(text,tok)
        if n>1024 or ident in source_id:raise ValueError('Out-of-scope Guo stimulus')
        docs[ident]={'id':ident,'text':text,'words':ws};guo_ids[row['id']]=ident
    originals={r['id']:r['text'] for r in stream_rows(PREVIOUS/'stimuli.jsonl')};hist={}
    for key,r in read_json(PREVIOUS/'qwen.json')['rows'].items():
        if r['kind'] not in ('ab','controls'):continue
        text=originals[key];ident=sha256(text.encode()).hexdigest();ws,n=word_geometry(text,tok)
        if ident in source_id or n>1024:raise ValueError('Conflicting historical stimulus')
        docs[ident]={'id':ident,'text':text,'words':ws};hist[key]=ident
    sealed_rows(ROOT/'stimuli.jsonl',[docs[k] for k in sorted(docs)])
    for k,v in truth.items():sealed_rows(ROOT/(k+'.jsonl'),v)
    write_json(ROOT/'external-keys.json',{'jargon':guo_ids,'historical':hist})
    counts={}
    for k,items in truth.items():
        counts[k]={'passages':len(items),'subjects':sorted({r['subject'] for r in items}),'sources':len({r['source'] for r in items}),
                   'terms':sum(len(r['terms']) for r in items),'forward_links':sum(len(r['links']) for r in items)}
    record={'protocol_sha256':digest(PROTOCOL),'source_revision':REVISION,'release_sha256':digest(ROOT/'deft.zip'),
            'inputs_sha256':digest(ROOT/'stimuli.jsonl'),'external_ids_sha256':digest(ROOT/'external-keys.json'),
            'grades_sha256':{k:digest(ROOT/(k+'.jsonl')) for k in truth},'counts':counts,'excluded_or_unmapped':dict(warnings),
            'guo_abstracts':len(guo_ids),'historical':len(hist),'guo_outcome_scores_loaded':False}
    write_json(OUT/'input-seal.json',record);print(counts,record['excluded_or_unmapped'],len(docs),flush=True)


if __name__=='__main__':prepare()
