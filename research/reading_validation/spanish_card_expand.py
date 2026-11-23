"""Add native onset coverage for unobserved Spanish response-card vocabulary; print no responses."""
from research.backbone_transfer.extract import read_json,write_json,digest,stream_rows
from research.reading_validation.identity_prepare import sealed_rows
from research.reading_validation.spanish_clue_data import ROOT,OUT,TOKENIZER,PUNCT,norm,logframes,build,dictionary
from pathlib import Path
from tokenizers import Tokenizer


def prepare():
    tok=Tokenizer.from_file(str(TOKENIZER));lex=dictionary();src=set();cards=[];old=dict()
    for filename in read_json(ROOT/'eligible-logs.json'):
        p=ROOT/'logs'/filename;_,design,main,_=build(p,lex);src.update(norm(r['pseudo']) for r in main)
        candidates={}
        for r in design:
            if r['condition'] in ('High','Low'):candidates.setdefault((r['block'],norm(r['source'])),set()).add(r['id'])
        _,records=logframes(p)
        for block,r in records:
            if r['Procedure']!='TestProc':continue
            pseudo=norm(r['TargetPseudoword']);ids=candidates.get((block,pseudo),set())
            if len(ids)==1:cards.append((pseudo,next(iter(ids)),[r['Word'+str(j)] for j in range(1,5)]))
    old=set(read_json(ROOT/'vocabulary.json')['onsets']);new=set(old);needed=set();covered=set();families=set()
    for pseudo,k,options in cards:
        if pseudo in src:continue
        v={tok.encode(' '+o.strip(PUNCT),add_special_tokens=False).ids[0] for o in options};new.update(v);needed.add(k);families.add(pseudo)
        if v<=old:covered.add(pseudo)
    path=ROOT/'expanded';path.mkdir(exist_ok=True);sealed_rows(path/'stimuli.jsonl',(r for r in stream_rows(ROOT/'stimuli.jsonl') if r['id'] in needed));write_json(path/'vocabulary.json',{'onsets':sorted(new)})
    write_json(OUT/'card-expansion-lock.json',dict(fit=digest(OUT/'choice-joint-scores.json'),plan=digest(Path(__file__).with_name('spanish-card-expansion.md')),stimuli=digest(path/'stimuli.jsonl'),vocabulary=digest(path/'vocabulary.json'),source=sorted(families),overlap=sorted(covered),fresh=sorted(families-covered),original_vocabulary=len(old),expanded_vocabulary=len(new),documents=len(needed)))
    print('Additional card-bearing texts',len(needed),'new source families',len(families-covered),'old and new vocabulary',len(old),len(new))

if __name__=='__main__':prepare()
