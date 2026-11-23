"""Prepare strictly disjoint public books, silver game training and gold game articles."""
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np

from research.reading_validation.transition_architecture_prepare import load_tokenizer,word_indices
from research.reading_validation.transition_direct_data import OUT as DIRECT,PD,PDFINGER,PRON,ROOT,display,gold_docs
from research.reading_validation.transition_direct_game import disagreement,expert_entities
from research.reading_validation.transition_score import dump,sha

OUT=Path('research/results/reading-validation-transition-game')
PROTOCOL=Path('research/reading_validation/transition-game-supervision-protocol.md')
FORM=tuple(sorted(PRON))


def silver_docs():
    with ZipFile(PD) as z:
        for f in sorted(z.namelist()):
            if not(f.startswith('silver/') and f.endswith('-game.xml')):continue
            part,fold,genre,_,filename=f.split('/')
            tree=ET.fromstring(z.read(f));wels=list(tree.iter('W'));tokens=[''.join(w.itertext()).strip() for w in wels]
            crac=[r.split('\t') for r in z.read(f.replace('masxml/','CRAC/').replace('-game.xml','.CRAC')).decode().splitlines()[1:] if r.strip()]
            if tokens!=[r[0] for r in crac]:raise ValueError('Silver identity labels and player text do not share a token stream')
            yield 'silver-'+genre+'-'+filename.removesuffix('-game.xml'),genre,fold,tree,crac,wels,tokens


def work(d,genre):
    prefix,_,item=d.split('-',2)
    if genre=='gutenberg':
        m=re.fullmatch(r'(.+?\d+)[a-z]+',item.casefold())
        if m:return genre+'-'+m[1]
    return genre+'-'+item.casefold()


def shingles(text):
    toks=re.findall('[a-z0-9]+',text.casefold())
    return {tuple(toks[i:i+20]) for i in range(len(toks)-19)}


def votes_for(tree,crac,wels,tokens,coords,words,unseen=None):
    chains,spans=expert_entities(crac,wels,tree)
    positions={(w[0],w[1]):j for j,w in enumerate(words)};cands=defaultdict(list);target={};people=set()
    for pd in tree.iter('PDante'):
        for an in pd.iter('anchor'):
            if an.get('mode')=='a':people.add(an.attrib['user_id'])
        ix=spans.get(pd.attrib['id'],())
        if len(ix)==1 and tokens[ix[0]].casefold() in PRON:cands[ix[0]].append(pd)
    for i,blocks in cands.items():
        if len(blocks)!=1 or coords[i] not in positions:continue
        pd=blocks[0]
        if any(k in pd.attrib for k in ('start_chr','end_chr','visibility')):continue
        choices=defaultdict(set);unmapped=False
        for a in pd.iter('anchor'):
            who=a.get('user_id')
            if a.get('mode')!='a' or (unseen is not None and who in unseen):continue
            kind=a.get('type')
            if kind=='DO':
                ante=a.get('ante')
                if ante not in chains or not spans[ante] or max(spans[ante])>=i:
                    unmapped=True;break
                choice='entity:'+chains[ante]
            elif kind in ('DN','PR','NR'):choice='reading:'+kind
            else:raise ValueError('Unexpected independent game choice')
            choices[who].add(choice)
        if unmapped or any(len(v)>1 for v in choices.values()):continue
        earlier=[next(iter(v)) for v in choices.values() if next(iter(v)).startswith('entity:')]
        if len(earlier)>=4:target[positions[coords[i]]]={'p':disagreement(earlier),'entity_readers':len(earlier),'players':len(choices)}
    return target,people


def prepare():
    if sha(PD)!=PDFINGER:raise ValueError('Wrong public reader archive')
    specs=json.loads(Path('research/backbone_transfer/models.json').read_text())['models']
    tok=load_tokenizer('qwen',specs['qwen']);direct=ROOT/'snac-direct-stimuli.jsonl'
    direct_meta=json.loads((DIRECT/'input-seal.json').read_text())
    if sha(direct)!=direct_meta['input_sha256']:raise ValueError('Word source changed')
    base={r['id']:r for r in map(json.loads,direct.read_text().splitlines())}
    held={d:r for d,r in base.items() if r['split']=='independent'}
    gold_docs_by_work={work(d,direct_meta['independent_articles'][d]['genre']) for d in held}
    goldsh=set().union(*(shingles(r['text']) for r in held.values()))
    catalog=[];excluded=[];devsh=set();devworks=set()
    for d,genre,fold,tree,crac,wels,tokens in silver_docs():
        text,coords,words=display(tree,wels,tokens);count=len(tok.encode(text,add_special_tokens=False).ids)
        title=work(d,genre);reason=None
        if title in gold_docs_by_work:reason='same_gold_work'
        elif count>2048:reason='native_context_limit'
        elif shingles(text)&goldsh:reason='gold_twenty_word_overlap'
        if reason:excluded.append([d,reason]);continue
        catalog.append((d,genre,fold,title))
        if fold=='dev':devsh.update(shingles(text));devworks.add(title)
    selected={d:(g,fold,title) for d,g,fold,title in catalog};data=[];people={'train':set(),'dev':set(),'gold':set()};itemmeta={}
    for d,genre,fold,tree,crac,wels,tokens in silver_docs():
        if d not in selected:continue
        title=selected[d][2];text,coords,words=display(tree,wels,tokens)
        if fold=='train' and (title in devworks or shingles(text)&devsh):
            excluded.append([d,'silver_development_overlap']);continue
        y,players=votes_for(tree,crac,wels,tokens,coords,words)
        ix=[i for i,w in enumerate(words) if text[w[0]:w[1]].casefold() in PRON]
        if not ix or not y:excluded.append([d,'no_four_reader_entity_outcome']);continue
        encoded=tok.encode(text,add_special_tokens=False)
        ge=word_indices(text,[words[i][:2] for i in ix],tok,encoded)
        use=[(i,p) for i,p in zip(ix,ge,strict=True) if p is not None]
        if not any(i in y for i,p in use):excluded.append([d,'no_certified_four_reader_outcome']);continue
        people[fold]|=players
        data.append({'id':d,'domain':'game','fold':fold,'group':title,'text':text,'words':words,
                     'save':[i for i,p in use],'token':[p for i,p in use],'token_count':len(encoded.ids),
                     'labels':[[-1,-1,-1,y[i]['p'] if i in y else -1] for i,p in use]})
        itemmeta[d]={'fold':fold,'genre':genre,'work':title,'observed':sum(i in y for i,p in use),
                     'reader_assignments':sum(y[i]['entity_readers'] for i,p in use if i in y)}
    new_gold=[];canonical=json.loads((DIRECT/'external-reader-outcomes.json').read_text())
    known={(r['doc'],r['word']):r for r in canonical['rows'] if r['antecedent_disagreement'] is not None}
    recovered={}
    for d,genre,tree,crac,wels,tokens in gold_docs():
        if d not in held:continue
        r=held[d];text,coords,words=display(tree,wels,tokens)
        ix=[i for i,w in enumerate(r['words']) if text[w[0]:w[1]].casefold() in PRON]
        maps=r['token_ranges']['qwen'];y,players=votes_for(tree,crac,wels,tokens,coords,r['words'])
        people['gold']|=players
        new,_=votes_for(tree,crac,wels,tokens,coords,r['words'],unseen=people['train'])
        for i,v in y.items():
            recovered[(d,i)]=v['p']
            if i in new:new_gold.append({'doc':d,'word':i,'form':text[words[r['indices'][i]][0]:words[r['indices'][i]][1]].casefold(),
                                         'd':new[i]['p'],'entity_readers':new[i]['entity_readers']})
        data.append({'id':d,'domain':'game','fold':'test','group':work(d,genre),'text':text,'words':r['words'],
                     'save':ix,'token':[maps[i] for i in ix],'token_count':r['token_counts']['qwen']})
    if set(known)!=set(recovered) or any(abs(known[k]['antecedent_disagreement']-p)>1e-12 for k,p in recovered.items()):
        raise ValueError('New independent reader extraction changed the already registered primary gold estimand')
    for r in base.values():
        if r['split']=='train':
            data.append({'id':r['id'],'domain':'snac','fold':'train','group':r['group'],'text':r['text'],'words':r['words'],
                         'save':list(range(len(r['words']))),'token':r['token_ranges']['qwen'],
                         'token_count':r['token_counts']['qwen'],'labels':[v+[-1] for v in r['labels']]})
    OUT.mkdir(parents=True,exist_ok=True);packet=ROOT/'game-native-stimuli.jsonl'
    packet.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in data))
    overlap={'training_players':len(people['train']),'development_players':len(people['dev']),
             'gold_players':len(people['gold']),'training_gold_intersection':len(people['train']&people['gold']),
             'development_gold_intersection':len(people['dev']&people['gold'])}
    dump(OUT/'new-reader-outcomes.json',{'rows':new_gold,'criterion':'gold antecedent votes from players absent from every independent response in retained training docs'})
    counts=Counter((r['domain'],r['fold']) for r in data)
    dump(OUT/'input-seal.json',{'protocol_sha256':sha(PROTOCOL),'input_sha256':sha(packet),'archive_sha256':PDFINGER,
         'registered_gold_sha256':sha(DIRECT/'external-reader-outcomes.json'),'supplement_gold_sha256':sha(OUT/'new-reader-outcomes.json'),
         'source_articles':itemmeta,'excluded_silver_docs':excluded,'source_gold_player_overlap':overlap,
         'public_inputs':{'/'.join(k):v for k,v in counts.items()},'new_player_gold_mentions':len(new_gold)})
    print(sha(packet),dict(counts),overlap,len(new_gold),Counter(x[1] for x in excluded),flush=True)


if __name__=='__main__':prepare()
