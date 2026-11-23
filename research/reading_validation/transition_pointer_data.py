"""Assemble original clicked regions without passing gold candidates to the observer."""
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
from itertools import chain

import numpy as np

from research.reading_validation.transition_architecture_prepare import load_tokenizer,word_indices
from research.reading_validation.transition_direct_data import PRON,display,gold_docs
from research.reading_validation.transition_direct_game import expert_entities
from research.reading_validation.transition_game_prepare import OUT as GAME,ROOT,silver_docs,votes_for
from research.reading_validation.transition_score import dump,sha
from research.reading_validation.transition_pointer_features import sentence_ids
from research.reading_validation.transition_snac_ner import FILES,POS


OUT=Path('research/results/reading-validation-transition-pointer')
PROTOCOL=Path('research/reading_validation/transition-pointer-protocol.md')


def extract(row,tree,crac,els,tokens,old_words,coords,token_indices,seen,reader_key):
    entities,spans=expert_entities(crac,els,tree)
    known,_=votes_for(tree,crac,els,tokens,coords,old_words)
    raw_to_new={old:new for new,old in enumerate(token_indices)}
    pos={(a,b):j for j,(a,b,*_) in enumerate(old_words)}
    candidates={}
    geo=np.array([old_words[j][:2] for j in token_indices])
    def interval(ne):
        if ne not in candidates:
            raw=spans[ne]
            a=coords[min(raw)][0];b=coords[max(raw)][1]
            ix=np.flatnonzero((geo[:,0]<b)&(geo[:,1]>a))
            candidates[ne]=(int(ix[0]),int(ix[-1])+1) if len(ix) else None
        return candidates[ne]
    audit=Counter();ans={}
    for pd in tree.iter('PDante'):
        members=spans[pd.get('id')]
        if len(members)!=1:continue
        orig=members[0];j=pos.get(coords[orig])
        if j not in known:continue
        new=raw_to_new.get(j)
        if new is None or new==0:audit['no_candidate_or_anchor_alignment']+=1;continue
        players=defaultdict(set)
        for a in pd.iter('anchor'):
            if a.get('mode')=='a' and a.get('type')=='DO':players[a.get('user_id')].add(a.get('ante'))
        votes=Counter();fresh=Counter();by_entity=defaultdict(set);bad=False;names=set()
        for person,choices in players.items():
            if len(choices)>1:audit['ambiguous_repeated_surface_clicks']+=1;continue
            ref=next(iter(choices));rng=interval(ref)
            if not rng or rng[1]>new:bad=True;break
            ent=entities[ref];key=(rng[0],rng[1],ent);votes[key]+=1;fresh[key]+=person not in seen
            names.add(person)
            if ent not in by_entity:
                for other,chain_id in entities.items():
                    if chain_id==ent and spans[other] and max(spans[other])<orig:
                        v=interval(other)
                        if v and v[1]<=new:by_entity[ent].add(v)
        if bad:audit['unmappable_or_nonprefix_click']+=1;continue
        if sum(votes.values())<4:audit['fewer_four_exact_clickers']+=1;continue
        if new in ans:raise ValueError('Two candidate references at the same visible word')
        reader_key.update(names);audit['eligible_queries']+=1;audit['reader_clicks']+=sum(votes.values())
        audit['new_reader_clicks']+=sum(fresh.values());audit['multi_entity_queries']+=len(by_entity)>1
        es={k:i for i,k in enumerate(sorted(by_entity))}
        ans[new]={'spans':[[a,b,c,es[e],fresh[(a,b,e)]] for (a,b,e),c in sorted(votes.items())],
                  'equivalent':[[list(h) for h in sorted(by_entity[e])] for e in sorted(by_entity)],
                  'people':len(names),'source_query_group':row['group']}
    return ans,audit


def prepare():
    import nltk
    from nltk.tag import PerceptronTagger
    if sha(ROOT/'game-native-stimuli.jsonl')!=json.loads((GAME/'input-seal.json').read_text())['input_sha256']:
        raise ValueError('Unexpected previously separated game source')
    use={r['id']:r for r in map(json.loads,(ROOT/'game-native-stimuli.jsonl').read_text().splitlines()) if r['domain']=='game'}
    specs=json.loads(Path('research/backbone_transfer/models.json').read_text())['models'];tok=load_tokenizer('qwen',specs['qwen'])
    pos_data=ROOT/'nltk';fn=next(pos_data.rglob('averaged_perceptron_tagger_eng.zip'))
    if sha(fn)!=FILES[fn.name]:raise ValueError('POS tagger weights changed')
    nltk.data.path.insert(0,str(pos_data));tag=PerceptronTagger()
    fit_people=set()
    for d,genre,fold,tree,cr,els,tokens in silver_docs():
        if d in use and fold=='train':
            fit_people.update(a.get('user_id') for a in tree.iter('anchor') if a.get('mode')=='a')
    data=[];gold={};stats=defaultdict(Counter);people=defaultdict(set);new_people=set()
    catalog=chain(silver_docs(),((d,g,'test',tr,cr,el,ts) for d,g,tr,cr,el,ts in gold_docs()))
    for d,g,fold,tr,cr,els,ts in catalog:
        if d not in use:continue
        r=use[d];text,coords,_=display(tr,els,ts)
        if text!=r['text']:raise ValueError('Original visible source changed')
        old=r['words'];encoding=tok.encode(text,add_special_tokens=False);maps=word_indices(text,[w[:2] for w in old],tok,encoding)
        kept=[i for i,p in enumerate(maps) if p];words=[old[i] for i in kept];positions=[maps[i] for i in kept];segs=[]
        segs=sentence_ids(text,words)
        people_doc=set();targets,audit=extract(r,tr,cr,els,ts,old,coords,kept,fit_people,people_doc)
        stats[fold].update(audit);people[fold]|=people_doc
        if fold=='test':new_people|=people_doc-fit_people
        forms=[text[a:b] for a,b,*_ in words];pos=[t for _,t in tag.tag(forms)]
        anchors=[j for j,f in enumerate(forms) if f.casefold() in PRON and j] if fold=='test' else sorted(targets)
        fixes=[]
        for j in anchors:
            prefix=[t for _,t in tag.tag(forms[:j+1])]
            if prefix[:max(0,j-1)]!=pos[:max(0,j-1)]:
                raise ValueError('Full-text POS uses information beyond the first two subsequent words')
            fixes.append(prefix[j-1])
        item={'id':d,'fold':fold,'group':r['group'],'text':text,'words':[w[:2] for w in words],
              'tokens':positions,'sentences':segs,'pos':pos,'queries':anchors,'pos_last':fixes,'token_count':len(encoding.ids)}
        if fold=='test':gold[d]=targets
        else:
            item['clicks']={str(j):[[a,b,n] for a,b,n,_,_ in q['spans']] for j,q in targets.items()}
        data.append(item)
    out=ROOT/'pointer-stimuli.jsonl';out.write_text(''.join(json.dumps(s,separators=(',',':'))+'\n' for s in data))
    OUT.mkdir(parents=True,exist_ok=True);dump(OUT/'external-clicks.json',{'articles':gold,'participant_id_columns':False})
    summary={'protocol_sha256':sha(PROTOCOL),'input_sha256':sha(out),'gold_sha256':sha(OUT/'external-clicks.json'),
             'documents':dict(Counter(r['fold'] for r in data)),'target_intake':{k:dict(v) for k,v in stats.items()},
             'distinct_eligible_clickers':{k:len(v) for k,v in people.items()},'game_train_annotators':len(fit_people),
             'unseen_eligible_gold_clickers':len(new_people),'pos_model_sha256':FILES[fn.name],
             'offered_gold_query_positions':sum(len(r['queries']) for r in data if r['fold']=='test')}
    dump(OUT/'source-seal.json',summary);print(summary,flush=True)


if __name__=='__main__':prepare()
