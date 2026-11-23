"""Recover independent crowd interpretations, collapsing alternate mentions of one gold entity."""
from collections import Counter,defaultdict
import json
import re

from research.reading_validation.transition_direct_data import OUT,PD,PDFINGER,PRON,ROOT,display,gold_docs
from research.reading_validation.transition_score import dump,sha


def expert_entities(crac,wels,tree):
    markables=defaultdict(list);entities={}
    for i,(_,annotation,_) in enumerate(crac):
        for tag in annotation.split('@'):
            if not tag:continue
            match=re.fullmatch(r'([BI])-(markable_\d+)=(set_\d+)',tag)
            if not match:raise ValueError('Unknown gold annotation syntax')
            _,index,entity=match.groups()
            if entities.setdefault(index,entity)!=entity:raise ValueError('Gold entity changes within one span')
            markables[index].append(i)
    spans=defaultdict(set)
    for mid,word in markables.items():spans[tuple(word)].add(entities[mid])
    coords={w:i for i,w in enumerate(wels)};ids={};runs={}
    for ne in tree.iter('ne'):
        key=ne.attrib['id'];indices=tuple(coords[w] for w in ne.iter('W'));runs[key]=indices
        if len(spans[indices])==1:ids[key]=next(iter(spans[indices]))
    return ids,runs


def disagreement(choices):
    counts=Counter(choices);n=len(choices)
    return 1-sum(q*(q-1) for q in counts.values())/(n*(n-1)) if n>=2 else None


def recover():
    seal=json.loads((OUT/'input-seal.json').read_text())
    packet=ROOT/'snac-direct-stimuli.jsonl'
    if sha(packet)!=seal['input_sha256'] or sha(PD)!=PDFINGER:raise ValueError('Inputs or human game archive changed')
    docs={r['id']:r for r in map(json.loads,packet.read_text().splitlines()) if r['split']=='independent'}
    final=[];audit=Counter();denoms=Counter()
    for d,genre,tree,crac,wels,tokens in gold_docs():
        if d not in docs:continue
        text,coords,_=display(tree,wels,tokens)
        row=docs[d];ends={(w[0],w[1]):i for i,w in enumerate(row['words'])}
        if text!=row['text']:raise ValueError('Independent article differs from the sealed presentation')
        chains,spans=expert_entities(crac,wels,tree)
        cands=defaultdict(list)
        for pd in tree.iter('PDante'):
            key=pd.attrib['id'];idx=spans.get(key,())
            if len(idx)==1 and tokens[idx[0]].casefold() in PRON:cands[idx[0]].append(pd)
        for i,pds in cands.items():
            audit['visible_pronoun_markables']+=len(pds)
            if len(pds)!=1:audit['multiply_nested_pronoun_locations']+=1;continue
            pd=pds[0]
            if any(k in pd.attrib for k in ('start_chr','end_chr','visibility')):
                audit['edited_or_hidden']+=1;continue
            coordinate=ends.get(coords[i])
            if coordinate is None:audit['uncertified_native_word']+=1;continue
            choices=defaultdict(set);bad=False;unmapped=0
            for a in pd.iter('anchor'):
                if a.get('mode')!='a':continue
                who=a.get('user_id');typ=a.get('type')
                if not who or typ not in ('DO','DN','PR','NR'):raise ValueError('Unidentified independent response')
                if typ=='DO':
                    antecedent=a.get('ante')
                    if antecedent not in chains or not spans[antecedent] or max(spans[antecedent])>=i:
                        unmapped+=1;continue
                    value='entity:'+chains[antecedent]
                else:value='reading:'+typ
                choices[who].add(value)
            if unmapped:audit['at_least_one_unmapped_choice']+=1;continue
            if any(len(a)>1 for a in choices.values()):audit['conflicting_repeated_player']+=1;continue
            all_votes=[next(iter(a)) for a in choices.values()]
            previous=[v for v in all_votes if v.startswith('entity:')]
            if len(all_votes)<4:audit['fewer_than_four_readers']+=1;continue
            if len(previous)<4:audit['fewer_than_four_entity_readers']+=1
            denoms[len(all_votes)]+=1
            final.append({'doc':d,'genre':genre,'word':int(coordinate),'start':coords[i][0],'end':coords[i][1],
                          'form':tokens[i].casefold(),'players':len(all_votes),'prior_entity_players':len(previous),
                          'antecedent_disagreement':disagreement(previous) if len(previous)>=4 else None,
                          'all_disagreement':disagreement(all_votes)})
    out={'source_sha256':PDFINGER,'packet_sha256':seal['input_sha256'],'counts':dict(audit),
         'reader_denominators':dict(sorted(denoms.items())),'rows':final}
    dump(OUT/'external-reader-outcomes.json',out)
    print(audit,len(final),sum(r['antecedent_disagreement'] is not None for r in final),
          len(set(r['doc'] for r in final)),Counter(r['genre'] for r in final))


if __name__=='__main__':recover()
