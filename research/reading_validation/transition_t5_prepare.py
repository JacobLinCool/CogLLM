"""Render source-only specialist training targets with exact human word coordinates."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import nltk
from nltk.tokenize import PunktTokenizer

from research.reading_validation.transition_score import dump, sha
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import ROOT,source

PROTOCOL=Path('research/reading_validation/transition-t5-protocol.md')
RESULT=Path('research/results/reading-validation-transition-t5')
PUNKT={
    'sent_starters.txt':'f3f8535483e1dba487241b764945168123bca3209a9645e59acd1225dc76edac',
    'abbrev_types.txt':'92a3e070f43d9b4c5534758ca40ad7343b04e7e29bfe0c2eb658a39445a4f779',
    'ortho_context.tab':'4bbcca25ed3d3f06c02402abf8419b9f033b8adc06e7b482eca4e45f81a5dc4c',
    'collocations.tab':'8e2da1225e4dd2cc9dba261ee231ccb134859e21b46006e7f472c5ee269af0cf',
}


def target(text,indices,words,positive):
    positive_indices=indices[positive[indices]]
    if not len(positive_indices):
        return 'true'
    runs=np.split(positive_indices,np.flatnonzero(np.diff(positive_indices)!=1)+1)
    names=[text[words[run[0],0]:words[run[-1],1]] for run in runs]
    return 'false character '+' <extra_id_1> '.join(names)


def prepare():
    docs,meta=inputs();raw=source()
    punkt=Path(str(nltk.data.find('tokenizers/punkt_tab/english/')))
    if {p.name:sha(p) for p in punkt.iterdir()}!=PUNKT:
        raise ValueError('Punkt sentence resources do not match the version used in this study')
    segmenter=PunktTokenizer('english')
    packet=ROOT/'snac-t5-stimuli.jsonl';manifest=[];stats=Counter()
    with packet.open('w') as out:
        for m in meta['documents']:
            d=m['doc'];row=docs[d];text=row['text'];words=np.asarray(row['words'])
            if m['split']!='test':
                valid,votes,_=outcomes(row,m,raw[d],'CharE')
            covered=np.zeros(len(words),dtype=int)
            for seg in m['segments']:
                origin=seg['start']
                for sent,(a,b) in enumerate(segmenter.span_tokenize(text[origin:seg['end']])):
                    a+=origin;b+=origin
                    indices=np.flatnonzero((words[:,0]>=a)&(words[:,1]<=b))
                    if not len(indices):
                        continue
                    covered[indices]+=1
                    if m['split']!='test' and not valid[indices].all():
                        stats['ambiguous_source_sentences_excluded']+=1
                        continue
                    key=f"{d}/{seg['index']}/{sent}"
                    item={'id':key,'doc':d,'split':m['split'],'sentence':text[a:b],
                          'context':' '.join(text[:a].split()[-921:])}
                    if m['split']!='test':
                        item['target']=target(text,indices,words,votes>0)
                        if item['target']!='true':stats[m['split']+'_positive_sentences']+=1
                    stats[m['split']+'_sentences']+=1
                    manifest.append({'id':key,'doc':d,'split':m['split'],'start':a,'end':b,
                                     'word_index_start':int(indices[0]),'word_index_stop':int(indices[-1]+1)})
                    print(json.dumps(item,separators=(',',':')),file=out)
            if not (covered==1).all():
                raise ValueError('Sentence segmentation left a human word uncovered or counted it twice')
    RESULT.mkdir(exist_ok=True,parents=True)
    dump(RESULT/'input-seal.json',{'protocol_sha256':sha(PROTOCOL),'input_sha256':sha(packet),
        'author_training_args_sha256':'e82cfda85f236a9b0f6c9ad38fd66064b3e9838666765fa57029bc1cc571ac24',
        'punkt_english_sha256':PUNKT,'counts':dict(stats),'sentences':manifest})
    print(sha(packet),stats)


if __name__=='__main__':prepare()
