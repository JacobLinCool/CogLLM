"""Generate gold-term query geometry for the explicitly supplied-term assay without any definition labels."""

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.definition_data import ROOT,OUT
from research.reading_validation.identity_prepare import DATA,PREVIOUS,TOKENIZER,sealed_rows
from tokenizers import Tokenizer


def prepare():
    src=read_json(OUT/'input-seal.json');info={}
    if digest(ROOT/'test.jsonl')!=src['grades_sha256']['test']:raise ValueError('Test geometry registration changed')
    for r in stream_rows(ROOT/'test.jsonl'):info[r['id']]={'kind':'test','query_terms':r['terms']}
    external=read_json(ROOT/'external-keys.json')
    for r in stream_rows(DATA/'jargon/geometry.jsonl'):
        terms={t['term']:[t['start'],t['end']] for t in r['terms']}
        info[external['jargon'][r['id']]]={'kind':'jargon','query_terms':terms}
    token=Tokenizer.from_file(str(TOKENIZER));layout=read_json(PREVIOUS/'qwen.json')['rows']
    docs={q['id']:q['text'] for q in stream_rows(ROOT/'stimuli.jsonl')}
    for old,key in external['historical'].items():
        e=token.encode(docs[key],add_special_tokens=False);r=layout[old]
        anchors={k:[e.offsets[a][0],e.offsets[b-1][1]] for k,(a,b) in r['anchor_tokens'].items()}
        text=docs[key]
        for name,(a,b) in anchors.items():
            while a<b and text[a].isspace():a+=1
            while b>a and text[b-1].isspace():b-=1
            anchors[name]=[a,b]
        info[key]={'kind':'historical','query_terms':anchors}
    sealed_rows(ROOT/'probe-layout-v2.jsonl',[{'id':k,**v} for k,v in sorted(info.items())]);print(len(info),digest(ROOT/'probe-layout-v2.jsonl'))


if __name__=='__main__':prepare()
