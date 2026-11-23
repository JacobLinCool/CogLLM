"""Score the frozen term-specific event decoder on the W00 positive and negative scientific sentences."""
from argparse import ArgumentParser
from collections import Counter,defaultdict
from pathlib import Path

from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.definition_data import OUT,ROOT
from research.reading_validation.definition_features import Corpus
from research.reading_validation.definition_stock import evidence
from research.reading_validation.definition_stock_fit import ARMS,gold,load_models,results,score_one


def evaluate(bank_dir):
    seal=read_json(OUT/'stock-external-lock.json');f=ROOT/'w00-structural.jsonl';src=ROOT/'w00-stimuli.jsonl'
    if digest(f)!=seal['grades'] or digest(src)!=seal['stimuli']:raise ValueError('Independent annotations changed')
    bank=Corpus(bank_dir,src);models=load_models();th=read_json(OUT/'stock-thresholds.json')['thresholds']
    groups=defaultdict(lambda:defaultdict(Counter));kinds=defaultdict(lambda:defaultdict(Counter));items=defaultdict(lambda:defaultdict(Counter))
    n=Counter()
    for r in stream_rows(f):
        if r['kind'] not in ('negative','single_forward'):continue
        pg=bank.page(r['id']);kw={'terms':{'t':r['terms'][0]} if r['terms'] else {},
            'links':[{'term':'t','definition':r['definitions'][0]}] if r['definitions'] else []}
        g=gold(pg,kw);n[r['kind']]+=1
        for a in ARMS:
            e=evidence(pg,*models[a],a);c=score_one(pg,e,th[a]['threshold'],g)
            groups[r['source_section']][a].update(c);kinds[r['kind']][a].update(c);items[r['id']][a].update(c)
    out={'source_lock':digest(OUT/'stock-external-lock.json'),'training_calibration':digest(OUT/'stock-thresholds.json'),
         'inputs':dict(n),'section_stratified':results(groups),'sentence_resampled':results(items),
         'kind_contingencies':{k:{a:dict(c) for a,c in v.items()} for k,v in kinds.items()}}
    write_json(OUT/'stock-external-events.json',out)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('bank_dir',type=Path);evaluate(p.parse_args().bank_dir)
