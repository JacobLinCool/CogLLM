"""Freeze prediction-only offline references for the single-article source and curve API."""
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tokenizers import Tokenizer

from research.backbone_transfer.extract import read_json,stream_rows,write_json
from research.reading_validation.definition_data import ROOT
from research.reading_validation.definition_features import Corpus
from research.reading_validation.definition_stock import evidence,render
from research.reading_validation.definition_stock_fit import load_models
from research.reading_validation.definition_data import OUT
from research.reading_validation.identity_prepare import TOKENIZER


def prepare(bank_path):
    bank=Corpus(bank_path);proposal,link=load_models()['semantic']
    threshold=read_json(OUT/'stock-thresholds.json')['thresholds']['semantic']['threshold']
    tok=Tokenizer.from_file(str(TOKENIZER));refs={}
    for candidate in sorted(stream_rows(ROOT/'stock-native-stimuli.jsonl'),key=lambda r:sha256(('api'+r['id']).encode()).digest()):
        pg=bank.page(candidate['id']);enc=tok.encode(pg.text,add_special_tokens=False)
        if pg.n<10:continue
        stop=int(pg.words[pg.n//2,1]);prefix=tok.encode(pg.text[:stop],add_special_tokens=False)
        if prefix.ids!=enc.ids[:len(prefix.ids)]:continue
        e=evidence(pg,proposal,link,'semantic');trace=render(pg,e,threshold,enc.offsets)
        if not trace['mentions'] and len(refs)>0:continue
        mentions=[[m['chars'], m['opening_token'], m['closing_token']] for m in trace['mentions']]
        refs[candidate['id']]={'mentions':mentions,'curve':[t['open_mentions'] for t in trace['tokens']],
                               'evidence':[w['explanatory_evidence'] for w in trace['words']],'stop_char':stop}
        if len(refs)==10:break
    if not any(e[-1] is not None for v in refs.values() for e in v['mentions']):raise ValueError('No closure in serving checks')
    write_json(ROOT/'stock-serving.json',refs)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('bank',type=Path);prepare(p.parse_args().bank)
