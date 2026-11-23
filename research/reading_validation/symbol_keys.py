"""Certify extra native key positions for display equations encountered in training material."""
from bisect import bisect_left, bisect_right
from pathlib import Path
from tokenizers import Tokenizer
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.identity_prepare import TOKENIZER,sealed_rows
from research.reading_validation.symbol_geometry import ROOT,OUT
from research.reading_validation.symbol_syntax import proposals


def run():
    tokenizer=Tokenizer.from_file(str(TOKENIZER));updated=[];old=0;new=0
    for r in stream_rows(ROOT/'stimuli.jsonl'):
        enc=tokenizer.encode(r['text'],add_special_tokens=False);starts,ends=zip(*enc.offsets);k=set(r['keys']);old+=len(k)
        for s in proposals(r['text']):
            k.update(range(bisect_right(ends,s.start),bisect_left(starts,s.end)))
        new+=len(k);updated.append({**r,'keys':sorted(k)})
    path=ROOT/'stimuli-native.jsonl';sealed_rows(path,updated)
    write_json(OUT/'equation-key-addition.json',{'previous_packet':digest(ROOT/'stimuli.jsonl'),'all_paragraphs_identical':True,'stimuli':digest(path),'old_keys':old,'expanded_keys':new,'syntax':digest(Path(__file__).with_name('symbol_syntax.py')),'reason':'Original training examples contain display equations outside inline delimiters. Capture their source positions while preserving all frozen cohorts and examples.'})


if __name__=='__main__':run()
