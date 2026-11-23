"""Seal the Dutch meaning-judgment materials while leaving all participant rows unopened."""
from collections import Counter
from pathlib import Path
import re
from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.exposure_data import ROOT,read_csv,key
from research.reading_validation.identity_prepare import TOKENIZER,sealed_rows

DIR=ROOT/'judgment'
OUT=Path('research/results/reading-validation-meaning')
PROTOCOL=Path(__file__).with_name('meaning-protocol.md')
COLUMNS={'CorrectUnfamiliarTestitem':(True,False),'CorrectFamiliarTestitem':(True,True),
         'IncorrectUnfamiliarTestitem':(False,False),'IncorrectFamiliarTestitem':(False,True)}


def prepare():
    DIR.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    tok=Tokenizer.from_file(str(TOKENIZER));rows=read_csv(ROOT/'stimuli_exp3.csv')
    subjects=[];texts={};excluded=[];onsets=[];lengths=[]
    for r in rows:
        word=r['Word'].strip();meaning=r['Translation'].strip()
        onset=tok.encode(' '+meaning.casefold(),add_special_tokens=False).ids[0];onsets.append(onset)
        for col,(fit,familiar) in COLUMNS.items():
            sentence=r[col].strip();matches=list(re.finditer(r'(?<!\w)'+re.escape(word)+r'(?!\w)',sentence,re.I))
            if len(matches)!=1:
                excluded.append({'word':word,'field':col,'matches':len(matches)});continue
            a,b=matches[0].span();host=(sentence[:a]+' <woord> '+sentence[b:]).casefold();ids={}
            gloss=f'Het woord {word} betekent {meaning}.\n\n'
            designs={'raw':(sentence,a,b),'known':(gloss+sentence,len(gloss)+a,len(gloss)+b),
                     'native':(sentence[:a]+meaning+sentence[b:],a,a+len(meaning))}
            for arm,(s,lo,hi) in designs.items():
                enc=tok.encode(s,add_special_tokens=False);js=[i for i,(l,h) in enumerate(enc.offsets) if h>lo and l<hi]
                if not js:raise ValueError('Experimental target has no native tokens')
                start=enc.offsets[js[0]][0]
                if tok.encode(s[:start],add_special_tokens=False).ids!=enc.ids[:js[0]]:
                    raise ValueError('Inconsistent last genuine pre-noun state')
                idx=key(s);texts[idx]={'id':idx,'text':s};ids[arm]={'id':idx,'first':js[0],'last':js[-1],'length':len(enc.ids)}
                lengths.append(len(enc.ids))
            subjects.append({'sentence':sentence,'word':word,'gloss':meaning,'fit':fit,'familiar':familiar,'frame':' '.join(host.split()),
                             'onset':onset,'ids':ids})
    sealed_rows(DIR/'stimuli.jsonl',[texts[k] for k in sorted(texts)]);write_json(DIR/'vocabulary.json',{'onsets':sorted(set(onsets))})
    write_json(DIR/'cases.json',subjects)
    write_json(OUT/'input-seal.json',{'protocol':digest(PROTOCOL),'stimuli':digest(DIR/'stimuli.jsonl'),'vocabulary':digest(DIR/'vocabulary.json'),
        'mapping':digest(DIR/'cases.json'),'source_hash':digest(ROOT/'stimuli_exp3.csv'),'trials_hash':digest(next(ROOT.glob('Experiment 3*csv'))),
        'codebook':digest(next(ROOT.glob('*.ods'))),'judgments_unopened':True,'source_words':len(rows),'eligible_cases':len(subjects),
        'excluded':excluded,'unique_texts':len(texts),'max_tokens':max(lengths),'prior_inspected':'codebook, paper, stimuli, trial headers, experiments 1 and 2'})


if __name__=='__main__':prepare()
