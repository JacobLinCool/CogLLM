"""Seal full posterior count and source-specific flows from ten actual offline passages."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import digest,read_json,stream_rows,write_json
from research.reading_validation.definition_data import OUT,ROOT
from research.reading_validation.definition_features import Corpus
from research.reading_validation.definition_stock_fit import load_models
from research.reading_validation.identity_prepare import sealed_rows
from research.reading_validation.stock_probability import predict


def prepare(bank_dir):
    bank=Corpus(bank_dir);refs=read_json(ROOT/'stock-serving.json');data=[];arrays={}
    fit=read_json(OUT/'stock-probability-weights.json')['fits']['semantic'];models=load_models()['semantic']
    factor=read_json(OUT/'stock-count-scales.json')['scales']['semantic']['scale']
    for k in sorted(refs):
        page=bank.page(k);t=predict(page,'semantic',fit,models)
        data.append({'id':k,'text':page.text,'cut':refs[k]['stop_char']})
        arrays[k+'/source_char_onsets']=page.words[t.starts,0];arrays[k+'/open']=factor*t.occurrence
        arrays[k+'/active']=factor*t.unresolved;arrays[k+'/resolved']=factor*t.resolved
    src=ROOT/'stock-replay.jsonl';dest=ROOT/'stock-replay.npz';sealed_rows(src,data);np.savez_compressed(dest,**arrays)
    write_json(OUT/'stock-probability-replay-inputs.json',{'articles':digest(src),'references':digest(dest),'count':len(refs),
                'checksum_source':digest(OUT/'stock-probability-weights.json'),'calibration':digest(OUT/'stock-count-scales.json')})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('bank',type=Path);prepare(p.parse_args().bank)
