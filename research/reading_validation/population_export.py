"""Export the source-only multinomial readout and its outcome-blind replay fixtures."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from research.backbone_transfer.extract import digest,read_json,write_json,stream_rows,write_rows
from research.reading_validation.cloze_population_data import ROOT,OUT
from research.reading_validation.cloze_population_fit import data
from research.reading_validation.cloze_population_model import batches
from research.reading_validation.identity_prepare import DATA
from research.reading_validation.population_inference import DEFAULT

PLAN=Path(__file__).with_name("population-serving-protocol.md")


def export(native):
    raw,X,U,V,_,meta=data(native)
    source=read_json(OUT/"fold-5.json")
    lock=read_json(OUT/"forecast-lock.json")
    if digest(OUT/"fold-5.json")!=lock["source_fits"][-1]:raise ValueError("Source head changed")
    DEFAULT.mkdir(exist_ok=True)
    names=[r["word"] for r in read_json(ROOT/"dictionary.json")]
    write_json(DEFAULT/"dictionary.json",names)
    g=source["geometry"]
    rng=np.random.default_rng(725)
    projection=[rng.choice([-1.,1.],(2048,32)).astype(np.float32)/np.sqrt(np.float32(2048)) for _ in range(2)]
    with np.load(native/"lexicon.npz") as z:
        np.savez_compressed(DEFAULT/"weights.npz",co=np.array(source["models"]["context"]["coefficients"]),U=U,V=V,
            basis=np.array(g["vectors"]),mean=np.array(g["mean"]),scale=np.array(g["scale"]),unique=z["unique"],route=z["route"],
            share=z["share"].astype(np.float32),mid_proj=projection[0],last_proj=projection[1])
    write_json(DEFAULT/"spec.json",dict(source=digest(OUT/"fold-5.json"),protocol=digest(PLAN),word_categories=len(names)+1,
        revision=read_json(Path('research/backbone_transfer/models.json'))['models']['qwen']['revision'],trained_coefficients=275,
        files={n:digest(DEFAULT/n) for n in ('weights.npz','dictionary.json')},
        training_sentences=205,training_word_positions=1726,training_reader_choices=135905))
    geometry=read_json(ROOT/"geometry.json");docs={d['id']:d for d in stream_rows(ROOT/"semantic-stimuli.jsonl")}
    packets=[]; lookup={}
    for id0,d in geometry.items():
        if d['source']!='Provo':continue
        r=docs[id0];packets.append(dict(id=id0,text=r['text'],ends=r['ends']))
        for t in d['items']:
            lookup[t['source_id']]=(id0,t['word']-1)
    file=ROOT/"serving-materials.jsonl";write_rows(file,packets)
    ext=np.flatnonzero(~meta['source']);refs=[]
    for ix,probs in batches(ext,raw,X,g,{'context':source['models']['context']},U,V):
        for i,lp in zip(ix,probs['context'],strict=True):
            uid=meta['ids'][i]; id0,w=lookup[uid];p=np.exp(lp.astype(np.float32).astype(float))
            refs.append(dict(entity=id0,word=w,source_id=uid,disagreement=float(1-p@p),other=float(p[-1]),top=np.argsort(-p,kind='stable')[:3].tolist()))
    expected=ROOT/"serving-reference.jsonl";write_rows(expected,refs)
    write_json(OUT/"serving-export.json",dict(protocol=digest(PLAN),native=digest(native/'receipt.json'),readout=digest(DEFAULT/'spec.json'),
            articles=len(packets),word_comparisons=len(refs),stimuli=digest(file),references=digest(expected)))
    print(len(packets),len(refs),digest(DEFAULT/"spec.json"))


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('native',type=Path);export(p.parse_args().native)
