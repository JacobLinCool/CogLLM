"""Apply already-fitted reader-fold coefficients to all 22 reserved pseudoword families."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from scipy.special import softmax
from research.backbone_transfer.extract import read_json,write_json,digest,stream_rows
from research.reading_validation.spanish_clue_data import ROOT,OUT,norm
from research.reading_validation.spanish_clue_score import responses,grouped
from research.reading_validation.spanish_choice_joint import external,material,pref
from research.reading_validation.meaning_score import folds


def run(folder):
    record=read_json(OUT/'card-expansion-lock.json');rec=read_json(folder/'receipt.json');fit=read_json(OUT/'choice-joint-scores.json')
    if (digest(OUT/'choice-joint-scores.json')!=record['fit'] or rec['input']!=record['stimuli'] or rec['vocabulary']!=record['vocabulary'] or digest(folder/'signals.npz')!=rec['signals']):raise ValueError('Changed frozen feature matrix or human training weights')
    emitted={r['id'] for r in stream_rows(ROOT/'expanded/stimuli.jsonl')};tr,_=responses();extra,_=external(tr);src=read_json(ROOT/'expanded/vocabulary.json')['onsets'];V={v:k for k,v in enumerate(src)}
    with np.load(folder/'signals.npz') as store:X,te,ex=material(extra,store,emitted,V)
    if ex or len(te)!=len(extra):raise ValueError('Expanded vocabulary still drops a displayed card')
    rows=tr+te;N=len(tr);items=np.array([r['source'] for r in te]);persons=np.array([r['person'] for r in rows]);words=np.asarray([[norm(o) for o in r['options']] for r in rows]);y=np.array([r['response'] for r in rows]);pfold=folds(persons,'spanish:reader:');pf=np.array([pfold[s] for s in persons]);idx=np.arange(len(rows))
    old_folds={r['person_fold']:r for r in fit['folds'] if r['lexical_fold']==-1};probs={a:np.full((len(te),4),np.nan) for a in ('surface','semantic')}
    for j in range(5):
        train=idx[(idx<N)&(pf!=j)];test=idx[(idx>=N)&(pf==j)]
        if len(train)!=old_folds[j]['train'] or set(persons[train])&set(persons[test]):raise ValueError('Estimation cohort changed')
        feat=pref(train,test,y,words);target=test-N
        for arm in probs:
            z=feat if arm=='surface' else np.concatenate((feat,X[target]),axis=2);w=np.asarray(old_folds[j]['models'][arm]['coefficients']);probs[arm][target]=softmax(z@w,axis=1)
    if not all(np.isfinite(z).all() for z in probs.values()):raise ValueError('Missing response forecasts')
    losses={a:-np.log(z[np.arange(len(te)),y[N:]]) for a,z in probs.items()};delta=losses['surface']-losses['semantic']
    old_v=set(read_json(ROOT/'vocabulary.json')['onsets'])
    from tokenizers import Tokenizer
    from research.reading_validation.spanish_clue_data import TOKENIZER,PUNCT
    tok=Tokenizer.from_file(str(TOKENIZER));overlap=np.array([all(tok.encode(' '+o.strip(PUNCT),add_special_tokens=False).ids[0] in old_v for o in r['options']) for r in te])
    summaries={}
    for name,mask in [('all',np.ones(len(te),bool)),('unscored_14_families',np.isin(items,record['fresh'])),('old_scored_cards',overlap),('new_cards',~overlap)]:
        summaries[name]={'choices':int(mask.sum()),'words':len(set(items[mask])),'readers':len(set(persons[N:][mask])),
             'nll':{a:float(l[mask].mean()) for a,l in losses.items()},'gain_word':grouped(delta[mask],items[mask]),'gain_reader':grouped(delta[mask],persons[N:][mask])}
    old=fit['result']['additional_strict_families'];replay=summaries['old_scored_cards']
    if replay['choices']!=old['choices'] or max(abs(replay['nll'][a]-old['models'][a]['nll']) for a in probs)>.001:raise ArithmeticError('Added vocabulary changed a previously computed prediction')
    write_json(OUT/'card-expansion-scores.json',dict(registered=digest(OUT/'card-expansion-lock.json'),prediction=digest(folder/'receipt.json'),models_changed=0,results=summaries))

if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);run(p.parse_args().folder)
