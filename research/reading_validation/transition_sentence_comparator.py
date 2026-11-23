"""Summarize character alerts at their study sentence and word coordinates."""
from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_author_precision import measures
from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_score import dump, sha, with_interval
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import RESULTS, source
from research.reading_validation.transition_t5_score import OUT, mark


PROTOCOL = Path('research/reading_validation/transition-sentence-comparator-protocol.md')


def record(label, prediction, category):
    gold = bool(label.any())
    correct = category and gold
    starts = np.flatnonzero(prediction & ~np.r_[False, prediction[:-1]])
    stops = np.flatnonzero(prediction & ~np.r_[prediction[1:], False]) + 1
    overlapping = sum(label[a:b].any() for a, b in zip(starts, stops, strict=True))
    return np.array([correct, category and not gold, gold and not category, not gold and not category,
                     correct and overlapping > 0, category and overlapping > 0,
                     overlapping if correct else 0, len(starts) if correct else 0,
                     overlapping, len(starts)], dtype=int)


def summarize(partials, ids, draws):
    groups = np.zeros((int(ids.max()) + 1, 10), dtype=int)
    np.add.at(groups, ids, partials)
    total = groups.sum(0)
    boot = draws @ groups
    point = measures(total[:4][None, :])
    samples = measures(boot[:, :4])
    out = {'tp_fp_fn_tn':total[:4].tolist(),
           **{k:with_interval(point[k][0],samples[k]) for k in ('precision','recall','f1','false_alarms_per_100_negatives')}}
    for key, num, den in (('overlap_per_correct_sentence',4,total[0]),
                          ('overlap_per_alerted_sentence',5,total[:2].sum()),
                          ('overlap_per_span_in_correct_sentences',6,total[7]),
                          ('overlap_per_span_in_alerted_sentences',8,total[9])):
        out[key]={'overlapping':int(total[num]),'evaluated':int(den),
                  'fraction':float(total[num]/den) if den else None}
    return out


def score(variant=None):
    meta=json.loads((OUT/'input-seal.json').read_text())
    material=defaultdict(list)
    for sentence in meta['sentences']:
        if sentence['split']=='test':
            material[sentence['doc']].append(sentence)
    raw=source(); docs,folds=inputs()
    qlock=json.loads((RESULTS/'prediction-lock.json').read_text())
    nlock=json.loads((RESULTS/'entity-control-lock.json').read_text())
    qfile=RESULTS/'heldout-probabilities.npz'; nfile=RESULTS/'entity-control-probabilities.npz'
    if sha(qfile)!=qlock['forecast_sha256'] or sha(nfile)!=nlock['forecast_sha256']:
        raise ValueError('Existing word forecast changed')
    qnpz=np.load(qfile); nnpz=np.load(nfile)
    qcut=json.loads((RESULTS/'readouts.json').read_text())['readouts']['CharE']['antecedents']['threshold']
    tq={}
    if variant:
        checked=json.loads((OUT/variant/'receipt.json').read_text())
        forecast=OUT/variant/'movie-generations.jsonl'
        if sha(forecast)!=checked['forecast_sha256'] or checked['input_sha256']!=meta['input_sha256']:
            raise ValueError('Specialist predictions changed')
        tq={s['id']:s['generation'] for s in map(json.loads,forecast.read_text().splitlines())}
    groups=[];partial=defaultdict(list);omitted=0;words=0
    for m in folds['documents']:
        if m['split']!='test':continue
        d=m['doc']; doc=docs[d]; valid,votes,_=outcomes(doc,m,raw[d],'CharE')
        causal=qnpz['CharE__'+d][:,3]>qcut; ner=nnpz[d][:,0]>nlock['cutoff']
        for row in material[d]:
            start,stop=row['word_index_start'],row['word_index_stop']
            s=slice(start,stop)
            if not valid[s].all():
                omitted+=1
                continue
            gold=votes[s]>0; words+=stop-start
            groups.append(m['group'])
            partial['qwen'].append(record(gold,causal[s],bool(causal[s].any())))
            partial['ner_surprisal'].append(record(gold,ner[s],bool(ner[s].any())))
            if variant:
                prediction,stats=mark(tq[row['id']], [doc['text'][a:b] for a,b,*_ in doc['words'][s]])
                category=bool(stats['well_formed_character_sentence'] or stats['empty_character_sentence'])
                partial['t5_'+variant].append(record(gold,prediction,category))
    titles,ids,draws=grouped_distribution(groups)
    result={'protocol_sha256':sha(PROTOCOL),'input_sha256':meta['input_sha256'],'variant':variant,
            'movie_groups':len(titles),'sentences':len(groups),'eligible_words':words,
            'unusable_sentences':omitted,'methods':{k:summarize(np.asarray(v),ids,draws) for k,v in partial.items()}}
    filename=OUT/(variant or '')/'sentence-scale.json'
    dump(filename,result)
    print(filename,len(groups),{k:[v['tp_fp_fn_tn'],v['f1']] for k,v in result['methods'].items()})


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('adapter','full'))
    score(parser.parse_args().variant)
