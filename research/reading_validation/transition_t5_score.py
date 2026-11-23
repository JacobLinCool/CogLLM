"""Score generated span sets on the exact movie word universe and compare frozen policies."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import re
from unicodedata import normalize

import numpy as np

from research.reading_validation.transition_author_precision import counts,measures
from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_score import dump, sha, with_interval
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_ner import POS,TYPES
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import RESULTS, ROOT, WORD, source

OUT=Path('research/results/reading-validation-transition-t5')
PROTOCOL=Path('research/reading_validation/transition-t5-protocol.md')


def conventional(s):
    return normalize('NFC',s).lower().replace('’',"'").replace('–','-')


def mark(decoded,words):
    """Mark all literal sequence matches, retaining empty and unmatchable model output counts."""
    words=[conventional(w) for w in words]
    mask=np.zeros(len(words),dtype=bool);status=Counter()
    normalized=decoded.strip()
    if normalized=='true':
        status['well_formed_clear_sentence']+=1
        return mask,status
    prefix=re.match(r'^false\s+character(?:\s+|$)',normalized)
    if prefix is None:
        status['malformed_sentence']+=1
        return mask,status
    body=normalized[prefix.end():].strip()
    if not body:
        status['empty_character_sentence']+=1
        return mask,status
    status['well_formed_character_sentence']+=1
    for phrase in body.split('<extra_id_1>'):
        tokens=[conventional(m.group()) for m in WORD.finditer(phrase)]
        if not tokens:
            status['empty_span']+=1
            continue
        pos=[k for k in range(len(words)-len(tokens)+1) if words[k:k+len(tokens)]==tokens]
        if not pos:status['unmatched_span']+=1
        else:status['matched_span']+=1
        if len(pos)>1:status['repeated_match_span']+=1
        for k in pos:mask[k:k+len(tokens)]=True
    return mask,status


def compare(y,methods,groups,selection):
    idx_groups,idx,draws=grouped_distribution(groups[selection])
    obs=y[selection]
    rows={};boots={}
    for arm,flag in methods.items():
        count=counts(obs,flag[selection],idx);total=count.sum(axis=0,keepdims=True)
        found=measures(total);samples=measures(draws@count)
        boots[arm]=samples
        rows[arm]={'tp_fp_fn_tn':total.ravel().astype(int).tolist(),
                   'movies_with_alerts':int(np.count_nonzero(count[:,:2].sum(axis=1))),
                   **{k:with_interval(v[0],samples[k]) for k,v in found.items()}}
    contrasts={}
    for other in ('qwen_word','ner_control'):
        for key in ('precision','recall','f1'):
            contrasts['t5_minus_'+other+'_'+key]=with_interval(
                rows['t5_span'][key]['estimate']-rows[other][key]['estimate'],
                boots['t5_span'][key]-boots[other][key])
    t=methods['t5_span'][selection];q=methods['qwen_word'][selection]
    return {'movie_groups':len(idx_groups),'words':len(obs),'human_positive':int(obs.sum()),'methods':rows,
            'paired_differences':contrasts,'word_alert_overlap':{
                'both_true':int((t&q&obs).sum()),'t5_only_true':int((t&~q&obs).sum()),
                'qwen_only_true':int((q&~t&obs).sum()),'neither_true':int((~q&~t&obs).sum())}}


def score(directory,variant):
    meta=json.loads((OUT/'input-seal.json').read_text())
    run=json.loads((directory/'receipt.json').read_text())
    forecasts=directory/'movie-generations.jsonl'
    active_protocol=PROTOCOL if variant=='full' else Path('research/reading_validation/transition-t5-adapter-protocol.md')
    if variant=='adapter' and (run.get('variant')!='T5-Large all-attention Q/V LoRA'
            or run.get('parent_protocol_sha256')!=meta['protocol_sha256']
            or sha(directory/'selected-adapter.pt')!=run.get('best_adapter_sha256')):
        raise ValueError('Unexpected specialist adapter or parent comparison')
    if (sha(PROTOCOL)!=meta['protocol_sha256'] or sha(active_protocol)!=run['protocol_sha256']
            or sha(forecasts)!=run['forecast_sha256'] or run['input_sha256']!=meta['input_sha256']
            or run['optimizer_updates']!=1840):
        raise ValueError('The specialist run does not match the predetermined movie source')
    predictions={r['id']:r['generation'] for r in map(json.loads,forecasts.read_text().splitlines())}
    material=[r for r in meta['sentences'] if r['split']=='test']
    if set(predictions)!={r['id'] for r in material} or len(material)!=run['external_sentences']:
        raise ValueError('T5 did not generate every held-out sentence')
    docs,folds=inputs();all_groups={m['doc']:m for m in folds['documents'] if m['split']=='test'}
    as_words={d:np.zeros(len(docs[d]['words']),dtype=bool) for d in all_groups}
    tally=Counter()
    for s in material:
        d=s['doc'];text=docs[d]['text']
        rows=docs[d]['words'][s['word_index_start']:s['word_index_stop']]
        matches,info=mark(predictions[s['id']],[text[a:b] for a,b,*_ in rows])
        as_words[d][s['word_index_start']:s['word_index_stop']]=matches
        tally.update(info)
    original=RESULTS/'heldout-probabilities.npz';tags=RESULTS/'entity-control-probabilities.npz'
    nerlock=json.loads((RESULTS/'entity-control-lock.json').read_text())
    if sha(original)!=json.loads((RESULTS/'prediction-lock.json').read_text())['forecast_sha256'] or sha(tags)!=nerlock['forecast_sha256']:
        raise ValueError('Qwen or strong NER external forecasts changed')
    qwen=np.load(original);ner=np.load(tags);ner_cut=nerlock['cutoff']
    qwen_cut=json.loads((RESULTS/'readouts.json').read_text())['readouts']['CharE']['antecedents']['threshold']
    y=[];first=[];g=[];raw=source();values={k:[] for k in ('qwen_word','ner_control','t5_span')}
    for d,m in all_groups.items():
        ok,v,_=outcomes(docs[d],m,raw[d],'CharE')
        y.extend(v[ok]>0);g.extend([m['group']]*int(ok.sum()));first.extend(ner[d][ok,2]>0)
        values['qwen_word'].extend(qwen['CharE__'+d][ok,3]>qwen_cut)
        values['ner_control'].extend(ner[d][ok,0]>ner_cut)
        values['t5_span'].extend(as_words[d][ok])
    y=np.array(y);g=np.array(g);values={k:np.array(arr) for k,arr in values.items()}
    out={'gpu_receipt_sha256':sha(directory/'receipt.json'),'generation_format':dict(tally),
         'all':compare(y,values,g,np.ones(len(y),dtype=bool)),
         'first_person_mention':compare(y,values,g,np.array(first))}
    destination=OUT/variant
    destination.mkdir(exist_ok=True)
    for filename in ('receipt.json','training.json','movie-generations.jsonl')+(('selected-adapter.pt',) if variant=='adapter' else ()):
        file=destination/filename
        file.write_bytes((directory/filename).read_bytes())
        if sha(file)!=sha(directory/filename):raise ValueError('Archived T5 comparator changed')
    out['variant']=variant;out['protocol_sha256']=sha(active_protocol)
    dump(destination/'scores.json',out)
    write={k:v['tp_fp_fn_tn'] for k,v in out['all']['methods'].items()}
    print(write,{k:v['f1'] for k,v in out['all']['methods'].items()})


if __name__=='__main__':
    command=ArgumentParser(description=__doc__);command.add_argument('--features',required=True,type=Path)
    command.add_argument('--variant',choices=('full','adapter'),required=True)
    args=command.parse_args();score(args.features,args.variant)
