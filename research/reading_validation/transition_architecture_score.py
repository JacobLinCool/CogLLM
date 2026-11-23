"""Freeze source-only cross-backbone estimators and score common held-out movie words."""
from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_external_fit import binomial, predict
from research.reading_validation.transition_external_score import evaluate_one, grouped_distribution
from research.reading_validation.transition_score import dump, sha, with_interval
from research.reading_validation.transition_snac_fit import inputs
from research.reading_validation.transition_snac_ner import POS, TYPES
from research.reading_validation.transition_snac_outcomes import outcomes
from research.reading_validation.transition_snac_prepare import RESULTS as HISTORICAL, ROOT, source

CANDIDATES = ('qwen','lfm3','gemma4')
RESULT = Path('research/results/reading-validation-transition-architecture')
PROTOCOL = Path('research/reading_validation/transition-architecture-protocol.md')


def common():
    meta = json.loads((RESULT/'common-support.json').read_text())
    packets = ROOT/'cross-snac-stimuli.jsonl'
    if sha(packets) != meta['packet_sha256'] or sha(PROTOCOL) != meta['protocol_sha256']:
        raise ValueError('Common support or prospective analysis changed')
    rows = {r['id']:r for r in map(json.loads,packets.read_text().splitlines())}
    original, fold = inputs()
    return meta, rows, original, fold['documents']


def native_arms(matrix, ner, local_width, candidate):
    projected = local_width // 2
    if projected not in (48,64) or matrix.shape[1] != 17+2*projected+15:
        raise ValueError('Unexpected feature dimensionality')
    matched = np.column_stack((matrix[:,:17+48],matrix[:,17+projected:17+projected+48]))
    arms = {'ner': np.column_stack((matrix[:,:17],ner)), 'local':matched,
            'relations':np.column_stack((matched,matrix[:,-15:]))}
    if candidate == 'gemma4':
        arms['relations_full_width'] = matrix
    return arms


def fit(candidate, directory):
    manifest, prepared, original, groups = common()
    meta = json.loads((directory/'receipt.json').read_text())
    features = directory/'word-features.npz'
    if (meta['candidate'] != candidate or sha(features) != meta['feature_sha256']
            or meta['input_sha256'] != manifest['packet_sha256']
            or meta['protocol_sha256'] != manifest['protocol_sha256']
            or meta['forwards'] != meta['documents'] or meta['documents'] != len(prepared)):
        raise ValueError('Unexpected architecture L4 receipt')
    raw = source()
    tags = np.load(ROOT/'snac-ner.npz')
    cache = np.load(features)
    if set(cache.files) != set(prepared):
        raise ValueError('Missing L4 architecture narratives')
    vectors = {}
    labels = {}
    material = {s['doc']:s for s in groups if s['doc'] in prepared}
    for doc,row in prepared.items():
        idx = np.asarray(row['indices'])
        vectors[doc] = native_arms(cache[doc],tags[doc][idx],meta['local_width'],candidate)
        if material[doc]['split'] != 'test':
            valid, votes, _ = outcomes(original[doc],material[doc],raw[doc],'CharE')
            labels[doc] = (valid[idx],votes[idx]>0)
    split = {k:[d for d,m in material.items() if m['split']==k] for k in ('train','dev','test')}
    estimators = {}
    predictions = {d:{} for d in split['test']}
    for arm in next(iter(vectors.values())):
        def source_matrices(fold):
            X = np.concatenate([vectors[d][arm][labels[d][0]] for d in split[fold]])
            y = np.concatenate([labels[d][1][labels[d][0]] for d in split[fold]]).astype(int)
            return X.astype(float),y
        X,Y = source_matrices('train')
        weights = binomial(X,Y,np.ones(len(Y),dtype=int))
        X,Y = source_matrices('dev')
        cutoff = float(np.quantile(predict(X,weights)[Y==0],.95,method='higher'))
        estimators[arm] = {'weights':weights,'cutoff':cutoff}
        for d,p in predictions.items():
            p[arm] = predict(vectors[d][arm],weights)
    file = RESULT/(candidate+'-movie-forecasts.npz')
    np.savez_compressed(file,**{d:np.column_stack([vals[a] for a in estimators]) for d,vals in predictions.items()})
    dump(RESULT/(candidate+'-lock.json'),{'candidate':candidate,'protocol_sha256':sha(PROTOCOL),
         'feature_receipt':meta,'forecast_sha256':sha(file),'named_entity_sha256':sha(ROOT/'snac-ner.npz'),
         'arm_order':list(estimators),'estimators':estimators})


def score():
    _, prepared, original, groups = common()
    raw = source()
    tags = np.load(ROOT/'snac-ner.npz')
    data = {}; specs = {}; controls = {}; labels = []; grouper = [];names=[];first=[]
    for c in CANDIDATES:
        specs[c] = json.loads((RESULT/(c+'-lock.json')).read_text())
        f = RESULT/(c+'-movie-forecasts.npz')
        if sha(f) != specs[c]['forecast_sha256'] or specs[c]['protocol_sha256'] != sha(PROTOCOL):
            raise ValueError('An estimator or movie forecast changed')
        data[c] = np.load(f)
        for arm in specs[c]['arm_order']:
            controls[c+'/'+arm] = []
    for m in groups:
        d = m['doc']
        if m['split'] != 'test' or d not in prepared:
            continue
        idx = np.asarray(prepared[d]['indices'])
        valid,votes,_ = outcomes(original[d],m,raw[d],'CharE')
        use = valid[idx]
        labels.extend(votes[idx][use]>0)
        grouper.extend([m['group']]*int(use.sum()))
        names.extend(tags[d][idx][use,len(POS)+TYPES.index('PERSON')]>0)
        first.extend(tags[d][idx][use,-4]>0)
        for c,table in data.items():
            for k,a in enumerate(specs[c]['arm_order']):
                controls[c+'/'+a].extend(table[d][use,k])
    historical_path = HISTORICAL/'heldout-probabilities.npz'
    historical_lock = json.loads((HISTORICAL/'prediction-lock.json').read_text())
    if sha(historical_path) != historical_lock['forecast_sha256'] or sha(HISTORICAL/'readouts.json') != historical_lock['model_sha256']:
        raise ValueError('Historic Qwen prediction or source threshold changed')
    historic = np.load(historical_path)
    prob = []
    for m in groups:
        d = m['doc']
        if m['split'] == 'test' and d in prepared:
            ids = np.asarray(prepared[d]['indices'])
            ok,_,_ = outcomes(original[d],m,raw[d],'CharE')
            prob.extend(historic['CharE__'+d][ids,3][ok[ids]])
    controls['production/relations'] = prob
    original_cut = json.loads((HISTORICAL/'readouts.json').read_text())['readouts']['CharE']['antecedents']['threshold']
    specs['production'] = {'estimators':{'relations':{'cutoff':original_cut}}}
    y,groups = np.asarray(labels).astype(int),np.asarray(grouper)
    out = {}
    for region,mask in [('all',np.ones(len(y),dtype=bool)),('recognized_person',np.asarray(names)),
                        ('first_person_mention',np.asarray(first))]:
        yy = y[mask];gg,idx,draws = grouped_distribution(groups[mask]);n=np.ones_like(yy)
        methods = {}; ap = {}
        for arm,full in controls.items():
            c,a=arm.split('/')
            prob = np.asarray(full)[mask]
            methods[arm],boot = evaluate_one(yy,n,prob,specs[c]['estimators'][a]['cutoff'],idx,draws)
            err = (yy-prob)**2
            rows = np.bincount(idx,weights=err,minlength=len(gg))
            denom = np.bincount(idx,minlength=len(gg))
            methods[arm]['brier'] = with_interval(err.mean(),(draws@rows)/(draws@denom))
            ap[arm] = boot['ap']
        def difference(a,b):
            point = methods[a]['ap']['estimate']-methods[b]['ap']['estimate']
            return with_interval(point,ap[a]-ap[b])
        contrast = {}
        for candidate in CANDIDATES:
            for reference in (candidate+'/ner',candidate+'/local','qwen/ner'):
                if reference != candidate+'/relations':
                    contrast[candidate+'/relations - '+reference] = difference(candidate+'/relations',reference)
        for a,b in [('lfm3','qwen'),('gemma4','qwen'),('gemma4','lfm3')]:
            contrast[a+'/relations - '+b+'/relations'] = difference(a+'/relations',b+'/relations')
        contrast['production/relations - qwen/relations'] = difference('production/relations','qwen/relations')
        contrast['gemma4/relations_full_width - gemma4/relations'] = difference('gemma4/relations_full_width','gemma4/relations')
        out[region] = {'movie_groups':len(gg),'words':len(yy),'positive_words':int(yy.sum()),
                       'methods':methods,'paired_ap_differences':contrast}
        print(region,{a:round(v['ap']['estimate'],3) for a,v in methods.items()},flush=True)
    dump(RESULT/'scores.json',out)


if __name__ == '__main__':
    cmd = ArgumentParser(description=__doc__)
    cmd.add_argument('mode',choices=['fit','score'])
    cmd.add_argument('--candidate',choices=CANDIDATES)
    cmd.add_argument('--features',type=Path)
    args = cmd.parse_args()
    if args.mode == 'fit':
        fit(args.candidate,args.features)
    else:
        score()
