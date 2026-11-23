"""Estimate the repeatable part of reader-pair disagreement using four distinct voters."""
from collections import Counter, defaultdict
from itertools import permutations
import json

import numpy as np

from research.reading_validation.transition_direct_data import ROOT, display, gold_docs
from research.reading_validation.transition_direct_game import expert_entities
from research.reading_validation.transition_external_score import grouped_distribution
from research.reading_validation.transition_game_prepare import DIRECT, OUT
from research.reading_validation.transition_score import dump, sha, with_interval


def factorial(n, k):
    ans=1
    for j in range(k):ans*=n-j
    return ans


def moments(choices):
    counts=Counter(choices)
    n=sum(counts.values())
    if n<4:raise ValueError('Four distinct people are required to form two disjoint pairs')
    pairs=[factorial(v,2) for v in counts.values()]
    agre=sum(pairs)/factorial(n,2)
    both_agree=(sum(factorial(v,4) for v in counts.values())+sum(pairs)**2-sum(a*a for a in pairs))/factorial(n,4)
    return 1-agre,1-2*agre+both_agree


def verify_formula():
    for labels in ('aaaa','abab','aabc','abcde','aabbbc'):
        draws=list(permutations(labels,4))
        agree=np.mean([a!=b for a,b,c,d in draws])
        both=np.mean([(a!=b) and (c!=d) for a,b,c,d in draws])
        np.testing.assert_allclose(moments(labels),[agree,both],atol=1e-15)


def recover():
    packet=ROOT/'snac-direct-stimuli.jsonl'
    books={r['id']:r for r in map(json.loads,packet.read_text().splitlines()) if r['split']=='independent'}
    bound=DIRECT/'external-reader-outcomes.json'
    gold={(r['doc'],r['word']):r for r in json.loads(bound.read_text())['rows'] if r['antecedent_disagreement'] is not None}
    eligible=[]
    for d,genre,tree,crac,wels,tokens in gold_docs():
        if d not in books:continue
        _,coordinates,_=display(tree,wels,tokens)
        tok={(r[0],r[1]):i for i,r in enumerate(books[d]['words'])}
        entities,allspans=expert_entities(crac,wels,tree)
        for label in tree.iter('PDante'):
            words=allspans[label.get('id')]
            if len(words)!=1:continue
            coord=tok.get(coordinates[words[0]]); previous=gold.get((d,coord))
            if previous is None:continue
            people=defaultdict(set)
            for a in label.iter('anchor'):
                if a.get('mode')=='a' and a.get('type')=='DO':
                    people[a.get('user_id')].add(entities[a.get('ante')])
            if any(len(x)!=1 for x in people.values()):raise ValueError('Ambiguous repeated voter')
            choices=[next(iter(x)) for x in people.values()];q,joint=moments(choices)
            if len(people)!=previous['prior_entity_players'] or abs(q-previous['antecedent_disagreement'])>1e-15:
                raise ValueError('Disjoint-pair eligibility differs from primary outcome')
            eligible.append((d,coord,genre,len(choices),q,joint))
    if {(r[0],r[1]) for r in eligible}!=set(gold) or len(eligible)!=len(gold):
        raise ValueError('Not every independent primary pronoun reproduced')
    return eligible,sha(bound)


def evaluate(rows,prior):
    cols=np.array([[r[4],r[5]] for r in rows])
    groups,idx,draws=grouped_distribution([r[0] for r in rows])
    def summarize(mu,second):
        best=mu-second
        baseline=mu*(1-prior)**2+(1-mu)*prior**2
        return {'pair_disagreement':mu,'replicate_pair_joint_disagreement':second,
                'expected_item_known_brier':best,'expected_dev_constant_brier':baseline,
                'potential_brier_reduction_from_source_prior':baseline-best,
                'between_item_variance':second-mu**2,
                'distinct_pair_correlation':(second-mu**2)/(mu*(1-mu))}
    est=summarize(*cols.mean(axis=0));cnt=draws@np.bincount(idx)
    gathered=(draws@np.stack([np.bincount(idx,weights=c) for c in cols.T],axis=1))/cnt[:,None]
    with np.errstate(invalid='ignore',divide='ignore'):
        samples=summarize(*gathered.T)
    return {'documents':len(groups),'mentions':len(rows),'unique_voter_mention_assignments':sum(r[3] for r in rows),
            **{k:with_interval(x,samples[k]) for k,x in est.items()}}


def main():
    verify_formula();rows,digest=recover()
    training=list(map(json.loads,(ROOT/'game-native-stimuli.jsonl').read_text().splitlines()))
    dev=np.array([y[-1] for r in training if r['fold']=='dev' for y in r['labels'] if y[-1]>=0])
    prior=float(dev.mean())
    result={'external_human_sha256':digest,'source_development_prior':prior,'disjoint_four_voter_moments':evaluate(rows,prior)}
    dump(OUT/'reader-replication.json',result)
    print(json.dumps(result['disjoint_four_voter_moments']))


if __name__=='__main__':main()
