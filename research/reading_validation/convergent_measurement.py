"""Recover aggregate human choices from the authors' reported Laplace-smoothed vocabulary."""
from collections import Counter
from csv import DictReader
from pathlib import Path
import ast
import re
import warnings
import numpy as np
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.convergent_word_prepare import ROOT,OUT

RATING_LIST_SIZE=[60,59,56,58,57,57,60,59]


class MissingCandidate(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id!='nan':raise ValueError('Unexpected symbol in archived statistical dictionary')
        return ast.copy_location(ast.Constant('__missing_option__'),node)


def measurement():
    release=ast.literal_eval(MissingCandidate().visit(ast.parse((ROOT/'distribution.txt').read_text(),mode='eval')))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',SyntaxWarning);source=ast.parse((ROOT/'preprocess.py').read_text())
    fix=next(ast.literal_eval(x.value) for x in source.body if isinstance(x,ast.Assign) and any(isinstance(k,ast.Name) and k.id=='punct_words' for k in x.targets))
    rows=list(DictReader((ROOT/'outcomes.csv').open()));pool=Counter(r['item'] for r in rows);obs={};counts=Counter()
    for r in rows:
        counts['eligible']+=1;v=release.get(r['item']);item=int(r['item_id']);q=r['word'];q=fix.get(re.sub(r'[.,?!:;]','',q),re.sub(r'[.,?!:;]','',q));s=float(r['rating_sd']);n_r=RATING_LIST_SIZE[int(r['list'])-1];se2=s*s/n_r
        if not np.isfinite(se2) or se2<0:raise ValueError('Rating variance absent')
        a=n=None
        if v and pool[r['item']]==1 and 'unk' in v:
            denominator=round(1/v['unk']);n_=denominator-len(v);K=len(v);raw={word:denominator*pv-1 for word,pv in v.items() if word!='unk'}
            exact=all(abs(c-round(c))<1e-5 and c>=-.00001 for c in raw.values()) and abs(sum(raw.values())-n_)<1e-5
            p=v.get(q,v['unk']);a_=int(round(denominator*p-1));matches=abs(p-float(r['cloze_p_smoothed']))<1e-7
            if exact and matches and 1<=n_<=80 and 0<=a_<=n_:n=n_;a=a_;counts['certified_unique_cloze']+=1
            elif not exact:counts['nonrecoverable_count']+=1
            elif not matches:counts['disagrees_with_original_item']+=1
            else:counts['extra_or_pooled_trials']+=1
        elif pool[r['item']]>1:counts['shared_prefix_excluded']+=1
        else:counts['no_matching_distribution']+=1
        obs[item]={'trials':n,'hits':a,'rating_sampling_variance':se2,'rating_n_upper_bound':n_r}
    return obs,counts


if __name__=='__main__':
    obs,c=measurement();d=[v for v in obs.values() if v['trials']];print(c,'participants per cloze item',Counter(x['trials'] for x in d));write_json(OUT/'measurement-geometry.json',dict(sources={f:digest(ROOT/f) for f in ('distribution.txt','outcomes.csv','preprocess.py','paper.xml')},coverage=dict(c),trial_counts={str(k):v for k,v in Counter(x['trials'] for x in d).items()},pooled_prefixes='excluded from binomial supervision, retained on other endpoints',rating_sampling_variance='Released rating SD squared divided by enrolled minus published list-specific exclusions; item-level nonresponse is unobserved in these aggregates.'))
