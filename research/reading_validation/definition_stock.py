"""Infer a causal ledger of term mentions and the specific later words that explain them."""

from dataclasses import dataclass
import numpy as np
from scipy.special import expit

from research.reading_validation.definition_fit import all_spans
from research.reading_validation.definition_predict import feature_series


@dataclass
class Evidence:
    spans: list
    opened: np.ndarray
    winners: np.ndarray
    scores: np.ndarray
    mention_prob: np.ndarray


def evidence(page,proposal,relation,arm):
    design='semantic' if arm=='joint' else 'surface' if arm=='name' else arm
    left,right,logits=all_spans(page,proposal,design);opened=np.full(page.n,-1,dtype=int);accepted=[];scores=[]
    for t in range(page.n):
        ids=np.flatnonzero(right==t);best=ids[np.argmax(logits[ids])]
        if logits[best]>=proposal['threshold']:
            opened[t]=len(accepted);accepted.append((int(left[best]),t));scores.append(float(expit(logits[best])))
    best=np.zeros(page.n,dtype='float32');winners=np.full(page.n,-1,dtype=int)
    for i,((a,b),score) in enumerate(zip(accepted,scores,strict=True)):
        if b+1>=page.n:continue
        if arm=='name':
            marker=tuple(page.forms[a:b+1]);width=len(marker);p=np.zeros(page.n,dtype='float32')
            for t in range(b+width,page.n):
                if tuple(page.forms[t-width+1:t+1])==marker:p[t]=1.
            series=p
        else:
            _,p=feature_series(page,(a,b),relation,design);series=score*p
        improve=series>best;winners[improve]=i;best[improve]=series[improve]
    return Evidence(accepted,opened,winners,best,np.array(scores))


def events(e,threshold):
    closed=np.full(len(e.spans),-1,dtype=int);by_word=np.full(len(e.opened),-1,dtype=int)
    for t in np.flatnonzero((e.winners>=0)&(e.scores>=threshold)):
        i=e.winners[t]
        if closed[i]<0:closed[i]=t;by_word[t]=i
    stock=np.cumsum((e.opened>=0).astype(int)-(by_word>=0).astype(int))
    if np.any(stock<0) or any(t>=0 and t<=e.spans[i][1] for i,t in enumerate(closed)):
        raise ArithmeticError('A concept was closed before it was available')
    return closed,by_word,stock


def render(page,e,threshold,offsets):
    closed,closures,word_values=events(e,threshold);unresolved=[];tokens=[];words=[]
    endpoints={int(word[3]):i for i,word in enumerate(page.words)};level=0
    def bounds(i):
        a,b=e.spans[i];return [int(page.words[a,0]),int(page.words[b,1])]
    for i,(left,right) in enumerate(offsets):
        j=endpoints.get(i);opening=None;answer=None;winner=None;evidence_score=None
        if j is not None:
            opening=int(e.opened[j]) if e.opened[j]>=0 else None;answer=int(closures[j]) if closures[j]>=0 else None
            winner=int(e.winners[j]) if e.winners[j]>=0 else None;evidence_score=float(e.scores[j]);level=int(word_values[j])
            words.append({'word_index':j,'token_index':i,'chars':page.words[j,:2].tolist(),'opened_mention':opening,'closed_mention':answer,
                          'leading_earlier_mention':winner,'explanatory_evidence':evidence_score,'open_mentions':level})
        tokens.append({'token_index':i,'chars':[left,right],'open_mentions':level,'opened_mention':opening,'closed_mention':answer})
    for i,sp in enumerate(e.spans):
        start,end=bounds(i);unresolved.append({'mention_id':i,'chars':[start,end],'text':page.text[start:end],
            'opening_token':int(page.words[sp[1],3]),'closing_token':int(page.words[closed[i],3]) if closed[i]>=0 else None,
            'proposal_score':float(e.mention_prob[i])})
    return {'tokens':tokens,'words':words,'mentions':unresolved,'threshold':threshold,
            'unit':'count of automatically proposed terminology mentions awaiting explanatory evidence'}
