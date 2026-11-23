"""Read observed words against predicted vocabulary alternatives and visible earlier words."""

import re

import numpy as np
import torch
from torch.nn import functional as F

from research.reading_validation.identity_causal_features import capture
from research.backbone_transfer.extract import forward

LEXEME = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*|\d+(?:[.,]\d+)*")
FUNCTION = frozenset(
    "a an the of at in on to from by for with into about as and or so if then when while but than "
    "he she it they we you i me him her them us my our their his its your this these that those "
    "is are were was be being been do does did have has had will would should can could may might "
    "there here also very not who which what how where".split()
)
COLUMNS = ("log_characters", "log_word_index", "pieces", "earlier_occurrences", "word_surprisal",
           "first_surprisal", "middle_first_surprisal", "top_alternative_log_odds",
           "alternative_semantic_fit", "alternative_mass", "earlier_alternative_log_odds",
           "earlier_semantic_log_odds", "earlier_semantic_similarity", "earlier_attention",
           "earlier_exists", "eligible")


def coordinates(text, encoding):
    offsets = np.asarray(encoding.offsets)
    rows = []
    for i, match in enumerate(LEXEME.finditer(text)):
        a,b = match.span()
        hit = np.flatnonzero((offsets[:,0] < b) & (offsets[:,1] > a))
        if not len(hit):
            raise ValueError("Source lexical item has no visible model token")
        first, last = map(int, (hit[0],hit[-1]))
        extra_left = text[offsets[first,0]:a]
        extra_right = text[b:offsets[last,1]]
        clean = not extra_left.strip() and not extra_right.strip()
        rows.append((a,b,first,last,int(clean and first > 0)))
    return np.asarray(rows,dtype=np.int32).reshape(-1,5)


def earlier_candidates(text, ranges, ids):
    """Each position receives the most recent mention of every distinct earlier lexical type."""
    inventory = {}
    counts = {}
    banks, repeats = [], []
    for a,b,start,end,_ in ranges:
        word = text[a:b].casefold()
        repeats.append(counts.get(word,0))
        counts[word] = counts.get(word,0) + 1
        banks.append([v for form,v in inventory.items() if form != word and v[0] != ids[start]])
        if len(word) >= 3 and word not in FUNCTION and word.isalpha():
            inventory[word] = (ids[start],start,end)
    return banks, repeats


def values(runtime, seq, words, text, final, middle, att):
    dev = final.device
    tokens = torch.as_tensor(seq,device=dev)
    embedding = runtime.encoder.embed_tokens.weight
    ranges = torch.as_tensor(words.astype("int64"),device=dev)
    previous = (ranges[:,2]-1).clamp(min=0)
    target = tokens[ranges[:,2]]
    signals = torch.zeros(len(words),len(COLUMNS),device=dev)
    observed = torch.zeros(len(seq),device=dev)
    for lo in range(1,len(seq),32):
        stop=min(len(seq),lo+32)
        logits=F.linear(final[lo-1:stop-1],embedding)
        observed[lo:stop]=F.cross_entropy(logits,tokens[lo:stop],reduction="none")
    cumulative=F.pad(observed.cumsum(0),(1,0))
    first,last=ranges[:,2],ranges[:,3]
    banks,repeat=earlier_candidates(text,words,seq)
    signals[:,0]=torch.log1p((ranges[:,1]-ranges[:,0]).float())
    signals[:,1]=torch.log1p(torch.arange(len(words),device=dev).float())
    signals[:,2]=last-first+1
    signals[:,3]=torch.log1p(torch.as_tensor(repeat,device=dev).float())
    signals[:,4]=cumulative[last+1]-cumulative[first]
    signals[:,5]=observed[first]
    signals[:,-1]=ranges[:,-1]
    for lo in range(0,len(words),24):
        stop=min(lo+24,len(words))
        indices=previous[lo:stop]
        labels=target[lo:stop]
        z=F.linear(final[indices],embedding)
        middle_z=F.linear(runtime.encoder.norm(middle[indices]),embedding)
        signals[lo:stop,6]=F.cross_entropy(middle_z,labels,reduction="none")
        real=z.gather(1,labels[:,None])
        denom=torch.logsumexp(z,dim=1,keepdim=True)
        z.scatter_(1,labels[:,None],-torch.inf)
        alternate, ids=z.topk(128)
        lexical=F.normalize(embedding[labels].float(),dim=-1)
        vectors=F.normalize(embedding[ids].float(),dim=-1)
        cos=torch.einsum("nd,nkd->nk",lexical,vectors)
        signals[lo:stop,7]=(alternate[:,:1]-real).flatten()
        signals[lo:stop,8]=(alternate.softmax(dim=1)*cos).sum(1)
        signals[lo:stop,9]=(alternate-denom).exp().sum(1)
        for j,history in enumerate(banks[lo:stop]):
            if not history:
                continue
            previous_ids=torch.as_tensor([v[0] for v in history],device=dev)
            odds=z[j,previous_ids]-real[j]
            similarity=F.normalize(embedding[previous_ids].float(),dim=-1)@lexical[j]
            # A fixed von Mises–Fisher log kernel favors similar lexical embeddings.
            related=odds+8*(similarity-1)
            winner=int(related.argmax())
            _,begin,end=history[winner]
            signals[lo+j,10]=odds.max()
            signals[lo+j,11]=related[winner]
            signals[lo+j,12]=similarity[winner]
            signals[lo+j,13]=att[int(words[lo+j,3]),begin:end+1].sum()
            signals[lo+j,14]=1
    if not torch.isfinite(signals).all():
        raise ArithmeticError("Nonfinite lexical expectation")
    return signals.cpu().numpy()


@torch.inference_mode()
def native(runtime, texts):
    enc=[runtime.tokenizer.encode(text,add_special_tokens=False) for text in texts]
    seqs=[v.ids for v in enc]
    states=[]
    def keep(module, args, output):
        states.append(output)
    with runtime.encoder.layers[runtime.mid_index].register_forward_hook(keep),capture(runtime,seqs) as weights:
        final=forward(runtime.encoder,seqs,248044,torch)
        att,_=weights()
    if len(states)!=1:
        raise RuntimeError("Missing the unique intermediate pass")
    result=[]
    for i,(text,encoding) in enumerate(zip(texts,enc,strict=True)):
        words=coordinates(text,encoding)
        if not len(words) or not 0<len(encoding.ids)<=2048:
            raise ValueError("Current text exceeds verified inference geometry")
        n=len(encoding.ids)
        result.append((words,values(runtime,encoding.ids,words,text,final[i,:n],states[0][i,:n],att[i,:n,:n]),n))
    return result
