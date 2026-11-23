"""Locate candidate unclear character references in English narrative summaries."""
from bisect import bisect_right
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from unicodedata import category

import numpy as np

from research.reading_validation.transition_inference import CHECKPOINT, REVISION

BUNDLE=Path(__file__).resolve().parents[1]/'results/reading-validation-transition-snac/serving/character-word.json'
BUNDLE_SHA='a8351be38b5e5a5b3ac7e2e0752d086db5d97e646ed4ebeec6f3d0b466e4600e'


def word_spans(text,start,end):
    """Keep Unicode letters and their original combining marks at the same code-point offsets."""
    i=start
    while i<end:
        a=i
        if text[i].isalpha():
            i+=1
            while i<end:
                if text[i].isalpha() or category(text[i]).startswith('M'):
                    i+=1
                elif text[i] in "'’-" and i+1<end and text[i+1].isalpha():
                    i+=2
                else:
                    break
        elif text[i].isdecimal():
            i+=1
            while i<end:
                if text[i].isdecimal():
                    i+=1
                elif text[i] in '.,' and i+1<end and text[i+1].isdecimal():
                    i+=2
                else:
                    break
        else:
            i+=1
            continue
        yield a,i


def certified_words(text,tokenizer,token_ids,offsets,segments):
    """Return word endpoints only when future characters cannot affect their representation."""
    ends=[b for _,b in offsets]
    result=[]
    previous_end=0
    for segment_index,(start,end) in enumerate(segments):
        if not 0<=previous_end<=start<end<=len(text):
            raise ValueError('Narrative segment coordinates must be ordered and refer to original text')
        previous_end=end
        for position,(a,b) in enumerate(word_spans(text,start,end)):
            prefix=tokenizer.encode(text[:b],add_special_tokens=False).ids
            left=bisect_right(ends,a);right=len(prefix)-1
            if (not prefix or right>=len(offsets) or left>right or offsets[left][0]>a
                    or tuple(prefix)!=token_ids[:right+1]):
                continue
            token_end=offsets[right][1]
            # Tokenizer NFC normalization can map the terminal character plus a combining
            # mark to a token whose recorded end stops just before the source's mark.
            if (token_end>b or any(not category(c).startswith('M') for c in text[token_end:b])
                    or (right+1<len(offsets) and offsets[right+1][0]<b)):
                continue
            result.append([a,b,left,right,segment_index,position])
    return result


@dataclass(frozen=True)
class NarrativeInference:
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float
    threshold: float
    inclusive_threshold: bool = False

    def flags(self, probabilities):
        scores = np.asarray(probabilities)
        return scores >= self.threshold if self.inclusive_threshold else scores > self.threshold

    @classmethod
    def load(cls,path=BUNDLE):
        return cls._checked_bundle(path,BUNDLE_SHA,'cogllm-narrative-character-en-v1')

    @classmethod
    def _checked_bundle(cls,path,checksum,schema):
        raw=Path(path).read_bytes()
        if sha256(raw).hexdigest()!=checksum:
            raise ValueError('Unexpected narrative character checkpoint')
        bundle=json.loads(raw)
        if (bundle['schema']!=schema or bundle['revision']!=REVISION
                or bundle['reader_checkpoint_sha256']!=CHECKPOINT):
            raise ValueError('Narrative bundle refers to another backbone or reader')
        fit=bundle['weights'];fitted=cls(np.array(fit['mean']),np.array(fit['std']),np.array(fit['weights']),
                                         float(fit['intercept']),float(bundle['threshold']))
        if (any(x.shape!=(128,) or not np.isfinite(x).all() for x in (fitted.mean,fitted.scale,fitted.weight))
                or not (fitted.scale>0).all() or not np.isfinite(fitted.bias) or not 0<fitted.threshold<1):
            raise ValueError('Invalid narrative probability head')
        return fitted

    def probabilities(self,matrix):
        X=np.asarray(matrix)
        if X.ndim!=2 or X.shape[1]!=128 or not np.isfinite(X).all():
            raise ValueError('Narrative features must supply 128 finite values at each word')
        logits=(X-self.mean)/self.scale @ self.weight+self.bias
        return np.exp(-np.logaddexp(0.,-logits))

    def score(self,text,features,tokenizer,reader,runtime,sentence_spans,*,checkpoint_sha256,token_surprisal=None):
        """Group successive sentences by threes, as in the held-out movie-summary study."""
        spans=list(sentence_spans)
        if any(not 0<=a<b<=len(text) or (i and spans[i-1][1]>a) for i,(a,b) in enumerate(spans)):
            raise ValueError('Expected ordered nonoverlapping sentence spans')
        segments=[(spans[i][0],spans[min(i+2,len(spans)-1)][1]) for i in range(0,len(spans),3)]
        return self.score_segments(text,features,tokenizer,reader,runtime,segments,
                                   checkpoint_sha256=checkpoint_sha256,token_surprisal=token_surprisal)

    def score_segments(self,text,features,tokenizer,reader,runtime,segments,*,checkpoint_sha256,token_surprisal=None):
        """Score all certified words in source-defined reading segments with no encoder call.

        A flag indicates an above-threshold English narrative reference for author review.
        An input with no eligible word or no above-threshold score yields no flags.
        """
        if (runtime.candidate!='qwen' or runtime.revision!=REVISION or checkpoint_sha256!=CHECKPOINT
                or features.relations is None or not runtime.encoder.config.tie_word_embeddings):
            raise ValueError('Narrative reading requires the selected native-attention Qwen3.5-2B checkpoint')
        enc=tokenizer.encode(text,add_special_tokens=False)
        ids=tuple(features.token_ids);offsets=tuple(features.offsets)
        if tuple(enc.ids)!=ids or tuple(map(tuple,enc.offsets))!=offsets:
            raise ValueError('Narrative features and article do not agree')
        words=certified_words(text,tokenizer,ids,offsets,segments)
        if not words:
            return []
        from research.reading_validation.transition_snac_forecast import design_from_features
        import torch
        with torch.inference_mode():
            matrix=design_from_features(runtime,reader,{'text':text,'words':words},features,token_surprisal)
        probabilities = self.probabilities(matrix)
        return [{'start':w[0],'end':w[1],'risk':float(p),'flagged':bool(flag)}
                for w,p,flag in zip(words,probabilities,self.flags(probabilities),strict=True)]


MARGINAL_BUNDLE=Path(__file__).resolve().parents[1]/'results/reading-validation-transition-partial/serving/character-word.json'
MARGINAL_SHA='571895011d9c618e13f2bff0a5f172e33fa5a1e7f825bd79f1ffebb18b99d22c'
DECISIONS_BUNDLE = MARGINAL_BUNDLE.with_name('character-operating-points.json')
DECISIONS_SHA = 'be8df42a4db2326118704f1f50811c9636b2bc729f72fbb4d4902b2d30a0b062'


class MarginalNarrativeInference(NarrativeInference):
    """Locate insufficiently introduced narrative characters with ambiguity-aware book supervision."""

    @classmethod
    def load(cls,path=MARGINAL_BUNDLE):
        return cls._checked_bundle(path,MARGINAL_SHA,'cogllm-narrative-character-en-marginal-v1')

    @classmethod
    def for_writers(cls, operating_point='balanced'):
        """Load the source-selected word screen for English narrative character introductions."""
        if operating_point not in ('balanced', 'high_recall'):
            raise ValueError("Expected 'balanced' or 'high_recall'")
        raw = DECISIONS_BUNDLE.read_bytes()
        if sha256(raw).hexdigest() != DECISIONS_SHA:
            raise ValueError('Unexpected narrative word decision policy')
        decisions = json.loads(raw)
        if (decisions['schema'] != 'cogllm-narrative-character-en-decisions-v1'
                or decisions['weights_sha256'] != MARGINAL_SHA):
            raise ValueError('Narrative policy and probability model differ')
        rule = decisions['settings'][operating_point]
        return replace(cls.load(), threshold=rule['cutoff'], inclusive_threshold=rule['inclusive'])
