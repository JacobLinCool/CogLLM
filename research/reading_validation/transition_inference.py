"""English sentence coherence judgments from an existing, single causal CogLLM pass."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

CHECKPOINT = 'c1704d4436d767fa43bdb59e9216698514e1f6bd1d45cad4b60d0f018d84abb5'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
BUNDLE = Path(__file__).resolve().parents[1]/'results/reading-validation-transition-external/readouts.json'
BUNDLE_SHA = '7cfbc7b6d89e56995e8a909294c91db619eb41aaf30fa8dd3b1a2cac8e7bbf11'


@dataclass(frozen=True)
class LinearProbability:
    columns: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    weight: np.ndarray
    bias: float
    threshold: float

    def predict(self, features):
        values = (features[self.columns] - self.mean) / self.scale
        return float(np.exp(-np.logaddexp(0., -(values @ self.weight + self.bias))))


@dataclass(frozen=True)
class TransitionInference:
    primary: LinearProbability
    student: LinearProbability
    surprisal_control: LinearProbability

    @classmethod
    def load(cls, path=BUNDLE):
        payload=Path(path).read_bytes()
        if sha256(payload).hexdigest()!=BUNDLE_SHA:
            raise ValueError('Unrecognized English coherence readout weights')
        table=json.loads(payload)
        def read(arm):
            fit=table['models'][arm]
            return LinearProbability(np.array(table['columns'][arm],dtype=int),np.array(fit['mean']),
                                     np.array(fit['std']),np.array(fit['weights']),fit['intercept'],
                                     table['thresholds'][arm])
        return cls(read('integration'),read('student'),read('lexical_surprisal'))

    def score_feature(self, feature_row):
        features=np.r_[np.log1p(feature_row['position']),feature_row['features'],feature_row['representation']]
        if features.shape!=(298,) or not np.isfinite(features).all():
            raise ValueError('Coherence input dimensions or finiteness differ from the study')
        risk=self.primary.predict(features)
        return {'risk':risk, 'flagged':risk>self.primary.threshold,'threshold':self.primary.threshold,
                'surprisal_control_risk':self.surprisal_control.predict(features),
                'distilled_risk':self.student.predict(features)}

    def score(self, text, features, tokenizer, reader, runtime, spans, *, token_trace, checkpoint_sha256, token_surprisal=None):
        """Evaluate each original transition at the completed sentence's final visible token.

        The first span represents the opening sentence. Subsequent positions carry one-based indices.
        A sentence whose tokenization uses later text receives a null score. Risk is calibrated to
        incremental coherence flags on short English stories. It is not a personal load rating.
        """
        if (runtime.candidate!='qwen' or runtime.revision!=REVISION or checkpoint_sha256!=CHECKPOINT
                or features.relations is None or not runtime.encoder.config.tie_word_embeddings):
            raise ValueError('This readout requires the selected native-attention Qwen3.5-2B checkpoint')
        encoding=tokenizer.encode(text,add_special_tokens=False)
        if tuple(encoding.ids)!=tuple(features.token_ids) or tuple(map(tuple,encoding.offsets))!=tuple(features.offsets):
            raise ValueError('Coherence readout received features for another article')
        offsets=tuple(map(tuple,features.offsets))
        if [(r['start'],r['end']) for r in token_trace]!=list(offsets):
            raise ValueError('Coherence token scores differ from the article source offsets')
        output=[]
        certified=[]
        previous_end=0
        for index,(start,end) in enumerate(spans,1):
            if not 0<=previous_end<=start<end<=len(text) or text[start:end]!=text[start:end].strip():
                raise ValueError('Expected ordered, nonoverlapping, trimmed sentence spans in the source article')
            previous_end=end
            if index==1:
                continue
            result={'start':start,'end':end,'sentence_index':index,'risk':None,'flagged':None,
                    'threshold':self.primary.threshold,'surprisal_control_risk':None,'distilled_risk':None}
            output.append(result)
            prefix=tokenizer.encode(text[:end],add_special_tokens=False).ids
            if (not 0<len(prefix)<=len(offsets) or offsets[len(prefix)-1][1]!=end
                    or tuple(prefix)!=tuple(features.token_ids[:len(prefix)])):
                continue
            # Sentence splits inside an orthographic word cannot match the study's word pooling.
            if start>0 and not text[start-1].isspace():
                continue
            certified.append({'id':index,'index':index,'start':start,'end':end,'archive_final':False})
        if certified:
            import torch
            from research.reading_validation.transition_external_forecast import pooled_boundary_features
            with torch.inference_mode():
                final,middle=reader.project(features)
                design,_=pooled_boundary_features(runtime,features,middle,final,token_trace,{'text':text,'stops':certified},token_surprisal)
            for row in design:
                output[row['id']-2].update(self.score_feature(row))
        return output
