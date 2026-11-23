"""Map released human selections to unambiguous source words with their original offsets."""
from collections import Counter
import re

import numpy as np


def outcomes(doc, material, original, target):
    words=np.asarray(doc['words'],dtype=int);strength=np.zeros(len(words),dtype=np.int16);valid=np.ones(len(words),dtype=bool)
    log=Counter();positive_segments=0
    for segment in material['segments']:
        i=segment['index'];start=segment['start'];end=segment['end']
        sentence=doc['text'][start:end];block=original[str(i)]
        if sentence!=' '.join(block['text'].split()):raise ValueError('Token bounds no longer locate this source segment')
        local=words[:,4]==i
        if not local.any():raise ValueError('Annotated segment contains no certified words')
        errors=[e for e in block['errors'] if e['error_type']==target]
        log['source_segments']+=1;log['source_annotations']+=len(errors)
        ambiguous=False
        for error in errors:
            span=' '.join(error['span'].split());votes=error['votes']
            if type(votes) is not int or votes<1:raise ValueError('Invalid human vote count')
            matches=list(re.finditer(re.escape(span),sentence)) if span else []
            if len(matches)!=1:
                log['unmatched_spans' if not matches else 'multiply_matched_spans']+=1;ambiguous=True;continue
            a,b=matches[0].span();hit=local&(words[:,0]<start+b)&(words[:,1]>start+a)
            if not hit.any():
                log['spans_without_word']+=1;ambiguous=True;continue
            strength[hit]=np.maximum(strength[hit],votes)
        if ambiguous:
            valid[local]=False;log['excluded_segments']+=1
        else:
            log['eligible_segments']+=1;positive_segments+=bool(errors)
    log['eligible_words']=int(valid.sum());log['confirmed_positive_words']=int((valid&(strength>0)).sum())
    log['eligible_segments_with_target']=positive_segments
    return valid,strength,log
