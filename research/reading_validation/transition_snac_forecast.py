"""Word-local antecedent evidence from one original-summary Qwen forward per document."""
from hashlib import sha256
import json
import math
from pathlib import Path
from unicodedata import normalize

import numpy as np
import torch
from torch.nn import functional as F

from research.reading_validation.bridge_forecast import STOP

SURFACE=('log_word_length','capitalized','all_caps','has_apostrophe','numeric','document_first','segment_first',
         'log_segment_number','log_word_in_segment','log_word_ordinal','log_repetitions','first_mention',
         'log_previous_mention_distance','previous_capitalized','earlier_capitalized_rate','starts_with_article')
RELATIONS=tuple(f'{source}_{feature}' for source in ('same_form','other_capitalized','other_content')
                for feature in ('exists','max_middle_cos','max_final_cos','attention_mass','max_attention'))


def design(runtime,reader,row):
    features=runtime.encode(row['text'])
    return design_from_features(runtime,reader,row,features)


def design_from_features(runtime,reader,row,f,token_surprisal=None):
    from research.reading_validation.transition_external_forecast import token_surprisals
    text=row['text']; words=np.asarray(row['words'],dtype=int)
    if words.ndim!=2 or words.shape[1]!=6 or not len(words):raise ValueError('Invalid certified word coordinates')
    starts,ends,t0,t1,segments,j=words.T
    if np.any(t1<t0) or np.any(t1[1:]<=t1[:-1]) or np.any(ends[1:]<=ends[:-1]):raise ValueError('Word ends do not advance')
    losses=token_surprisals(runtime,f) if token_surprisal is None else token_surprisal
    if np.shape(losses)!=(len(f.token_ids),) or not np.isfinite(losses).all():
        raise ValueError('A cached token surprisal vector must match the original words')
    loss_prefix=np.r_[0.,np.cumsum(losses)];surp=loss_prefix[t1+1]-loss_prefix[t0]
    idx=torch.as_tensor(t1,device=f.hidden.device)
    final,middle=reader.project(f)
    local=torch.cat([middle[idx],final[idx]],dim=1).cpu().numpy()
    a=F.normalize(f.mid[idx].float(),dim=1);b=F.normalize(f.hidden[idx].float(),dim=1)
    middle_cos=(a@a.T).cpu().numpy(); final_cos=(b@b.T).cpu().numpy()
    scale=(idx+1).float()[:,None]
    attention=(f.relations[idx,:,0]+math.log(2.)).expm1()/scale
    cumsum=F.pad(attention.cumsum(dim=1),(1,0));
    att_words=(cumsum[:,torch.as_tensor(t1+1,device=idx.device)]-cumsum[:,torch.as_tensor(t0,device=idx.device)]).cpu().numpy()
    names=[text[a:b] for a,b in zip(starts,ends,strict=True)];norm=np.array([normalize('NFC',w).casefold() for w in names]);cap=np.array([w[0].isupper() for w in names]);content=np.array([w not in STOP and any(c.isalpha() for c in w) for w in norm]);
    surface=np.zeros((len(words),len(SURFACE)));relationships=np.zeros((len(words),len(RELATIONS)))
    history={};order=np.arange(len(words));prior_caps=np.r_[0,np.cumsum(cap)]
    for i,(w,key) in enumerate(zip(names,norm,strict=True)):
        previous=history.get(key,[]);earlier=order<i;same=(norm==key)&earlier
        prev=norm[i-1] if i else ''
        surface[i]=[math.log1p(len(w)),cap[i],w.isupper(),"'" in w or '’' in w,w[0].isdigit(),i==0,j[i]==0,
                    math.log1p(segments[i]+1),math.log1p(j[i]+1),math.log1p(i+1),math.log1p(len(previous)),
                    not previous,math.log1p(i-previous[-1]) if previous else 0,cap[i-1] if i else False,
                    prior_caps[i]/max(i,1),prev in ('a','an','the')]
        for block,mask in enumerate((same,earlier&~same&cap,earlier&~same&content)):
            if mask.any():
                mass=att_words[i,mask]
                relationships[i,5*block:5*(block+1)]=[1,middle_cos[i,mask].max(),final_cos[i,mask].max(),mass.sum(),mass.max()]
        history.setdefault(key,[]).append(i)
    matrix=np.column_stack([surface,surp,local,relationships]).astype('float32')
    if not np.isfinite(matrix).all():raise ValueError('Nonfinite causal word forecast')
    return matrix


@torch.inference_mode()
def run(runtime,reader,checkpoint,input_path,output_dir):
    if runtime.candidate!='qwen' or not runtime.encoder.config.tie_word_embeddings:raise ValueError('Requires the selected Qwen')
    packet=Path(input_path).read_bytes();data=[json.loads(s) for s in packet.splitlines()];output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    all_rows={};count=[0]
    def hook(_m,_x):count[0]+=1
    with runtime.encoder.register_forward_pre_hook(hook):
        for i,row in enumerate(data):
            all_rows[row['id']]=design(runtime,reader,row)
            if count[0]!=i+1:raise ValueError('One and only one encoder pass per entire narrative')
            if i%25==24:print(f'SNaC {i+1}/{len(data)}',flush=True)
    features=output/'word-features.npz';np.savez_compressed(features,**all_rows)
    (output/'receipt.json').write_text(json.dumps({'input_sha256':sha256(packet).hexdigest(),'feature_sha256':sha256(features.read_bytes()).hexdigest(),
      'candidate':runtime.candidate,'revision':runtime.revision,'documents':len(data),'forwards':count[0],'words':sum(len(v) for v in all_rows.values()),
      'surface_features':SURFACE,'local_width':96,'relation_features':RELATIONS},indent=2))
