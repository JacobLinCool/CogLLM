"""Extract book/movie word features under a single pinned text encoder per assignment."""
from argparse import ArgumentParser
from importlib import import_module
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.transition_snac_forecast import RELATIONS, SURFACE, design_from_features
from research.token_readout.backbone import Backbone
from research.token_readout.readout import PortableReadout

CHECKPOINTS = {
    'qwen':'57e0d5fda366fc088dbe07f19faf5eac0638c22952fb635c3a7073c51ae1219c',
    'lfm3':'1e37db7f0ce9d4ab2f5a3248ce3790c043db0f038cbc6730754ac5f63f52ce41',
    'gemma4':'198741728ec5b9f222c0dccf8f722aac4848424f85aeac54d3ec49d57b372933',
}
CAUSAL_CLASS = {'qwen':'Qwen3_5ForCausalLM','lfm3':'Lfm3ForCausalLM','gemma4':'Gemma4ForCausalLM'}


def native_logits(runtime, hidden):
    logits = F.linear(hidden,runtime.encoder.embed_tokens.weight)
    if runtime.candidate == 'gemma4':
        cap = runtime.encoder.config.final_logit_softcapping
        if cap != 30.0:
            raise ValueError('Unexpected Gemma final logits softcap')
        logits = torch.tanh(logits / cap) * cap
    return logits


def native_losses(runtime, features):
    targets = torch.tensor(features.token_ids[1:],device=runtime.device)
    losses = np.zeros(len(features.token_ids),dtype=np.float64)
    for start in range(0,len(targets),32):
        y = targets[start:start+32]
        logits = native_logits(runtime,features.hidden[start:start+len(y)])
        losses[start+1:start+len(y)+1] = F.cross_entropy(logits,y,reduction='none').cpu().numpy()
    return losses


def verify_logits(runtime):
    """Compare to the architecture's actual HF forward and tied native head."""
    import inspect
    config = runtime.encoder.config
    if not config.tie_word_embeddings:
        raise ValueError('The snapshot does not tie input and output embeddings')
    cls = getattr(import_module(type(runtime.encoder).__module__),CAUSAL_CLASS[runtime.candidate])
    if cls._tied_weights_keys != {'lm_head.weight':'model.embed_tokens.weight'}:
        raise ValueError('Unexpected language-model output projection binding')
    with torch.device('meta'):
        reference = cls(config)
    reference.model = runtime.encoder
    reference.lm_head.weight = runtime.encoder.embed_tokens.weight
    reference.eval()
    probe = 'A new librarian welcomed Ella. Later she asked him for a book.'
    factors = runtime.encode(probe)
    ids = torch.tensor([factors.token_ids],device=runtime.device)
    actual = reference(input_ids=ids,use_cache=False).logits[0]
    replay = native_logits(runtime,factors.hidden)
    max_error = float((actual-replay).abs().max().item())
    if max_error > .005 or not np.isfinite(max_error):
        raise ValueError(f'Native causal-LM logits disagree with block reconstruction: {max_error}')
    return {'absolute_error':max_error, 'vocabulary':actual.shape[-1],
            'official_forward':cls.__name__, 'source_sha256':digest(Path(inspect.getfile(cls))),
            'technical_forwards':2, 'final_softcapping':getattr(config,'final_logit_softcapping',None)}


@torch.inference_mode()
def run(model, candidate, stimuli, checkpoint, output, protocol_sha256):
    if digest(checkpoint) != CHECKPOINTS[candidate]:
        raise ValueError('Expected the historic, matched span projector')
    packet = torch.load(checkpoint,map_location='cpu',weights_only=True)
    runtime = Backbone.load(candidate, model, native_attention=True)
    if (packet['revision'] != runtime.revision or packet['candidate'] != candidate
            or packet['variant'] != 'span' or packet['relations'] is not True):
        raise ValueError('Unexpected backbone/span readout identity')
    reader = PortableReadout(runtime.width,packet['variant'],packet['relations']).to(runtime.device)
    reader.load_state_dict(packet['reader'],strict=True)
    reader.eval().requires_grad_(False)
    native = verify_logits(runtime)
    data = [json.loads(line) for line in stimuli.read_text().splitlines()]
    output.mkdir(parents=True,exist_ok=True)
    count = 0
    def count_call(_module,_args):
        nonlocal count
        count += 1
    result = {}
    started = perf_counter()
    with runtime.encoder.register_forward_pre_hook(count_call):
        for row in data:
            words = [w[:2]+tuple_range+w[2:] for w,tuple_range in
                     zip(row['words'],row['token_ranges'][candidate],strict=True)]
            feat = runtime.encode(row['text'])
            if len(feat.token_ids) != row['token_counts'][candidate]:
                raise ValueError('Complete article tokenizer IDs differ from local common support')
            if any(feat.offsets[last][1] != end for _,end,_,last,_,_ in words):
                raise ValueError('Remote word end disagrees with the public text')
            matrix = design_from_features(runtime,reader,{'text':row['text'],'words':words},
                                          feat,token_surprisal=native_losses(runtime,feat))
            result[row['id']] = matrix
            if count != len(result):
                raise ValueError('Expected one whole-document causal forward per narrative')
            if count%25 == 0:
                print(f'{candidate}: {count}/{len(data)} narratives',flush=True)
    archive = output/'word-features.npz'
    np.savez_compressed(archive,**result)
    write_json(output/'receipt.json',{'candidate':candidate,'revision':runtime.revision,
        'device':torch.cuda.get_device_name(), 'protocol_sha256':protocol_sha256,
        'checkpoint_sha256':digest(checkpoint),'input_sha256':digest(stimuli),
        'feature_sha256':digest(archive),'documents':len(result),'forwards':count,
        'words':sum(len(v) for v in result.values()),'seconds':perf_counter()-started,
        'surface_features':SURFACE,'local_width':2*reader.middle.out_features,
        'relation_features':RELATIONS,'global_layers':len(runtime.global_layers),
        'vocab_probe':native})


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',choices=tuple(CHECKPOINTS),required=True)
    for key in ('model','stimuli','checkpoint','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--protocol-sha256',required=True)
    run(**vars(parser.parse_args()))
