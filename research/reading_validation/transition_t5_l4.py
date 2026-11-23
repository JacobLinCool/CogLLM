"""Book-only re-fit of the published T5-Large generative character-span architecture."""
from argparse import ArgumentParser
from collections import Counter
from contextlib import contextmanager
from hashlib import file_digest, sha1
import json
import math
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, set_seed

REPO='google-t5/t5-large'
REVISION='150ebc2c4b72291e770f58e6057481c8d2ed331a'
WEIGHTS='bb566a699a6939ffa2ac2fa7eb26fd02b13d4d14956186d7af1e46ce19c67d32'
SMALL={'config.json':'e4be09f2bfa7ed03596de9bdad4ddc0c7bdff4c1',
       'generation_config.json':'d52815623b46b7db1c4b957b5a83a8ad30b0146a',
       'tokenizer.json':'41911369be79611d82d152725113e66206bd7b79',
       'spiece.model':'4e28ff6ebdf584f5372d9de68867399142435d9a'}
SEED=4703
INPUT_SHA='6acf4a262716687a591fbd0522b743bd820c4fa99bd7b3ea1c3be00309f759bb'
PROTOCOL_SHA='7aa3d0c1dc825755862ed91097c13bc7d9d7b069b2ed6a4ab61ffbb3f2a4d51f'


def digest(path):
    with Path(path).open('rb') as file:
        return file_digest(file,'sha256').hexdigest()


def write_json(path,data):
    tmp=path.with_suffix('.pending')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def model_path(preloaded):
    if not preloaded:
        from huggingface_hub import snapshot_download
        preloaded=Path(snapshot_download(repo_id=REPO,revision=REVISION,allow_patterns=list(SMALL)+['model.safetensors']))
    for name,expected in SMALL.items():
        body=(preloaded/name).read_bytes()
        got=sha1(f'blob {len(body)}\0'.encode()+body).hexdigest()
        if got!=expected:raise ValueError(f'{name} does not match the pinned T5 Git object')
    if digest(preloaded/'model.safetensors')!=WEIGHTS:
        raise ValueError('The original T5 public base tensors changed')
    return preloaded


def examples(file,tokenizer):
    rows={'train':[],'dev':[],'test':[]};stats=Counter()
    sentinel=tokenizer.convert_tokens_to_ids('<extra_id_0>')
    for line in file.read_text().splitlines():
        row=json.loads(line); split=row['split']
        if (split=='test')!=('target' not in row):raise ValueError('Outcome entered the external GPU payload')
        raw=tokenizer(' '.join((row['context'],'<extra_id_0>',row['sentence'])),
                      return_attention_mask=False)['input_ids']
        stats[split+'_sentences']+=1
        if len(raw)>1024:stats[split+'_inputs_truncated']+=1
        encoded=raw if len(raw)<=1024 else raw[:1023]+[tokenizer.eos_token_id]
        if sentinel not in encoded:stats[split+'_target_sentence_hidden_by_authors_limit']+=1
        entry={'id':row['id'],'input':encoded}
        if split!='test':
            label=tokenizer(row['target'],return_attention_mask=False)['input_ids']
            if len(label)>80:stats[split+'_targets_truncated']+=1
            entry['target']=label if len(label)<=80 else label[:79]+[tokenizer.eos_token_id]
        rows[split].append(entry)
    if tuple(len(rows[s]) for s in ('train','dev','test'))!=(2938,716,1661):
        raise ValueError('The matching source-only specialist split changed')
    return rows,dict(stats)


def batch(examples,training):
    def padded(key,value):
        width=max(len(r[key]) for r in examples)
        return torch.tensor([r[key]+[value]*(width-len(r[key])) for r in examples],
                            dtype=torch.long,device='cuda')
    data=padded('input',0)
    xyz={'input_ids':data,'attention_mask':data.ne(0)}
    if training:
        xyz['labels']=padded('target',-100)
    return xyz


def batches(cases,size):
    for start in range(0,len(cases),size):
        yield cases[start:start+size]


def schedule(cases,seed):
    rng=random.Random(seed)
    shuffled=list(cases);rng.shuffle(shuffled)
    groups=[]
    for pool in batches(shuffled,64):
        pool.sort(key=lambda r:len(r['input']))
        groups.extend(batches(pool,8))
    rng.shuffle(groups)
    return groups


@torch.inference_mode()
def evaluate(model,cases):
    model.eval();nll=0.;total=0
    for block in batches(sorted(cases,key=lambda e:len(e['input'])),4):
        features=batch(block,True); tokens=int(features['labels'].ne(-100).sum())
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=model(**features,use_cache=False).loss
        nll+=float(loss)*tokens;total+=tokens
    if not math.isfinite(nll):raise ArithmeticError('Nonfinite book development loss')
    return nll/total


@torch.inference_mode()
def generate(model,tokenizer,movies,output):
    model.eval();forecasts=[]
    for group in batches(sorted(movies,key=lambda r:len(r['input'])),8):
        data=batch(group,False)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            out=model.generate(**data,max_length=80,do_sample=False,num_beams=1,use_cache=True)
        for row,ids in zip(group,out.tolist(),strict=True):
            keep=[t for t in ids if t not in (tokenizer.pad_token_id,tokenizer.eos_token_id)]
            forecasts.append({'id':row['id'],'generation':tokenizer.decode(keep,skip_special_tokens=False)})
    with output.open('w') as handle:
        for row in forecasts:print(json.dumps(row),file=handle)
    if len({r['id'] for r in forecasts})!=len(movies):raise ValueError('Dropped or repeated movie sentence')


def run(stimuli,output,protocol_sha256,model=None):
    if digest(stimuli)!=INPUT_SHA or protocol_sha256!=PROTOCOL_SHA:
        raise ValueError('Unexpected source archive or study protocol')
    if torch.cuda.device_count()!=1 or torch.cuda.get_device_name()!='NVIDIA L4' or not torch.cuda.is_bf16_supported():
        raise RuntimeError('This study is budgeted for one BF16-capable Colab NVIDIA L4')
    import transformers
    set_seed(SEED);torch.set_num_threads(2); torch.set_float32_matmul_precision('high')
    output.mkdir(parents=True,exist_ok=True)
    snapshot=model_path(model)
    tokenizer=T5Tokenizer.from_pretrained(snapshot,local_files_only=True)
    encoder=T5ForConditionalGeneration.from_pretrained(snapshot,use_safetensors=True,local_files_only=True,
                                                         dtype=torch.float32).to('cuda')
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    data,geometry=examples(stimuli,tokenizer)
    optimizer=torch.optim.Adam(encoder.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0.,fused=True)
    epochs=5;steps=epochs*math.ceil(len(data['train'])/8)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:max(0.,(steps-step)/steps))
    initial=evaluate(encoder,data['dev']);started=perf_counter()
    print(f'T5-Large initial book development NLL {initial:.6f}; {steps} total steps',flush=True)
    history=[];best_loss=float('inf');best_epoch=None;updates=0;ratios=[]
    # Only small metrics and generated spans are returned from the GPU. Keep the selected 2.95 GB state on Colab.
    with TemporaryDirectory(dir='/content') as tmp:
        best=Path(tmp)/'source-selected.pt'
        for epoch in range(1,epochs+1):
            encoder.train();nll=0.;tokens=0;recent=[]
            for group in schedule(data['train'],SEED+epoch):
                target_tokens=sum(len(r['target']) for r in group)
                optimizer.zero_grad(set_to_none=True)
                mean_nll=0.
                for small in batches(group,2):
                    feat=batch(small,True)
                    count=sum(len(r['target']) for r in small)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=encoder(**feat,use_cache=False).loss
                    if not torch.isfinite(loss):raise ArithmeticError('Nonfinite specialist training loss')
                    (loss*(count/target_tokens)).backward()
                    mean_nll+=float(loss.detach())*count/target_tokens
                grad=torch.nn.utils.clip_grad_norm_(encoder.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();scheduler.step();updates+=1;nll+=mean_nll*target_tokens;tokens+=target_tokens
                recent.append(float(grad));ratios.append(mean_nll)
                if updates%100==0:print(f'T5 epoch {epoch}, step {updates}/{steps}, NLL {np.mean(ratios[-30:]):.5f}, elapsed {perf_counter()-started:.1f}s',flush=True)
            validation=evaluate(encoder,data['dev'])
            history.append({'epoch':epoch,'training_tokens':tokens,'training_nll':nll/tokens,
                            'development_nll':validation,'mean_gradient_norm':float(np.mean(recent)),
                            'last_gradient_norm':recent[-1],'end_learning_rate':scheduler.get_last_lr()[0],
                            'cumulative_seconds':perf_counter()-started})
            if validation<best_loss:
                best_epoch=epoch;best_loss=validation
                torch.save(encoder.state_dict(),best)
            write_json(output/'training.json',{'initial_development_nll':initial,'epochs':history,'best_epoch':best_epoch})
            print(f'T5 epoch {epoch}/{epochs}, dev NLL {validation:.6f}, best {best_epoch}',flush=True)
        if updates!=steps or best_epoch is None:raise ArithmeticError('Incomplete specialist training')
        del optimizer,scheduler
        encoder.load_state_dict(torch.load(best,weights_only=True,map_location='cpu'),strict=True)
        encoder.gradient_checkpointing_disable()
        output_file=output/'movie-generations.jsonl'
        generate(encoder,tokenizer,data['test'],output_file)
        source_check=digest(best)
    write_json(output/'receipt.json',{'protocol_sha256':protocol_sha256,'input_sha256':digest(stimuli),
        'pretrained':REPO,'revision':REVISION,'pretrained_weight_sha256':WEIGHTS,'source_selected_state_sha256':source_check,
        'torch':str(torch.__version__),'transformers':transformers.__version__,'device':torch.cuda.get_device_name(),
        'optimizer_updates':updates,'training_parameters':sum(p.numel() for p in encoder.parameters()),
        'initial_development_nll':initial,'best_epoch':best_epoch,'best_development_nll':best_loss,
        'relative_development_nll_reduction':(initial-best_loss)/initial,
        'last_training_nll':float(np.mean(ratios[-30:])),'geometry':geometry,
        'external_sentences':len(data['test']),'forecast_sha256':digest(output_file),'seconds':perf_counter()-started,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated()})


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    for key in ('stimuli','output'):parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--protocol-sha256',required=True);parser.add_argument('--model',type=Path)
    run(**vars(parser.parse_args()))
