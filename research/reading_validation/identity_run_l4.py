"""Collect outcome-blind, single-pass reading features for four registered studies."""

from argparse import ArgumentParser
import gc
import json
from pathlib import Path
from time import perf_counter
import traceback

import numpy as np
import torch

from research.backbone_transfer.extract import digest, stream_rows, write_json
from research.reading_validation import bridge_forecast, external_forecast, local_forecast
from research.reading_validation.identity_features import NAMES, pairs
from research.token_readout.backbone import Backbone
from research.token_readout.readout import PortableReadout


CHECKPOINT = 'c1704d4436d767fa43bdb59e9216698514e1f6bd1d45cad4b60d0f018d84abb5'


def identity(runtime, reader, packet, source, output):
    output.mkdir()
    example = 'The sensor reading fluctuated. The cable connector was loose.'
    full = example + ' Next, the maintenance team replaced the cable.'
    prefix = runtime.tokenizer.encode(example, add_special_tokens=False)
    extension = runtime.tokenizer.encode(full, add_special_tokens=False)
    if extension.ids[:len(prefix.ids)] != prefix.ids:
        raise AssertionError('Technical prefix tokenization mismatch')
    anchor = {'issue': [1,4]}
    a = pairs(runtime, reader, packet, {'text': example,'anchor_tokens': anchor}, runtime.encode(example))
    b = pairs(runtime, reader, packet, {'text': full,'anchor_tokens': anchor}, runtime.encode(full))
    max_error = float(np.max(np.abs(a-b[:, :a.shape[1]])))
    if not np.isfinite(max_error) or max_error > 0.004:
        raise ArithmeticError(f'Feature prefix validation failed: {max_error}')
    counter = 0
    def count(_module, _args):
        nonlocal counter
        counter += 1
    rows = list(stream_rows(source))
    total = 0
    entries = []
    start = perf_counter()
    with runtime.encoder.register_forward_pre_hook(count):
        for step,row in enumerate(rows,1):
            feat = runtime.encode(row['text'])
            data = pairs(runtime,reader,packet,row,feat)
            if counter != step:
                raise RuntimeError(f'Expected {step} calls, found {counter}')
            np.savez_compressed(output/f"{row['id']}.npz", pair=data)
            total += len(feat.token_ids)
            entries.append({'id': row['id'], 'anchors':sorted(row['anchor_tokens']), 'tokens':len(feat.token_ids)})
            if step % 100 == 0:
                print(f'Identity features {step}/{len(rows)}, {total} tokens',flush=True)
    write_json(output/'geometry.json', {'documents':entries, 'channels':NAMES, 'prefix_max_abs_error':max_error,
                                        'forward_count':counter, 'technical_forward_count':2, 'seconds':perf_counter()-start,
                                        'tokens':total,'input_sha256':digest(source)})


def run_phases(model, candidate, output, protocol_sha256, jobs):
    if candidate != 'qwen' or torch.cuda.device_count() != 1 or torch.cuda.get_device_name() != 'NVIDIA L4':
        raise RuntimeError('Expected a single Colab NVIDIA L4')
    output.mkdir()
    weights = Path('curve.pt')
    if digest(weights) != CHECKPOINT:
        raise ValueError('Different joint checkpoint')
    packet = torch.load(weights, map_location='cpu',weights_only=True)
    runtime = Backbone.load(candidate,model,native_attention=True)
    if runtime.revision != packet['revision']:
        raise ValueError('Different Qwen revision')
    reader = PortableReadout(runtime.width,packet['variant'],packet['relations']).to(runtime.device)
    reader.load_state_dict(packet['reader'],strict=True)
    reader.eval().requires_grad_(False)
    provenance = {'model_revision':runtime.revision,'checkpoint':CHECKPOINT,'device':torch.cuda.get_device_name(),
                  'device_total_bytes':torch.cuda.get_device_properties(0).total_memory,'torch':str(torch.__version__)}
    phases = {}
    # Each independently sealed study writes its receipts. Preserve successful
    # outputs if another study reports a specific data or numerical error.
    for name, render, path in jobs:
        start = perf_counter()
        try:
            print(f'Begin {name}',flush=True)
            with torch.no_grad():
                render(runtime,reader,packet,path,output/(name+'-forecast'))
            phases[name] = {'completed':True,'seconds':perf_counter()-start,'input_sha256':digest(path)}
        except Exception as exc:
            # Failure remains explicit in the downloaded status and the report.
            phases[name] = {'completed':False,'exception':type(exc).__name__,'message':str(exc),'seconds':perf_counter()-start}
            traceback.print_exc()
        print(json.dumps({name:phases[name]}),flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    write_json(output/'causality.json',{**provenance,'studies':phases})
    files={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'seal.json', {**provenance,'protocol_sha256':protocol_sha256,'study_status':phases,
                                    'files_sha256':files, 'all_completed':all(p['completed'] for p in phases.values())})


def run(model, candidate, stimuli, output, protocol_sha256):
    jobs = [('local', local_forecast.run, Path('local-stimuli.jsonl')),
            ('bridge', bridge_forecast.run, Path('bridge-stimuli.jsonl')),
            ('external', external_forecast.run, Path('external-stimuli.jsonl')),
            ('identity', identity, stimuli)]
    run_phases(model, candidate, output, protocol_sha256, jobs)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for name in ('model','stimuli','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('candidate','protocol-sha256'):
        parser.add_argument('--'+name,required=True)
    run(**vars(parser.parse_args()))
