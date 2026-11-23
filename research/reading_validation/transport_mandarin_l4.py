"""Use one owned device for original, independently sealed native reading endpoints."""
from argparse import ArgumentParser
import gc
from pathlib import Path
import traceback
import json
from time import perf_counter
import torch
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation import convergent_word_l4,state_transport_l4,transition_mandarin_l4,word_response_verify
from research.reading_validation import writer_reference_l4,transition_null_l4,transition_pointer_serving_verify


def run(model,candidate,stimuli,output,protocol_sha256):
    jobs=[('word-responses',word_response_verify.run,Path('word-stimuli.jsonl')),
          ('manual-writing',writer_reference_l4.run,Path('manual.json')),
          ('multilab',convergent_word_l4.run,Path('multilab.jsonl')),
          ('state-transport',state_transport_l4.run,stimuli),
          ('reasoning',state_transport_l4.run,Path('reasoning.jsonl'))]
    run_phases(model,candidate,output,protocol_sha256,jobs)
    origin=json.loads((output/'causality.json').read_text());(output/'causality.json').rename(output/'shared-status.json');(output/'seal.json').rename(output/'shared-seal.json')
    tasks=[
        ('mandarin-reading',transition_mandarin_l4.run,[Path('mandarin.jsonl')],['b41e5c9cf82fbc3d513ff9ea2fef642c1268312b592c2c2800059d84fe1a9a3c','e54793dc7ef88b5db3b1d5fe54995a2c5ef7a1e51c8b1ef40688cc195a74a5f7']),
        ('pointer-writer',transition_pointer_serving_verify.run,[Path('pointer-writer.jsonl'),Path('pointer-serving-reference.npz')],['2d7c1b9c759524cb605008338b77f769c05f9701396a871de5ae42d1f253ed61','e2d10e82405c6625b21711ae5a5183ae27c18956eb9d72d1cb6589b2759690ca']),
        ('null-choice',transition_null_l4.run,[Path('source-pointer'),Path('source-gate'),Path('null-choice.jsonl')],['d57549e3452409c31e4368d5f10cbeed06a40d08b468ade0675feae272878f20','d82ea4e9228d8deefd28a56fdb834ac6e3190251e5aff6048e9b11d3df168ece'])]
    for name,op,args,hashes in tasks:
        gc.collect();torch.cuda.empty_cache();start=perf_counter()
        try:
            destination=output/(name+'-forecast');op(model,*args,destination,*hashes);origin['studies'][name]=dict(completed=True,seconds=perf_counter()-start)
        except Exception as exc:
            traceback.print_exc();origin['studies'][name]=dict(completed=False,message=str(exc),exception=type(exc).__name__)
    write_json(output/'causality.json',origin)
    write_json(output/'seal.json',dict(protocol_sha256=protocol_sha256,all_completed=all(v['completed'] for v in origin['studies'].values()),files_sha256={str(f.relative_to(output)):digest(f) for f in sorted(output.rglob('*')) if f.is_file()}))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for k in ('model','stimuli','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('candidate','protocol-sha256'):p.add_argument('--'+k,required=True)
    run(**vars(p.parse_args()))
