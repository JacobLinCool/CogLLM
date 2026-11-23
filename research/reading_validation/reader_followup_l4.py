"""Run individual blinded reading experiments sequentially on one owned Colab L4."""
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
import gc
import json
import sys
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import traceback
import torch
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation import argument_l4,convergent_word_l4,expectation_l4,nns_writer_l4,semlex_l4
from research.reading_validation import transition_dpr_l4,transition_gate_l4,transition_jones_l4,transition_pointer_serving_verify


def run(model,candidate,stimuli,output,protocol_sha256):
    inputs=json.loads(Path('studies.json').read_text());start=perf_counter()
    shared=[('learner-writer',nns_writer_l4.run),('discourse-order',expectation_l4.run),('convergent-original',convergent_word_l4.run),('semlex',semlex_l4.run),('argument',argument_l4.run)]
    run_phases(model,candidate,output,protocol_sha256,[(k,op,Path(inputs[k]['source'])) for k,op in shared]);result=json.loads((output/'causality.json').read_text())
    (output/'causality.json').rename(output/'shared-studies.json');(output/'seal.json').rename(output/'shared-seal.json');status=result['studies']
    individual=[('jones',transition_jones_l4.run),('gate-pointer',transition_gate_l4.run),('pointer-writer',transition_pointer_serving_verify.run),('semantic-pronoun',transition_dpr_l4.run)]
    for key,op in individual:
        gc.collect();torch.cuda.empty_cache();began=perf_counter();row=inputs[key];source=Path(row['source'])
        try:
            if key=='pointer-writer':op(model,source,Path('pointer-serving-reference.npz'),output/(key+'-forecast'),row['sha256'],row['reference_sha256'])
            elif key=='semantic-pronoun':op(model,source,output/(key+'-forecast'),row['sha256'],row['protocol_sha256'])
            else:op(model,Path('source-pointer'),source,output/(key+'-forecast'),row['sha256'],row['protocol_sha256'])
            status[key]={'completed':True,'seconds':perf_counter()-began,'input_sha256':digest(source)}
        except Exception as exc:
            traceback.print_exc();status[key]={'completed':False,'seconds':perf_counter()-began,'exception':type(exc).__name__,'message':str(exc)}
    write_json(output/'causality.json',dict(studies=status,python=sys.version.split()[0],seconds=perf_counter()-start,model_revision=result['model_revision'],checkpoint=result['checkpoint'],device=torch.cuda.get_device_name()))
    write_json(output/'seal.json',dict(protocol_sha256=protocol_sha256,all_completed=all(r['completed'] for r in status.values()),files_sha256={str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for k in ('model','stimuli','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('candidate','protocol-sha256'):p.add_argument('--'+k,required=True)
    run(**vars(p.parse_args()))
