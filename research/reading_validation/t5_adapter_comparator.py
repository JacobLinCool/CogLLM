"""Exact-resumable T5-LoRA comparator using separately archived book-source checkpoints."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.transition_t5_adapter_l4 import run as fit
from research.reading_validation.transition_t5_l4 import REVISION
from research.reading_validation.t5_checkpoint_transport import start_transport


def run(model,candidate,stimuli,output,protocol_sha256):
    if candidate!='t5':raise ValueError('Unexpected noncausal comparator')
    output.mkdir();resume=Path('prior-snapshot.pt');prior=Path('prior-manifest.json')
    if resume.exists()!=prior.exists():raise ValueError('Unmatched resume weights and receipt')
    state=read_json(prior) if prior.exists() else None
    destination=output/'t5-adapter-forecast';stop,worker,errors=start_transport(destination,Path('checkpoint-transport'))
    try:fit(stimuli,destination,protocol_sha256,model,resume if state else None,state['sha256'] if state else None)
    finally:stop.set();worker.join()
    if errors:raise RuntimeError('Could not certify optimizer-state transport') from errors[0]
    write_json(output/'causality.json',{'candidate':candidate,'revision':REVISION,'observes_complete_target_sentence':True,'initial_checkpoint':state})
    write_json(output/'seal.json',{'protocol_sha256':protocol_sha256,'files_sha256':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for n in ('model','stimuli','output'):p.add_argument('--'+n,required=True,type=Path)
    for n in ('candidate','protocol-sha256'):p.add_argument('--'+n,required=True)
    run(**vars(p.parse_args()))
