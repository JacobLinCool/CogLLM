"""Package the book-trained full-sentence T5 comparator under the common L4 receipts."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.transition_t5_l4 import REVISION,run as fit


def run(model,candidate,stimuli,output,protocol_sha256):
    if candidate!='t5':raise ValueError('This comparison requires the published T5-Large base')
    output.mkdir();fit(stimuli,output/'t5-forecast',protocol_sha256,model)
    write_json(output/'causality.json',{'candidate':candidate,'revision':REVISION,'observes_complete_target_sentence':True,
        'description':'Full-sentence supervised generation is an external comparator for the causal word-level system.'})
    write_json(output/'seal.json',{'protocol_sha256':protocol_sha256,'files_sha256':{
        str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for n in ('model','stimuli','output'):p.add_argument('--'+n,required=True,type=Path)
    for n in ('candidate','protocol-sha256'):p.add_argument('--'+n,required=True)
    run(**vars(p.parse_args()))
