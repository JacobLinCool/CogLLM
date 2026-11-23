"""Collect blinded outputs of the fixed T5 decision comparator."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.extract import digest,write_json
from research.reading_validation.transition_t5_decision_l4 import run as apply
from research.reading_validation.transition_t5_l4 import REVISION


def run(model,candidate,stimuli,output,protocol_sha256):
    if candidate!='t5':raise ValueError('Expected pinned T5 reference')
    output.mkdir();adapter=Path('selected-adapter.pt')
    apply(stimuli,adapter,output/'t5-decision-forecast','cc7c979d6ad7ecb176e6440a80c8c415441766fb91c6503a299ba49c394682e3',protocol_sha256,model)
    write_json(output/'causality.json',dict(candidate=candidate,revision=REVISION,supervised_updates=0,complete_sentence_comparator=True))
    write_json(output/'seal.json',dict(protocol_sha256=protocol_sha256,files_sha256={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for k in ('model','stimuli','output'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('candidate','protocol-sha256'):p.add_argument('--'+k,required=True)
    run(**vars(p.parse_args()))
