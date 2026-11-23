"""Seal article-only causal predictions from one participant-blind architecture replication."""
from argparse import ArgumentParser
from pathlib import Path

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.transition_direct_l4 import run as fit


def run(model, candidate, stimuli, output, protocol_sha256):
    output.mkdir()
    destination = output / 'direct-forecast'
    fit(model, candidate, stimuli, Path('span.pt'), Path('ner.npz'), destination, protocol_sha256)
    receipt = read_json(destination / 'receipt.json')
    write_json(output / 'causality.json', {'candidate':candidate, 'feature_metadata':receipt['feature_metadata'],
                                          'source_supervision':'book training/development only'})
    write_json(output / 'seal.json', {'protocol_sha256':protocol_sha256, 'files_sha256':{
        str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})


if __name__ == '__main__':
    p=ArgumentParser(description=__doc__)
    for n in ('model','stimuli','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('candidate','protocol-sha256'):p.add_argument('--'+n,required=True)
    run(**vars(p.parse_args()))
