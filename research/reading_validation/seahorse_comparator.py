"""Wrap the official Q1 specialist without adapting its parameters."""
from argparse import ArgumentParser
from pathlib import Path

from research.backbone_transfer.extract import digest, write_json
from research.reading_validation.transition_seahorse_specialist_l4 import REV, run as evaluate

SOURCE = 'e759af76a71078eb2e181fcfec276dd888f8623b4c1ca87eb558c190e5d4e295'


def run(model, candidate, stimuli, output, protocol_sha256):
    if candidate != 'seahorse':
        raise ValueError('Expected the public SEAHORSE Q1 specialist')
    output.mkdir()
    evaluate(stimuli, output / 'specialist-forecast', SOURCE, protocol_sha256, model)
    write_json(output / 'causality.json', dict(candidate=candidate, revision=REV, supervised_updates=0,
                                               visible_source_article=True, decode_steps=1))
    files = {str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'seal.json', dict(protocol_sha256=protocol_sha256, files_sha256=files))


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    for key in ('model','stimuli','output'):
        parser.add_argument('--' + key,type=Path,required=True)
    for key in ('candidate','protocol-sha256'):
        parser.add_argument('--' + key,required=True)
    run(**vars(parser.parse_args()))
