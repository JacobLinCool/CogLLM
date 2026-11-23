"""Stage and release one full-width backbone for the reference tracking experiment."""
from argparse import ArgumentParser
from pathlib import Path

from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest
from research.reading_validation.architecture_colab import SPAN
from research.reading_validation.identity_colab import DATA, ROOT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PROTOCOL = Path(__file__).with_name('transition-direct-protocol.md')
SOURCES = {
    'snac-direct-stimuli.jsonl':'5ba394da96bd18b4859aed77d5c0b154980f18bd0eed5908855ff12ef0ce4467',
    'snac-ner.npz':'9f906e30f9f998385c4fb4a76c67be95c2c98e7822a7f191a4920362400d64be',
}


def prepare(candidate):
    if digest(PROTOCOL) != '7701cbe2cea263e635aeeab518314e108dd5be2560e256137bcff79e0ca1dcb4':
        raise ValueError('Source-supervised study specification changed')
    if any(digest(DATA/n) != h for n,h in SOURCES.items()):
        raise ValueError('Data no longer match the sealed extraction packet')
    base = ROOT / '.cache/cogllm/token-readout/transfer-l4' / ('cogllm-portable-'+SPAN[candidate])
    inputs = {'stimuli.jsonl':DATA/'snac-direct-stimuli.jsonl', 'ner.npz':DATA/'snac-ner.npz', 'span.pt':base/'extraction/span.pt'}
    tag='reading-validation-direct'
    entry='research/reading_validation/direct_l4.py'
    if candidate=='qwen':
        inp=DATA/'game-native-stimuli.jsonl'
        if digest(inp)!='3b711a247e6b7372ad8772aa2e877efc20f35d5ee38f3cac9186e6cf200b410a':
            raise ValueError('Independent silver-source transfer packet changed')
        inputs['game-native-stimuli.jsonl']=inp
        tag='reading-validation-direct-game';entry='research/reading_validation/direct_qwen_l4.py'
    pack_job(DATA/'l4', PROTOCOL, 'reading-direct-'+candidate, tag,
             dependencies(entry), inputs, {}, 5400,
             {'purpose':'Direct word-level transfer across book, movie and independent reader-game texts'}, candidate=candidate)


if __name__ == '__main__':
    p=ArgumentParser(description=__doc__); sp=p.add_subparsers(dest='action',required=True)
    sp.add_parser('prepare').add_argument('candidate',choices=tuple(SPAN));sp.add_parser('run').add_argument('job',type=Path)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.candidate)
    else:run(a.job,protocol=PROTOCOL)
