"""Run two independently specified human-reference studies on the same staged Qwen checkpoint."""
from argparse import ArgumentParser
import gc
from pathlib import Path

import torch

from research.backbone_transfer.extract import digest, read_json, write_json
from research.reading_validation.transition_direct_l4 import run as direct
from research.reading_validation.transition_game_l4 import run as game

GAME_PROTOCOL = '373bb5cf8cd007f08613d1b7846bd3e9fe47bde2977a815d366bf3086bcb89b6'
GAME_STIMULI = '3b711a247e6b7372ad8772aa2e877efc20f35d5ee38f3cac9186e6cf200b410a'


def run(model,candidate,stimuli,output,protocol_sha256):
    if candidate!='qwen':raise ValueError('The paired game transfer was registered on Qwen')
    output.mkdir()
    direct(model,candidate,stimuli,Path('span.pt'),Path('ner.npz'),output/'direct-forecast',protocol_sha256)
    gc.collect();torch.cuda.empty_cache()
    game(model,Path('game-native-stimuli.jsonl'),output/'game-forecast',GAME_PROTOCOL,GAME_STIMULI)
    checks={tag:read_json(output/(tag+'-forecast')/'receipt.json') for tag in ('direct','game')}
    write_json(output/'causality.json',{'candidate':candidate,'studies':list(checks),'direct_protocol':protocol_sha256,'game_protocol':GAME_PROTOCOL})
    write_json(output/'seal.json',{'protocol_sha256':protocol_sha256,'files_sha256':{str(p.relative_to(output)):digest(p)
        for p in sorted(output.rglob('*')) if p.is_file()}})


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for n in ('model','stimuli','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('candidate','protocol-sha256'):p.add_argument('--'+n,required=True)
    run(**vars(p.parse_args()))
