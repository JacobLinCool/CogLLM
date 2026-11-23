"""Verify a portable readout through actual Gradio HTTP on its pinned backbone."""

import argparse
from pathlib import Path

from research.token_readout.live import verify_candidate
from research.token_readout.portable_runner import load


def run(candidate, model, stimuli, output, protocol_sha256, checkpoint):
    runner = load(model=model, checkpoint=checkpoint)
    if runner.runtime.candidate != candidate:
        raise ValueError('The portable checkpoint does not match this run')
    verify_candidate(runner, stimuli, output, protocol_sha256)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    for name in ('model', 'stimuli', 'output', 'checkpoint'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(**vars(parser.parse_args()))
