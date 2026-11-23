"""Extract and certify readouts for implicit definitions in textbook prose."""

from argparse import ArgumentParser
from pathlib import Path

from research.reading_validation.external_forecast import run as discourse
from research.reading_validation.definition_forecast import run
from research.reading_validation.identity_run_l4 import run_phases


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for name in ('model','stimuli','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('candidate','protocol-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run_phases(a.model,a.candidate,a.output,a.protocol_sha256,[('discourse',discourse,Path('discourse-stimuli.jsonl')),('definitions',run,a.stimuli)])
