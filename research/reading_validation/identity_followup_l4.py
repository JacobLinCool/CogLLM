"""Run four prospective external transfer extractions on a single owned GPU."""

from argparse import ArgumentParser
from pathlib import Path

from research.reading_validation import external_forecast,local_forecast,transition_external_forecast
from research.reading_validation.identity_run_l4 import identity,run_phases


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    for name in ('model','output','stimuli'):
        parser.add_argument('--'+name,required=True,type=Path)
    for name in ('candidate','protocol-sha256'):
        parser.add_argument('--'+name,required=True)
    options=parser.parse_args()
    sources=[('rewrite',external_forecast.run,Path('rewrite-stimuli.jsonl')),
             ('jargon',local_forecast.run,Path('jargon/local-stimuli.jsonl')),
             ('conflict',external_forecast.run,Path('conflict-stimuli.jsonl')),
             ('cohesion',transition_external_forecast.run,Path('cohesion-stimuli.jsonl')),
             ('multirc',identity,options.stimuli)]
    run_phases(options.model,options.candidate,options.output,options.protocol_sha256,sources)
