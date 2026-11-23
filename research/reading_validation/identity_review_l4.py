"""Run independent reading, revision, and word-localization studies on one owned L4."""

from argparse import ArgumentParser
from pathlib import Path

from research.reading_validation.event_forecast import run as events
from research.reading_validation.identity_run_l4 import identity,run_phases
from research.reading_validation.revision_forecast import run as revision
from research.reading_validation.transition_serving_verify import run as transition_serving
from research.reading_validation.transition_snac_forecast import run as snac
from research.reading_validation.writer_verify import run as writer


def run(model,candidate,stimuli,output,protocol_sha256):
    jobs=[('writer',writer,Path('transition-serving-stimuli.jsonl')),
          ('transition-serving',transition_serving,Path('transition-serving-stimuli.jsonl')),
          ('review',identity,stimuli),('revision',revision,Path('revision/stimuli.jsonl')),
          ('snac',snac,Path('snac-stimuli.jsonl')),('events',events,Path('event-stimuli.jsonl'))]
    run_phases(model,candidate,output,protocol_sha256,jobs)


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for name in ('model','output','stimuli'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('candidate','protocol-sha256'):p.add_argument('--'+name,required=True)
    run(**vars(p.parse_args()))
