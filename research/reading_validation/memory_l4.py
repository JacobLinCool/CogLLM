"""Verify sealed serving and recognition features, then fit the last-layer memory adaptation."""

from argparse import ArgumentParser
from pathlib import Path
from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation.memory_train import run as train
from research.reading_validation.recognition_forecast import run as recognition
from research.reading_validation.transition_narrative_verify import run as narrative
from research.reading_validation.writer_narrative_verify import run as writer
from research.reading_validation.revision_forecast import run as comprehension

if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for name in ('model','stimuli','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('candidate','protocol-sha256'):p.add_argument('--'+name,required=True)
    a=p.parse_args();run_phases(a.model,a.candidate,a.output,a.protocol_sha256,[('narrative-serving',narrative,Path('narrative-serving-stimuli.jsonl')),
        ('writer',writer,Path('narrative-serving-stimuli.jsonl')),('recognition',recognition,a.stimuli),('ats-comprehension',comprehension,Path('ats-comprehension/stimuli.jsonl')),('memory',train,Path('memory/cad-train.jsonl'))])
