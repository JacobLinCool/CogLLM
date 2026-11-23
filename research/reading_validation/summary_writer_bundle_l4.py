"""Verify the normal English writer and measure event accessibility using the same causal encoder."""
from argparse import ArgumentParser
from pathlib import Path
from research.backbone_transfer.extract import read_json

from research.reading_validation import dispersion_stems_l4, reinstatement_l4, writer_summary_l4
from research.reading_validation.identity_run_l4 import run_phases


if __name__ == '__main__':
    p = ArgumentParser(description=__doc__)
    for k in ('model','stimuli','output'):
        p.add_argument('--'+k, type=Path, required=True)
    for k in ('candidate','protocol-sha256'):
        p.add_argument('--'+k, required=True)
    a=p.parse_args()
    phases=[('summary-writer', writer_summary_l4.run, a.stimuli)]
    companion=read_json(Path('study-set.json'))['companion']
    if companion:
        phases.extend([('event-reinstatement', reinstatement_l4.run, Path('event-reinstatement.jsonl')),
                       ('dispersion-stems', dispersion_stems_l4.run, Path('reader-stems.jsonl'))])
    run_phases(a.model, a.candidate, a.output, a.protocol_sha256, phases)
