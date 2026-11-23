"""Run the frozen public Q1 specialist on one owned and budgeted Colab L4."""
from argparse import ArgumentParser
from pathlib import Path

from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest, read_json
from research.reading_validation.identity_colab import DATA, ROOT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PLAN = Path(__file__).with_name('transition-seahorse-specialist.md')


def prepare():
    seal = read_json(ROOT/'research/results/reading-validation-transition-external/seahorse/specialist-input.json')
    source = DATA / 'seahorse-specialist-stimuli.jsonl'
    if digest(PLAN) != seal['protocol'] or digest(source) != seal['source']:
        raise ValueError('Specialist stimulus or protocol changed')
    pack_job(DATA/'l4', PLAN, 'reading-summary-specialist', 'reading-validation-summary-specialist',
             dependencies('research/reading_validation/seahorse_comparator.py'), {'stimuli.jsonl':source}, {}, 2400,
             {'purpose':'Registered external published specialist. Zero parameter updates and full test coverage.'},
             candidate='seahorse', model_spec={'repo':'google/seahorse-large-q1','revision':'a39a2c372923817c2a857b7953cb10827ec0e810'})


if __name__ == '__main__':
    parser=ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    commands.add_parser('prepare')
    commands.add_parser('run').add_argument('job',type=Path)
    args=parser.parse_args()
    if args.action=='prepare':
        prepare()
    else:
        run(args.job,protocol=PLAN)
