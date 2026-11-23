"""Seal public writer replay stimuli and independent character-accessibility materials for one L4."""
from argparse import ArgumentParser
from pathlib import Path

from research.backbone_transfer.colab import run
from research.backbone_transfer.extract import digest, read_json
from research.reading_validation.identity_colab import DATA, ROOT, CHECKPOINT
from research.reading_validation.memory_colab import dependencies
from research.token_readout.colab import pack_job

PLAN = Path(__file__).with_name('transition-seahorse-protocol.md')


def prepare(replay_only=False):
    src = ROOT/'research/results/reading-validation-transition-external/seahorse'
    source = DATA/'seahorse-writer-stimuli.jsonl'
    native = read_json(src/'serving-replay.json')
    if digest(source) != native['native_packet'] or digest(src/'serving/spec.json') != native['receipt']:
        raise ValueError('Registered summary writer changed')
    states = DATA/'event-reinstatement.jsonl'
    stems = DATA/'reader-history/verified-trial-stimuli.jsonl'
    seal = read_json(ROOT/'research/results/reading-validation-reader-history/verified-input-seal.json')
    if digest(stems) != seal['packet']:
        raise ValueError('Original independent sentence beginnings changed')
    paths = {'stimuli.jsonl':source, 'curve.pt':CHECKPOINT, 'event-reinstatement.jsonl':states, 'reader-stems.jsonl':stems}
    for dirname in ('research/results/reading-validation-local/serving',
                    'research/results/reading-validation-transition-external/seahorse/serving',
                    'research/results/reading-validation-cloze-population/serving'):
        for file in (ROOT/dirname).iterdir():
            if file.is_file():
                paths[str(file.relative_to(ROOT))]=file
    transition='research/results/reading-validation-transition-external/readouts.json'
    paths[transition]=ROOT/transition
    code=dependencies('research/reading_validation/summary_writer_bundle_l4.py')
    if digest(ROOT/'research/reading_validation/writer_summary_l4.py')!='0117ae2b38a38a75d924ed8cbaabb5acbb152740567921804d137539431f9437':
        raise ValueError('Native writer replay changed')
    if replay_only:
        paths.pop('reader-stems.jsonl'); paths.pop('event-reinstatement.jsonl')
    pack_job(DATA/'l4',PLAN,'reading-summary-writer','reading-validation-summary-writer',code,paths,{'study-set.json':{'companion':not replay_only}},1200,
             {'purpose':'Verify the final factual English summary author view; separate character accessibility observation',
              'event_stimuli_sha256':digest(states),'summary_stimuli_sha256':digest(source)})


if __name__ == '__main__':
    p=ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    sub.add_parser('prepare').add_argument('--replay-only',action='store_true');sub.add_parser('run').add_argument('job',type=Path)
    a=p.parse_args()
    if a.action=='prepare':
        prepare(a.replay_only)
    else:
        run(a.job,protocol=PLAN)
