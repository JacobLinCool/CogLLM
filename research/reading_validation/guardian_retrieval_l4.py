"""Run two registered studies with one Qwen load and separate source/forecast receipts."""
from argparse import ArgumentParser
from pathlib import Path
from research.reading_validation.identity_run_l4 import run_phases
from research.reading_validation.guardian_forecast import run as guardian
from research.reading_validation.relay_summary_l4 import run as summaries
from research.reading_validation.stock_replay import run as stock
from research.reading_validation.meaning_forecast import run as meaning
from research.reading_validation.retrieval_forecast import run as retrieval


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    for n in ('model','stimuli','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('candidate','protocol-sha256'):p.add_argument('--'+n,required=True)
    a=p.parse_args();run_phases(a.model,a.candidate,a.output,a.protocol_sha256,
                               [('guardian',guardian,Path('guardian-stimuli.jsonl')),
                                ('stock-replay',stock,Path('stock-replay.jsonl')),
                                ('relay-summary',summaries,Path('relay-summary-stimuli.jsonl')),
                                ('meaning',meaning,Path('judgment/stimuli.jsonl')),
                                ('retrieval',retrieval,a.stimuli)])
