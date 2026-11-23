"""Extract bounded causal Qwen states and segment means for QASPER."""

import argparse
from pathlib import Path

from research.core_features.extract import run
from research.stage3.probe import _jsonl


MAX_PADDED_BATCH_TOKENS = 6144


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stimuli', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dense-tokens', action='store_true', help='Also store projected states for all paper tokens')
    args = parser.parse_args()
    projected_tokens = None
    if args.dense_tokens:
        from .qasper_fit import projection

        paper_ids = {row['id'] for row in _jsonl(args.stimuli) if row['id'].startswith('paper-')}
        projected_tokens = paper_ids, projection()
    run(args.stimuli, args.output, max_batch_tokens=MAX_PADDED_BATCH_TOKENS, segment_means=True,
        projected_tokens=projected_tokens)
