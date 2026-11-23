"""Read only IIRC paragraphs and cache their projected causal Qwen token states."""

import argparse
from pathlib import Path

from research.core_features.extract import run as extract
from research.stage3.probe import _jsonl

from .qasper_fit import projection


MAX_PADDED_BATCH_TOKENS = 6144
MAX_BATCH_EXAMPLES = 24


def run(stimuli, output):
    ids = set()
    for row in _jsonl(stimuli):
        if (row.keys() != {'id', 'text', 'stops'} or not isinstance(row['id'], str) or not row['id']
                or not isinstance(row['text'], str) or not row['text'].strip()
                or row['stops'] != [len(row['text'])] or row['id'] in ids):
            raise ValueError('Expected one final character stop per unique plain IIRC paragraph')
        ids.add(row['id'])
    if not ids:
        raise ValueError('The IIRC stimulus file is empty')
    extract(stimuli, output, max_batch_tokens=MAX_PADDED_BATCH_TOKENS, max_batch_examples=MAX_BATCH_EXAMPLES,
            projected_tokens=(ids, projection()), projected_only=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stimuli', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(**vars(parser.parse_args()))
