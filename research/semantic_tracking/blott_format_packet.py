"""Certify grade-free Blott 2023 prefix stops for the frozen L4 projector."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.semantic_tracking import beukers_extract as extractor


def prepare(stimuli, tokenizer, output):
    from tokenizers import Tokenizer

    if extractor.digest(tokenizer) != extractor.TOKENIZER_SHA256:
        raise ValueError('Expected the frozen Qwen tokenizer')
    if output.exists():
        raise FileExistsError(output)
    rust = Tokenizer.from_file(str(tokenizer))

    def encode(text):
        value = rust.encode(text, add_special_tokens=False)
        return value.ids, value.offsets

    def encode_prefixes(texts):
        return [value.ids for value in rust.encode_batch(texts, add_special_tokens=False)]

    rows = extractor.read_stimuli(stimuli)
    pending = []
    excluded = Counter()
    for row in rows:
        certified = extractor.certify(row, encode, encode_prefixes)
        stops = [end for end, _ in certified.certified]
        if not stops or stops[-1] != len(row['text']):
            raise ValueError('A Blott sentence is missing the exact whole-sentence endpoint')
        omitted = sorted(set(row['stops']) - set(stops))
        if len(omitted) != sum(certified.rejected.values()):
            raise ValueError('The list of uncertified Blott stops does not reconcile')
        excluded.update(certified.rejected)
        pending.append(({**row, 'stops': stops}, omitted))

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.blott-packet-') as temporary:
        stage = Path(temporary) / 'packet'
        stage.mkdir()
        certified_input = stage / 'stimuli.jsonl'
        with certified_input.open('x', encoding='utf-8') as stream:
            for row, _ in pending:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        summary, filtered, _ = extractor.inventory(certified_input, encode, encode_prefixes)
        extractor.validate_preflight(summary, filtered, extractor.digest(certified_input))
        receipt = {
            'candidate_id': extractor.CANDIDATE,
            'stimuli_sha256': extractor.digest(certified_input),
            'original_stimuli_sha256': extractor.digest(stimuli),
            'documents': {row['id']: row['stops'] for row, _ in pending},
            'excluded_stops': {row['id']: omitted for row, omitted in pending if omitted},
            'exclusion_counts_by_reason': dict(excluded),
        }
        for name, value in [('preflight.json', receipt), ('extractor-preflight.json', summary)]:
            with (stage / name).open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stimuli', 'tokenizer', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    prepare(**vars(parser.parse_args()))
