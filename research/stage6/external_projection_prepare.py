"""Bind grade-free GENTLE and Turkish inputs to their fixed source and test receipts."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.semantic_tracking.gum_bridge_extract import TOKENIZER_SHA256, certify, digest


RESULTS = Path(__file__).resolve().parents[1] / 'results'
GENTLE_AUDIT = RESULTS / 'gum-bridge-gentle-audit.json'
GUM_FIT = RESULTS / 'gum-bridge-l4/fit.json'
PROTOCOL = Path(__file__).with_name('external-projection-protocol.md')
STIMULI = 'model_input/stimuli.jsonl'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def prepare(gentle, turkish, tokenizer_path, output):
    from tokenizers import Tokenizer

    upstream = read(GENTLE_AUDIT)
    tr_receipt = turkish / 'assay.json'
    tr_manifest = read(tr_receipt)
    if (digest(tokenizer_path) != TOKENIZER_SHA256
            or digest(gentle / upstream['source_file']) != upstream['source_sha256']
            or any(digest(gentle / name) != sha for name, sha in upstream['derived_sha256'].items())
            or digest(turkish / STIMULI) != tr_manifest['file_sha256'][STIMULI]):
        raise ValueError('The tokenizer or public material differs from the original source audit')
    rust = Tokenizer.from_file(str(tokenizer_path))

    def tokenize(text):
        encoding = rust.encode(text, add_special_tokens=False)
        return encoding.ids, encoding.offsets

    def prefixes(texts):
        return [encoding.ids for encoding in rust.encode_batch(texts, add_special_tokens=False)]

    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, partitions = {}, {}
    gentle_rows = read_rows(gentle / 'articles.jsonl')
    if len(gentle_rows) != upstream['documents']:
        raise ValueError('The GENTLE packet must include every audited article')
    material = {'gentle': gentle_rows, 'turkish': read_rows(turkish / STIMULI)}
    seen_ids, seen_text = set(), set()
    with TemporaryDirectory(dir=output.parent, prefix='.external-projection-') as temporary:
        destination = Path(temporary) / 'assay'
        (destination / 'model_input').mkdir(parents=True)
        with (destination / STIMULI).open('x', encoding='utf-8') as stream:
            for assay, rows in material.items():
                partitions[assay] = []
                counts[assay] = Counter()
                for row in rows:
                    key, text, stops = row['id'], row['text'], row['stops']
                    if key in seen_ids or text in seen_text or stops[-1] != len(text):
                        raise ValueError(f'Each reading must be distinct and request its complete terminal state: {key}')
                    seen_ids.add(key)
                    seen_text.add(text)
                    certified = certify(row, tokenize, prefixes)
                    coordinates = [end for end, _ in certified.certified]
                    if not coordinates or coordinates[-1] != len(text):
                        raise ValueError(f'The complete text has no certified terminal state: {key}')
                    if assay == 'turkish' and (coordinates != [len(text)] or stops != coordinates):
                        raise ValueError('Expected one full terminal stop per Turkish sentence')
                    counts[assay].update(certified.rejected)
                    counts[assay].update(documents=1, tokens=len(certified.tokens), certified_stops=len(coordinates),
                                         original_source_stops=len(stops))
                    partitions[assay].append(key)
                    instance = {'id': key, 'text': text, 'stops': coordinates}
                    stream.write(json.dumps(instance, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')
        audit = {'schema': 1, 'file_sha256': {STIMULI: digest(destination / STIMULI)}, 'counts': counts,
                 'partitions': partitions, 'gentle_audit_sha256': digest(GENTLE_AUDIT),
                 'gentle_material_sha256': upstream['derived_sha256'], 'gum_fit_sha256': digest(GUM_FIT),
                 'turkish_assay_sha256': digest(tr_receipt), 'protocol_sha256': digest(PROTOCOL),
                 'tokenizer_sha256': TOKENIZER_SHA256, 'prepare_sha256': digest(Path(__file__))}
        with (destination / 'assay.json').open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(audit, sort_keys=True, indent=2, allow_nan=False) + '\n')
        destination.rename(output)
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('gentle', 'turkish', 'tokenizer-path', 'output'):
        parser.add_argument(f'--{option}', type=Path, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), sort_keys=True))
