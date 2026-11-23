"""Certify the independent ARRAU model-input packet without loading Qwen weights."""

import argparse
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.semantic_tracking.arrau_bridge_prepare import COLLECTIONS, REVISION
from research.semantic_tracking.gum_bridge_extract import CANDIDATE, TOKENIZER_SHA256, certify, digest
from research.semantic_tracking.gum_bridge_prepare import jsonl_rows, write_row


def prepare(material, tokenizer_path, output):
    from tokenizers import Tokenizer

    if output.exists():
        raise FileExistsError(output)
    source = material / 'audit.json'
    audit = json.loads(source.read_text(encoding='utf-8'))
    if audit['source']['revision'] != REVISION or digest(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError('Expected the pinned ARRAU release and Qwen tokenizer')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(text):
        row = tokenizer.encode(text, add_special_tokens=False)
        return row.ids, row.offsets

    def prefixes(texts):
        return [row.ids for row in tokenizer.encode_batch(texts, add_special_tokens=False)]

    seen_ids, collection_counts, documents, inputs_sha256 = set(), {}, {}, {}
    with TemporaryDirectory(dir=output.parent, prefix='.arrau-projection-') as scratch:
        destination = Path(scratch) / 'packet'
        destination.mkdir()
        stimuli = destination / 'stimuli.jsonl'
        with stimuli.open('x', encoding='utf-8') as stream:
            for collection in COLLECTIONS:
                counts = Counter()
                name = f'model_input/{collection}.jsonl'
                source_path = material / name
                inputs_sha256[name] = digest(source_path)
                if inputs_sha256[name] != audit['output_sha256'][name]:
                    raise ValueError(f'Input changed after source audit for {collection}')
                for _, row in jsonl_rows(source_path):
                    if (row.keys() != {'id', 'text', 'stops', 'paragraphs'} or row['paragraphs']
                            or row['id'] in seen_ids or row['stops'] != sorted(set(row['stops']))
                            or not row['stops'] or not 0 < row['stops'][0] <= row['stops'][-1] == len(row['text'])):
                        raise ValueError(f'Invalid grade-free ARRAU row in {source_path}')
                    seen_ids.add(row['id'])
                    doc = certify(row, encode, prefixes)
                    stops = [end for end, _ in doc.certified]
                    if len(stops) + sum(doc.rejected.values()) != len(row['stops']):
                        raise ValueError('Source and certified token counts differ')
                    counts.update(documents=1, model_tokens=len(doc.tokens), original_stops=len(row['stops']),
                                  certified_stops=len(stops), extractable_documents=bool(stops))
                    counts.update(doc.rejected)
                    documents[doc.identifier] = {'collection': collection, 'model_tokens': len(doc.tokens),
                                                 'certified_stops': len(stops), 'rejected': dict(doc.rejected)}
                    if stops:
                        write_row(stream, {'id': doc.identifier, 'text': row['text'], 'stops': stops})
                if counts['documents'] != audit['collections'][collection]['documents'] or not counts['certified_stops']:
                    raise ValueError(f'Missing documents or exact stops in {collection}')
                collection_counts[collection] = dict(counts)

        receipt = {'candidate_id': CANDIDATE, 'tokenizer_sha256': TOKENIZER_SHA256,
                   'source_audit_sha256': digest(source), 'builder_sha256': digest(Path(__file__)),
                   'certifier_sha256': digest(Path(certify.__code__.co_filename)),
                   'inputs_sha256': inputs_sha256, 'stimuli_sha256': digest(stimuli),
                   'collections': collection_counts, 'documents': documents}
        with (destination / 'preflight.json').open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
        destination.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('material', 'tokenizer-path', 'output'):
        parser.add_argument(f'--{option}', required=True, type=Path)
    prepare(**vars(parser.parse_args()))
