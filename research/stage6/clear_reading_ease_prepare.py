"""Seal CLEAR passage families, original stops, and separate teacher outcomes."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata
from urllib.parse import urlsplit

from openpyxl import load_workbook
from tokenizers import Tokenizer

from research.gradio_app.causal_plaba import digest, sentence_position
from research.gradio_app.checkpoint import DEFAULT_CHECKPOINT, load_checkpoint
from research.gradio_app.combined import paragraphs
from research.gradio_app.plaba import TOKENIZER_SHA256, sentence_spans
from research.stage2.records import write_jsonl
from research.stage6.turkish_sentence_prepare import connected_components


WORKBOOK_SHA256 = 'e87a19a29c12a441e0c4f11a4e49897ff24ac7dff240ee7397283a050a902cc7'
PROTOCOL = Path(__file__).with_name('clear-reading-ease-protocol.md')
LICENSES = frozenset({'CC BY 3.0', 'CC BY 4.0', 'CC BY-SA 3.0'})
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def words(text):
    return tuple(WORD.findall(unicodedata.normalize('NFKC', text).casefold()))


def shingles(sequence, size):
    return {' '.join(sequence[i:i + size]) for i in range(len(sequence) - size + 1)}


def workbook_rows(path):
    if digest(path) != WORKBOOK_SHA256:
        raise ValueError('Expected the pinned author CLEAR workbook')
    with path.open('rb') as stream:
        workbook = load_workbook(stream, read_only=True, data_only=True)
        try:
            rows = workbook['Data'].values
            header = next(rows)
            required = {'ID', 'Excerpt', 'License', 'Author', 'Title', 'URL',
                        'Categ', 'BT_easiness', 's.e.'}
            if len(set(header)) != len(header) or not required <= set(header):
                raise ValueError('Unexpected CLEAR schema')
            result = [dict(zip(header, row, strict=True)) for row in rows]
            if len(result) != 4724 or not all(type(row['ID']) is int for row in result):
                raise ValueError('Unexpected CLEAR row inventory or ID type')
            if len({row['ID'] for row in result}) != len(result):
                raise ValueError('Duplicate CLEAR ID')
            for row in result:
                row['ID'] = str(row['ID'])
            return result
        finally:
            workbook.close()


def normalized_url(value):
    parsed = urlsplit(value or '')
    if (parsed.scheme not in {'http', 'https'} or not parsed.netloc
            or (not parsed.path.strip('/') and not parsed.query)):
        return None
    return parsed.netloc.casefold().removeprefix('www.'), parsed.path.rstrip('/'), parsed.query


def material_rows(rows):
    known = set()
    materials = {}
    exclusions = Counter()
    for row in sorted(rows, key=lambda item: item['ID']):
        if row['License'] not in LICENSES:
            exclusions['excluded_excerpt_license'] += 1
            continue
        key = row['ID']
        text = row['Excerpt']
        sequence = words(text)
        url = normalized_url(row['URL'])
        author = words(row['Author'] or '')
        title = words(row['Title'] or '')
        if not (url or (author and title)):
            exclusions['no_source_key'] += 1
            continue
        if sequence in known:
            exclusions['duplicate_text'] += 1
            continue
        known.add(sequence)
        materials[key] = (row, sequence, url, (author, title) if author and title else None)
    return materials, exclusions


def prior_overlap(materials, paths):
    index = defaultdict(set)
    for key, (_, sequence, _, _) in materials.items():
        for window in shingles(sequence, 12):
            index[window].add(key)
    strong = set()
    short = set()
    inventory = []
    for path in paths:
        n_rows = 0
        matched = set()
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                sequence = words(json.loads(line)['text'])
                n_rows += 1
                matches = {key for window in shingles(sequence, 12) for key in index.get(window, ())}
                if matches:
                    windows = shingles(sequence, 30)
                    for key in matches:
                        if windows & shingles(materials[key][1], 30):
                            strong.add(key)
                        else:
                            short.add(key)
                    matched.update(matches)
        if n_rows == 0:
            raise ValueError(f'No overlap references in {path}')
        inventory.append({'path': path.as_posix(), 'sha256': digest(path), 'documents': n_rows,
                          'excerpts_with_12_word_overlap': len(matched)})
    return strong, short - strong, inventory


def groups(materials):
    edges = {key: set() for key in materials}
    first = {}
    for key, (_, sequence, url, author_title) in materials.items():
        identifiers = [('text_overlap', window) for window in shingles(sequence, 30)]
        if url:
            identifiers.append(('url', url))
        if author_title:
            identifiers.append(('author_title', author_title))
        for identity in identifiers:
            neighbor = first.setdefault(identity, key)
            if neighbor != key:
                edges[key].add(neighbor)
                edges[neighbor].add(key)
    return {key: min(members) for members in connected_components(edges).values() for key in members}


def certified_spans(text, tokenizer, segmenter):
    spans = [(start + left, start + right) for start, end, _ in paragraphs(text)
             for left, right in sentence_spans(text[start:end], segmenter)]
    if not spans:
        return None
    encoding = tokenizer.encode(text, add_special_tokens=False)
    tokens = []
    for start, end in spans:
        try:
            position = sentence_position(text, {'start': start, 'end': end, 'id': f'{start}:{end}'}, encoding.offsets)
        except ValueError:
            return None
        # App sentence boundaries have already stripped trailing whitespace.
        prefix = tokenizer.encode(text[:end], add_special_tokens=False).ids
        if encoding.ids[:position + 1] != prefix:
            return None
        tokens.append(position)
    return spans, tokens, len(encoding.ids)


def prepare(workbook, prior, checkpoint, tokenizer_path, output):
    if output.exists():
        raise FileExistsError(output)
    if digest(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError('Expected the frozen Gradio tokenizer')
    materials, exclusions = material_rows(workbook_rows(workbook))
    strong, short, overlap_inventory = prior_overlap(materials, prior)
    exclusions['previous_30_word_overlap'] = len(strong)
    families = groups(materials)
    candidate = load_checkpoint(checkpoint)
    segmenter = candidate.segmenter()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    stimuli, metadata, outcomes = [], [], []
    for key, (row, _, _, _) in materials.items():
        if key in strong:
            continue
        text = row['Excerpt']
        aligned = certified_spans(text, tokenizer, segmenter)
        if aligned is None:
            exclusions['uncertified_sentence'] += 1
            continue
        spans, tokens, token_count = aligned
        target, error = row['BT_easiness'], row['s.e.']
        if not (type(target) in (float, int) and isfinite(target) and type(error) in (float, int)
                and isfinite(error) and error >= 0 and row['Categ'] in ('Lit', 'Info')):
            raise ValueError(f'Invalid published rating or genre for excerpt {key}')
        family = families[key]
        split = 'holdout' if int(sha256(f'clear-ease-v1:{family}'.encode()).hexdigest()[:8], 16) % 5 < 2 else 'calibration'
        identifier = sha256(text.encode()).hexdigest()
        stimuli.append({'id': identifier, 'text': text, 'stops': [end for _, end in spans]})
        metadata.append({'id': identifier, 'family': family, 'excerpt_id': key, 'split': split,
                         'license': row['License'], 'genre': row['Categ'], 'spans': spans,
                         'sentence_tokens': tokens, 'input_tokens': token_count, 'soft_overlap': key in short})
        outcomes.append({'id': identifier, 'bt_easiness': target, 'standard_error': error})
    if len(stimuli) + sum(exclusions.values()) != 4724:
        raise ValueError('CLEAR material accounting is incomplete')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.clear-preparation-') as temporary:
        stage = Path(temporary) / 'prepared'
        stage.mkdir()
        parts = {'stimuli.jsonl': stimuli, 'materials.jsonl': metadata, 'outcomes.jsonl': outcomes}
        for name, records in parts.items():
            write_jsonl(stage / name, records)
        details = {}
        for split in ('calibration', 'holdout'):
            selected = [row for row in metadata if row['split'] == split]
            details[split] = {'passages': len(selected), 'families': len({row['family'] for row in selected}),
                              'input_tokens': sum(row['input_tokens'] for row in selected),
                              'stops': sum(len(row['spans']) for row in selected),
                              'by_genre': dict(Counter(row['genre'] for row in selected))}
        (stage / 'assay.json').write_text(json.dumps({
            'candidate_id': candidate.identifier, 'workbook_sha256': WORKBOOK_SHA256,
            'tokenizer_sha256': TOKENIZER_SHA256, 'protocol_sha256': digest(PROTOCOL),
            'file_sha256': {name: digest(stage / name) for name in parts},
            'max_input_tokens': max(row['input_tokens'] for row in metadata),
            'overlap_references': overlap_inventory, 'exclusions': dict(exclusions), 'split': details,
        }, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', required=True, type=Path)
    parser.add_argument('--prior', required=True, type=Path, nargs='+')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--tokenizer', required=True, type=Path, dest='tokenizer_path')
    parser.add_argument('--output', required=True, type=Path)
    prepare(**vars(parser.parse_args()))
