"""Prepare the open ARRAU 2.1 collections for a sealed, single-anchor assay."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

from .gum_bridge_prepare import digest, write_row


REVISION = '764d5e53c10656ff1858503c1d32752b03241174'
SOURCE = f'https://raw.githubusercontent.com/arrauproject/data/{REVISION}'
FILES = {
    'ARRAU2.1_Gnome_Subset2.zip': 'a59b0d8aeb0fc333a1a67fd337470b03db6b6f7add66ea3117ff94788d9c57ea',
    'ARRAU2.1_Pear_Stories.zip': 'd52f4da0480aae43311aa9c1239030b19e02907044931cd064f559f35ee1bc2d',
    'LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
}
COLLECTIONS = {
    'gnome': ('ARRAU2.1_Gnome_Subset2.zip', 'Gnome_Subset2', 'x_'),
    'pear': ('ARRAU2.1_Pear_Stories.zip', 'Pear_Stories', 'pear_'),
}
LAYERS = ('model_input', 'scoring', 'private_metadata')
WORD_RANGE = re.compile(r'word_([1-9][0-9]*)(?:\.\.word_([1-9][0-9]*))?')


@dataclass(frozen=True)
class Document:
    text: str
    token_spans: list[tuple[int, int]]
    sentence_spans: list[tuple[int, int]]
    sentence_by_token: list[int]


def token_spans(attribute, size):
    """Convert closed MMAX word-number ranges to half-open token ranges."""
    parts = []
    for component in attribute.split(','):
        match = WORD_RANGE.fullmatch(component)
        if not match:
            raise ValueError(f'Invalid MMAX span: {attribute}')
        start = int(match[1]) - 1
        end = int(match[2] or match[1])
        if start >= end or end > size or parts and start < parts[-1][1]:
            raise ValueError(f'Unordered or out-of-range MMAX span: {attribute}')
        parts.append((start, end))
    return parts


def document_layout(words, sentences):
    text_chunks, offsets, sentence_offsets, sentence_indices = [], [], [], []
    for index, element in enumerate(words, start=1):
        if (element.tag != 'word' or element.attrib != {'id': f'word_{index}'} or len(element)
                or not element.text or any(character.isspace() for character in element.text)):
            raise ValueError(f'Invalid content for MMAX word_{index}')
    if not len(words) or not len(sentences):
        raise ValueError('Missing annotated words or sentences')
    position = 0
    for sentence_index, sentence in enumerate(sentences):
        ranges = token_spans(sentence.attrib['span'], len(words))
        if len(ranges) != 1 or ranges[0][0] != len(offsets):
            raise ValueError('MMAX sentences must be a contiguous token partition')
        start, end = ranges[0]
        if sentence_index:
            text_chunks.append('\n')
            position += 1
        sentence_start = position
        for index in range(start, end):
            if index > start:
                text_chunks.append(' ')
                position += 1
            word = words[index].text
            text_chunks.append(word)
            offsets.append((position, position + len(word)))
            sentence_indices.append(sentence_index)
            position += len(word)
        sentence_offsets.append((sentence_start, position))
    if len(words) != len(offsets):
        raise ValueError('MMAX sentences do not cover every source word')
    return Document(''.join(text_chunks), offsets, sentence_offsets, sentence_indices)


def character_span(span, document):
    start, end = span
    return document.token_spans[start][0], document.token_spans[end - 1][1]


def containing_sentence(span, document):
    start, end = span
    sentence = document.sentence_by_token[start]
    return sentence if sentence == document.sentence_by_token[end - 1] else None


def resolve_links(elements, document, counts):
    markables = {element.attrib['id']: element for element in elements}
    if len(markables) != len(elements):
        raise ValueError('Duplicate MMAX markable IDs')
    by_trigger = defaultdict(list)
    source_records = []
    for trigger_id, element in markables.items():
        for suffix in ('', '_2'):
            if element.get(f'related_object{suffix}') != 'yes':
                continue
            counts['source_bridge_links'] += 1
            raw_target = element.attrib[f'related_phrase{suffix}']
            anchor_ids = [] if raw_target == 'empty' else raw_target.split(';')
            if any(anchor_id not in markables for anchor_id in anchor_ids) or len(set(anchor_ids)) != len(anchor_ids):
                raise ValueError(f'Unknown or duplicate ARRAU bridge anchor for {trigger_id}')
            trigger = token_spans(element.attrib['span'], len(document.token_spans))
            anchors = [token_spans(markables[anchor].attrib['span'], len(document.token_spans))
                       for anchor in anchor_ids]
            if not anchors:
                status = 'missing_anchor'
            elif len(anchors) > 1:
                status = 'multiple_anchor_markables'
            elif len(trigger) > 1 or len(anchors[0]) > 1:
                status = 'discontinuous_span'
            elif anchors[0][0][1] > trigger[0][0]:
                status = 'anchor_not_wholly_earlier'
            else:
                status = 'eligible_link'
                by_trigger[trigger[0]].append((trigger_id, anchor_ids[0], anchors[0][0]))
            counts[status] += 1
            source_records.append({'trigger_id': trigger_id, 'anchor_ids': anchor_ids,
                                   'trigger_parts': trigger, 'anchor_parts': anchors, 'status': status})
    return by_trigger, source_records


def bridge_rows(identifier, document, by_trigger, counts):
    for trigger, relations in sorted(by_trigger.items()):
        anchors = {anchor for _, _, anchor in relations}
        if len(anchors) != 1:
            counts['trigger_with_competing_anchor_spans'] += 1
            continue
        anchor, = anchors
        trigger_span = character_span(trigger, document)
        trigger_sentence = containing_sentence(trigger, document)
        anchor_sentence = containing_sentence(anchor, document)
        cross_sentence = (trigger_sentence is not None and anchor_sentence is not None
                          and anchor_sentence < trigger_sentence)
        counts['scoring_rows'] += 1
        counts['cross_sentence_rows'] += cross_sentence
        yield {'id': identifier, 'event_stop': trigger_span[1], 'trigger_span': trigger_span,
               'antecedent_spans': [character_span(anchor, document)],
               'trigger_sentence': trigger_sentence, 'antecedent_sentences': [anchor_sentence],
               'cross_sentence': cross_sentence,
               'source_markable_links': sorted({(trigger_id, anchor_id) for trigger_id, anchor_id, _ in relations})}


def prepare_collection(source, output, collection):
    filename, directory, prefix = COLLECTIONS[collection]
    xml_root = f'{directory}/MMAX'
    words_root = f'{xml_root}/Basedata/{prefix}'
    counts = Counter()
    with ExitStack() as stack:
        archive = stack.enter_context(ZipFile(source / filename))
        names = set(archive.namelist())
        outputs = {layer: stack.enter_context((output / layer / f'{collection}.jsonl').open('w', encoding='utf-8'))
                   for layer in LAYERS}
        for file in sorted(path for path in names if path.startswith(words_root) and path.endswith('_words.xml')):
            name = file.removeprefix(words_root).removesuffix('_words.xml')
            identifier = f'arrau21-{collection}-{name}'
            markable_root = f'{xml_root}/markables/{prefix}{name}'
            document = document_layout(ET.fromstring(archive.read(file)),
                                       ET.fromstring(archive.read(f'{markable_root}_sentence_level.xml')))
            raw_base = f'{directory}/RAW/{name}'
            raw_text = archive.read(f'{raw_base}.txt').decode('mac-roman' if name == 'byzantine' else 'utf-8')
            compact_raw, compact_tokens = ''.join(raw_text.split()), ''.join(document.text.split())
            if compact_tokens in compact_raw:
                raise ValueError(f'{identifier} requires a new audit for verbatim paragraph alignment')
            raw_html = f'{raw_base}.html' in names
            stops = [end for _, end in document.token_spans]
            write_row(outputs['model_input'], {'id': identifier, 'text': document.text,
                                               'stops': stops, 'paragraphs': []})
            records = ET.fromstring(archive.read(f'{markable_root}_coref_level.xml'))
            by_trigger, source_records = resolve_links(records, document, counts)
            rows = list(bridge_rows(identifier, document, by_trigger, counts))
            for row in rows:
                write_row(outputs['scoring'], row)
            write_row(outputs['private_metadata'], {'id': identifier, 'source_archive': filename,
                                                    'source_token_file': file, 'raw_text_file': f'{raw_base}.txt',
                                                    'raw_html_file': f'{raw_base}.html' if raw_html else None,
                                                    'token_spans': document.token_spans,
                                                    'sentence_spans': document.sentence_spans,
                                                    'source_links': source_records})
            counts['documents'] += 1
            counts['source_tokens'] += len(stops)
            counts['source_sentences'] += len(document.sentence_spans)
            counts['documents_with_raw_html'] += raw_html
            counts['documents_with_cross_sentence_rows'] += any(row['cross_sentence'] for row in rows)
    if not counts['documents']:
        raise ValueError(f'No documents in {filename}')
    return dict(counts)


def prepare(source, output):
    for name, checksum in FILES.items():
        if digest(source / name) != checksum:
            raise ValueError(f'ARRAU source digest mismatch: {name}')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.arrau-bridge-', dir=output.parent) as temporary:
        destination = Path(temporary) / 'prepared'
        for layer in LAYERS:
            (destination / layer).mkdir(parents=True)
        inventory = {collection: prepare_collection(source, destination, collection) for collection in COLLECTIONS}
        audit = {
            'source': {'release': 'ARRAU 2.1 GNOME Subset 2 and PEAR Stories', 'revision': REVISION,
                       'license': 'Apache-2.0', 'official_repository': 'https://github.com/arrauproject/data',
                       'files': {name: {'url': f'{SOURCE}/{name}', 'sha256': sha} for name, sha in FILES.items()}},
            'collections': inventory,
            'splits': 'Neither public archive supplies a partition. Each output preserves a whole released collection.',
            'coordinates': 'Zero-based Unicode code points, half-open spans, and exclusive prefix stop indices.',
            'text': 'Literal MMAX word strings with single separating spaces and native MMAX sentence line breaks.',
            'paragraphs': 'No certified paragraph layer. RAW text does not reproduce the MMAX token sequence verbatim '
                          'for any document, even after removing whitespace.',
            'scoring': 'Positive expert annotations only. Single, contiguous, strictly earlier annotated anchor '
                       'mention per trigger. All prior MMAX word ends are available as possible anchors. Do not '
                       'interpret other tokens as explicitly judged negatives or expand gold to the coreference chain.',
            'output_sha256': {str(path.relative_to(destination)): digest(path)
                              for path in sorted(destination.glob('*/*.jsonl'))},
        }
        (destination / 'audit.json').write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        destination.rename(output)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.source, args.output)
    print(json.dumps(report['collections'], indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
