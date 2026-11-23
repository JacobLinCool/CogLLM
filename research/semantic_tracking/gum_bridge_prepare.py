"""Reconstruct pinned UD English-GUM documents and keep Bridge outcomes separate."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata
from urllib.parse import urlsplit


REVISION = 'e92f501b037ebd38387d7479b99aa2a0c45b779e'
SOURCE_URL = f'https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/{REVISION}'
SPLITS = ('train', 'dev', 'test')
TABLES = ('model_input', 'scoring', 'private_metadata')
SOURCE_HASHES = {
    'en_gum-ud-train.conllu': '4672bc446b6f516877b023384d2e33952bb012aca052506b197c92ddba67b78e',
    'en_gum-ud-dev.conllu': 'ed955aa38fe62d3c2670701ef92dd577608a9a10c5ee452e6990dc3d1fe3c7ce',
    'en_gum-ud-test.conllu': '40854b004a1475412c8cdd0ba9536d77e4876467238d8e035718dd33bd29c1e3',
    'README.md': '50a06b28291872657dd7e9b5ba925acd78562700c68681085b46007c08d174ec',
    'LICENSE.txt': 'b6310e2df084d7cf04d592a748a1449439768afe8c68d49a4248237db646915f',
}
BLOCK = re.compile(r'([a-z]+)(?: .+)? \((\d+) s\)')
ENTITY_PART = re.compile(r'\((\d+)-[^()]+(\))?|(\d+)\)')
BRIDGE = re.compile(r'(\d+)<(\d+):([a-z-]+(?:;[a-z-]+)*)')
ENTITY_FORMAT = 'GRP-etype-infstat-salience-centering-minspan-link-identity'


@dataclass(frozen=True)
class Sentence:
    identifier: str
    text: str
    new_block: bool
    blocks: tuple[tuple[str, int], ...]
    tokens: tuple[tuple[str, str, str], ...]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_documents(path):
    """Yield the original documents with their sentence text and block metadata."""
    document_id = None
    sentences = []
    metadata = {}
    comments = {}
    tokens = []

    def finish_sentence():
        if not tokens:
            if comments:
                raise ValueError(f'Unexpected comments without tokens in {path.name}')
            return
        text = comments['text']
        if not text or text.strip() != text:
            raise ValueError(f'Invalid original sentence text in {comments["sent_id"]}')
        block_entries = comments.get('newpar_block', '')
        descriptors = []
        if block_entries:
            for entry in block_entries.split(' | '):
                match = BLOCK.fullmatch(entry)
                if not match:
                    raise ValueError(f'Unknown TEI block format: {entry}')
                descriptors.append((match[1], int(match[2])))
        if descriptors and 'newpar' not in comments:
            raise ValueError('Expected UD newpar marker for a documented block')
        sentences.append(Sentence(comments['sent_id'], text, 'newpar' in comments,
                                  tuple(descriptors), tuple(tokens)))
        comments.clear()
        tokens.clear()

    with path.open(encoding='utf-8') as stream:
        for line in stream:
            line = line.rstrip('\n')
            if not line:
                finish_sentence()
            elif line.startswith('# '):
                key, _, value = line[2:].partition(' = ')
                if key == 'newdoc id':
                    if comments or tokens or (document_id is not None and not sentences):
                        raise ValueError('Repeated document marker or document marker inside a sentence')
                    if document_id is not None:
                        yield document_id, metadata, sentences
                    document_id, metadata, sentences = value, {}, []
                elif key == 'global.Entity':
                    if value != ENTITY_FORMAT:
                        raise ValueError(f'Unsupported entity format: {value}')
                    metadata[key] = value
                elif key.startswith('meta::'):
                    metadata[key[6:]] = value
                elif key in {'sent_id', 'text', 'newpar', 'newpar_block'}:
                    if key in comments:
                        raise ValueError(f'Repeated document structure field: {key}')
                    comments[key] = value
            else:
                fields = line.split('\t')
                if len(fields) != 10:
                    raise ValueError('CoNLL-U token lines must have ten columns')
                tokens.append((fields[0], fields[1], fields[9]))
    finish_sentence()
    if not document_id or not sentences:
        raise ValueError('No final GUM document found')
    yield document_id, metadata, sentences


def text_layout(sentences, block_counts):
    paragraph_sentences = []
    for i, sentence in enumerate(sentences):
        for tag, length in sentence.blocks:
            block_counts[tag] += 1
            if i + length > len(sentences):
                raise ValueError('TEI block extends beyond its document')
            if tag == 'p':
                if length < 1:
                    raise ValueError('Declared GUM prose paragraphs must contain a sentence')
                if paragraph_sentences and i < paragraph_sentences[-1][1]:
                    raise ValueError('Declared GUM prose paragraphs overlap')
                paragraph_sentences.append((i, i + length))
    boundaries = {i for span in paragraph_sentences for i in span}
    chunks = []
    starts = []
    ends = []
    position = 0
    for i, sentence in enumerate(sentences):
        if i:
            separator = '\n\n' if i in boundaries else '\n' if sentence.new_block else ' '
            chunks.append(separator)
            position += len(separator)
        starts.append(position)
        chunks.append(sentence.text)
        position += len(sentence.text)
        ends.append(position)
    paragraphs = [(starts[left], ends[right - 1]) for left, right in paragraph_sentences]
    return ''.join(chunks), starts, paragraphs


def attributes(raw):
    return dict(item.split('=', 1) for item in raw.split('|') if '=' in item)


def align_tokens(sentence, document_offset, stats):
    """Align orthographic FORM and require exact surface composition for MWT components."""
    cursor = 0
    mwt = None
    next_token = 1
    for identifier, form, raw_misc in sentence.tokens:
        if '.' in identifier:
            if {'Bridge', 'Entity'} & attributes(raw_misc).keys():
                raise ValueError('An empty CoNLL-U node carries a reference annotation')
            stats['empty_nodes'] += 1
            continue
        if '-' in identifier:
            first, last = map(int, identifier.split('-'))
            if first != next_token or first >= last or mwt is not None:
                raise ValueError('Invalid CoNLL-U multiword token range')
            if {'Bridge', 'Entity'} & attributes(raw_misc).keys():
                raise ValueError('A CoNLL-U range line carries an unexpected reference annotation')
            start = cursor
            while start < len(sentence.text) and sentence.text[start].isspace():
                start += 1
            if not form or not sentence.text.startswith(form, start):
                raise ValueError(f'Cannot align multiword FORM in {sentence.identifier}: {form!r}')
            mwt = (last, start + len(form))
            cursor = start
            stats['multiword_surface_ranges'] += 1
            continue
        if int(identifier) != next_token:
            raise ValueError(f'Unexpected token order in {sentence.identifier}')
        if mwt is None:
            while cursor < len(sentence.text) and sentence.text[cursor].isspace():
                cursor += 1
        start, end = cursor, cursor + len(form)
        if not form or not sentence.text.startswith(form, start):
            raise ValueError(f'Cannot align FORM in {sentence.identifier}: {form!r}')
        if mwt is not None and (end > mwt[1] or (next_token == mwt[0] and end != mwt[1])):
            raise ValueError(f'Multiword FORM does not match its components in {sentence.identifier}')
        if mwt is not None and next_token == mwt[0]:
            mwt = None
        stats['source_tokens'] += 1
        yield (document_offset + start, document_offset + end), attributes(raw_misc)
        cursor = end
        next_token += 1
    if mwt is not None or cursor != len(sentence.text):
        raise ValueError(f'Incomplete alignment in {sentence.identifier}')


def references(sentences, sentence_starts, stats):
    mentions = defaultdict(list)
    openings = defaultdict(list)
    edges = []
    offsets = []
    for sentence, offset in zip(sentences, sentence_starts, strict=True):
        for (left, right), misc in align_tokens(sentence, offset, stats):
            offsets.append((left, right))
            entity = misc.get('Entity', '')
            consumed = 0
            for match in ENTITY_PART.finditer(entity):
                if match.start() != consumed:
                    raise ValueError(f'Unexpected Entity syntax in {sentence.identifier}: {entity}')
                opening, singleton, closing = match.groups()
                if singleton:
                    mentions[opening].append((left, right))
                elif opening:
                    openings[opening].append(left)
                else:
                    if not openings[closing]:
                        raise ValueError(f'Unmatched entity closure in {sentence.identifier}')
                    mentions[closing].append((openings[closing].pop(), right))
                consumed = match.end()
            if consumed != len(entity):
                raise ValueError(f'Incomplete Entity annotation in {sentence.identifier}: {entity}')
            if 'Bridge' in misc:
                for value in misc['Bridge'].split(','):
                    match = BRIDGE.fullmatch(value)
                    if not match:
                        raise ValueError(f'Unexpected Bridge syntax in {sentence.identifier}: {value}')
                    edges.append((left, match[1], match[2], tuple(match[3].split(';'))))
    if any(openings.values()):
        raise ValueError('Unclosed entity at document end')
    return offsets, mentions, edges


def paragraph_index(span, paragraphs, starts):
    index = bisect_right(starts, span[0]) - 1
    return index if index >= 0 and span[1] <= paragraphs[index][1] else None


def outcomes(identifier, edges, mentions, paragraphs, stats):
    groups = defaultdict(list)
    for start, antecedent_id, trigger_id, subtypes in edges:
        groups[start, trigger_id].append((antecedent_id, subtypes))
    stats['source_bridge_edges'] += len(edges)
    stats['source_bridge_triggers'] += len(groups)
    starts = [span[0] for span in paragraphs]
    annotations = defaultdict(lambda: {'trigger_ids': set(), 'bridge_edges': set(), 'antecedent_spans': set()})
    for (start, trigger_id), relations in groups.items():
        triggers = sorted({span for span in mentions[trigger_id] if span[0] == start})
        if len(triggers) != 1:
            stats['trigger_without_unique_same_start_entity_span'] += 1
            continue
        trigger = triggers[0]
        referents = [set(span for span in mentions[source] if span[1] <= start) for source, _ in relations]
        if not all(referents):
            stats['trigger_with_an_edge_without_earlier_entity_mention'] += 1
            continue
        annotation = annotations[trigger]
        annotation['trigger_ids'].add(trigger_id)
        for (source, subtypes), referent_spans in zip(relations, referents, strict=True):
            annotation['bridge_edges'].add((source, trigger_id, subtypes))
            annotation['antecedent_spans'].update(referent_spans)

    for trigger, annotation in sorted(annotations.items()):
        antecedents = sorted(annotation['antecedent_spans'])
        target_paragraph = paragraph_index(trigger, paragraphs, starts)
        source_paragraphs = [paragraph_index(span, paragraphs, starts) for span in antecedents]
        earlier = [par for par in source_paragraphs if target_paragraph is not None and par is not None and par < target_paragraph]
        stats['retained_annotation_rows'] += 1
        stats['rows_with_earlier_paragraph_gold'] += bool(earlier)
        stats['rows_with_at_least_two_earlier_gold_paragraphs'] += len(set(earlier)) >= 2
        stats['rows_with_earlier_paragraph_gold_and_no_same_paragraph_gold'] += bool(earlier) and target_paragraph not in source_paragraphs
        stats['rows_with_only_earlier_paragraph_gold'] += bool(earlier) and len(earlier) == len(source_paragraphs)
        yield {
            'id': identifier, 'event_stop': trigger[1], 'trigger_span': trigger, 'antecedent_spans': antecedents,
            'trigger_paragraph': target_paragraph, 'antecedent_paragraphs': source_paragraphs,
            'trigger_ids': sorted(annotation['trigger_ids']),
            'bridge_edges': [{'antecedent_id': source, 'trigger_id': target, 'subtypes': subtypes}
                             for source, target, subtypes in sorted(annotation['bridge_edges'])],
        }


def write_row(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def source_family(url):
    parsed = urlsplit(url)
    return parsed._replace(fragment='').geturl() if parsed.netloc else None


def prepare_split(source, split, model_path, scoring_path, metadata_path):
    stats, blocks = Counter(), Counter()
    keys = {'document_id': set(), 'normalized_text': set(), 'source_url_without_fragment': set()}
    with (model_path.open('w', encoding='utf-8', newline='\n') as model,
          scoring_path.open('w', encoding='utf-8', newline='\n') as scoring,
          metadata_path.open('w', encoding='utf-8', newline='\n') as metadata_stream):
        for identifier, metadata, sentences in read_documents(source):
            if identifier in keys['document_id'] or metadata.get('global.Entity') != ENTITY_FORMAT:
                raise ValueError(f'Duplicate document or missing entity schema: {identifier}')
            keys['document_id'].add(identifier)
            text, starts, paragraphs = text_layout(sentences, blocks)
            offsets, mentions, edges = references(sentences, starts, stats)
            stops = [end for _, end in offsets]
            if not stops or stops != sorted(set(stops)) or stops[-1] != len(text):
                raise ValueError(f'Invalid source token stops for {identifier}')
            stats['documents'] += 1
            stats['sentences'] += len(sentences)
            stats['true_paragraphs'] += len(paragraphs)
            stats['documents_with_at_least_two_true_paragraphs'] += len(paragraphs) >= 2
            keys['normalized_text'].add(' '.join(unicodedata.normalize('NFKC', text).casefold().split()))
            source_url = metadata.get('sourceURL', '')
            if family := source_family(source_url):
                keys['source_url_without_fragment'].add(family)
            write_row(model, {'id': identifier, 'text': text, 'stops': stops, 'paragraphs': paragraphs})
            write_row(metadata_stream, {'id': identifier, 'split': split, 'genre': metadata.get('genre'),
                                        'author': metadata.get('author'), 'source_url': source_url,
                                        'token_spans': offsets})
            for row in outcomes(identifier, edges, mentions, paragraphs, stats):
                write_row(scoring, row)
    return {**stats, 'source_tei_blocks_by_tag': dict(sorted(blocks.items()))}, keys


def prepare(source, output):
    for name, checksum in SOURCE_HASHES.items():
        if digest(source / name) != checksum:
            raise ValueError(f'GUM v2.19 source hash mismatch: {name}')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.gum-bridge-intake-', dir=output.parent) as temporary:
        destination = Path(temporary) / 'prepared'
        for directory in TABLES:
            (destination / directory).mkdir(parents=True)
        reports = {}
        families = {}
        for split in SPLITS:
            reports[split], families[split] = prepare_split(source / f'en_gum-ud-{split}.conllu', split,
                destination / 'model_input' / f'{split}.jsonl', destination / 'scoring' / f'{split}.jsonl',
                destination / 'private_metadata' / f'{split}.jsonl')
        overlaps = {
            f'{left}/{right}': {key: len(values & families[right][key]) for key, values in families[left].items()}
            for i, left in enumerate(SPLITS) for right in SPLITS[i + 1:]
        }
        if any(cross['document_id'] or cross['normalized_text'] for cross in overlaps.values()):
            raise ValueError(f'Official source document or exact text crosses splits: {overlaps}')
        audit = {
            'source': {'release': 'UD English GUM r2.19', 'commit': REVISION,
                       'license_name': 'CC BY-NC-SA 4.0',
                       'license': f'{SOURCE_URL}/LICENSE.txt', 'readme': f'{SOURCE_URL}/README.md',
                       'paper': 'https://aclanthology.org/2026.lrec-1.543.pdf#page=4',
                       'files': {name: {'url': f'{SOURCE_URL}/{name}', 'sha256': sha}
                                 for name, sha in SOURCE_HASHES.items()}},
            'text': 'Verbatim CoNLL-U # text sentence strings, with deterministic spaces and structural line breaks.',
            'paragraphs': 'Only nonoverlapping, sentence-spanning TEI p blocks declared in newpar_block. '
                          'Other blocks remain in the article but do not certify a true paragraph.',
            'stops': 'All source syntactic token ends in increasing Unicode code-point coordinates (0 based, end exclusive). '
                     'Prepared without consulting Bridge or Entity annotations.',
            'endpoint': 'Published retrospective expert bridging annotations, placed at the trigger span end by assay '
                        'convention. Earlier mentions of the annotated antecedent entity are alternate valid spans.',
            'negative_labels': 'The intake has positive annotated bridges only. Do not interpret exclusions or '
                               'unannotated tokens as realtime human assessments.',
            'splits': reports, 'cross_split_overlaps': overlaps,
            'output_sha256': {str(path.relative_to(destination)): digest(path)
                              for path in sorted(destination.glob('*/*.jsonl'))},
        }
        (destination / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                                                encoding='utf-8')
        destination.rename(output)
    return audit


def jsonl_rows(path):
    with path.open('rb') as stream:
        for line in stream:
            yield line, json.loads(line)


def freeze(prepared, output):
    """Keep official test intact while separating the earlier splits by source URL."""
    intake_path = prepared / 'audit.json'
    intake = json.loads(intake_path.read_text(encoding='utf-8'))
    inputs = {f'{table}/{split}.jsonl': digest(prepared / table / f'{split}.jsonl')
              for table in TABLES for split in SPLITS}
    if intake['source']['commit'] != REVISION or inputs != intake['output_sha256']:
        raise ValueError('Prepared inputs do not match the pinned, audited GUM intake')
    metadata = {
        split: {row['id']: source_family(row['source_url'])
                for _, row in jsonl_rows(prepared / 'private_metadata' / f'{split}.jsonl')}
        for split in SPLITS
    }
    families = {split: set(documents.values()) - {None} for split, documents in metadata.items()}
    exclude = {
        'train': families['dev'] | families['test'],
        'dev': families['test'],
        'test': set(),
    }
    keep = {split: {identifier for identifier, family in documents.items() if family not in exclude[split]}
            for split, documents in metadata.items()}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.gum-bridge-freeze-', dir=output.parent) as temporary:
        destination = Path(temporary) / 'assay'
        for table in TABLES:
            (destination / table).mkdir(parents=True)
        splits = {}
        output_hashes = {}
        for split in SPLITS:
            splits[split] = {
                'original_documents': len(metadata[split]),
                'retained_documents': len(keep[split]),
                'excluded_document_ids': sorted(metadata[split].keys() - keep[split]),
                'unavailable_source_url_documents': sum(family is None for family in metadata[split].values()),
                'retained_rows': {},
            }
            for table in TABLES:
                name = f'{table}/{split}.jsonl'
                retained = 0
                with (destination / name).open('wb') as stream:
                    for line, row in jsonl_rows(prepared / name):
                        if row['id'] not in metadata[split]:
                            raise ValueError(f'Unknown {split} document in {table}: {row["id"]}')
                        if row['id'] in keep[split]:
                            stream.write(line)
                            retained += 1
                if table != 'scoring' and retained != len(keep[split]):
                    raise ValueError(f'Expected one {table} row per retained document')
                splits[split]['retained_rows'][table] = retained
                output_hashes[name] = digest(destination / name)
                if split == 'test' and output_hashes[name] != inputs[name]:
                    raise ValueError('Official test inputs must be byte-identical')
        test_rows = [row for _, row in jsonl_rows(destination / 'scoring' / 'test.jsonl')
                     if row['trigger_paragraph'] is not None and all(
                         index is not None and index < row['trigger_paragraph']
                         for index in row['antecedent_paragraphs'])]
        audit = {
            'upstream_intake_audit_sha256': digest(intake_path),
            'source_revision': REVISION,
            'family_key': 'Published source URL with fragment removed. Missing URL does not define a family.',
            'exclusion_rules': {'train': 'Exclude URLs in original official dev or test.',
                                'dev': 'Exclude URLs in official test.', 'test': 'Preserve official test.'},
            'splits': splits,
            'official_test_literal_cross_only': {
                'rows': len(test_rows), 'distinct_stops': len({(row['id'], row['event_stop']) for row in test_rows}),
                'documents': len({row['id'] for row in test_rows}),
                'rule': 'Trigger and all retained gold spans belong to literal paragraphs. All gold paragraphs precede '
                        'the trigger paragraph. No minimum count of earlier paragraphs is imposed.',
            },
            'input_sha256': inputs, 'output_sha256': output_hashes,
        }
        (destination / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                                                encoding='utf-8')
        destination.rename(output)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--source', type=Path, help='Prepare all official splits from fixed CoNLL-U files')
    inputs.add_argument('--freeze', type=Path, help='Freeze source-disjoint splits from an audited prepared directory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.freeze is None:
        prepare(args.source, args.output)
    else:
        freeze(args.freeze, args.output)


if __name__ == '__main__':
    main()
