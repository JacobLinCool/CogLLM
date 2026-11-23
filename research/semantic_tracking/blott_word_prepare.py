"""Build grade-free lexical prefix stops and audit author cues only after packet creation."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.semantic_tracking import blott_format_packet
from research.semantic_tracking.blott_format_prepare import TOKEN


STIMULUS_PINS = {
    '2021': ('02b0537c8ee7d0348d24c97251bce4ac4692f82e34d7c8c14bc56ef9586887f2', 188),
    '2023': ('a191df40dcd3cc77c33b1e5f096fb2778f2fd2ec6a9ee3c134c683b524a2dbf1', 88),
}
DESIGN_PINS = {
    '2021': '6698c694bfae671d47e7b35fd691d3a326ae62f4ba4ff9d670232350fe826372',
    '2023': 'c18e3a2b41d2f873a498909f8ba402cda82bad0670d3842c2e930683bbb7b47e',
}
ARTICLES = {'2021': 'https://doi.org/10.1037/xlm0000936', '2023': 'https://doi.org/10.1080/23273798.2022.2159992'}
PROJECTS = {'2021': 'https://osf.io/hn3bu/', '2023': 'https://osf.io/m87vg/'}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def write_rows(path, values):
    with path.open('x', encoding='utf-8') as stream:
        for row in values:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def grade_free_candidates(stimulus_sources):
    inputs, words, by_source = [], {}, {}
    for study, path in stimulus_sources.items():
        expected_sha, count = STIMULUS_PINS[study]
        if sha256(path) != expected_sha:
            raise ValueError(f'The frozen {study} stimulus file differs from its published project receipt')
        original = rows(path)
        if len(original) != count:
            raise ValueError(f'The frozen {study} stimulus count differs')
        by_source[study] = set()
        for row in original:
            identifier, text = row['id'], row['text']
            id_material = ('blott-2021:' if study == '2021' else '') + text
            if identifier != hashlib.sha256(id_material.encode()).hexdigest() or identifier in words:
                raise ValueError('A row has a repeated ID or disagrees with its frozen literal text')
            spans = [list(match.span()) for match in TOKEN.finditer(text)]
            if not spans:
                raise ValueError('The displayed sentence has no orthographic words')
            words[identifier] = spans
            stops = sorted({span[1] for span in spans} | {len(text)})
            inputs.append({'id': identifier, 'text': text, 'stops': stops})
            by_source[study].add(identifier)
    return inputs, words, by_source


def local_layout(requested, spans, certified):
    for row in requested:
        identifier = row['id']
        stops = set(certified[identifier])
        if not stops <= set(row['stops']):
            raise ValueError('A certificate contains an unrequested source coordinate')
        previous_certified = False
        words = []
        for span in spans[identifier]:
            current_certified = span[1] in stops
            words.append({'span': span, 'certified': current_certified,
                          'update_eligible': previous_certified and current_certified})
            previous_certified = current_certified
        sentence_stop = len(row['text'])
        yield {'id': identifier, 'words': words, 'sentence_stop': sentence_stop,
               'sentence_certified': sentence_stop in stops}


def build_grade_free(stimulus_sources, tokenizer, output):
    if output.exists():
        raise FileExistsError(output)
    requested, spans, by_source = grade_free_candidates(stimulus_sources)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.blott-word-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'build'
        local, packet = stage / 'prepared', stage / 'packet'
        local.mkdir(parents=True)
        write_rows(local / 'stimuli.jsonl', requested)
        blott_format_packet.prepare(local / 'stimuli.jsonl', tokenizer, packet)
        certificate = json.loads((packet / 'preflight.json').read_text())
        certified = certificate['documents']
        layout = list(local_layout(requested, spans, certified))
        write_rows(local / 'layout.jsonl', layout)
        stage.rename(output)
    return by_source


def cue_status(cue, words):
    if cue is None:
        return 'source_cue_undefined'
    for i, row in enumerate(words):
        if row['span'] != cue:
            continue
        if i == 0:
            return 'first_word_has_no_update'
        if row['update_eligible']:
            return 'eligible'
        if not row['certified'] and not words[i - 1]['certified']:
            return 'both_word_ends_uncertified'
        if not row['certified']:
            return 'cue_word_end_uncertified'
        return 'prior_word_end_uncertified'
    return 'not_an_exact_single_word_span'


def audit_sealed_packet(output, design_sources, source_ids):
    prepared, packet = output / 'prepared', output / 'packet'
    layout = {row['id']: row for row in rows(prepared / 'layout.jsonl')}
    preflight = json.loads((packet / 'extractor-preflight.json').read_text())
    source_audit = {}
    for study, ids in source_ids.items():
        design_file = design_sources[study]
        if sha256(design_file) != DESIGN_PINS[study]:
            raise ValueError(f'The frozen {study} cue design file differs from its project receipt')
        design = {row['id']: row for row in rows(design_file)}
        if design.keys() != ids:
            raise ValueError(f'Frozen {study} design and body IDs disagree')
        by_condition = defaultdict(Counter)
        for identifier in ids:
            row = design[identifier]
            condition = row['condition'] if study == '2021' else row['ambiguity']
            by_condition[condition][cue_status(row['cue_span'], layout[identifier]['words'])] += 1
        words = [word for identifier in ids for word in layout[identifier]['words']]
        source_audit[study] = {
            'article': ARTICLES[study], 'public_data': PROJECTS[study],
            'stimuli': len(ids), 'orthographic_words': len(words),
            'certified_word_ends': sum(word['certified'] for word in words),
            'eligible_word_updates': sum(word['update_eligible'] for word in words),
            'author_cue_coverage_by_condition': {key: dict(sorted(value.items()))
                                                for key, value in sorted(by_condition.items())},
        }
    paths = ('prepared/stimuli.jsonl', 'prepared/layout.jsonl', 'packet/stimuli.jsonl',
             'packet/preflight.json', 'packet/extractor-preflight.json')
    return {
        'article_source_audit': source_audit,
        'license': 'https://creativecommons.org/licenses/by/4.0/',
        'word_definition': 'Unicode alphanumeric spans with internal straight or curly apostrophes.',
        'coordinates': 'Zero-based half-open Unicode character spans. Stops are exclusive prefix lengths.',
        'eligible_update': 'Both the word endpoint and the immediately preceding lexical word endpoint are certified. The first word has no prior word. Sentence end alone never fills this role.',
        'evidence_scope': 'Candidate packet created from the unchanged literal text and IDs before reading local author condition or cue positions. The subsequent audit counts syntactic alignment of that stimulus cue, not a localized human report.',
        'frozen_source_sha256': {study: {'stimuli': STIMULUS_PINS[study][0], 'design': DESIGN_PINS[study]}
                                 for study in STIMULUS_PINS},
        'packet_sha256': {name: sha256(output / name) for name in paths},
        'pinned_tokenizer_sha256': preflight['tokenizer_sha256'],
        'certificate': {key: preflight[key] for key in ('input_documents', 'input_tokens', 'max_input_tokens',
                                                        'requested_stops', 'certified_stops')},
        'omitted_offsets': sum(len(value) for value in json.loads((packet / 'preflight.json').read_text())['excluded_stops'].values()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-2021', 'source-2023', 'tokenizer', 'output', 'report'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    source_directories = {'2021': args.source_2021, '2023': args.source_2023}
    ids = build_grade_free({year: path / 'stimuli.jsonl' for year, path in source_directories.items()},
                           args.tokenizer, args.output)
    report = audit_sealed_packet(args.output, {year: path / 'layout.jsonl'
                                               for year, path in source_directories.items()}, ids)
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with (args.output / 'prepared' / 'source-audit.json').open('x', encoding='utf-8') as stream:
        stream.write(content)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x', encoding='utf-8') as stream:
        stream.write(content)
    print(content, end='')


if __name__ == '__main__':
    main()
