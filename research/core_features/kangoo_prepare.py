"""Audit uniquely attributable lexical difficulty marks from Kangoo's v4 data."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unicodedata
from xml.etree import ElementTree
from zipfile import ZipFile

from research.core_features.cwi_news_prepare import read_complex
from research.semantic_tracking.iirc_anchor_fit import WORD
from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3.run import save_json


SOURCE = 'https://data.mendeley.com/datasets/p2wrs7hm4z/4'
DONOR = 'https://egyankosh.ac.in/index.jsp'
CSV = 'dataset_english.csv'
CSV_SHA256 = 'c3d31a4075c150277d754177a5b9848985e33ca7a712101c8a7e423e7bafcd37'
CSV_COLUMNS = ['', 'word', 'fre', 'len', 'ps', 'difficult_ug', 'difficult_pg']
NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
FORMS = {
    '1': (72, 'afc6375d698ace5e1ae1cd394bc43cc7a951b018770dde1ecc5bff98942d32e8'),
    '2': (304, 'bc231d54528224e659704703595ad10e5d3df61709e8fa2dd22a159a2bc298dd'),
    '3': (60, '7d12e0056366e2fae991a43f7226cd80e3784a0e171257805edef11f4cba440e'),
    '4': (103, '08edf80d17142aa143ac445ec5c97a7cdf2887270526b85b836b14ff8fd95045'),
    '5': (122, '1d071478375f2d9d477edd81351d1ce8b04dd81932dfe73725dc6e598844f138'),
    '6': (106, 'f0f90943a511c9c407df7ca084598ea8e630a92afff7781ee8de49d3e5b94eee'),
    'PG1': (52, 'f4d503b478446f21d21b0ed6151fa5b93d1889475ac9136de45a15a57bae2111'),
    'PG2': (54, 'af523fdec41950a248532d552718c8c5e4d7f29083d0a16c8df1255acba89777'),
    'PG3': (37, 'f2bd7b67223c49504726bfcb371d62f5d15bc3eb578fe65ab886c5bad2886ee5'),
    'PG4': (50, '015f74bfa9415874ff45ab8ec52217eabebd04b081842e363225324bb2321f8f'),
}


@dataclass(frozen=True)
class Location:
    form: str
    page: int
    start: int
    end: int


def normalized(text):
    return unicodedata.normalize('NFKC', text).casefold()


def read_form(path):
    with ZipFile(path) as source:
        root = ElementTree.fromstring(source.read('word/document.xml'))
    body = root.find(NS + 'body')
    pages = [[]]
    pagebreaks = Counter()
    unsupported = {'altChunk', 'del', 'drawing', 'endnoteReference', 'fldChar',
                   'footnoteReference', 'ins', 'instrText', 'noBreakHyphen', 'object',
                   'softHyphen', 'sym', 'tbl', 'txbxContent'}
    if body is None or any(node.tag in {NS + tag for tag in unsupported} for node in body.iter()):
        raise ValueError('Unexpected text-bearing Word structure')
    for paragraph in body:
        if paragraph.tag == NS + 'sectPr':
            continue
        if paragraph.tag != NS + 'p':
            raise ValueError('Expected literal body paragraphs')
        parts = []
        for element in paragraph.iter():
            if element.tag == NS + 'lastRenderedPageBreak':
                if parts or not pages[-1]:
                    raise ValueError('Stored page split is not at a noninitial paragraph boundary')
                pages.append([])
                pagebreaks['rendered'] += 1
            elif element.tag == NS + 't':
                parts.append(element.text or '')
            elif element.tag == NS + 'tab':
                parts.append('\t')
            elif element.tag == NS + 'br':
                if element.get(NS + 'type') != 'page':
                    raise ValueError('Unexpected Word line break in fixed documents')
                pagebreaks['explicit'] += 1
        text = ''.join(parts)
        if text.strip():
            pages[-1].append(text)
    if len(pages) != 10 or pagebreaks['rendered'] != 9 or not all(pages):
        raise ValueError('Expected ten text pages per questionnaire')
    return tuple('\n'.join(parts) for parts in pages), pagebreaks


def parse_count(value):
    value = value.strip()
    if value in {'', '-'}:
        return None
    if not value.isascii() or not value.isdigit():
        raise ValueError('Expected a published whole-number count or an explicit absence marker')
    return int(value)


def read_counts(path):
    with path.open(encoding='utf-8', newline='') as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != CSV_COLUMNS:
            raise ValueError('Unexpected Mendeley v4 CSV field names')
        rows = []
        for index, row in enumerate(reader):
            if row[''] != str(index) or not row['word'] or not row['fre'].isascii() or not row['fre'].isdigit():
                raise ValueError('Expected unique release row numbers and a positive reported frequency')
            frequency = int(row['fre'])
            if frequency < 1:
                raise ValueError('Expected a positive reported frequency')
            rows.append((normalized(row['word'].strip()), frequency, parse_count(row['difficult_ug']),
                         parse_count(row['difficult_pg'])))
    if len(rows) != 5368:
        raise ValueError('Unexpected number of word-type rows')
    return rows


def locate(rows, pages, frequencies):
    observed = defaultdict(list)
    for (form, page), text in pages.items():
        for match in WORD.finditer(text):
            observed[normalized(match.group())].append(Location(form, page, *match.span()))
    distinct_rows = Counter(word for word, *_ in rows)
    exclusions = Counter()
    candidates = defaultdict(list)
    for word, count, undergrad, postgrad in rows:
        if count != 1:
            exclusions['source_frequency_not_one'] += 1
            continue
        if distinct_rows[word] != 1:
            exclusions['ambiguous_csv_normalized_form'] += 1
            continue
        if len(observed[word]) != 1:
            exclusions['no_unique_literal_whole_word'] += 1
            continue
        location, = observed[word]
        form = location.form
        applicable, opposite = (postgrad, undergrad) if form.startswith('PG') else (undergrad, postgrad)
        if applicable is None:
            exclusions['own_cohort_unobserved'] += 1
            continue
        if opposite is not None:
            exclusions['contradictory_opposite_cohort_count'] += 1
            continue
        if applicable > FORMS[form][0]:
            exclusions['count_above_cohort_enrollment'] += 1
            continue
        candidates[form, location.page].append({
            'start': location.start, 'end': location.end, 'selections': applicable,
            'form_count': frequencies[word],
        })
    if sum(map(len, candidates.values())) + exclusions.total() != len(rows):
        raise ValueError('Every published word-type row must be accounted for')
    return candidates, exclusions, sum(map(len, observed.values()))


def prepare(raw, complex_stimuli, output):
    if output.exists():
        raise FileExistsError(output)
    source_hashes = {CSV: CSV_SHA256, **{f'questionnaire-{form}.docx': sha for form, (_, sha) in FORMS.items()}}
    if any(digest(raw / name) != sha for name, sha in source_hashes.items()):
        raise ValueError('Expected the exact Mendeley v4 source files')
    _, frequencies = read_complex(complex_stimuli)
    pages, pagination = {}, {}
    for form in FORMS:
        texts, pagination[form] = read_form(raw / f'questionnaire-{form}.docx')
        pages.update(((form, page), text) for page, text in enumerate(texts, 1))
    rows = read_counts(raw / CSV)
    candidates, exclusions, source_word_tokens = locate(rows, pages, frequencies)
    stimuli, outcomes = [], []
    inventory = defaultdict(Counter)
    for (form, page), text in pages.items():
        identifier = 'kangoo-v4-' + hashlib.sha256(text.encode('utf-8')).hexdigest()
        words = sorted(candidates[form, page], key=lambda row: row['start'])
        results = Counter(row['selections'] for row in words)
        inventory[form].update({
            'pages': 1, 'candidate_words': len(words),
            'words_with_a_mark': len(words) - results[0],
            'pages_with_orderable_pairs': int(len(results) > 1),
            'unequal_count_pairs': sum(n * (len(words) - n) for n in results.values()) // 2,
        })
        stimuli.append({'id': identifier, 'text': text})
        outcomes.append({'id': identifier, 'questionnaire': form, 'page': page,
                         'cohort_enrollment': FORMS[form][0], 'words': words})
    if len({row['id'] for row in stimuli}) != len(stimuli):
        raise ValueError('Unexpected duplicate literal questionnaire pages')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.kangoo-v4-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'prepared'
        stage.mkdir()
        write_jsonl(stage / 'stimuli.jsonl', stimuli)
        write_jsonl(stage / 'outcomes.jsonl', outcomes)
        audit = {
            'schema': 1, 'source': SOURCE, 'source_file_sha256': source_hashes,
            'preparation_code_sha256': digest(Path(__file__)),
            'stimuli_sha256': digest(stage / 'stimuli.jsonl'),
            'outcomes_sha256': digest(stage / 'outcomes.jsonl'),
            'frequency_source_sha256': digest(complex_stimuli), 'frequency_source_splits': ['train', 'trial'],
            'word_type_rows': len(rows), 'questionnaire_pages': len(stimuli),
            'cohort_enrollment': {form: count for form, (count, _) in FORMS.items()},
            'source_word_tokens': source_word_tokens, 'excluded_word_type_rows': dict(sorted(exclusions.items())),
            'inventory_by_questionnaire': dict(inventory), 'word_document_pagination': pagination,
            'outcome_scope': 'Word mark count, with occurrence and exposure verified under the singleton rule.',
            'count_denominator_scope': 'Published cohort enrollment; completion counts by page are unavailable.',
            'mode_scope': 'Paper underlining and online word checklists are pooled in the source.',
            'release_license': 'CC BY 4.0 for the Mendeley record',
            'prose_rights': {'resource_policy': DONOR, 'exact_passage_reuse_verified': False,
                             'model_transfer_authorized': False},
        }
        save_json(stage / 'source-audit.json', audit)
        stage.rename(output)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--complex-stimuli', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.raw, args.complex_stimuli, args.output)


if __name__ == '__main__':
    main()
