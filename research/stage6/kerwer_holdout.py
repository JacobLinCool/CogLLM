"""Prepare the public Kerwer article materials and private scoring records."""

import argparse
import csv
from hashlib import file_digest, sha256
from html.entities import html5
import json
from pathlib import Path
import re
import unicodedata
from xml.etree import ElementTree

from research.stage2.records import write_jsonl


PAPER = 'https://doi.org/10.1525/collabra.18898'
OSF = 'https://doi.org/10.17605/OSF.IO/A9QSY'
DOI_PREFIX = '10.5964/jspp.'
STUDIES = (
    ('v5i1.642', 4989, '00030d755ca6c49a60f5388dba95ff23e6a3cad11c22cae1abb71effff28109a'),
    ('v5i2.713', 5013, 'd2425e9b07f7ceaf24c456079eead5f99f7fe154fcb0536b06c8e79cf8ddbcfb'),
    ('v6i1.741', 5043, 'baa0bcea458a4cd38b3fce127723f80715ef4d42c4da1225494a2d4494610fe4'),
    ('v5i2.733', 5011, 'dbb082b5efd624b8e14d6f6e1715adf36e9787a023831273780fbc3efe110ac3'),
    ('v6i1.601', 5037, 'bc64d9bbe3419e55b28389dc62d84e0e1006910d69d6d4f9609305f2f5a023e5'),
    ('v6i2.859', 5065, '2aa45a488f16e5921686df4411231674e723f9020ec7a7853bc82f999afd9962'),
    ('v7i1.1063', 5163, 'd404ca572828369fa69710bc6c3d1337947c78e02dbc584f5d8bcb4ee94b3822'),
    ('v7i1.557', 5151, 'dc7b418184ffb0179fdcca15d196b0fed3b9a46f6c49136a1103f56686a9cc14'),
    ('v7i1.746', 5129, '19e58dd0f157590e13b763daab05ff3d64ed31ee508cee38e8b97be519a2cf7c'),
    ('v7i1.822', 5127, '04f77b7a81022ef1fe75c62871f3ca9ec7d89bcc75a9c3e01ef55b12f00a8ebb'),
    ('v7i1.933', 5153, '5100d49709d67ee9226d9e47e3eac0fb38b1a9394bfbc0eaed908a123a5ae25d'),
    ('v7i1.981', 5159, '2da0117aef1387ae41c710b2ae74bd545767af5b2deec48e7670423002167862'),
)
ARCHIVE = {
    'ratings.csv': ('b34fj', 'd4f4387a1ffb98060a6d066547d2095da90cea175d87f565b1114b31a2852782'),
    'codebook.csv': ('9nqsd', '79692bc963fbce3e2d2c6a363490d90f6b49854d86bf7806f05f5f39dc6d06b7'),
    'participants.csv': ('csqyv', '62f73e77306f3b4081c711af04e06903d022447f294363f75cbdefabc9135934'),
    'participants-codebook.csv': ('q4chk', '43d5c7991e6a60c3f3d68c273c58e076aed8993ba558a4972e5bf841089299a9'),
    'license.txt': ('tc7e9', 'ebd500c63f7fb056ca2e94388840f1878eea8f03111d2e5ec97af426ed629be1'),
}
ARMS = {'Non PLS': 'scientific_abstract', 'No Heading': 'plain_without_headings', 'PLS': 'plain_with_headings'}
HEADINGS = ('Background', 'Why was this study done?', 'What did the researchers do and find?', 'What do these findings mean?')
ENDPOINTS = {'vst': ('comprehensibility', 8), 'con': ('confusion', 5), 'frust': ('frustration', 5)}
XML_ENTITIES = {'amp;', 'apos;', 'gt;', 'lt;', 'quot;'}


def digest(path):
    with path.open('rb') as source:
        return file_digest(source, 'sha256').hexdigest()


def check_file(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Upstream data changed: {path.name}')


def element_text(element):
    return ' '.join(''.join(element.itertext()).split())


def entity(match):
    name = match[1] + ';'
    if name in XML_ENTITIES:
        return match[0]
    return ''.join(f'&#{ord(character)};' for character in html5[name])


def article(raw, suffix, article_id, checksum):
    path = raw / f'{suffix}.xml'
    check_file(path, checksum)
    source = re.sub(r'&([A-Za-z][A-Za-z0-9]+);', entity, path.read_text(encoding='utf-8'))
    meta = ElementTree.fromstring(source).find('front/article-meta')
    doi = DOI_PREFIX + suffix
    if meta.findtext("article-id[@pub-id-type='doi']") != doi:
        raise ValueError(f'Publisher DOI differs from the study key: {doi}')
    ordinary = next(entry for entry in meta.findall('abstract') if 'abstract-type' not in entry.attrib)
    plain = meta.find("abstract[@abstract-type='non-technical']")
    sections = plain.findall('sec')
    headings = tuple(element_text(section.find('title')) for section in sections)
    if (ordinary.attrib or plain.get('{http://www.w3.org/XML/1998/namespace}lang') != 'en'
            or tuple(re.sub(r'^\d+\. ', '', heading) for heading in headings) != HEADINGS):
        raise ValueError(f'Unexpected publisher abstract structure: {doi}')
    ordinary_paragraphs = [element_text(p) for p in ordinary.findall('p')]
    plain_paragraphs = [[element_text(p) for p in section.findall('p')] for section in sections]
    if not ordinary_paragraphs or any(not paragraphs for paragraphs in plain_paragraphs):
        raise ValueError(f'Missing body text: {doi}')
    license_url = 'https://creativecommons.org/licenses/by/' + ('4.0/' if suffix.startswith('v7') else '3.0/')
    if license_url.rstrip('/') not in (raw / f'{suffix}.html').read_text(encoding='utf-8'):
        raise ValueError(f'Publisher page does not verify the expected CC BY license: {doi}')
    publication = {
        'doi': doi,
        'publisher_xml': f'https://jspp.psychopen.eu/index.php/jspp/article/view/{article_id}/{article_id}.xml',
        'xml_sha256': checksum,
        'license': license_url,
        'title': element_text(meta.find('title-group/article-title')),
        'authors': [' '.join((name.findtext('given-names', ''), name.findtext('surname', ''))).strip()
                    for name in meta.findall("contrib-group/contrib[@contrib-type='author']/name")],
    }
    variants = {
        'scientific_abstract': [(part, True) for part in ordinary_paragraphs],
        'plain_without_headings': [(part, True) for group in plain_paragraphs for part in group],
        'plain_with_headings': [(part, index > 0) for heading, group in zip(headings, plain_paragraphs, strict=True)
                                for index, part in enumerate([heading, *group])],
    }
    return publication, variants


def make_input(case, blocks):
    text = ''
    body = []
    for block, is_body in blocks:
        if text:
            text += '\n\n'
        start = len(text)
        text += block
        if is_body:
            body.append([start, len(text)])
    return {'id': case, 'text': text, 'body_spans': body}


def normalized_words(text):
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text).casefold())


def fivegrams(words):
    return set(zip(words, words[1:], words[2:], words[3:], words[4:]))


def overlap_max(inputs, prior_files):
    current = [fivegrams(normalized_words(item['text'])) for item in inputs]
    result = {}
    for path in prior_files:
        maximum = 0.0
        count = 0
        with path.open(encoding='utf-8') as source:
            for line in source:
                prior = fivegrams(normalized_words(json.loads(line)['text']))
                if prior:
                    maximum = max(maximum, max(len(prior & candidate) / len(candidate) for candidate in current))
                count += 1
        if not count or maximum >= 0.25:
            raise ValueError(f'Unusable distinct-source check at {path}: {count} records, overlap {maximum:.3f}')
        result[str(path)] = {'rows': count, 'max_candidate_fivegram_coverage': maximum, 'sha256': digest(path)}
    return result


def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        return list(csv.DictReader(source))


def scoring_records(raw, cases):
    for filename, (_, expected) in ARCHIVE.items():
        check_file(raw / filename, expected)
    book = {row['name']: row for row in csv_rows(raw / 'codebook.csv')}
    source_map = re.findall(r'\{(\d+)=(10\.5964/jspp\.[^}]+)\}', book['study']['values'])
    if source_map != [(str(index), DOI_PREFIX + item[0]) for index, item in enumerate(STUDIES, 1)]:
        raise ValueError('Codebook DOI order differs from Table 1')
    participants = {row['id']: row for row in csv_rows(raw / 'participants.csv')}
    outcomes = []
    observed = set()
    for row in csv_rows(raw / 'ratings.csv'):
        reader = row['id']
        source_key = (reader, row['study'])
        participant = participants[reader]
        if (source_key in observed or row['Englisch'] != participant['Englisch']
                or row['vws_wiss'] != participant['vws_wiss']):
            raise ValueError('Participant-by-study keys or prereading English proficiency do not match')
        observed.add(source_key)
        grades = {label: int(row[column]) for column, (label, _) in ENDPOINTS.items()}
        if set(grades.values()) == {-999}:
            continue
        if any(not 1 <= grades[label] <= maximum for label, maximum in ENDPOINTS.values()):
            raise ValueError('Invalid ratings or discordant missingness in source outcomes')
        outcomes.append({'id': cases[row['study'], row['PLS']], 'reader': reader, **grades})
    if (len(participants), len(observed), len(outcomes)) != (166, 1992, 1947):
        raise ValueError('Published sample size or outcome exclusions changed')
    profiles = [{'reader': reader, 'english_e_pa': int(row['Englisch_unst']),
                 'research_familiarity': int(row['vws_wiss'])} for reader, row in sorted(participants.items())]
    return outcomes, profiles


def prepare(raw, output, prior_files):
    studies = []
    inputs, design, cases = [], [], {}
    for index, source in enumerate(STUDIES, 1):
        study, variants = article(raw, *source)
        studies.append(study)
        for label, arm in ARMS.items():
            case = sha256(f'{study["doi"]}/{arm}'.encode()).hexdigest()[:16]
            cases[str(index), label] = case
            design.append({'id': case, 'study': str(index), 'doi': study['doi'], 'arm': arm})
            inputs.append(make_input(case, variants[arm]))
    inputs.sort(key=lambda item: item['id'])
    overlap = overlap_max(inputs, prior_files)
    outcomes, profiles = scoring_records(raw, cases)
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / 'model_input'
    model_dir.mkdir()
    scoring_dir = output / 'scoring'
    scoring_dir.mkdir(mode=0o700)
    write_jsonl(model_dir / 'stimuli.jsonl', inputs)
    write_jsonl(scoring_dir / 'design.jsonl', design)
    write_jsonl(scoring_dir / 'public-study-ratings.jsonl', outcomes)
    write_jsonl(scoring_dir / 'prereading-profiles.jsonl', profiles)
    manifest = {
        'paper': PAPER, 'osf': OSF, 'articles': studies,
        'material_fidelity': 'Reconstructed publication bodies and heading deletion; session export unavailable.',
        'prior_corpora': overlap,
        'counts': {'families': len(studies), 'variants': len(inputs), 'readers': len(profiles), 'outcomes': len(outcomes)},
        'archive': {name: {'url': f'https://osf.io/download/{key}/', 'sha256': checksum}
                    for name, (key, checksum) in ARCHIVE.items()},
        'sha256': {str(path.relative_to(output)): digest(path) for path in output.glob('*/*.jsonl')},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest['counts']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path, required=True, nargs='+')
    args = parser.parse_args()
    prepare(args.raw, args.output, args.compare)
