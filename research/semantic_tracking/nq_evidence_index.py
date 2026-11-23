"""Select repeated-revision NQ pages disjoint from prior IIRC source and link titles."""

import argparse
from collections import defaultdict
from hashlib import file_digest
from html import unescape
import json
from pathlib import Path
import tarfile
import unicodedata
from urllib.parse import parse_qs, urlsplit

from research.semantic_tracking.iirc_prepare import RELEASE_SHA256 as IIRC_SHA256
from research.semantic_tracking.nq_source_positions import revision


INDEX_SHA256 = '4cedaff9a73694c8e4b4a81c08d59d9b30bb3c51932a2ac8a6141b7481269d5e'
SOURCE_REVISION = 'e8103d566bef4154c2c12b17c6095ec5275840cc'
FIELDS = {'id', 'document_url', 'revision', 'row_index', 'row_in_shard', 'shard', 'source_revision'}


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def normalized_title(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().replace('_', ' ').split())


def iirc_titles(path):
    if digest(path) != IIRC_SHA256:
        raise ValueError('Expected the same IIRC release used in CogLLM experiments')
    sources, links = set(), set()
    with tarfile.open(path, 'r:gz') as archive:
        for split in ('train', 'dev'):
            with archive.extractfile(f'iirc/{split}.json') as source:
                for passage in json.load(source):
                    sources.add(normalized_title(passage['title']))
                    links.update(normalized_title(link['target']) for link in passage['links'])
    return sources, links


def nq_revisions(path):
    if digest(path) != INDEX_SHA256:
        raise ValueError('The Natural Questions index differs from the pinned grade-free release projection')
    groups, title_by_revision, ids = defaultdict(list), {}, set()
    with path.open(encoding='utf-8') as stream:
        for expected, line in enumerate(stream):
            row = json.loads(line)
            url = row['document_url']
            page = revision(url)
            titles = parse_qs(urlsplit(unescape(url)).query).get('title', [])
            if (row.keys() != FIELDS or row['source_revision'] != SOURCE_REVISION or row['revision'] != page
                    or row['row_index'] != expected or row['id'] in ids or len(titles) != 1):
                raise ValueError('Natural Questions index row has inconsistent identity or provenance')
            title = normalized_title(titles[0])
            if not title or title_by_revision.setdefault(page, title) != title:
                raise ValueError('A Natural Questions revision must have a consistent title')
            ids.add(row['id'])
            groups[page].append(row)
    if (len(groups), len(ids)) != (7369, 7830):
        raise ValueError('Unexpected validation inventory')
    return groups, title_by_revision


def select(index, iirc, output, receipt):
    if output.exists() or receipt.exists():
        raise FileExistsError('Write the candidate index and receipt to new destinations')
    sources, links = iirc_titles(iirc)
    groups, titles = nq_revisions(index)
    repeated = {page: rows for page, rows in groups.items() if len(rows) >= 2}
    prior_titles = sources | links
    candidates = [row for page, rows in repeated.items() if titles[page] not in prior_titles for row in rows]
    candidates.sort(key=lambda row: row['row_index'])
    stats = {'source_revision': SOURCE_REVISION, 'input_index_sha256': INDEX_SHA256, 'iirc_sha256': IIRC_SHA256,
             'selection_code_sha256': digest(Path(__file__)), 'validation_records': sum(map(len, groups.values())),
             'validation_revisions': len(groups), 'repeated_revisions': len(repeated),
             'iirc_main_titles': len(sources), 'iirc_hyperlink_titles': len(links),
             'repeated_revisions_matching_iirc_main_title': sum(titles[page] in sources for page in repeated),
             'repeated_revisions_matching_iirc_hyperlink_title': sum(titles[page] in links for page in repeated),
             'candidate_revisions': len({row['revision'] for row in candidates}), 'candidate_records': len(candidates)}
    with output.open('x', encoding='utf-8') as stream:
        for row in candidates:
            stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n')
    stats['candidate_index_sha256'] = digest(output)
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(stats, stream, indent=2, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('index', 'iirc', 'output', 'receipt'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    select(**vars(parser.parse_args()))
