"""Fetch the locked Natural Questions revision sample without using annotations for selection."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
from hashlib import file_digest, sha256
from html import unescape
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from threading import Lock
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlsplit
from urllib.request import Request, urlopen


DATASET = 'google-research-datasets/natural_questions'
SOURCE_REVISION = 'e8103d566bef4154c2c12b17c6095ec5275840cc'
INDEX_SHA256 = '6a4a3921b7fa142eaf34db148e0f93a44fd0da6313a2e9c2b63b8853cad247b5'
SAMPLE_SALT = 'cogllm-nq-v1:'
GROUP_COUNT = 72
EXPECTED_ROWS = 153
MAX_WIRE_RESPONSE = 3_000_000
MAX_JSON_RESPONSE = 12_000_000
MAX_TOTAL_TRANSFER = 70_000_000
WORKERS = 4
INDEX_FIELDS = {'document_url', 'id', 'revision', 'row_in_shard', 'row_index', 'shard', 'source_revision'}
DOCUMENT_FIELDS = {'html', 'title', 'tokens', 'url'}
TOKEN_FIELDS = {'end_byte', 'is_html', 'start_byte', 'token'}
CANDIDATE_FIELDS = {'end_byte', 'end_token', 'start_byte', 'start_token', 'top_level'}
GRADE_FREE_FIELDS = {'id', 'document', 'question', 'long_answer_candidates'}


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def dump_line(stream, row):
    json.dump(row, stream, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    stream.write('\n')


def selected_groups(index):
    if digest(index) != INDEX_SHA256:
        raise ValueError('The grade-free IIRC-disjoint source index differs from the locked revision')
    by_revision, ids, offsets = defaultdict(list), set(), set()
    with index.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            revision = row['revision']
            original_url = urlsplit(unescape(row['document_url']))
            if (row.keys() != INDEX_FIELDS or row['source_revision'] != SOURCE_REVISION or not revision.isdecimal()
                    or parse_qs(original_url.query).get('oldid') != [revision] or original_url.netloc != 'en.wikipedia.org'
                    or row['id'] in ids or row['row_index'] in offsets):
                raise ValueError('The grade-free index has inconsistent revision, record or row identity')
            ids.add(row['id'])
            offsets.add(row['row_index'])
            by_revision[revision].append(row)
    order = sorted(by_revision, key=lambda revision: sha256(f'{SAMPLE_SALT}{revision}'.encode()).digest())[:GROUP_COUNT]
    groups = [sorted(by_revision[revision], key=lambda row: row['row_index']) for revision in order]
    if len(groups) != GROUP_COUNT or any(len(group) < 2 for group in groups) or sum(map(len, groups)) != EXPECTED_ROWS:
        raise ValueError('The fixed sample must contain 72 full revision groups and 153 official rows')
    return order, groups


class TransferBudget:
    def __init__(self, previous=0):
        self.bytes = previous
        self.lock = Lock()

    def record(self, count):
        with self.lock:
            self.bytes += count
            if self.bytes > MAX_TOTAL_TRANSFER:
                raise ValueError('The registered Natural Questions network byte budget was exceeded')


def get_payload(indexed, budget):
    query = urlencode({'dataset': DATASET, 'config': 'default', 'split': 'validation',
                       'offset': indexed['row_index'], 'length': 1})
    request = Request(f'https://datasets-server.huggingface.co/rows?{query}', headers={'Accept-Encoding': 'gzip'})
    with urlopen(request, timeout=15) as response:
        if response.headers.get('X-Revision') != SOURCE_REVISION or response.headers.get('Content-Encoding') != 'gzip':
            raise ValueError('Expected the pinned, gzip-compressed Hugging Face snapshot')
        wire = bytearray()
        while len(wire) <= MAX_WIRE_RESPONSE:
            piece = response.read(min(65536, MAX_WIRE_RESPONSE + 1 - len(wire)))
            if not piece:
                break
            budget.record(len(piece))
            wire.extend(piece)
        if len(wire) > MAX_WIRE_RESPONSE:
            raise ValueError('The gzip-compressed row exceeded its registered byte cap')
    with gzip.GzipFile(fileobj=BytesIO(wire)) as compressed:
        data = compressed.read(MAX_JSON_RESPONSE + 1)
    if len(data) > MAX_JSON_RESPONSE:
        raise ValueError('The decoded Natural Questions row exceeded its registered byte cap')
    return json.loads(data)


def fetch_row(indexed, budget):
    attempts = 12
    for attempt in range(attempts):
        try:
            page = get_payload(indexed, budget)
            break
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            if error.code == 429:
                retry_after = error.headers.get('Retry-After', '')
                delay = max(1, int(retry_after)) if retry_after.isdecimal() else min(20 * (attempt + 1), 60)
                print(f'Hugging Face rate limit at row {indexed["row_index"]}. Retrying in {delay} seconds.', flush=True)
                sleep(delay)
                continue
        except (TimeoutError, URLError):
            if attempt == attempts - 1:
                raise
        sleep(min(2**attempt, 30))
    if page['num_rows_total'] != 7830 or len(page['rows']) != 1:
        raise ValueError('Expected one row from the full official validation split')
    captured, = page['rows']
    row = captured['row']
    document = row['document']
    if (captured['row_idx'] != indexed['row_index'] or row['id'] != indexed['id'] or captured['truncated_cells']
            or row.keys() != GRADE_FREE_FIELDS | {'annotations'} or document.keys() != DOCUMENT_FIELDS
            or document['url'] != indexed['document_url'] or document['tokens'].keys() != TOKEN_FIELDS
            or row['question'].keys() != {'text', 'tokens'} or row['long_answer_candidates'].keys() != CANDIDATE_FIELDS):
        raise ValueError('The exact indexed, untruncated Natural Questions record was not available')
    retained = {'row_idx': captured['row_idx'], **{name: row[name] for name in GRADE_FREE_FIELDS}}
    return retained, {'id': row['id'], 'annotations': row['annotations']}


PAYLOAD_FILES = ('grade-free.jsonl', 'outcomes.jsonl')


def read_checkpoint(path, batch):
    receipt = json.loads((path / 'receipt.json').read_text(encoding='utf-8'))
    if (receipt['rows'] != batch or receipt['file_sha256'] != {name: digest(path / name) for name in PAYLOAD_FILES}
            or type(receipt['wire_bytes']) is not int or receipt['wire_bytes'] < 1):
        raise ValueError(f'Resumable download batch {path.name} does not match the locked row identities and bytes')
    return receipt['wire_bytes']


def write_checkpoint(path, batch, executor, budget):
    initial_bytes = budget.bytes
    with TemporaryDirectory(dir=path.parent, prefix='.batch-tmp-') as temporary:
        stage = Path(temporary) / 'rows'
        stage.mkdir(mode=0o700)
        grade_free, escrow = (stage / name for name in PAYLOAD_FILES)
        with grade_free.open('x', encoding='utf-8') as geometry:
            descriptor = os.open(escrow, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'w', encoding='utf-8') as labels:
                for document, outcome in executor.map(fetch_row, batch, [budget] * len(batch)):
                    dump_line(geometry, document)
                    dump_line(labels, outcome)
        receipt = {'rows': batch, 'wire_bytes': budget.bytes - initial_bytes,
                   'file_sha256': {name: digest(stage / name) for name in PAYLOAD_FILES}}
        (stage / 'receipt.json').write_text(json.dumps(receipt, sort_keys=True) + '\n', encoding='utf-8')
        stage.rename(path)


def fetch(index, output):
    if output.exists():
        raise FileExistsError(output)
    order, groups = selected_groups(index)
    rows = [row for group in groups for row in group]
    print(f'Locked {len(groups)} complete revisions and {len(rows)} rows before retrieval', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoints = output.with_name(f'.{output.name}.checkpoints')
    checkpoints.mkdir(mode=0o700, exist_ok=True)
    batches = [(checkpoints / f'{start:04d}', rows[start:start + WORKERS]) for start in range(0, len(rows), WORKERS)]
    completed = {path: read_checkpoint(path, batch) for path, batch in batches if path.exists()}
    budget = TransferBudget(sum(completed.values()))
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        for number, (path, batch) in enumerate(batches, start=1):
            if path not in completed:
                write_checkpoint(path, batch, executor, budget)
            if number % 6 == 0 or number == len(batches):
                print(f'Sealed {min(number * WORKERS, len(rows))}/{len(rows)} downloaded rows', flush=True)
    with TemporaryDirectory(dir=output.parent, prefix='.nq-holdout-stage-') as temporary:
        stage = Path(temporary) / 'sealed'
        stage.mkdir()
        for name in PAYLOAD_FILES:
            descriptor = os.open(stage / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                                 0o600 if name == 'outcomes.jsonl' else 0o644)
            with os.fdopen(descriptor, 'wb') as stream:
                for path, _ in batches:
                    with (path / name).open('rb') as parts:
                        shutil.copyfileobj(parts, stream)
        manifest = {
            'dataset': DATASET, 'config': 'default', 'split': 'validation', 'source_revision': SOURCE_REVISION,
            'index_sha256': INDEX_SHA256, 'selection': {'salt': SAMPLE_SALT, 'revision_order': order, 'rows': rows},
            'groups': len(groups), 'records': len(rows), 'wire_bytes': budget.bytes,
            'file_sha256': {name: digest(stage / name) for name in PAYLOAD_FILES},
            'fetch_script_sha256': digest(Path(__file__)),
        }
        with (stage / 'manifest.json').open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
        stage.rename(output)
    shutil.rmtree(checkpoints)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    fetch(**vars(parser.parse_args()))
