"""Prepare exact, supplied-question ExpMRC comparisons for one causal forward per article."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json, write_rows


ASSAY_SHA256 = '6c5f8689b8ac76ceda4cb655a5bcbe775803775149a685386b40066211049a33'
TOKENIZER_SHA256 = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
MAX_TOKENS = 35_000
MAX_ARTICLE_TOKENS = 2048


def rank(namespace, *parts):
    return sha256('\0'.join(('cogllm-same-question-evidence-v1', namespace, *parts)).encode()).hexdigest()


def table(path):
    result = {}
    for row in stream_rows(path):
        key = row['id']
        if key in result:
            raise ValueError(f'Duplicate identity in {path.name}')
        result[key] = row
    return result


class Geometry:
    def __init__(self, text, tokenizer):
        self.text = text
        self.tokenizer = tokenizer
        encoded = tokenizer.encode(text, add_special_tokens=False)
        self.ids = encoded.ids
        self.ends = [right for _, right in encoded.offsets]
        if (not self.ids or self.ends[-1] != len(text) or self.ends != sorted(self.ends)
                or any(left >= right for left, right in encoded.offsets)):
            raise ValueError('Invalid or incomplete rendered token offsets')
        self.cache = {}

    def stop(self, char_end):
        if char_end not in self.cache:
            count = bisect_right(self.ends, char_end)
            exact = count and self.ends[count - 1] == char_end
            if exact:
                prefix = self.tokenizer.encode(self.text[:char_end], add_special_tokens=False).ids
                exact = prefix == self.ids[:count]
            self.cache[char_end] = count if exact else None
        return self.cache[char_end]

    def span(self, start, end):
        lo, hi = self.stop(start), self.stop(end)
        if lo is None or hi is None or lo >= hi:
            return None
        return {'start': start, 'end': end, 'start_token': lo, 'end_token': hi, 'token_count': hi - lo}

    def evidence_window(self, references, article_start):
        first = min(start for start, _ in references)
        last = max(end for _, end in references)
        index = bisect_right(self.ends, first) - 1
        # A bound may leave whitespace between itself and the original evidence.
        if index < 0 or self.text[self.ends[index]:first].strip():
            return None
        left = self.ends[index]
        index = bisect_left(self.ends, last)
        if index == len(self.ends) or self.text[last:self.ends[index]].strip() or left < article_start:
            return None
        return self.span(left, self.ends[index])

    def equal_width_before(self, end, width, lower):
        hi = self.stop(end)
        if hi is None or hi <= width:
            return None
        left = self.ends[hi - width - 1]
        window = self.span(left, end) if left >= lower else None
        return window if window is not None and window['token_count'] == width else None


def candidate(article, control, earlier, later, tokenizer):
    chosen = sorted((earlier, later), key=lambda row: rank('display-order', article['id'], row['id']))
    header, queries = '', []
    for index, row in enumerate(chosen, start=1):
        text = row['question']
        if not text.strip() or '\n' in text or '\r' in text:
            return None
        header += f'Question {index}: '
        start = len(header)
        header += text
        queries.append({'id': row['id'], 'span': [start, len(header)]})
        header += '\n'
    header += '\n'
    article_start = len(header)
    text = header + article['text']
    geometry = Geometry(text, tokenizer)
    if len(geometry.ids) > MAX_ARTICLE_TOKENS or geometry.stop(len(text)) is None:
        return None
    windows = {}
    for query, row in zip(queries, chosen, strict=True):
        _, end = query['span']
        words_visible = geometry.stop(end)
        if words_visible is None:
            # A punctuation token may also consume the author-independent separator.
            next_end = geometry.ends[bisect_left(geometry.ends, end)]
            if text[end:next_end].strip():
                return None
            words_visible = geometry.stop(next_end)
        if words_visible is None:
            return None
        query['end_token'] = words_visible - 1
        query['references'] = [[article_start + lo, article_start + hi] for lo, hi in row['reference_spans']]
        window = geometry.evidence_window(query['references'], article_start)
        if window is None:
            return None
        windows[row['id']] = window
    first, second = windows[earlier['id']], windows[later['id']]
    if not first['end'] < second['start']:
        return None
    width = first['token_count']
    sham = geometry.equal_width_before(first['start'], width, article_start)
    later_equal = geometry.equal_width_before(second['end'], width, first['end'])
    if later_equal is not None and later_equal['start'] > second['start']:
        later_equal = None
    key = rank('envelope', article['id'], earlier['id'], later['id'])[:32]
    return {'id': key, 'text': text}, {
        'id': key, 'group_id': article['id'], 'split': control['split'], 'article_start': article_start,
        'queries': queries, 'earlier_query': earlier['id'], 'later_query': later['id'],
        'earlier_window': first, 'later_window': second, 'earlier_sham': sham, 'later_equal_window': later_equal,
        'input_tokens': len(geometry.ids),
    }


def prepare(source, tokenizer_path, output):
    from tokenizers import Tokenizer

    if output.exists():
        raise FileExistsError(output)
    if digest(source / 'assay.json') != ASSAY_SHA256 or digest(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError('Expected the frozen ExpMRC source receipt and Qwen3.5 tokenizer')
    source_receipt = read_json(source / 'assay.json')
    if any(digest(source / name) != expected for name, expected in source_receipt['file_sha256'].items()):
        raise ValueError('The prepared ExpMRC source files differ from their sealed digests')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    articles = table(source / 'model_input/stimuli.jsonl')
    layout = table(source / 'layout.jsonl')
    labels = table(source / 'labels.jsonl')
    if articles.keys() != layout.keys():
        raise ValueError('Original split inventory mismatch')
    controls = defaultdict(list)
    for row in stream_rows(source / 'controls.jsonl'):
        key = row['article_id']
        earlier, later = (labels[row[name]] for name in ('earlier_id', 'later_id'))
        if (key != earlier['article_id'] or key != later['article_id']
                or row['window'] != [earlier['consensus_end'], later['first_marked_start']]
                or row['window'][0] >= row['window'][1]):
            raise ValueError('Invalid same-document reference correspondence')
        controls[key].append((earlier, later))
    inputs, pairs = [], []
    inventory = defaultdict(Counter)
    total = 0
    for key, article in articles.items():
        meta = layout[key]
        split = meta['split']
        counts = inventory[split]
        if split not in ('train', 'test') or meta['official_split'] != 'dev' or meta['group_id'] != key:
            raise ValueError('Unexpected original ExpMRC article grouping')
        counts['source_articles'] += 1
        options = sorted(controls[key], key=lambda rows: rank('pair', key, *(row['id'] for row in rows)))
        for earlier, later in options:
            row = candidate(article, meta, earlier, later, tokenizer)
            if row is not None:
                break
            counts['unavailable_candidate_pairs'] += 1
        else:
            counts['no_eligible_pair'] += 1
            continue
        text, pair = row
        if total + pair['input_tokens'] >= MAX_TOKENS:
            raise ValueError('Selected article cohort exceeds the fixed causal token budget')
        total += pair['input_tokens']
        inputs.append(text)
        pairs.append(pair)
        counts['articles_with_one_pair'] += 1
        counts['prefix_body_tokens'] += pair['input_tokens']
        counts['earlier_sham_available'] += pair['earlier_sham'] is not None
        counts['later_equal_width_available'] += pair['later_equal_window'] is not None
        counts['earlier_question_displayed_first'] += pair['queries'][0]['id'] == pair['earlier_query']
    if any(not inventory[split]['articles_with_one_pair'] for split in ('train', 'test')):
        raise ValueError('Both frozen article groups need exact matched evidence pairs')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as directory:
        staged = Path(directory) / 'assay'
        staged.mkdir()
        write_rows(staged / 'inputs.jsonl', inputs)
        write_rows(staged / 'pairs.jsonl', pairs)
        write_json(staged / 'assay.json', {
            'schema': 1, 'prepare_sha256': digest(Path(__file__)), 'source_assay_sha256': ASSAY_SHA256,
            'source_license': 'CC BY-SA 4.0', 'tokenizer_sha256': TOKENIZER_SHA256,
            'file_sha256': {name: digest(staged / name) for name in ('inputs.jsonl', 'pairs.jsonl')},
            'selection': ('Retain original train/test article groups within official dev. One SHA-ordered '
                          'disjoint pair per article, conditional on exact observable geometry only.'),
            'endpoint': ('Retrospective human-marked evidence for a supplied question. At the earlier window, '
                         'the same visible text completes its retained references and precedes the later references.'),
            'sham_scope': ('An equally long preceding token window with no selected marked support. Unannotated '
                           'support could still exist. The separate later window records the later marked support.'),
            'positions': 'Unicode character offsets and zero-based half-open token intervals. Query end_token is an index.',
            'documents': len(inputs), 'input_tokens': total, 'splits': inventory,
        })
        staged.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.tokenizer, args.output)


if __name__ == '__main__':
    main()
