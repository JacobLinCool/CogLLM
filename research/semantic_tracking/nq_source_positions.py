"""Map Natural Questions source tokens and byte intervals to the visible text view."""

from bisect import bisect_left
from dataclasses import dataclass
from html import unescape
from urllib.parse import parse_qs, urlsplit


TOKEN_COLUMNS = {'token', 'is_html', 'start_byte', 'end_byte'}
CANDIDATE_COLUMNS = {'start_token', 'end_token', 'start_byte', 'end_byte', 'top_level'}


def revision(url):
    parsed = urlsplit(unescape(url))
    ids = parse_qs(parsed.query).get('oldid', [])
    if (parsed.scheme != 'https' or parsed.hostname != 'en.wikipedia.org'
            or len(ids) != 1 or not ids[0].isascii() or not ids[0].isdigit()):
        raise ValueError('Expected a single archived revision of an English Wikipedia page')
    return ids[0]


@dataclass(frozen=True)
class TextView:
    text: str
    source_positions: tuple[int, ...]
    character_spans: tuple[tuple[int, int], ...]
    byte_spans: tuple[tuple[int, int], ...]

    def support(self, start_token, end_token, start_byte, end_byte):
        if (type(start_token) is not int or type(end_token) is not int or type(start_byte) is not int
                or type(end_byte) is not int or not 0 <= start_token < end_token <= len(self.byte_spans)
                or (start_byte, end_byte) != (self.byte_spans[start_token][0], self.byte_spans[end_token - 1][1])):
            raise ValueError('Natural Questions support byte and source token intervals disagree')
        left = bisect_left(self.source_positions, start_token)
        right = bisect_left(self.source_positions, end_token)
        if left == right:
            return None
        return self.character_spans[left][0], self.character_spans[right - 1][1]


def build_text(document):
    tokens = document['tokens']
    if tokens.keys() != TOKEN_COLUMNS:
        raise ValueError('Unknown Natural Questions document token layout')
    source_bytes = len(document['html'].encode('utf-8'))
    positions, character_spans, byte_spans, words = [], [], [], []
    offset, previous_end = 0, 0
    for position, (token, is_html, start, end) in enumerate(zip(
            tokens['token'], tokens['is_html'], tokens['start_byte'], tokens['end_byte'], strict=True)):
        if (type(token) is not str or type(is_html) is not bool or type(start) is not int or type(end) is not int
                or not previous_end <= start < end <= source_bytes):
            raise ValueError('Natural Questions token coordinates are missing, overlapping or out of bounds')
        byte_spans.append((start, end))
        previous_end = end
        if is_html:
            continue
        if not token or any(character.isspace() for character in token):
            raise ValueError('Expected a single nonempty visible token in the upstream text view')
        positions.append(position)
        character_spans.append((offset, offset + len(token)))
        words.append(token)
        offset += len(token) + 1
    if not words:
        raise ValueError('No visible tokens in Natural Questions document')
    return TextView(' '.join(words), tuple(positions), tuple(character_spans), tuple(byte_spans))


def candidate_spans(record, view):
    columns = record['long_answer_candidates']
    if columns.keys() != CANDIDATE_COLUMNS:
        raise ValueError('Unknown Natural Questions candidate structure')
    spans = []
    for start_token, end_token, start_byte, end_byte, top_level in zip(
            columns['start_token'], columns['end_token'], columns['start_byte'], columns['end_byte'],
            columns['top_level'], strict=True):
        if type(top_level) is not bool:
            raise ValueError('Expected the authors’ candidate hierarchy indicator')
        spans.append(view.support(start_token, end_token, start_byte, end_byte))
    return tuple(spans)
