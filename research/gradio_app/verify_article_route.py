"""Verify the long-article route, combined overview, and cached Gradio views."""

import argparse
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
import json
from math import floor, isfinite
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urljoin


PARAGRAPHS = (
    'The 🌧 field team visited a small watershed after three days of rain. They wanted to understand why the '
    'stream became cloudy even after the wind had settled. At the first bridge, they took samples from the '
    'surface and from the water near the bed, where fine sediment can remain suspended. Each bottle had the '
    'time, the depth, and the location written on its label. A sign at the nearby cafe\u0301 directed them '
    'to a dry shelter where they could sort the bottles before carrying them back to the laboratory. '
    'The team kept a separate log that could connect every reading to the bottle from which it came.',
    'At the laboratory, the first step was to let the sealed bottles reach room temperature. The technicians '
    'measured cloudiness by shining a beam through each sample and recording how much light scattered. They '
    'repeated the reading after gently turning each bottle. The notebook used the label <sample> & the original '
    'station code so that the notes could be compared with the export from the instrument. Some samples came '
    'from water only 10\u00a0cm deep. Those samples were collected away from submerged branches because the '
    'water there could carry small fragments of bark. A second technician checked all of the labels before '
    'the group entered the measurements in a shared table.',
    'The group then compared the dates when the samples had been collected. The darkest water was found at '
    'the bridge below a gravel road, but the difference was smaller in samples taken near the surface. A '
    'shallow ditch next to the road drained through a pipe into the stream. The team walked as far as the '
    'outlet to see whether the ditch could explain the pattern. They photographed the pipe and marked the '
    'distance to the bridge. The notes retained the original station names, including São Bento and the '
    'field sign marked नदी. The station names made it possible to match records even when the order of the '
    'measurements changed between exports.',
    'The final inspection began with a new sample at the downstream footbridge. The stream had been clear '
    'for a week, and the readings from the surface and bed were nearly the same. The team moved the equipment '
    'back to the ditch to repeat the comparison during the next storm. They drew a simple map of the proposed '
    'stations and recorded which access points were safe after rain. The next report will include the new '
    'measurements together with the original observations. The final map identifies the downstream lantern '
    'post so that another team can find the same sampling location.',
)
ARTICLE = '  ' + '\n\n'.join(PARAGRAPHS) + '\n  '


def text_digest(text):
    return sha256(text.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class Mark:
    kind: str
    attrs: dict[str, str | None]
    start: int
    end: int
    text: str
    parent_start: int | None


@dataclass(frozen=True)
class OpenMark:
    kind: str
    attrs: dict[str, str | None]
    start: int
    chunk_start: int
    parent_start: int | None


class SourceMarkup(HTMLParser):
    """Record each mark's measured position in the decoded, rendered source."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.source_count = 0
        self.in_source = False
        self.chunks = []
        self.length = 0
        self.marks = []
        self.active = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'div' and 'source-text' in (attributes.get('class') or '').split():
            if self.source_count or len(attrs) != len(attributes):
                raise AssertionError('Expected exactly one rendered source block')
            self.source_count += 1
            self.in_source = True
        elif self.in_source:
            if tag != 'mark' or len(attrs) != len(attributes):
                raise AssertionError(f'Unexpected markup inside article: {tag}')
            kinds = {'sentence', 'lexical'}.intersection((attributes.get('class') or '').split())
            if len(kinds) != 1:
                raise AssertionError('Each source mark must identify exactly one heatmap layer')
            kind, = kinds
            parent = self.active[-1] if self.active else None
            if parent is not None and (parent.kind, kind) != ('sentence', 'lexical'):
                raise AssertionError('Only a lexical mark can be nested inside a sentence mark')
            self.active.append(OpenMark(kind, attributes, self.length, len(self.chunks), parent.start if parent else None))

    def handle_endtag(self, tag):
        if not self.in_source:
            return
        if tag == 'mark' and self.active:
            mark = self.active.pop()
            if mark.start == self.length:
                raise AssertionError('Source marks must contain a nonempty span')
            self.marks.append(Mark(mark.kind, mark.attrs, mark.start, self.length,
                                   ''.join(self.chunks[mark.chunk_start:]), mark.parent_start))
        elif tag == 'div' and not self.active:
            self.in_source = False
        else:
            raise AssertionError(f'Unexpected closing tag inside article: {tag}')

    def handle_data(self, data):
        if self.in_source:
            self.chunks.append(data)
            self.length += len(data)


def verify_layer(marks, kind, expected, offsets):
    if len(marks) != len(expected) or not expected:
        raise AssertionError(f'{kind} heatmap omitted or added a certified mark')
    paragraphs = [(ARTICLE.index(text), ARTICLE.index(text) + len(text)) for text in PARAGRAPHS]
    counts = [0] * len(paragraphs)
    direction = 1 if kind == 'sentence' else -1
    score_min, score_max = (1, 5) if kind == 'sentence' else (0, 1)
    if any(not isfinite(value) or not score_min <= value <= score_max for _, _, _, value in expected):
        raise AssertionError(f'{kind} scores must be finite and in range')
    scores = [round(value, 2) for _, _, _, value in expected]
    eases = [direction * score for score in scores]
    low, high = min(eases), max(eases)
    previous_end = 0
    for mark, (start, end, token, _), score, ease in zip(marks, expected, scores, eases, strict=True):
        attrs = mark.attrs
        classes = (attrs.get('class') or '').split()
        if (any(type(value) is not int for value in (start, end, token))
                or not previous_end <= start < end <= len(ARTICLE)
                or (mark.start, mark.end) != (start, end) or mark.text != ARTICLE[start:end]
                or (attrs.get('data-start'), attrs.get('data-end'), attrs.get('data-token'), attrs.get('data-score'))
                != (str(start), str(end), str(token), f'{score:.2f}')):
            raise AssertionError(f'{kind} heatmap mark differs from its source coordinates or certified score')
        tail = start + len(mark.text.rstrip()) - 1
        if not 0 <= token < len(offsets):
            raise AssertionError(f'{kind} terminal token is outside the original tokenizer output')
        left, right = offsets[token]
        if (any(type(offset) is not int for offset in (left, right))
                or not 0 <= left <= tail < right <= len(ARTICLE)
                or ARTICLE[end:right].strip() or (kind == 'sentence' and right > end)):
            raise AssertionError(f'{kind} terminal token does not cover the mark end in the original tokenizer output')
        tones = [name for name in classes if name.startswith('tone-')]
        tone = 2 if low == high else floor(4 * (ease - low) / (high - low) + 0.5)
        if tones != [f'tone-{tone}']:
            raise AssertionError(f'{kind} heatmap tone does not match its own displayed score scale')
        for paragraph, (left, right) in enumerate(paragraphs):
            if left <= start < end <= right:
                counts[paragraph] += 1
                break
        previous_end = end
    if not all(counts) or sum(counts) != len(expected):
        raise AssertionError(f'{kind} heatmap must cover every paragraph and preserve its boundaries')
    return counts


def verify_markup(markup, layers, offsets):
    parsed = SourceMarkup()
    parsed.feed(markup)
    parsed.close()
    if parsed.source_count != 1 or parsed.in_source or ''.join(parsed.chunks) != ARTICLE:
        raise AssertionError('Heatmap changed or omitted original text')
    if {mark.kind for mark in parsed.marks} != set(layers):
        raise AssertionError('Unexpected source heatmap layer inventory')
    marks = {kind: [mark for mark in parsed.marks if mark.kind == kind] for kind in layers}
    counts = {kind: verify_layer(marks[kind], kind, ratings, offsets) for kind, ratings in layers.items()}

    nested = 0
    layered = 'sentence' in layers
    sentences = iter(marks.get('sentence', ()))
    sentence = next(sentences, None)
    for word in marks.get('lexical', ()):
        classes = (word.attrs.get('class') or '').split()
        if classes.count('lexical-underline') != int(layered):
            raise AssertionError('Lexical marks must use underlines only in the combined overview')
        while sentence is not None and sentence.end <= word.start:
            sentence = next(sentences, None)
        parent = None
        if sentence is not None and sentence.start < word.end:
            if not sentence.start <= word.start < word.end <= sentence.end:
                raise AssertionError('A lexical span crosses a certified sentence boundary')
            parent = sentence.start
        if word.parent_start != parent:
            raise AssertionError('Lexical span is detached from its containing sentence or marked inside the wrong sentence')
        nested += parent is not None
    return counts, nested


def verify_http(client, post, after_analysis):
    from .app import EASE, LEXICAL, UNDERSTANDING

    parameters = client.view_api(print_info=False, return_format='dict')['named_endpoints']['/analyze']['parameters']
    if (len(parameters) != 2 or parameters[0]['component'] != 'Textbox' or parameters[0]['parameter_has_default']
            or parameters[1]['component'] != 'Radio' or not parameters[1]['parameter_has_default']
            or parameters[1]['parameter_default'] != EASE):
        raise AssertionError('Expected an article-only public call with the sentence heatmap as its default')
    _, ease_markup, _, _, _, _ = client.submit(ARTICLE, api_name='/analyze').result(timeout=100)
    after_analysis()
    analyze, = [row for row in client.config['dependencies'] if row['api_name'] == 'analyze']
    text_input, radio = analyze['inputs']
    components = {component['id']: component for component in client.config['components']}
    if (components[text_input]['type'] != 'textbox' or components[text_input]['props'].get('value') is not None
            or components[radio]['type'] != 'radio' or components[radio]['props']['value'] != EASE):
        raise AssertionError('The Gradio page must start with an empty article and the sentence heatmap selected')
    switch, = [row for row in client.config['dependencies'] if [radio, 'input'] in row['targets']]
    address = urljoin(client.src, f'{client.api_prefix}run/{switch["api_name"]}')

    def select(metric):
        reply = post(address, timeout=15, json={'data': [None, metric, None, None], 'fn_index': switch['id'],
                                               'session_hash': client.session_hash, 'trigger_id': radio})
        reply.raise_for_status()
        markup, = reply.json()['data']
        after_analysis()
        return markup

    understanding_markup = select(UNDERSTANDING)
    lexical_markup = select(LEXICAL)
    if select(EASE) != ease_markup:
        raise AssertionError('Returning to the default overview must restore the exact cached view')
    return ease_markup, understanding_markup, lexical_markup


def verify_route(reader, port, *, save_markup: Path | None = None):
    import httpx
    from gradio_client import Client

    from .app import CSS, build_app

    if save_markup is not None and save_markup.exists():
        raise FileExistsError(save_markup)
    ui = build_app(reader)
    analyze = reader.analyze
    recorded = []
    forwards = 0

    def count_forward(*_):
        nonlocal forwards
        forwards += 1

    def record_analysis(text):
        if text != ARTICLE or recorded:
            raise AssertionError('Submit the complete original article exactly once')
        result = analyze(text)
        if result.text != text:
            raise AssertionError('The backbone result changed the source text')
        recorded.append(result)
        return result

    def after_analysis():
        if forwards != 1 or len(recorded) != 1:
            raise AssertionError('Expected one backbone forward for the initial Gradio request')

    hook = reader.backbone.encoder.register_forward_hook(count_forward)
    try:
        with patch.object(reader, 'analyze', side_effect=record_analysis):
            _, address, _ = ui.launch(server_name='127.0.0.1', server_port=port, css=CSS, quiet=True,
                                      share=False, prevent_thread_lock=True, show_error=True)
            with closing(Client(address, verbose=False, download_files=False)) as client:
                ease_markup, understanding_markup, lexical_markup = verify_http(client, httpx.post, after_analysis)
                if forwards != 1 or len(recorded) != 1:
                    raise AssertionError('Switching the same Gradio session re-ran analysis')
    finally:
        hook.remove()
        ui.close()

    result, = recorded
    tokenization = reader.backbone.tokenizer.encode(ARTICLE, add_special_tokens=False)
    ease = [(row.start, row.end, token, row.ease)
            for row, token in zip(result.sentences, result.sentence_tokens, strict=True)]
    understanding = [(row.start, row.end, token, row.understanding)
                     for row, token in zip(result.sentences, result.sentence_tokens, strict=True)]
    lexical = [(row.start, row.end, row.token, row.difficulty) for row in result.lexical]
    ease_counts, nested = verify_markup(ease_markup, {'sentence': ease, 'lexical': lexical}, tokenization.offsets)
    understanding_counts, understanding_nested = verify_markup(
        understanding_markup, {'sentence': understanding, 'lexical': lexical}, tokenization.offsets,
    )
    lexical_counts, _ = verify_markup(lexical_markup, {'lexical': lexical}, tokenization.offsets)
    if not 0 < nested == understanding_nested <= len(lexical):
        raise AssertionError('Both combined views must preserve nested lexical evidence from the cached analysis')
    default_sha256 = text_digest(ease_markup)
    if save_markup is not None:
        with save_markup.open('xb') as output:
            output.write(ease_markup.encode('utf-8'))
        if sha256(save_markup.read_bytes()).hexdigest() != default_sha256:
            raise AssertionError('Saved HTML differs from the verified default Gradio view')
    return {
        'fixture_sha256': text_digest(ARTICLE), 'characters': len(ARTICLE), 'tokens': len(tokenization.ids),
        'paragraphs': len(PARAGRAPHS), 'default_article_only': True, 'verified_exact_source_and_coordinates': True,
        'sentence_marks': len(ease), 'lexical_marks': len(lexical),
        'sentence_marks_by_paragraph': ease_counts['sentence'],
        'lexical_marks_by_paragraph': lexical_counts['lexical'],
        'unavailable_sentences': len(result.unavailable_sentences), 'unavailable_words': len(result.unavailable_words),
        'model_forwards': forwards, 'model_forwards_for_cached_view': forwards - 1,
        'sentence_markup_sha256': default_sha256, 'lexical_markup_sha256': text_digest(lexical_markup),
        'verified_combined_overview': True, 'cached_view_switches': 3,
        'combined_ease_lexical_marks_by_paragraph': ease_counts['lexical'],
        'combined_understanding_sentence_marks_by_paragraph': understanding_counts['sentence'],
        'combined_understanding_lexical_marks_by_paragraph': understanding_counts['lexical'],
        'combined_lexical_marks_nested': nested, 'combined_lexical_marks_in_gaps': len(lexical) - nested,
        'understanding_markup_sha256': text_digest(understanding_markup),
    }


def main():
    from . import app
    from .causal_plaba import digest
    from .checkpoint import DEFAULT_CHECKPOINT
    from .combined import CombinedReader
    from .plaba import BASE_REVISION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--cuda', action='store_true', required=True, help='Use the verified NVIDIA L4 backbone')
    args = parser.parse_args()
    markup = args.output.with_name('reading-overview.html')
    if args.output == markup:
        parser.error('--output must name the verification receipt, separate from reading-overview.html')
    for destination in (args.output, markup):
        if destination.exists():
            raise FileExistsError(destination)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reader = CombinedReader(args.model, checkpoint=args.checkpoint, cuda=args.cuda)
    receipt = {**verify_route(reader, args.port, save_markup=markup), 'candidate_id': reader.checkpoint.identifier,
               'model_revision': BASE_REVISION, 'device': 'NVIDIA L4 FP32',
               'source_code_sha256': digest(Path(__file__)), 'app_sha256': digest(Path(app.__file__))}
    with args.output.open('x', encoding='utf-8') as output:
        output.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
