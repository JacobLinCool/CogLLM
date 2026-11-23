"""Verify actual Gradio requests, original text, coordinates, and cached selection."""

from __future__ import annotations

import argparse
from contextlib import closing
from hashlib import sha256
from html.parser import HTMLParser
import json
from math import ceil, isclose
from pathlib import Path
import re
from urllib.parse import urljoin

from .app import CHANNELS, CSS, EXAMPLE_TEXT, build_app, load_runner, parse_analysis, render_article


class ArticleMarkup(HTMLParser):
    def __init__(self, fragment):
        super().__init__(convert_charrefs=True)
        self.blocks = 0
        self.in_text = False
        self.span = None
        self.characters = 0
        self.pieces = []
        self.segments = []
        self.checkpoint = None
        self.feed(fragment)
        self.close()
        if self.blocks != 1 or self.in_text or self.span is not None or not self.checkpoint:
            raise AssertionError('Expected a complete original article block')

    def handle_starttag(self, tag, fields):
        attributes = dict(fields)
        classes = (attributes.get('class') or '').split()
        if tag == 'section' and 'trace-panel' in classes:
            self.checkpoint = attributes.get('data-checkpoint')
        if tag == 'div' and 'trace-text' in classes:
            self.in_text = True
            self.blocks += 1
        elif self.in_text:
            if tag != 'mark' or self.span is not None or len(attributes) != len(fields):
                raise AssertionError('Unexpected tag or nested mark in the original text')
            self.span = self.characters, attributes

    def handle_data(self, text):
        if self.in_text:
            self.pieces.append(text)
            self.characters += len(text)

    def handle_endtag(self, tag):
        if not self.in_text:
            return
        if tag == 'div' and self.span is None:
            self.in_text = False
        elif tag == 'mark' and self.span is not None:
            start, attributes = self.span
            self.segments.append((start, self.characters, attributes))
            self.span = None
        else:
            raise AssertionError('Unbalanced original text markup')


def verify_article(fragment, analysis, index):
    rendered = ArticleMarkup(fragment)
    if ''.join(rendered.pieces) != analysis.text or rendered.checkpoint != analysis.checkpoint:
        raise AssertionError('Marked HTML changed the source text or candidate identity')
    current = analysis.tokens[index]
    selected, pointed = [], []
    for start, end, attributes in rendered.segments:
        if (start, end) != (int(attributes['data-start']), int(attributes['data-end'])):
            raise AssertionError('A source mark differs from its Unicode offsets')
        classes = attributes['class'].split()
        if 'point-mark' in classes:
            selected.extend(range(start, end))
        if 'context-mark' in classes:
            pointed.extend(range(start, end))
        if 'data-token' in attributes:
            token = analysis.tokens[int(attributes['data-token']) - 1]
            if not token.start <= start < end <= token.end:
                raise AssertionError('Source mark falls outside the token it represents')
            if not isclose(float(attributes['data-unresolved']), token.unresolved, abs_tol=5e-9):
                raise AssertionError('Source mark shows the wrong unresolved score')
    if selected != list(range(current.start, current.end)):
        raise AssertionError('Rendered selection differs from the chosen token span')
    expected_pointer = [] if current.pointer_start is None else list(range(current.pointer_start, current.pointer_end))
    if pointed != expected_pointer:
        raise AssertionError('Rendered source pointer differs from the chosen token pointer')
    return len(rendered.segments)


def verify_plot(plot_update, analysis):
    plot = plot_update.get('value', plot_update)
    if plot['columns'] != ['token', 'score', 'signal'] or plot['mark'] != 'line':
        raise AssertionError('Gradio did not return the native token curve')
    expected = [(index, getattr(token, channel), label)
                for index, token in enumerate(analysis.tokens, 1) for channel, (label, _) in CHANNELS.items()]
    for actual, (position, value, label) in zip(plot['data'], expected, strict=True):
        if actual[0] != position or actual[2] != label or not isclose(actual[1], value, abs_tol=5e-9):
            raise AssertionError('Curve points differ from the model token trace')
    return len(expected)


def preview_html(plot_update, article, detail, token_count):
    """Render the verified HTTP curve and exact article response without external assets."""
    data = plot_update.get('value', plot_update)['data']
    upper = max(1, ceil(max(value for _, value, _ in data)))
    left, right, top, bottom = 46, 425, 12, 166

    def x(position):
        return left + (position - 1) * (right - left) / max(token_count - 1, 1)

    def y(value):
        return bottom - value / upper * (bottom - top)

    marks = []
    for step in range(5):
        level = upper * step / 4
        row = y(level)
        marks.extend((f'<path class="axis" d="M{left},{row:.2f} H{right}"/>',
                      f'<text x="{left - 7}" y="{row + 4:.2f}" text-anchor="end">{level:.4g}</text>'))
    for position in sorted({1 + round((token_count - 1) * step / 4) for step in range(5)}):
        marks.append(f'<text x="{x(position):.2f}" y="184" text-anchor="middle">{position}</text>')
    marks.append('<text x="235" y="202" text-anchor="middle">Position in the passage</text>')
    legend = []
    for (label, color), dash in zip(CHANNELS.values(), ('none', '5 3', '1 3', '8 3 2 3'), strict=True):
        points = [(x(position), y(value)) for position, value, trace in data if trace == label]
        path = ' '.join(f'{px:.2f},{py:.2f}' for px, py in points)
        if len(points) == 1:
            px, py = points[0]
            marks.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.4" fill="{color}"/>')
        else:
            marks.append(f'<polyline points="{path}" fill="none" stroke="{color}" '
                         f'stroke-width="2" stroke-dasharray="{dash}"/>')
        legend.append(f'<span><svg viewBox="0 0 20 8" aria-hidden="true"><path d="M0,4 H20" '
                      f'stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/></svg>{label}</span>')

    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>CogLLM Reading Review</title><style>' + CSS + """
        :root { --body-text-color-subdued:#425752; background:#fbfcf9; color:#182b29; }
        body { margin:0; font:16px/1.5 system-ui,sans-serif; }
        main { max-width:760px; margin:auto; padding:1.2rem min(4vw,1.5rem); }
        h1 { font-size:1.55rem; margin:0 0 .9rem; }
        figure { margin:0 0 1rem; }
        figcaption { display:flex; gap:.55rem; align-items:baseline; flex-wrap:wrap; }
        figcaption span { color:#425752; font-size:.88rem; }
        .curve { display:block; width:100%; height:auto; }
        .curve text { fill:#425752; font-size:14px; }
        .axis { stroke:#dce3da; fill:none; }
        .curve-legend { display:flex; flex-wrap:wrap; gap:.3rem .95rem; font-size:.88rem; }
        .curve-legend span { display:inline-flex; align-items:center; gap:.3rem; }
        .curve-legend svg { width:1.15rem; height:.5rem; }
        #cogllm-details { margin:0 0 .75rem; }
        """ + '</style></head><body><main><h1>CogLLM Reading Review</h1><figure>'
        '<figcaption><strong>Signals by position</strong><span>Model scores</span></figcaption>'
        '<svg class="curve" viewBox="0 0 440 210" role="img" aria-labelledby="curve-title">'
        '<title id="curve-title">Model scores for each position in the reading passage</title>'
        + ''.join(marks) + '</svg><div class="curve-legend">' + ''.join(legend) + '</div></figure>'
        f'<div id="cogllm-details">{detail}</div><div id="cogllm-reading">{article}</div></main></body></html>'
    )


def verify_http(analyze, texts, *, port=None, forward_count=None, timeout=120, save_preview=None, preview_request=1,
                verify_example=False):
    """Exercise the real callback over localhost and optionally count actual backbone calls."""
    from gradio_client import Client
    import httpx

    texts = tuple(texts)
    if not texts:
        raise ValueError('At least one article is needed for HTTP verification')
    if save_preview is not None and not 1 <= preview_request <= len(texts):
        raise ValueError('Preview request must identify one of the HTTP inputs')
    if verify_example:
        texts += (EXAMPLE_TEXT,)
    analyses, requests = [], []
    checksum = None

    def record(text):
        data = analyze(text)
        analyses.append(parse_analysis(text, data))
        return data

    ui = build_app(record)
    try:
        _, address, _ = ui.launch(server_name='127.0.0.1', server_port=port, share=False, css=CSS,
                                  prevent_thread_lock=True, quiet=True, show_error=True)
        with closing(Client(address, verbose=False, download_files=False)) as client:
            endpoint = client.view_api(print_info=False, return_format='dict')['named_endpoints']['/analyze']
            parameters = endpoint['parameters']
            if len(parameters) != 1 or parameters[0]['component'] != 'Textbox' or parameters[0]['parameter_has_default']:
                raise AssertionError('Analyze must accept just the user-supplied article')
            if len(endpoint['returns']) != 5:
                raise AssertionError('Analyze must return the five public outputs')
            events = {row['api_name']: row for row in client.config['dependencies']}

            def call_ui(name, data):
                event = events[name]
                url = urljoin(client.src, f'{client.api_prefix}run/{name}')
                reply = httpx.post(url, timeout=timeout,
                                   json={'data': data, 'fn_index': event['id'],
                                         'session_hash': client.session_hash, 'trigger_id': event['targets'][0][0]})
                reply.raise_for_status()
                return reply.json()['data']

            for number, text in enumerate(texts, 1):
                before = forward_count() if forward_count else None
                if verify_example and number == len(texts):
                    loaded, _, article, plot, slider, detail, status = call_ui('load_example', [])
                    if loaded != EXAMPLE_TEXT or article or detail or status or plot['visible'] or slider['visible']:
                        raise AssertionError('Load example must load the passage and clear the old result')
                    if len(analyses) != number - 1 or forward_count and forward_count() != before:
                        raise AssertionError('Load example must not execute model inference')
                    text = loaded
                response = client.submit(text, api_name='/analyze').result(timeout=timeout)
                if len(analyses) != number or analyses[-1].text != text:
                    raise AssertionError('Analyze did not dispatch exactly once with the submitted text')
                after = forward_count() if forward_count else None
                if forward_count and after - before != 1:
                    raise AssertionError('Analyze must execute exactly one backbone forward')
                analysis = analyses[-1]
                if checksum is None:
                    checksum = analysis.checkpoint
                if analysis.checkpoint != checksum:
                    raise AssertionError('HTTP requests used different candidates')
                article, plot, slider, detail, _ = response
                index = slider['value'] - 1
                if not 0 <= index < len(analysis.tokens) or slider['maximum'] != max(2, len(analysis.tokens)):
                    raise AssertionError('Index control did not follow the current article')
                if f'data-selected="{index + 1}"' not in detail:
                    raise AssertionError('Initial detail and article selected different tokens')
                marks = verify_article(article, analysis, index)
                points = verify_plot(plot, analysis)

                for selection in (0, len(analysis.tokens) - 1):
                    selected_article, selected_detail = call_ui('select_token', [None, selection + 1])
                    verify_article(selected_article, analysis, selection)
                    if f'data-selected="{selection + 1}"' not in selected_detail:
                        raise AssertionError('A cached selection did not update the matching detail')
                if len(analyses) != number or forward_count and forward_count() != after:
                    raise AssertionError('Token inspection recomputed model scores')
                if number == preview_request and save_preview is not None:
                    with save_preview.open('x', encoding='utf-8') as preview:
                        preview.write(preview_html(plot, article, detail, len(analysis.tokens)))
                requests.append({'text_sha256': sha256(text.encode()).hexdigest(), 'characters': len(text),
                                 'tokens': len(analysis.tokens), 'plot_points': points, 'visible_marks': marks,
                                 'inference_callbacks': 1, 'cached_selections': 2,
                                 'backbone_forwards': after - before if forward_count else None})
    finally:
        ui.close()
    return {'checkpoint': checksum, 'requests': requests, 'input_and_offsets_verified': True,
            'cached_selections_verified': True, 'backbone_counter_verified': forward_count is not None,
            'example_load_verified': verify_example,
            'preview_request': preview_request if save_preview is not None else None}


def smoke_fixture(text):
    rows = []
    for index, word in enumerate(re.finditer(r'\S+', text)):
        duplicate = '🙂' in word[0]
        previous = rows[-1] if rows else None
        row = {'start': word.start(), 'end': word.end(), 'unresolved': (index % 4) / 4,
               'opened': (index % 3) / 4, 'resolved': (index % 5) / 5, 'transition': (index % 2) / 2,
               'pointer_start': previous['start'] if previous else None, 'pointer_end': previous['end'] if previous else None}
        rows.append(row)
        if duplicate:
            rows.append({**row, 'unresolved': 0.88})
    return {'text': text, 'checkpoint': 'synthetic-interface-fixture', 'tokens': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runner', help='Candidate factory as module:factory')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--articles', type=Path, nargs='+', help='UTF-8 test articles')
    parser.add_argument('--port', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.runner:
        if args.model is None or args.checkpoint is None or not args.articles:
            parser.error('--runner requires --model, --checkpoint and --articles')
        analyze = load_runner(args.runner, args.model, args.checkpoint)
        texts = [path.read_bytes().decode('utf-8') for path in args.articles]
        report = verify_http(analyze, texts, port=args.port, forward_count=getattr(analyze, 'forward_count', None))
    else:
        if args.model is not None or args.checkpoint is not None or args.articles:
            parser.error('Candidate inputs require --runner')
        texts = ['A <script> signal. 🙂文\r\n\r\nAn e\u0301cho then arrived.\n它指向 前文。', '較短的 正文。🙂']
        for text in texts:
            analysis = parse_analysis(text, smoke_fixture(text))
            for index in range(len(analysis.tokens)):
                verify_article(render_article(analysis, index), analysis, index)
        report = verify_http(smoke_fixture, texts, port=args.port, verify_example=True)
        report['synthetic'] = True
    output = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.write_text(output, encoding='utf-8')
    print(output, end='')


if __name__ == '__main__':
    main()
