"""Verify Gradio's paragraph boundaries and reference refresh over actual HTTP."""

import argparse
from contextlib import closing
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
from unittest.mock import patch
from urllib.parse import urljoin

from .verify_article_route import SourceMarkup, text_digest
from .verify_reference import ARTICLE as REFERENCE_ARTICLE, PHRASE, assert_markup


REFRESH_ARTICLE = (
    'The field crew went to a café and phoned Acme Inc.\r\n\r\n'
    'Green signal lights 🎐 labelled <active> & cafe\u0301 came on.'
)
REFRESH_SENTENCES = ((0, 50), (54, 110))
REFRESH_SENTENCE_TOKENS = (13, 29)


def verify_overview(markup, analysis, tokenizer, metric):
    from .app import EASE, LEXICAL, UNDERSTANDING

    parser = SourceMarkup()
    parser.feed(markup)
    parser.close()
    text = analysis.text
    if parser.source_count != 1 or parser.in_source or ''.join(parser.chunks) != text:
        raise AssertionError('The displayed article differs from the submitted Unicode text')
    layered = metric in (EASE, UNDERSTANDING)
    if not layered and metric != LEXICAL:
        raise ValueError('Expected a combined or lexical-only view')
    measure = 'ease' if metric == EASE else 'understanding'
    tokens = tokenizer.encode(text, add_special_tokens=False)
    sentences = [(row.start, row.end, token, getattr(row, measure))
                 for row, token in zip(analysis.sentences, analysis.sentence_tokens, strict=True)]
    words = [(row.start, row.end, row.token, row.difficulty) for row in analysis.lexical]
    if not sentences or not words:
        raise AssertionError('The fixture must have certified sentence and lexical markers')
    for kind, expected in (('sentence', sentences if layered else ()), ('lexical', words)):
        observed = [mark for mark in parser.marks if mark.kind == kind]
        if len(observed) != len(expected):
            raise AssertionError(f'Missing or extra {kind} markup')
        for mark, (start, end, token, score) in zip(observed, expected, strict=True):
            if ((mark.start, mark.end, mark.text) != (start, end, text[start:end])
                    or [mark.attrs.get(field) for field in ('data-start', 'data-end', 'data-token', 'data-score')]
                    != [str(start), str(end), str(token), f'{score:.2f}']):
                raise AssertionError(f'Inconsistent {kind} score, source coordinates or terminal token')
            if not 0 <= token < len(tokens.ids):
                raise AssertionError(f'{kind} token index is not in the original text')
            left, right = tokens.offsets[token]
            if not left < end <= right or text[end:right].strip():
                raise AssertionError(f'{kind} terminal token does not reach the source boundary')
            if kind == 'sentence':
                prefix = tokenizer.encode(text[:end], add_special_tokens=False)
                if right != end or tokens.ids[:token + 1] != prefix.ids:
                    raise AssertionError('A sentence score is not certified against its own causal prefix')
            else:
                containing = next((a for a, b, _, _ in sentences if a <= start < end <= b), None)
                if (mark.parent_start != (containing if layered else None)
                        or ('lexical-underline' in mark.attrs['class'].split()) != layered):
                    raise AssertionError('Lexical marks use the wrong sentence nesting or appearance')
    return {'sentence_marks': len(sentences) if layered else 0, 'lexical_marks': len(words)}


def verify_anchor_view(markup, analysis):
    fragment = markup.split('<div class="source-text" aria-label="已分析原文">', 1)[1].split('</div>', 1)[0]
    if unescape(re.sub(r'<[^>]+>', '', fragment)) != analysis.text:
        raise AssertionError('The cached anchor view changed the refreshed source text')
    marks = re.findall(r'<mark class="anchor" data-rank="(\d+)" data-start="(\d+)" data-end="(\d+)" '
                       r'data-stop="(\d+)" title="[^"]*">(.*?)</mark>', fragment, re.DOTALL)
    observed = [(int(rank), int(start), int(end), int(stop), unescape(span))
                for rank, start, end, stop, span in marks]
    expected = [(row.rank, row.start, row.end, row.visible_end, analysis.text[row.start:row.end])
                for row in sorted(analysis.anchors, key=lambda row: row.start)]
    if not expected or observed != expected:
        raise AssertionError('The cached anchor view failed to preserve its exact candidate spans')


def verify_route(reader, port, *, save_markup=None):
    import httpx
    from gradio_client import Client

    from .app import ANCHORS, CSS, EASE, LEXICAL, REFERENCE, UNDERSTANDING, build_app
    from .reference import phrase_locations

    if save_markup is not None and save_markup.exists():
        raise FileExistsError(save_markup)
    ui = build_app(reader)
    analyze = reader.analyze
    analyses, forward_counts, cached_calls = [], [], []
    forward_calls = 0

    def model_forward(*_):
        nonlocal forward_calls
        forward_calls += 1

    def record_analysis(text):
        before = forward_calls
        result = analyze(text)
        if result.text != text or forward_calls != before + 1:
            raise AssertionError('Every plain article request must preserve the source and do one forward')
        analyses.append(result)
        forward_counts.append(forward_calls - before)
        return result

    def call(client, event, inputs, trigger_id=None):
        before = forward_calls
        address = urljoin(client.src, f'{client.api_prefix}run/{event["api_name"]}')
        reply = httpx.post(address, timeout=20, json={
            'data': inputs, 'fn_index': event['id'], 'session_hash': client.session_hash,
            'trigger_id': event['targets'][0][0] if trigger_id is None else trigger_id,
        })
        reply.raise_for_status()
        if forward_calls != before:
            raise AssertionError('A cached Gradio request ran the model again')
        cached_calls.append(event['api_name'])
        return reply.json()['data']

    hook = reader.backbone.encoder.register_forward_hook(model_forward)
    try:
        with patch.object(reader, 'analyze', side_effect=record_analysis):
            _, address, _ = ui.launch(server_name='127.0.0.1', server_port=port, css=CSS, quiet=True,
                                      share=False, prevent_thread_lock=True, show_error=True)
            with closing(Client(address, verbose=False, download_files=False)) as client:
                api = client.view_api(print_info=False, return_format='dict')['named_endpoints']['/analyze']
                if (len(api['parameters']) != 2 or api['parameters'][0]['component'] != 'Textbox'
                        or api['parameters'][0]['parameter_has_default'] or api['parameters'][1]['component'] != 'Radio'
                        or api['parameters'][1]['parameter_default'] != EASE or len(api['returns']) != 6):
                    raise AssertionError('The public route no longer accepts an article with an optional view')
                events = {event['api_name']: event for event in client.config['dependencies']}
                analyze_event = events['analyze']
                reset = events['reset_reference_view']
                measure_id = analyze_event['inputs'][1]
                if (reset['inputs'] != [measure_id] or reset['outputs'][-1] != measure_id
                        or reset['trigger_after'] != analyze_event['id'] or reset['trigger_only_on_success'] is not True):
                    raise AssertionError('The browser success chain does not connect analysis and radio reset')

                _, opening, _, _, _, _ = client.submit(REFERENCE_ARTICLE, api_name='/analyze').result(timeout=100)
                if len(analyses) != 1 or analyses[0].text != REFERENCE_ARTICLE or forward_calls != 1:
                    raise AssertionError('The initial article must make exactly one forward')
                first = analyses[0]
                verify_overview(opening, first, reader.backbone.tokenizer, EASE)
                points = phrase_locations(first.text, PHRASE, first.reference)
                if len(points) != 2:
                    raise AssertionError('Both repeated reference occurrences must be independently selectable')
                occurrence, _, found, _, active_measure = call(
                    client, events['find_reference'], [None, first.text, PHRASE],
                )
                choices = [f'{start}:{end}' for start, end in points]
                if (occurrence['visible'] is not True or occurrence['value'] != choices[0]
                        or [option[1] for option in occurrence['choices']] != choices or active_measure != REFERENCE):
                    raise AssertionError('The first reference result failed to switch to its cached reference view')
                assert_markup(found, reader.reference_head.rank(first.text, first.reference, *points[0]))
                _, selected, _, active_measure = call(
                    client, events['select_reference'], [None, first.text, PHRASE, choices[1]],
                )
                if active_measure != REFERENCE:
                    raise AssertionError('A selected reference occurrence did not retain the reference mode')
                assert_markup(selected, reader.reference_head.rank(first.text, first.reference, *points[1]))

                table, refreshed, _, status, _, _ = client.submit(
                    REFRESH_ARTICLE, active_measure, api_name='/analyze',
                ).result(timeout=100)
                if (len(analyses) != 2 or analyses[-1].text != REFRESH_ARTICLE
                        or forward_counts != [1, 1] or forward_calls != 2):
                    raise AssertionError('Submitting a new article did not create a second independent single forward')
                second = analyses[-1]
                spans = tuple((row.start, row.end) for row in second.sentences)
                if (spans != REFRESH_SENTENCES or second.sentence_numbers != (1, 2)
                        or second.sentence_tokens != REFRESH_SENTENCE_TOKENS or second.unavailable_sentences
                        or len(table['data']) != 2 or status != '已分析 2 句。'):
                    raise AssertionError('The abbreviation before a paragraph break lacks its own certified sentence')
                counts = verify_overview(refreshed, second, reader.backbone.tokenizer, EASE)
                dropdown, note, measure_update = call(client, reset, [active_measure])
                if (dropdown.get('visible') is not False or dropdown.get('value') is not None
                        or dropdown.get('choices') != [] or note != '' or measure_update != EASE):
                    raise AssertionError('The HTTP success handler did not return the reference reset and new radio value')
                switch = events['update_measure']
                reference_view, = call(client, switch, [None, REFERENCE, None, None, None], measure_id)
                if ('class="reference-query"' in reference_view or 'class="anchor reference-word"' in reference_view
                        or '在上方輸入原文中的完整片語' not in reference_view):
                    raise AssertionError('The old reference remained after replacing the article')
                empty_reference = SourceMarkup()
                empty_reference.feed(reference_view)
                empty_reference.close()
                if (empty_reference.source_count != 1 or empty_reference.in_source
                        or ''.join(empty_reference.chunks) != REFRESH_ARTICLE):
                    raise AssertionError('The cleared reference view did not retain the new article')

                for metric in (UNDERSTANDING, LEXICAL, ANCHORS, EASE):
                    markup, = call(client, switch, [None, metric, None, None, None], measure_id)
                    if metric == ANCHORS:
                        verify_anchor_view(markup, second)
                    else:
                        verify_overview(markup, second, reader.backbone.tokenizer, metric)
                    if metric == EASE and markup != refreshed:
                        raise AssertionError('The cached combined view did not reproduce the refreshed overview')
                if forward_calls != 2:
                    raise AssertionError('Cached Gradio interaction changed the content-forward count')
    finally:
        hook.remove()
        ui.close()
    markup_sha256 = text_digest(refreshed)
    if save_markup is not None:
        with save_markup.open('xb') as destination:
            destination.write(refreshed.encode('utf-8'))
        if sha256(save_markup.read_bytes()).hexdigest() != markup_sha256:
            raise AssertionError('Saved HTML fragment differs from the HTTP response')
    return {
        'fixture_sha256': text_digest(REFRESH_ARTICLE), 'reference_fixture_sha256': text_digest(REFERENCE_ARTICLE),
        'model_forwards': forward_calls, 'model_forwards_per_article': forward_counts,
        'model_forwards_for_cached_views': forward_calls - sum(forward_counts),
        'paragraph_sentence_spans': spans, 'paragraph_sentence_tokens': second.sentence_tokens,
        'paragraph_sentence_marks': counts['sentence_marks'], 'refresh_lexical_marks': counts['lexical_marks'],
        'verified_exact_source_and_coordinates': True,
        'verified_reference_refresh': True, 'verified_radio_success_http': True, 'verified_other_views': True,
        'radio_reset_value': measure_update, 'radio_success_verification': 'same_session_http_callback',
        'refresh_markup_sha256': markup_sha256, 'cached_http_trace': cached_calls,
    }


def main():
    from . import app, combined, plaba, reference, verify_article_route, verify_reference
    from .causal_plaba import digest
    from .checkpoint import DEFAULT_CHECKPOINT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--cuda', action='store_true', required=True)
    args = parser.parse_args()
    markup = args.output.with_name('reading-overview.html')
    if args.output == markup:
        parser.error('--output must name the JSON receipt')
    for output in (args.output, markup):
        if output.exists():
            raise FileExistsError(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    reader = combined.CombinedReader(args.model, checkpoint=args.checkpoint, cuda=args.cuda, with_reference=True)
    sources = {
        'source_code_sha256': Path(__file__), 'app_sha256': Path(app.__file__),
        'combined_sha256': Path(combined.__file__), 'plaba_sha256': Path(plaba.__file__),
        'reference_code_sha256': Path(reference.__file__),
        'article_verifier_sha256': Path(verify_article_route.__file__),
        'reference_verifier_sha256': Path(verify_reference.__file__),
    }
    receipt = {
        **verify_route(reader, args.port, save_markup=markup),
        'candidate_id': reader.checkpoint.identifier, 'model_revision': plaba.BASE_REVISION,
        'reference_release_sha256': reader.reference_head.identifier, 'device': 'NVIDIA L4 FP32',
        **{name: digest(path) for name, path in sources.items()},
    }
    with args.output.open('x', encoding='utf-8') as destination:
        destination.write(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
