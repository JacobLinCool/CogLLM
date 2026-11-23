"""Verify the Gradio reference route against the same causal Qwen request."""

import argparse
from contextlib import closing
from hashlib import sha256
from html import unescape
import json
from pathlib import Path
import re
from unittest.mock import patch
from urllib.parse import urljoin


ARTICLE = (
    "The maintainer found an old brass handle near the stairwell. An illustrated plan hung inside "
    "the cabinet, where the page listed two spare parts. Its margin read <draft> by the blue stamp.\n\n"
    "At dawn 🎐 a courier examined the loading entrance. Beside the painted gate sat a wooden storage box "
    "with a small latch. After moving the rope, she raised the latch and saw the spare handle.\n\n"
    "In the evening the caretaker closed the cabinet and turned off the lamp. A note near the stairs "
    "asked him to check the latch before leaving. The hallway remained quiet."
)
PHRASE = 'the latch'


def fixture_sha256():
    return sha256(ARTICLE.encode()).hexdigest()


def assert_markup(markup, expected):
    fragment = markup.split('<div class="source-text" lang="en">', 1)[1].split('</div>', 1)[0]
    if unescape(re.sub(r'<[^>]+>', '', fragment)) != ARTICLE or '<draft>' in fragment:
        raise AssertionError('Reference markup did not preserve and escape the full source')
    selected = re.findall(r'<mark class="reference-query" data-start="(\d+)" data-end="(\d+)">(.*?)</mark>',
                          fragment, re.DOTALL)
    if selected != [(str(expected.start), str(expected.end), PHRASE)]:
        raise AssertionError('The selected reference is not at its exact Unicode offset')
    marks = re.findall(r'<mark class="anchor reference-word" data-rank="(\d+)" data-start="(\d+)" '
                       r'data-end="(\d+)">(.*?)</mark>', fragment, re.DOTALL)
    actual = sorted((int(rank), int(start), int(end), unescape(word)) for rank, start, end, word in marks)
    wanted = [(row.rank, row.start, row.end, ARTICLE[row.start:row.end]) for row in expected.candidates]
    if len(actual) != 5 or actual != wanted or any(end > expected.start for _, _, end, _ in actual):
        raise AssertionError('The view did not show every prior candidate at its exact source position')


def verify_route(reader, port):
    import httpx
    from gradio_client import Client

    from .app import CSS, EASE, LEXICAL, REFERENCE, build_app
    from .reference import phrase_locations

    ui = build_app(reader)
    recorded = []
    forwards = 0
    analyze = reader.analyze

    def count_forward(*_):
        nonlocal forwards
        forwards += 1

    def record_analysis(text):
        result = analyze(text)
        recorded.append(result)
        return result

    def call(client, event, inputs):
        reply = httpx.post(urljoin(client.src, f'{client.api_prefix}run/{event["api_name"]}'), timeout=20,
                           json={'data': inputs, 'fn_index': event['id'], 'session_hash': client.session_hash,
                                 'trigger_id': event['targets'][0][0]})
        reply.raise_for_status()
        return reply.json()['data']

    hook = reader.backbone.encoder.register_forward_hook(count_forward)
    try:
        with patch.object(reader, 'analyze', side_effect=record_analysis):
            _, address, _ = ui.launch(server_name='127.0.0.1', server_port=port, css=CSS, quiet=True,
                                      share=False, prevent_thread_lock=True, show_error=True)
            with closing(Client(address, verbose=False, download_files=False)) as client:
                parameters = client.view_api(print_info=False, return_format='dict')['named_endpoints']['/analyze']['parameters']
                if (len(parameters) != 2 or parameters[0]['component'] != 'Textbox'
                        or parameters[0]['parameter_has_default'] or parameters[1]['parameter_default'] != EASE):
                    raise AssertionError('The public route must still accept only a required article')
                response = client.submit(ARTICLE, api_name='/analyze').result(timeout=100)
                if len(response) != 6 or forwards != 1 or len(recorded) != 1:
                    raise AssertionError('The default full-article analysis must do one backbone forward')
                _, ordinary, _, _, _, _ = response
                if 'class="sentence ' not in ordinary or '<mark class="reference-query"' in ordinary:
                    raise AssertionError('The normal article view should start with its sentence heatmap')
                result = recorded[0]
                points = phrase_locations(ARTICLE, PHRASE, result.reference)
                if len(points) != 2:
                    raise AssertionError('Expected two exact repeated reference phrases')
                dependencies = client.config['dependencies']
                find = next(row for row in dependencies if row['api_name'] == 'find_reference')
                select = next(row for row in dependencies if row['api_name'] == 'select_reference')
                switch = next(row for row in dependencies if row['api_name'] == 'update_measure')
                found = call(client, find, [None, ARTICLE, PHRASE])
                alternatives, _, markup, _, display = found
                if (len(alternatives['choices']) != 2 or not alternatives['visible'] or display != REFERENCE):
                    raise AssertionError('Repeated occurrences must be independently selectable')
                assert_markup(markup, reader.reference_head.rank(ARTICLE, result.reference, *points[0]))
                start, end = points[1]
                _, selected, _, display = call(client, select, [None, ARTICLE, PHRASE, f'{start}:{end}'])
                if display != REFERENCE:
                    raise AssertionError('Selecting the second occurrence must display the reference view')
                assert_markup(selected, reader.reference_head.rank(ARTICLE, result.reference, start, end))
                lexical, = call(client, switch, [None, LEXICAL, None, None, None])
                restored, = call(client, switch, [None, REFERENCE, None, None, None])
                if 'class="lexical ' not in lexical or selected != restored or forwards != 1 or len(recorded) != 1:
                    raise AssertionError('Reference queries and cached view changes must use no added forward')
    finally:
        hook.remove()
        ui.close()
    return {'fixture_sha256': fixture_sha256(), 'model_forwards': forwards,
            'model_forwards_for_cached_reference': forwards - 1,
            'verified_exact_source_and_coordinates': True, 'reference_queries': len(points)}


def main():
    from . import app, combined, reference
    from .causal_plaba import digest
    from .checkpoint import DEFAULT_CHECKPOINT
    from .plaba import BASE_REVISION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--cuda', action='store_true', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    reader = combined.CombinedReader(args.model, checkpoint=args.checkpoint, cuda=args.cuda, with_reference=True)
    report = {**verify_route(reader, args.port), 'candidate_id': reader.checkpoint.identifier,
              'reference_release_sha256': reader.reference_head.identifier,
              'model_revision': BASE_REVISION, 'device': 'NVIDIA L4 FP32',
              'source_code_sha256': digest(Path(__file__)), 'app_sha256': digest(Path(app.__file__)),
              'combined_sha256': digest(Path(combined.__file__)), 'reference_code_sha256': digest(Path(reference.__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
