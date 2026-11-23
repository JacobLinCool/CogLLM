"""Replay the released cue readout and verify its cached Gradio view over HTTP."""

import argparse
from collections import defaultdict
from contextlib import closing
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urljoin

import numpy as np

from research.semantic_tracking.beukers_extract import read_stimuli
from research.semantic_tracking.gum_bridge_extract import certify, json_digest

from .causal_plaba import CausalEncoding, HIDDEN_WIDTH, digest
from .checkpoint import BACKBONE, DEFAULT_CHECKPOINT
from .combined import CombinedReader
from .verify_overview_http import REFRESH_ARTICLE, verify_overview
from .word_cues import RELEASE


ARTICLE = 'The field crew went to a café and phoned Acme Inc.'
ABSOLUTE_SCORE_TOLERANCE = 1e-11
CUDA_PREFIX_TOLERANCE = 0.001


def text_digest(text):
    return sha256(text.encode('utf-8')).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require_digest(path, expected):
    if digest(path) != expected:
        raise AssertionError(f'Artifact differs from its recorded digest: {path}')


class ReplayBackbone:
    """Only archived boundary states are evidence. Other positions exercise the UI."""

    def __init__(self, tokenizer, archived):
        self.tokenizer = tokenizer
        self.archived = archived
        self.encoder = self
        self.on_forward = None

    def encode(self, text):
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        tokens, offsets = tuple(encoded.ids), tuple(encoded.offsets)
        states = np.stack([
            np.random.default_rng([index, token]).standard_normal(HIDDEN_WIDTH, dtype=np.float32)
            for index, token in enumerate(tokens)
        ])
        if text in self.archived:
            positions, actual = self.archived[text]
            states[list(positions)] = actual
        if self.on_forward is not None:
            self.on_forward(self, tokens, states)
        return CausalEncoding(text, tokens, offsets, states)

    def register_forward_hook(self, callback):
        if self.on_forward is not None:
            raise RuntimeError('A replay forward observer is already installed')
        self.on_forward = callback
        return self

    def remove(self):
        self.on_forward = None


class CueMarkup(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks = 0
        self.in_source = False
        self.pieces = []
        self.position = 0
        self.open_mark = None
        self.marks = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = (attributes.get('class') or '').split()
        if tag == 'div' and 'source-text' in classes and not self.in_source:
            self.in_source = True
            self.blocks += 1
        elif self.in_source:
            if (tag != 'mark' or self.open_mark or 'cue-candidate' not in classes
                    or len(attributes) != len(attrs)):
                raise AssertionError('Unexpected tag or nested mark in the cue source')
            self.open_mark = self.position, len(self.pieces), attributes

    def handle_data(self, data):
        if self.in_source:
            self.pieces.append(data)
            self.position += len(data)

    def handle_endtag(self, tag):
        if not self.in_source:
            return
        if tag == 'div' and self.open_mark is None:
            self.in_source = False
        elif tag == 'mark' and self.open_mark is not None:
            start, segment, attrs = self.open_mark
            self.marks.append((start, self.position, ''.join(self.pieces[segment:]), attrs))
            self.open_mark = None
        else:
            raise AssertionError('The cue source contains unbalanced tags')


def verify_markup(markup, analysis, tokenizer):
    parser = CueMarkup()
    parser.feed(markup)
    parser.close()
    text = analysis.text
    if parser.blocks != 1 or parser.in_source or ''.join(parser.pieces) != text:
        raise AssertionError('The cue view changed the submitted Unicode or line endings')
    candidates = defaultdict(list)
    for item in analysis.cues:
        candidates[item.sentence_number].append(item)
    for rows in candidates.values():
        if [item.rank for item in sorted(rows, key=lambda item: (-item.score, item.start))] != list(range(1, len(rows) + 1)):
            raise AssertionError('Cue ranks do not follow descending sentence-local score')
    expected = sorted((item for item in analysis.cues if item.rank <= 3), key=lambda item: item.start)
    if not expected or len(expected) != len(parser.marks):
        raise AssertionError('The cue view did not display exactly the first three ranks per sentence')
    tokens = tokenizer.encode(text, add_special_tokens=False)
    for (start, end, source, attrs), item in zip(parser.marks, expected, strict=True):
        fields = ('start', 'end', 'token', 'sentence', 'rank', 'score')
        values = (item.start, item.end, item.token, item.sentence_number, item.rank, item.score)
        if ((start, end, source) != (item.start, item.end, text[item.start:item.end])
                or [attrs.get(f'data-{field}') for field in fields] != [str(value) for value in values]
                or f'cue-rank-{item.rank}' not in attrs['class'].split() or attrs.get('tabindex') != '0'):
            raise AssertionError('Cue coordinates, score, rank or keyboard access changed in HTML')
        left, right = tokens.offsets[item.token]
        if not left < right == end or tokens.ids[:item.token + 1] != tokenizer.encode(text[:end], add_special_tokens=False).ids:
            raise AssertionError('A rendered cue does not end on its exact causal token')
    return len(expected)


def verify_prefix(before, after, tokenizer):
    if not after.text.startswith(before.text + '\r\n\r\n'):
        raise AssertionError('Prefix control must append a distinct paragraph after the complete first sentence')
    prefix = tokenizer.encode(before.text, add_special_tokens=False).ids
    full = tokenizer.encode(after.text, add_special_tokens=False).ids
    if full[:len(prefix)] != prefix:
        raise AssertionError('Suffix control changed the first sentence tokenization')
    first, second = before.cues, tuple(row for row in after.cues if row.sentence_number == 1)
    if not first or len(first) != len(second):
        raise AssertionError('Suffix control changed the available prefix cues')
    differences = []
    for left, right in zip(first, second, strict=True):
        fields = ('start', 'end', 'token', 'sentence_number', 'rank')
        if any(getattr(left, field) != getattr(right, field) for field in fields):
            raise AssertionError('Suffix control changed a certified prefix cue coordinate or rank')
        differences.append(abs(left.score - right.score))
    max_difference = max(differences)
    if not np.isfinite(differences).all() or max_difference > CUDA_PREFIX_TOLERANCE:
        raise AssertionError('Suffix control changed an earlier raw cue score')
    return {'prefix_sha256': text_digest(before.text), 'prefix_tokens': len(prefix), 'compared_cues': len(first),
            'maximum_absolute_score_difference': max_difference, 'absolute_tolerance': CUDA_PREFIX_TOLERANCE}


def verify_http(reader, fixtures, port, save_markup, preview, *, check_prefix=False):
    import httpx
    from gradio_client import Client

    from .app import CSS, CUES, EASE, LEXICAL, build_app

    if CUES != '消歧線索候選':
        raise AssertionError('The cue measure changed its public name')
    ui = build_app(reader)
    analyze = reader.analyze
    analyses, checked = [], []
    forward_calls = 0

    def count_forward(*_):
        nonlocal forward_calls
        forward_calls += 1

    def observe(text):
        before = forward_calls
        result = analyze(text)
        if result.text != text or forward_calls - before != 1:
            raise AssertionError('Article analysis failed exact text or single-forward verification')
        analyses.append(result)
        return result

    def switch(client, event, measure_id, value):
        argument = [value if component == measure_id else None for component in event['inputs']]
        if argument.count(value) != 1:
            raise AssertionError('Cached Gradio route does not have exactly one measure input')
        address = urljoin(client.src, f'{client.api_prefix}run/{event["api_name"]}')
        response = httpx.post(address, timeout=20, json={
            'data': argument, 'fn_index': event['id'], 'session_hash': client.session_hash,
            'trigger_id': measure_id,
        })
        response.raise_for_status()
        markup, = response.json()['data']
        return markup

    hook = reader.backbone.encoder.register_forward_hook(count_forward)
    try:
        with patch.object(reader, 'analyze', side_effect=observe):
            _, address, _ = ui.launch(server_name='127.0.0.1', server_port=port, css=CSS, quiet=True,
                                      share=False, prevent_thread_lock=True, show_error=True)
            with closing(Client(address, verbose=False, download_files=False)) as client:
                events = {event['api_name']: event for event in client.config['dependencies']}
                measure_id = events['analyze']['inputs'][1]
                event = events['update_measure']
                for index, (identifier, article) in enumerate(fixtures, 1):
                    measure = CUES if index == 1 else EASE
                    _, initial, *_ = client.submit(article, measure, api_name='/analyze').result(timeout=100)
                    if len(analyses) != index or analyses[-1].text != article or forward_calls != index:
                        raise AssertionError('Gradio did not submit one fresh analysis for this exact article')
                    analysis = analyses[-1]
                    if measure == CUES:
                        verify_markup(initial, analysis, reader.backbone.tokenizer)
                    else:
                        verify_overview(initial, analysis, reader.backbone.tokenizer, EASE)
                    markup = switch(client, event, measure_id, CUES)
                    marks = verify_markup(markup, analysis, reader.backbone.tokenizer)
                    lexical = switch(client, event, measure_id, LEXICAL)
                    verify_overview(lexical, analysis, reader.backbone.tokenizer, LEXICAL)
                    if (switch(client, event, measure_id, CUES) != markup
                            or (measure == CUES and initial != markup)
                            or forward_calls != index or len(analyses) != index):
                        raise AssertionError('Cached view switching changed the cue output or triggered recomputation')
                    checked.append({'id': identifier, 'text_sha256': text_digest(article), 'cue_marks': marks,
                                    'offsets_verified': True, 'content_forwards': 1, 'cached_forwards': 0})
    finally:
        hook.remove()
        ui.close()
    if save_markup is not None:
        with save_markup.open('xb') as destination:
            destination.write(markup.encode('utf-8'))
        require_digest(save_markup, text_digest(markup))
    result = {'fixtures': checked, 'markup_sha256': text_digest(markup), 'content_forwards': forward_calls,
              'cached_forwards': forward_calls - len(checked)}
    if check_prefix:
        result['prefix_invariance'] = verify_prefix(*analyses, reader.backbone.tokenizer)
    if preview is not None:
        document = ('<!doctype html>\n<html lang="zh-Hant"><meta charset="utf-8">'
                    f'<title>CogLLM · {CUES}</title><style>{CSS}</style>'
                    f'<body><main id="article-result">{markup}</main></body></html>\n')
        with preview.open('xb') as destination:
            destination.write(document.encode('utf-8'))
        result['preview_sha256'] = text_digest(document)
        require_digest(preview, result['preview_sha256'])
    return result


def archived_replay(work, results, tokenizer_path, checkpoint, include_http, port, save_markup, preview):
    from tokenizers import Tokenizer

    extract_path = results / 'projection' / 'seal.json'
    forecast_path = results / 'forecast-seal.json'
    predictions_path = results / 'word-2023-forecasts.json'
    stimulus_path = work / 'packet' / 'stimuli.jsonl'
    release = load_json(RELEASE)
    require_digest(extract_path, release['evidence']['extraction_seal_sha256'])
    require_digest(forecast_path, release['evidence']['forecast_seal_sha256'])
    extraction = load_json(extract_path)
    forecasts = load_json(forecast_path)
    fit_path, score_path = results / 'fits.json', results / 'results.json'
    require_digest(fit_path, release['evidence']['fit_sha256'])
    require_digest(fit_path, forecasts['file_sha256'][fit_path.name])
    require_digest(score_path, release['evidence']['score_sha256'])
    scored = load_json(score_path)
    if (release['fit'] != load_json(fit_path)['fits_for_2023']['state']
            or scored['forecast_seal_sha256'] != digest(forecast_path)
            or forecasts['extraction_sha256'] != digest(extract_path)):
        raise AssertionError('The release, frozen all-2021 fit, external forecasts and archived scoring do not agree')
    require_digest(predictions_path, forecasts['file_sha256'][predictions_path.name])
    require_digest(stimulus_path, extraction['stimuli_sha256'])
    require_digest(tokenizer_path, extraction['tokenizer_sha256'])
    preflight_path = results / 'extractor-preflight.json'
    require_digest(preflight_path, extraction['preflight_sha256'])
    preflight = {row['id']: row for row in load_json(preflight_path)['documents']}
    if extraction['assay_sha256'] != digest(results / 'preflight.json') or extraction.get('raw_stops') is not True:
        raise AssertionError('The archived extraction is not bound to the raw-stop assay')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def tokenize(text):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return encoded.ids, encoded.offsets

    def tokenize_prefixes(texts):
        return [encoded.ids for encoded in tokenizer.encode_batch(texts, add_special_tokens=False)]

    state_path, coordinate_path = (results / 'projection' / name for name in ('states.npy', 'coordinates.jsonl'))
    for path in (state_path, coordinate_path):
        require_digest(path, extraction['file_sha256'][path.name])
    states = np.load(state_path, mmap_mode='r', allow_pickle=False)
    if (states.dtype != np.float32 or list(states.shape) != extraction['states_shape']
            or states.shape != (extraction['certified_stops'], HIDDEN_WIDTH)):
        raise AssertionError('The archived FP32 state inventory has changed')
    expected = load_json(predictions_path)
    expected_by_id = {row['id']: row for row in expected}
    if len(expected) != 88 or len(expected_by_id) != 88:
        raise AssertionError('The frozen 2023 cohort must contain all 88 isolated stimuli')
    archived, by_id = {}, {}
    begin = 0
    with coordinate_path.open(encoding='utf-8') as coordinates:
        for source, encoded in zip(read_stimuli(stimulus_path), map(json.loads, coordinates), strict=True):
            identifier = source['id']
            if encoded != {'id': identifier, 'stops': preflight[identifier]['stops']}:
                raise AssertionError('The archived coordinates differ from the preflight')
            stop = begin + len(encoded['stops'])
            if identifier in expected_by_id:
                certified = certify(source, tokenize, tokenize_prefixes)
                if (json_digest([certified.tokens, certified.offsets, certified.certified])
                        != preflight[identifier]['certificate_sha256']):
                    raise AssertionError('An archived stimulus changed its exact tokens or certified boundaries')
                text = source['text']
                by_id[identifier] = text
                archived[text] = (tuple(position for _, position in certified.certified), states[begin:stop])
            begin = stop
    if begin != len(states) or by_id.keys() != expected_by_id.keys():
        raise AssertionError('The state table does not cover all sealed replay rows')

    backbone = ReplayBackbone(tokenizer, archived)
    reader = CombinedReader(tokenizer_path.parent, checkpoint=checkpoint, backbone=backbone,
                            with_cues=True, with_reference=include_http)
    if reader.checkpoint.identifier != extraction['candidate_id'] or reader.cue_head.identifier != digest(RELEASE):
        raise AssertionError('Replay uses a different model candidate or cue release')
    max_difference, count = 0.0, 0
    for reference in expected:
        analysis = reader.analyze(by_id[reference['id']])
        scores = np.asarray(reference['logits']['state'], dtype=np.float64)
        ranks = np.empty(len(scores), dtype=int)
        ranks[np.argsort(-scores, kind='stable')] = np.arange(1, len(scores) + 1)
        observed = [(row.start, row.end, row.token, row.sentence_number, row.rank) for row in analysis.cues]
        recorded = [(start, end, token, 1, rank) for (start, end), token, rank in
                    zip(reference['spans'], reference['token_indices'], ranks, strict=True)]
        if observed != recorded or not len(scores):
            raise AssertionError(f'Cue coordinates, inventory or ranks differ on {reference["id"]}')
        if any(row.token not in archived[analysis.text][0] for row in analysis.cues):
            raise AssertionError('A primary replay cue used a synthetic end state')
        differences = np.abs(np.asarray([row.score for row in analysis.cues]) - scores)
        if not np.isfinite(differences).all() or np.max(differences) > ABSOLUTE_SCORE_TOLERANCE:
            raise AssertionError(f'Cue replay scores differ on {reference["id"]}')
        max_difference = max(max_difference, float(np.max(differences)))
        count += len(scores)
    if count != 1073:
        raise AssertionError('The sealed cohort does not contain 1,073 eligible cue candidates')
    result = {'device': 'cpu_archived_states', 'replayed_articles': len(expected), 'replayed_cues': count,
              'maximum_absolute_score_difference': max_difference, 'absolute_score_tolerance': ABSOLUTE_SCORE_TOLERANCE,
              'forecast_sha256': digest(predictions_path), 'extraction_sha256': digest(extract_path),
              'fit_sha256': digest(fit_path), 'scored_results_sha256': digest(score_path),
              **identity(reader)}
    if include_http:
        first_text = by_id[expected[0]['id']]
        fixtures = (('archived', first_text), ('unicode-crlf', REFRESH_ARTICLE))
        result['http'] = verify_http(reader, fixtures, port, save_markup, preview)
    return result


def identity(reader):
    directory = Path(__file__).parent
    return {'source_code_sha256': digest(Path(__file__)), 'app_sha256': digest(directory / 'app.py'),
            'combined_sha256': digest(directory / 'combined.py'), 'cue_release_sha256': digest(RELEASE),
            'cue_head_id': reader.cue_head.identifier, 'candidate_id': reader.checkpoint.identifier,
            'model_revision': BACKBONE['revision']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode', required=True)
    replay = modes.add_parser('replay')
    replay.add_argument('--work', type=Path, required=True)
    replay.add_argument('--results', type=Path, required=True)
    replay.add_argument('--tokenizer', type=Path, required=True)
    replay.add_argument('--http', action='store_true')
    http = modes.add_parser('http')
    http.add_argument('--model', type=Path, required=True)
    for mode in (replay, http):
        mode.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
        mode.add_argument('--port', type=int, default=7861)
        mode.add_argument('--save-markup', type=Path)
        mode.add_argument('--markup-output', type=Path)
        mode.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = [path.resolve() for path in (args.output, args.save_markup, args.markup_output) if path is not None]
    if len(paths) != len(set(paths)) or any(path.exists() for path in paths):
        raise FileExistsError('Verification outputs must use different fresh paths')
    if args.mode == 'replay':
        if (args.save_markup or args.markup_output) and not args.http:
            parser.error('replay HTML output requires --http')
        result = archived_replay(args.work, args.results, args.tokenizer, args.checkpoint, args.http,
                                 args.port, args.save_markup, args.markup_output)
    else:
        import torch

        if torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != 'NVIDIA L4':
            raise RuntimeError('Live verification requires one Colab NVIDIA L4')
        reader = CombinedReader(args.model, checkpoint=args.checkpoint, cuda=True, with_cues=True, with_reference=True)
        fixtures = (('plain', ARTICLE), ('unicode-crlf', REFRESH_ARTICLE))
        result = {**identity(reader), 'device': torch.cuda.get_device_name(0),
                  **verify_http(reader, fixtures, args.port, args.save_markup, args.markup_output, check_prefix=True)}
    with args.output.open('x', encoding='utf-8') as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write('\n')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
