"""Check the causal PLABA local runtime against the sealed L4 sentence states."""

import argparse
from html import unescape
import json
from pathlib import Path
import re

import numpy as np

from .causal_plaba import CHECKPOINT, CausalPlabaReader, digest


FIXTURES = ('text_01', 'text_14', 'text_38_simp')
SCORE_TOLERANCE = 0.02
CROSS_ENGINE_STATE_TOLERANCE = 0.01
CAUSAL_TOLERANCE = 0.001


def relative_rms(left, right):
    residual = np.sqrt(np.mean(np.square(left - right), axis=-1))
    scale = np.maximum(np.sqrt(np.mean(np.square(right), axis=-1)), 1e-8)
    return float(np.max(residual / scale))


def scores_of(analysis):
    return np.asarray([(row.ease, row.understanding) for row in analysis.sentences])


def run(model, features, stimuli, check_ui=False):
    provenance = json.loads((CHECKPOINT / 'manifest.json').read_text(encoding='utf-8'))
    if (digest(features) != provenance['data']['features_sha256']
            or digest(stimuli) != provenance['data']['current_stage2_sha256']['stimuli.jsonl']):
        raise ValueError('The verification fixtures must use the sealed Stage 3 and Stage 2 inputs')
    with stimuli.open(encoding='utf-8') as source:
        documents = {row['id']: row for line in source if (row := json.loads(line))['id'] in FIXTURES}
    with np.load(features, allow_pickle=False) as archive:
        reference = dict(zip(archive['ids'], archive['features'][:, :2048], strict=True))

    reader = CausalPlabaReader(model)
    encoder = reader.backbone.encoder
    calls = 0

    def counted(tokens):
        nonlocal calls
        calls += 1
        return encoder(tokens)

    reader.backbone.encoder = counted
    report = {}
    for document_id in FIXTURES:
        document = documents[document_id]
        text, segments = document['text'], document['segments']
        bounds = [(segment['start'], segment['end']) for segment in segments]
        encoded = reader.backbone.encode(text)
        analysis = reader.rate(encoded, bounds)
        if analysis.unavailable_sentences or analysis.sentence_numbers != tuple(range(1, 1 + len(bounds))):
            raise AssertionError(f'Unexpected uncertified L4 alignment for {document_id}')
        positions = [reader.position(encoded, start, end) for start, end in bounds]
        l4_states = np.stack([reference[segment['id']] for segment in segments])
        source_sentences = [text[start:end] for start, end in bounds]
        l4_scores = reader.score(l4_states, source_sentences)
        np.testing.assert_allclose(scores_of(analysis), l4_scores, atol=SCORE_TOLERANCE, rtol=0)
        state_error = relative_rms(encoded.states[positions], l4_states)
        if state_error > CROSS_ENGINE_STATE_TOLERANCE:
            raise AssertionError(f'Local hidden states differ from L4 on {document_id}: {state_error}')
        report[document_id] = {'sentences': len(bounds), 'hidden_max_relative_rms': state_error,
                               'scores_max_absolute_difference': float(np.max(np.abs(scores_of(analysis) - l4_scores)))}
    if calls != len(FIXTURES):
        raise AssertionError('Expected one backbone invocation per fixture')

    original = documents[FIXTURES[0]]
    prefix_end = original['segments'][0]['end']
    prefix = original['text'][:prefix_end]
    bounds = [(0, len(prefix))]
    inputs = (prefix, original['text'], prefix + ' A later sentence introduces entirely different information.')
    passes = [reader.backbone.encode(text) for text in inputs]
    if any(encoded.tokens[:len(passes[0].tokens)] != passes[0].tokens for encoded in passes[1:]):
        raise AssertionError('Selected suffix control changes the input token prefix')
    if passes[1].tokens[len(passes[0].tokens):] == passes[2].tokens[len(passes[0].tokens):]:
        raise AssertionError('Suffix control failed to change subsequent input tokens')
    scored = [scores_of(reader.rate(encoded, bounds)) for encoded in passes]
    np.testing.assert_allclose(scored[1:], np.broadcast_to(scored[0], (2, 1, 2)), atol=CAUSAL_TOLERANCE, rtol=0)
    causal_state_error = max(relative_rms(row.states[:len(passes[0].tokens)], passes[0].states) for row in passes[1:])
    if causal_state_error > CAUSAL_TOLERANCE:
        raise AssertionError(f'Suffix reaches FP32 prefix states: {causal_state_error}')

    cut = next(left + 1 for left, right in passes[1].offsets if right - left > 1)
    partial = reader.rate(passes[1], [(0, cut), (cut, len(passes[1].text))])
    if partial.sentence_numbers != (2,) or partial.unavailable_sentences != ((0, cut),):
        raise AssertionError('Uncertified tokenizer boundary was not held out at its original position')

    before = calls
    live = reader.analyze(original['text'])
    if calls != before + 1 or not live.sentences or live.text != original['text']:
        raise AssertionError('Pasted-text app path must preserve original text and invoke the model once')
    app_forwards = calls - before
    if check_ui:
        from gradio_client import Client
        from .app import CSS, EASE, build_app

        demo = build_app(reader)
        try:
            _, address, _ = demo.launch(server_name='127.0.0.1', share=False, css=CSS, quiet=True,
                                        prevent_thread_lock=True)
            before = calls
            table, article, _, _ = Client(address, verbose=False).predict(original['text'], EASE, api_name='/analyze')
            app_forwards = calls - before
            source = re.search(r'<div class="source-text"[^>]*>(.*?)</div>', article, re.DOTALL)
            if (app_forwards != 1 or source is None or unescape(re.sub(r'<[^>]+>', '', source[1])) != original['text']
                    or [row[0] for row in table['data']] != list(live.sentence_numbers)):
                raise AssertionError('Causal Gradio did not preserve article offsets or invoke the backbone once')
            np.testing.assert_allclose([row[2:] for row in table['data']], np.round(scores_of(live), 2), atol=1e-9, rtol=0)
        finally:
            demo.close()
    return {
        'checkpoint': provenance['revision'], 'head_sha256': provenance['head_sha256'],
        'device': str(reader.backbone.mx.default_device()), 'compute_dtype': 'float32',
        'l4_parity': report, 'l4_score_absolute_tolerance': SCORE_TOLERANCE,
        'l4_state_relative_rms_tolerance': CROSS_ENGINE_STATE_TOLERANCE,
        'causal_prefix_state_max_relative_rms': causal_state_error,
        'causal_prefix_score_max_absolute_difference': float(np.max(np.abs(np.asarray(scored[1:]) - scored[0]))),
        'causal_tolerance': CAUSAL_TOLERANCE, 'paste_path_sentences': len(live.sentences),
        'paste_path_unavailable_sentences': len(live.unavailable_sentences), 'forwards_per_app_request': app_forwards,
        'uncertified_boundary_retained_original_numbering': True,
        'gradio_roundtrip_checked': check_ui,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('model', 'features', 'stimuli'):
        parser.add_argument(f'--{key}', type=Path, required=True)
    parser.add_argument('--check-ui', action='store_true')
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
