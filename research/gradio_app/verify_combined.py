"""Verify shared-forward readouts against sealed L4 assets and the Gradio route."""

import argparse
import asyncio
from dataclasses import replace
from html import unescape
import json
from pathlib import Path
import re

import numpy as np

from research.semantic_tracking.iirc_anchor_score import run as score_anchors
from research.stage2.records import write_jsonl
from research.stage3.probe import _jsonl
from research.stage3.run import save_json

from .causal_plaba import CHECKPOINT, digest
from .checkpoint import DEFAULT_CHECKPOINT
from .combined import IIRC_CHECKPOINT, CombinedReader, paragraphs, project_states
from .verify_causal_plaba import SCORE_TOLERANCE, scores_of


IIRC_FIXTURES = ('p_4754', 'p_4861', 'p_4969', 'p_5076', 'p_5183')
UNICODE_FIXTURE = 'p_4172'
PLABA_FIXTURE = 'text_01'
IIRC_SCORE_TOLERANCE = 0.0002
SUFFIX_SCORE_TOLERANCE = 0.000003


def source_fragment(markup):
    match = re.search(r'<div class="source-text"[^>]*>(.*?)</div>', markup, re.DOTALL)
    if match is None:
        raise AssertionError('No source text in Gradio output')
    return match[1]


async def verify_cached_views(ui, fixture, expected, forward_count):
    from gradio import SelectData
    from gradio.state_holder import SessionState

    from .app import ANCHORS, EASE, LEXICAL, anchor_detail, render_anchor_article, render_article, render_lexical_article, sentence_detail

    functions = {block.fn.__name__: index for index, block in ui.fns.items() if block.fn is not None}
    state = SessionState(ui)
    before = forward_count()
    analysis = await ui.process_api(functions['analyze'], [fixture, EASE], state=state)
    outputs = analysis['data']
    if forward_count() != before + 1 or outputs[4] != render_article(expected, EASE, 0):
        raise AssertionError('Initial analysis must render the sentence view from one forward')
    change = await ui.process_api(functions['update_measure'], [None, LEXICAL, None, None], state=state)
    if change['data'] != [render_lexical_article(expected)]:
        raise AssertionError('Cached lexical view differs from the initial analysis')

    anchor_index = len(expected.anchors) - 1
    anchor = outputs[7]['data'][anchor_index]
    event = SelectData(None, {'index': [anchor_index, 0], 'value': anchor[0], 'row_value': anchor})
    selection = await ui.process_api(functions['select_anchor'], [None], state=state, event_data=event)
    if selection['data'][1:] != [render_anchor_article(expected, anchor_index), anchor_detail(expected, anchor_index), ANCHORS]:
        raise AssertionError('Selecting an anchor did not retain the exact source span and visible stop')

    sentence_index = len(expected.sentences) - 1
    sentence = outputs[3]['data'][sentence_index]
    event = SelectData(None, {'index': [sentence_index, 0], 'value': sentence[0], 'row_value': sentence})
    selection = await ui.process_api(functions['select_sentence'], [None, ANCHORS], state=state, event_data=event)
    if selection['data'][1:] != [render_article(expected, EASE, sentence_index), sentence_detail(expected, sentence_index), EASE]:
        raise AssertionError('Selecting a sentence did not retain its coordinates and distribution')
    if forward_count() != before + 1:
        raise AssertionError('Changing a view or selecting a span invoked the backbone again')


def verify_ui(reader, fixture, expected, forward_count=None):
    from gradio_client import Client

    from .app import ANCHORS, CSS, LEXICAL, anchor_table, build_app, sentence_table

    ui = build_app(reader)
    try:
        _, address, _ = ui.launch(server_name='127.0.0.1', server_port=7861, share=False, css=CSS, quiet=True,
                                  prevent_thread_lock=True)
        client = Client(address, verbose=False)

        def predict(measure):
            before = forward_count() if forward_count is not None else None
            result = client.predict(fixture, measure, api_name='/analyze')
            if forward_count is not None and forward_count() != before + 1:
                raise AssertionError(f'Expected one model forward in the {measure} analysis request')
            return result

        table, article, detail, _, anchors, _ = predict(ANCHORS)
        fragment = source_fragment(article)
        marks = [(int(rank), int(start), int(end), int(stop), unescape(text))
                 for rank, start, end, stop, text in re.findall(
                     r'<mark class="anchor(?: active)?" data-rank="(\d+)" data-start="(\d+)" '
                     r'data-end="(\d+)" data-stop="(\d+)"[^>]*>(.*?)</mark>', fragment, re.DOTALL)]
        actual = sorted(marks)
        wanted = sorted((row.rank, row.start, row.end, row.visible_end, expected.text[row.start:row.end])
                        for row in expected.anchors)
        if (unescape(re.sub(r'<[^>]+>', '', fragment)) != fixture or table['data'] != sentence_table(expected)
                or anchors['data'] != anchor_table(expected) or actual != wanted or '實驗性評分分布' not in detail):
            raise AssertionError('Gradio changed the source coordinates, anchor proposals or sentence ratings')
        _, article, _, _, _, _ = predict(LEXICAL)
        fragment = source_fragment(article)
        marks = [(score, int(start), int(end), int(token), unescape(text))
                 for score, start, end, token, text in re.findall(
                     r'<mark class="lexical tone-\d" data-score="(\d\.\d\d)" data-start="(\d+)" '
                     r'data-end="(\d+)" data-token="(\d+)"[^>]*>(.*?)</mark>', fragment, re.DOTALL)]
        wanted = [(f'{row.difficulty:.2f}', row.start, row.end, row.token, expected.text[row.start:row.end])
                  for row in expected.lexical]
        if not marks or marks != wanted or unescape(re.sub(r'<[^>]+>', '', fragment)) != fixture:
            raise AssertionError('Gradio changed the word spans, scores or original article')
        if forward_count is not None:
            asyncio.run(verify_cached_views(ui, fixture, expected, forward_count))
    finally:
        ui.close()


def replay_dev(reader, path, output, l4_manifest, runtime, export_only):
    stimuli = path / 'stimuli.jsonl'
    if digest(stimuli) != l4_manifest['evaluation_sha256']:
        raise ValueError('The local replay must use the original unmodified held-out dev paragraphs')
    l4 = {row['paragraph_id']: row['methods'] for row in _jsonl(IIRC_CHECKPOINT / 'predictions.jsonl')}
    score_rows, drifts = [], []
    same_order = same_set = 0
    for row in _jsonl(stimuli):
        identifier = row['id']
        anchors, unavailable = reader.locate(reader.backbone.encode(row['text']))
        if unavailable or not anchors or any(anchor.experimental_context for anchor in anchors):
            raise AssertionError(f'Invalid held-out IIRC paragraph {identifier}')
        candidate = [{'start': anchor.start, 'end': anchor.end, 'score': anchor.score} for anchor in anchors]
        sealed = l4[identifier]
        actual = {(item['start'], item['end']): item['score'] for item in candidate}
        expected = {(item['start'], item['end']): item['score'] for item in sealed['qwen']}
        same_order += list(actual) == list(expected)
        same_set += actual.keys() == expected.keys()
        drifts.extend(abs(actual[key] - expected[key]) for key in actual.keys() & expected.keys())
        score_rows.append({'paragraph_id': identifier, 'methods': {'form': sealed['form'], 'qwen': candidate}})
    if len(score_rows) != 430 or {row['paragraph_id'] for row in score_rows} != l4.keys():
        raise AssertionError('Local IIRC replay does not cover all 430 held-out paragraphs')

    output.mkdir(parents=True, exist_ok=False)
    predictions = output / 'predictions.jsonl'
    seal = output / 'serving-seal.json'
    write_jsonl(predictions, score_rows)
    save_json(seal, {
        'model': l4_manifest['model'], 'revision': l4_manifest['revision'], 'runtime': runtime,
        'heads_sha256': l4_manifest['heads_sha256'], 'evaluation_sha256': l4_manifest['evaluation_sha256'],
        'predictions_sha256': digest(predictions), 'l4_form_control_source_sha256': l4_manifest['predictions_sha256'],
    })
    local_report = {
        'paragraphs': len(score_rows), 'identical_ordered_spans': same_order, 'identical_span_sets': same_set,
        'common_candidate_spans': len(drifts), 'score_mean_absolute_difference': float(np.mean(drifts)),
        'score_max_absolute_difference': max(drifts),
        'predictions_sha256': digest(predictions), 'seal_sha256': digest(seal),
    }
    if not export_only:
        scores_path = output / 'scores.json'
        score_anchors(stimuli, path / 'visible-anchors.jsonl', path / 'questions.jsonl', path / 'pairs.jsonl',
                      predictions, IIRC_CHECKPOINT / 'heads.npz', seal, scores_path)
        scores = json.loads(scores_path.read_text(encoding='utf-8'))
        local_report.update({name: scores[name] for name in ('per_question_coverage', 'distinct_same_stop_pair_coverage', 'acceptance')})
        local_report['scores_sha256'] = digest(scores_path)
    save_json(output / 'replay-report.json', local_report)
    return local_report


def run(model, iirc_stimuli, plaba_features, plaba_stimuli, check_ui=False, iirc_dev=None, output=None, cuda=False,
        iirc_export_only=False, checkpoint=DEFAULT_CHECKPOINT):
    iirc_manifest = json.loads((IIRC_CHECKPOINT / 'fit-manifest.json').read_text(encoding='utf-8'))
    plaba_manifest = json.loads((CHECKPOINT / 'manifest.json').read_text(encoding='utf-8'))
    iirc_predictions = IIRC_CHECKPOINT / 'predictions.jsonl'
    if (digest(iirc_stimuli) != iirc_manifest['stimuli_sha256'] or digest(iirc_predictions) != iirc_manifest['predictions_sha256']
            or digest(plaba_features) != plaba_manifest['data']['features_sha256']
            or digest(plaba_stimuli) != plaba_manifest['data']['current_stage2_sha256']['stimuli.jsonl']):
        raise ValueError('Use the immutable source text and predictions from the sealed L4 runs')
    texts = {row['id']: row['text'] for row in _jsonl(iirc_stimuli) if row['id'] in (*IIRC_FIXTURES, UNICODE_FIXTURE)}
    predictions = {row['paragraph_id']: row['methods']['qwen'] for row in _jsonl(iirc_predictions)
                   if row['paragraph_id'] in IIRC_FIXTURES}
    document = next(row for row in _jsonl(plaba_stimuli) if row['id'] == PLABA_FIXTURE)
    with np.load(plaba_features, allow_pickle=False) as archive:
        plaba_states = dict(zip(archive['ids'], archive['features'][:, :2048], strict=True))

    calls = 0

    def register_call(*_):
        nonlocal calls
        calls += 1

    reader = CombinedReader(model, checkpoint=checkpoint, cuda=cuda)
    if cuda:
        reader.backbone.encoder.register_forward_hook(register_call)
        runtime = 'NVIDIA L4 FP32'
    else:
        encoder = reader.backbone.encoder

        def counted(tokens):
            register_call()
            return encoder(tokens)

        reader.backbone.encoder = counted
        runtime = f'{reader.backbone.mx.default_device()} FP32'
    if len(reader.backbone.tokenizer.encode(texts[UNICODE_FIXTURE], add_special_tokens=False).ids) != 277:
        raise AssertionError('The production AutoTokenizer Devanagari correction is absent')

    plaba = reader.analyze(document['text'])
    bounds = [(segment['start'], segment['end']) for segment in document['segments']]
    if ([(row.start, row.end) for row in plaba.sentences] != bounds or calls != 1
            or not plaba.anchors or not plaba.lexical):
        raise AssertionError('Shared forward did not produce all three readouts at certified source positions')
    reference_states = np.stack([plaba_states[segment['id']] for segment in document['segments']])
    sentence_text = [document['text'][start:end] for start, end in bounds]
    l4_scores = reader.score(reference_states, sentence_text)
    plaba_error = float(np.max(np.abs(scores_of(plaba) - l4_scores)))
    np.testing.assert_allclose(scores_of(plaba), l4_scores, atol=SCORE_TOLERANCE, rtol=0)
    ordinal = np.array([(row.ease, row.understanding) for row in plaba.distributions])
    reference_ordinal = np.array([(row.ease, row.understanding)
                                  for row in reader.ordinal_head.score(reference_states, sentence_text)])
    np.testing.assert_allclose(ordinal.sum(axis=-1), 1, atol=1e-12, rtol=0)
    np.testing.assert_allclose(ordinal, reference_ordinal, atol=0.005, rtol=0)

    iirc_report = {}
    sample = None
    for paragraph_id in IIRC_FIXTURES:
        before = calls
        text = texts[paragraph_id]
        analysis = reader.analyze(text)
        local = [(span.start, span.end) for span in analysis.anchors]
        reference = [(span['start'], span['end']) for span in predictions[paragraph_id]]
        local_scores = [span.score for span in analysis.anchors]
        sealed_scores = [span['score'] for span in predictions[paragraph_id]]
        if (calls != before + 1 or not local or analysis.unavailable_paragraphs
                or any(span.paragraph_number != 1 or span.paragraph_start != 0 or span.paragraph_end != len(text)
                       or span.visible_end != len(text) or span.experimental_context for span in analysis.anchors)):
            raise AssertionError(f'Combined model did not produce certified IIRC proposals at {paragraph_id}')
        actual_by_span = dict(zip(local, local_scores, strict=True))
        sealed_by_span = dict(zip(reference, sealed_scores, strict=True))
        common = actual_by_span.keys() & sealed_by_span.keys()
        difference = max((abs(actual_by_span[span] - sealed_by_span[span]) for span in common), default=None)
        iirc_report[paragraph_id] = {
            'same_ranked_source_spans': sum(a == b for a, b in zip(local, reference)),
            'common_source_spans': len(common), 'identical_ordered_spans': local == reference,
            'common_span_score_max_absolute_difference': difference,
            'numeric_parity_at_nominal_tolerance': (local == reference and difference is not None
                                                    and difference <= IIRC_SCORE_TOLERANCE),
        }
        if sample is None:
            sample = analysis

    # Use independent float64 row products as the reference for the serving projection.
    sample_states = reader.backbone.encode(sample.text).states[::13]
    matrix = reader.anchor_head['projection']
    reference_product = np.einsum('ij,jk->ik', sample_states.astype(np.float64), matrix.astype(np.float64))
    reference_projection = reference_product / np.linalg.norm(reference_product, axis=-1, keepdims=True)
    actual_projection = project_states(sample_states, matrix)
    np.testing.assert_allclose(actual_projection, reference_projection, atol=0.000002, rtol=0)

    prefix = sample.text.rstrip('\r\n') + '\n\n'
    text = prefix + 'The second passage examines the sea. Two groups tracked migrating whales.'
    replacement = prefix + 'The second passage examines plants. Two groups tracked spring blooms.'
    before = calls
    left, right = reader.analyze(text), reader.analyze(replacement)
    if calls != before + 2 or any(analysis.unavailable_paragraphs or {anchor.paragraph_number for anchor in analysis.anchors} != {1, 2}
                                   or not all(anchor.experimental_context for anchor in analysis.anchors)
                                   for analysis in (left, right)):
        raise AssertionError('Expected one model forward and certified experimental paragraphs per multi-paragraph analysis')
    a, b = ([span for span in analysis.anchors if span.paragraph_number == 1] for analysis in (left, right))
    if [(spot.start, spot.end, spot.visible_end) for spot in a] != [(spot.start, spot.end, spot.visible_end) for spot in b]:
        raise AssertionError('Later paragraph changed the earlier certified anchor coordinates')
    np.testing.assert_allclose([spot.score for spot in a], [spot.score for spot in b], atol=SUFFIX_SCORE_TOLERANCE, rtol=0)
    left_words, right_words = ([word for word in article.lexical if word.end <= len(prefix)] for article in (left, right))
    coordinates = [(word.start, word.end, word.token) for word in left_words]
    if not coordinates or coordinates != [(word.start, word.end, word.token) for word in right_words]:
        raise AssertionError('Later paragraph changed earlier certified word coordinates')
    word_diff = np.abs(np.array([word.difficulty for word in left_words]) -
                       np.array([word.difficulty for word in right_words]))
    np.testing.assert_array_less(word_diff, SUFFIX_SCORE_TOLERANCE)
    encoded = reader.backbone.encode(text)
    invalid_prefix = replace(encoded, tokens=(-1, *encoded.tokens[1:]))
    blocked, excluded = reader.locate(invalid_prefix)
    if blocked or excluded != tuple((start, end) for start, end, _ in paragraphs(text)):
        raise AssertionError('Proposals escaped the uncertified token-prefix guard')

    ui_forwards = None
    if check_ui:
        verify_ui(reader, sample.text, sample, forward_count=lambda: calls)
        ui_forwards = 1
    report = {
        'model': iirc_manifest['revision'], 'iirc_head_sha256': iirc_manifest['heads_sha256'],
        'checkpoint_sha256': reader.checkpoint.identifier,
        'ordinal_head_sha256': reader.ordinal_head.head_sha256,
        'ordinal_l4_max_probability_difference': float(np.max(np.abs(ordinal - reference_ordinal))),
        'plaba_head_sha256': plaba_manifest['head_sha256'], 'device': runtime,
        'plaba_sentences': len(plaba.sentences), 'plaba_score_max_absolute_difference': plaba_error,
        'iirc_fixtures': iirc_report, 'iirc_score_absolute_tolerance': IIRC_SCORE_TOLERANCE,
        'iirc_projection_max_absolute_difference_against_float64': float(np.max(np.abs(actual_projection - reference_projection))),
        'known_unicode_token_count': 277, 'uncertified_prefixes_withheld': len(excluded),
        'anchor_suffix_max_absolute_difference': float(np.max(np.abs(np.array([spot.score for spot in a]) -
                                                                   np.array([spot.score for spot in b])))),
        'model_forwards_per_combined_gradio_request': ui_forwards,
        'gradio_cached_views_verified': check_ui,
        'model_forwards_for_three_cached_interactions': 0 if check_ui else None,
        'lexical_head_sha256': reader.lexical_head.head_sha256,
        'lexical_words_in_plaba_fixture': len(plaba.lexical),
        'lexical_prefix_words': len(coordinates),
        'lexical_suffix_max_absolute_difference': float(word_diff.max()),
    }
    if iirc_dev is not None:
        report['iirc_serving_replay'] = replay_dev(reader, iirc_dev, output, iirc_manifest, runtime, iirc_export_only)
        save_json(output / 'verification.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'iirc_stimuli', 'plaba_features', 'plaba_stimuli'):
        parser.add_argument(f'--{name.replace("_", "-")}', type=Path, required=True)
    parser.add_argument('--check-ui', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--cuda', action='store_true', help='Verify the NVIDIA L4 serving backend')
    parser.add_argument('--iirc-dev', type=Path, help='Prepared IIRC dev directory for the complete local replay')
    parser.add_argument('--output', type=Path, help='New directory for replay artifacts')
    parser.add_argument('--iirc-export-only', action='store_true', help='Seal replay predictions without opening IIRC outcomes')
    args = parser.parse_args()
    if (args.iirc_dev is None) != (args.output is None):
        parser.error('--iirc-dev and --output must be supplied together')
    if args.iirc_export_only and args.iirc_dev is None:
        parser.error('--iirc-export-only requires --iirc-dev')
    print(json.dumps(run(**vars(args)), indent=2))
