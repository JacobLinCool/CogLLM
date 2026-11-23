"""Check paragraph geometry, score provenance, and cached views without loading Qwen."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .app import ANCHOR_LABEL, ANCHORS, EASE, LEXICAL, REFERENCE, UNDERSTANDING, build_app, render_article
from .causal_plaba import CausalEncoding, HIDDEN_WIDTH, digest
from .combined import CombinedReader
from .plaba import MAX_CHARACTERS, TOKENIZER_SHA256, sentence_spans
from .verify_article_route import SourceMarkup


class VerifiedTokenizerBackbone:
    """Return synthetic causal states so the serving geometry can be checked without Qwen."""

    def __init__(self, tokenizer_path):
        if digest(tokenizer_path) != TOKENIZER_SHA256:
            raise ValueError('Expected the frozen tokenizer')
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        position = np.arange(1, len(encoded.ids) + 1, dtype=np.float32)[:, None]
        dimensions = np.arange(1, HIDDEN_WIDTH + 1, dtype=np.float32)[None, :]
        states = np.cos(position * dimensions / HIDDEN_WIDTH).astype(np.float32)
        return CausalEncoding(text, tuple(encoded.ids), tuple(encoded.offsets), states)


def verify(checkpoint, tokenizer):
    backbone = VerifiedTokenizerBackbone(tokenizer)
    reader = CombinedReader(tokenizer.parent, checkpoint=checkpoint, backbone=backbone, with_reference=True)
    fixtures = (
        ('The installer was Acme Inc.', '\n\n', 'Green lights came on.'),
        ('He arrived at 3 p.m.', '\r\n\r\n', 'Green lights came on.'),
        ('The installer was Acme Inc.', '\r\n \r\n', 'Blue lights flashed!'),
    )
    for first, separator, second in fixtures:
        text = first + separator + second
        assert sentence_spans(text, reader.segmenter) == [(0, len(text))], 'Fixture no longer reproduces the old merge'
        expected = [(0, len(first)), (len(first) + len(separator), len(text))]
        analysis = reader.analyze(text)
        assert [(item.start, item.end) for item in analysis.sentences] == expected
        assert analysis.sentence_numbers == (1, 2) and not analysis.unavailable_sentences
        before_next = backbone.tokenizer.encode(first, add_special_tokens=False).ids
        assert analysis.sentence_tokens[0] == len(before_next) - 1

        parsed = SourceMarkup()
        parsed.feed(render_article(analysis, EASE, 0))
        assert ''.join(parsed.chunks) == text
        assert [(mark.start, mark.end) for mark in parsed.marks if mark.kind == 'sentence'] == expected
        assert any(mark.kind == 'lexical' for mark in parsed.marks)
        verify_score_scope(analysis)
    assert backbone.calls == [first + separator + second for first, separator, second in fixtures]

    for invalid in (' \r\n\t', 'first.\n\n' + 'a' * MAX_CHARACTERS):
        try:
            reader.analyze(invalid)
        except ValueError:
            continue
        raise AssertionError('Invalid article was accepted')
    assert len(backbone.calls) == len(fixtures), 'Invalid text performed a model forward'
    verify_reference_reset(reader)
    return len(fixtures)


def verify_score_scope(analysis):
    first = replace(analysis.sentences[0], ease=4.24, understanding=3.50)
    second = replace(analysis.sentences[1], ease=4.25, understanding=3.51)
    close = replace(analysis, sentences=(first, second))
    for metric, expected in ((EASE, ('篇內較難 4.24', '篇內較易 4.25')),
                             (UNDERSTANDING, ('自評較低 3.50', '自評較高 3.51'))):
        rendered = render_article(close, metric, 0)
        assert all(label in rendered for label in expected)
        assert f'模型估計{metric} · 1–5 分 · 篇內相對色階' in rendered
        assert '參照全文可見時的英語生醫摘要讀者群體自評' in rendered
        assert '每句估計只使用讀到該句句末的文字' in rendered
        if metric == UNDERSTANDING:
            assert '篇內較難' not in rendered and '篇內較易' not in rendered
        parsed = SourceMarkup()
        parsed.feed(rendered)
        assert ''.join(parsed.chunks) == analysis.text
        marks = [item for item in parsed.marks if item.kind == 'sentence']
        assert 'tone-0' in marks[0].attrs['class'] and 'tone-4' in marks[1].attrs['class']
        assert all(f'模型估計{metric}' in item.attrs['title'] for item in marks)
        assert [int(item.attrs['data-token']) for item in marks] == list(analysis.sentence_tokens)
    tied = replace(close, sentences=(first, replace(second, ease=first.ease)))
    assert 'tone-2">同分 4.24' in render_article(tied, EASE, 0)


def verify_reference_reset(reader):
    old_text = 'A guide checked the lamp. It flickered.'
    new_text = 'The first kettle boiled. The stove cooled.'
    ui = build_app(reader)
    try:
        callbacks = {block.fn.__name__: block for block in ui.fns.values() if block.fn is not None}
        analyze = callbacks['analyze'].fn
        query = callbacks['find_reference'].fn
        reset = callbacks['reset_reference_view']
        radio = callbacks['analyze'].inputs[1]
        assert callbacks['analyze'].api_name == 'analyze' and radio.value == EASE
        assert (ANCHOR_LABEL, ANCHORS) in radio.choices
        old_analysis = analyze(old_text, EASE)[0]
        _, reference, _, _, selected_mode = query(old_analysis, old_text, 'It')
        assert reference.candidates and selected_mode == REFERENCE
        outputs = analyze(new_text, selected_mode)
        assert outputs[0].text == new_text and outputs[-1] is None
        rendered = SourceMarkup()
        rendered.feed(outputs[4])
        assert ''.join(rendered.chunks) == new_text
        assert {mark.kind for mark in rendered.marks} == {'sentence', 'lexical'}
        assert reset.fn(selected_mode)[2] == EASE
        assert reset.inputs[0] is reset.outputs[2], 'The successful callback must reset the visible radio'
        for metric in (EASE, UNDERSTANDING, LEXICAL, ANCHORS):
            assert reset.fn(metric)[2] == {'__type__': 'update'}, f'Reanalysis changed {metric}'
        anchors = callbacks['update_measure'].fn(outputs[0], ANCHORS, 0, -1, None)
        assert ANCHOR_LABEL in anchors
        assert '假設已有提問，模型依段末可見文字排序本段的可能指向' in anchors
        assert reader.backbone.calls[-2:] == [old_text, new_text]
    finally:
        ui.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    print(f'Verified {verify(args.checkpoint, args.tokenizer)} paragraph and legend fixtures, Gradio API and cached views')
