"""Serve article readouts from the same causal Qwen states."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np

from research.semantic_tracking.iirc_anchor_fit import FORM_FEATURES, MAX_WORDS, TOP_SPANS, predict_passage
from research.stage3.run import MODEL

from .causal_plaba import CausalArticleRatings, CausalEncoding, CausalPlabaReader, HIDDEN_WIDTH, digest
from .checkpoint import DEFAULT_CHECKPOINT, load_checkpoint
from .lexical import LexicalRating, LexicalReadout
from .ordinal import OrdinalReadout, SentenceDistribution
from .plaba import BASE_REVISION, sentence_spans, validate_article_text
from .reference import ReferenceReadout, ReferenceStates
from .tokenizer import ExtractionTokenizer

if TYPE_CHECKING:
    from .word_cues import CueCandidate


IIRC_CHECKPOINT = Path(__file__).resolve().parents[1] / 'results' / 'iirc-anchor-run'
LINE_BREAK = r'(?:\r\n|\n|\r(?!\n))'
PARAGRAPH_BREAK = re.compile(rf'{LINE_BREAK}(?:[^\S\r\n]*{LINE_BREAK})+')
PROJECTION_WIDTH = 128


@dataclass(frozen=True)
class AnchorProposal:
    start: int
    end: int
    score: float
    paragraph_number: int
    paragraph_start: int
    paragraph_end: int
    visible_end: int
    rank: int
    experimental_context: bool


@dataclass(frozen=True)
class CombinedArticleRatings(CausalArticleRatings):
    anchors: tuple[AnchorProposal, ...]
    unavailable_paragraphs: tuple[tuple[int, int], ...]
    lexical: tuple[LexicalRating, ...]
    unavailable_words: tuple[tuple[int, int], ...]
    distributions: tuple[SentenceDistribution, ...]
    reference: ReferenceStates | None = None
    cues: tuple['CueCandidate', ...] = ()


def paragraphs(text: str) -> list[tuple[int, int, int]]:
    """Return source spans with the paragraph separator included in the visible stop."""
    bounds = []
    start = 0
    for separator in PARAGRAPH_BREAK.finditer(text):
        if text[start:separator.start()].strip():
            bounds.append((start, separator.start(), separator.end()))
        start = separator.end()
    if text[start:].strip():
        bounds.append((start, len(text), len(text)))
    return bounds


def load_anchor_head(path: Path) -> dict[str, np.ndarray]:
    manifest_path = path / 'fit-manifest.json'
    weights = path / 'heads.npz'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    scores = json.loads((path / 'scores.json').read_text(encoding='utf-8'))
    checksum = digest(weights)
    if (manifest['model'] != MODEL or manifest['revision'] != BASE_REVISION
            or manifest['heads_sha256'] != checksum or scores['input_sha256']['heads'] != checksum
            or scores['input_sha256']['seal'] != digest(manifest_path)
            or manifest['top_spans'] != TOP_SPANS or manifest['max_words'] != MAX_WORDS
            or tuple(manifest['form_features']) != FORM_FEATURES or not scores['acceptance']['all_conditions_passed']):
        raise ValueError('IIRC 權重與通過留出驗收的 checkpoint 不一致。')
    shapes = {'form': (len(FORM_FEATURES),), 'qwen_form': (len(FORM_FEATURES),),
              'qwen_start': (PROJECTION_WIDTH,), 'qwen_end': (PROJECTION_WIDTH,),
              'projection': (HIDDEN_WIDTH, PROJECTION_WIDTH)}
    with np.load(weights, allow_pickle=False) as artifact:
        if set(artifact.files) != set(shapes):
            raise ValueError('Unexpected IIRC readout parameters')
        values = {name: artifact[name] for name in shapes}
    if any(value.shape != shapes[name] or value.dtype != np.float32 or not np.isfinite(value).all()
           for name, value in values.items()):
        raise ValueError('Invalid IIRC readout dimensions or coefficients')
    return values


def project_states(states: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    projected = states @ matrix
    norms = np.linalg.norm(projected, axis=-1, keepdims=True)
    if not np.all(np.isfinite(norms) & (norms > 0)):
        raise RuntimeError('Invalid projected IIRC token states')
    return projected / norms


class CombinedReader(CausalPlabaReader):
    """Apply the fixed readouts to one plain-text forward."""

    def __init__(self, model_dir: Path, *, checkpoint: Path = DEFAULT_CHECKPOINT,
                 cuda: bool = False, backbone=None, with_reference: bool = False, with_cues: bool = False):
        self.checkpoint = load_checkpoint(checkpoint)
        self.anchor_head = load_anchor_head(checkpoint / 'anchor')
        self.reference_head = ReferenceReadout(self.checkpoint, self.anchor_head['projection']) if with_reference else None
        self.lexical_head = LexicalReadout(checkpoint / 'lexical')
        self.ordinal_head = OrdinalReadout(checkpoint / 'ordinal')
        self.cue_head = None
        if with_cues:
            from .word_cues import CueReadout

            self.cue_head = CueReadout(self.checkpoint)
        if cuda:
            if backbone is not None:
                raise ValueError('Select one combined backbone')
            from .cuda import CudaCausalQwen

            backbone = CudaCausalQwen(model_dir)
        super().__init__(model_dir, backbone=backbone, checkpoint=checkpoint / 'sentence',
                         segmenter=self.checkpoint.segmenter())
        if backbone is None:
            self.backbone.tokenizer = ExtractionTokenizer(model_dir)

    def locate(self, encoding: CausalEncoding, *, projected=None) -> tuple[tuple[AnchorProposal, ...], tuple[tuple[int, int], ...]]:
        """Compute candidates from states that were visible at each paragraph stop."""
        proposals, unavailable = [], []
        offsets = np.asarray(encoding.offsets, dtype=np.int64)
        if projected is None:
            projected = project_states(encoding.states, self.anchor_head['projection'])
        text = encoding.text
        spans = paragraphs(text)
        for number, (start, end, visible_end) in enumerate(spans, 1):
            prefix = self.backbone.tokenizer.encode(text[:visible_end], add_special_tokens=False).ids
            if (not prefix or encoding.tokens[:len(prefix)] != tuple(prefix)
                    or any(right > visible_end for _, right in encoding.offsets[:len(prefix)])):
                unavailable.append((start, end))
                continue
            visible = offsets[:len(prefix)]
            mask = (visible[:, 0] < end) & (visible[:, 1] > start)
            if not np.any(mask):
                unavailable.append((start, end))
                continue
            local_offsets = np.clip(visible[mask] - start, 0, end - start)
            proposals.extend(
                AnchorProposal(start + candidate['start'], start + candidate['end'], candidate['score'],
                               number, start, end, visible_end, rank,
                               experimental_context=len(spans) > 1 or start != 0 or end != len(text))
                for rank, candidate in enumerate(predict_passage(text[start:end], local_offsets,
                                                                  projected[:len(prefix)][mask], self.anchor_head)['qwen'], 1)
            )
        return tuple(proposals), tuple(unavailable)

    def analyze(self, text: str) -> CombinedArticleRatings:
        validate_article_text(text)
        spans = [(start + left, start + right)
                 for start, end, _ in paragraphs(text)
                 for left, right in sentence_spans(text[start:end], self.segmenter)]
        encoded = self.backbone.encode(text)
        projected = project_states(encoded.states, self.anchor_head['projection'])
        ratings = self.rate(encoded, spans, allow_empty=True)
        anchors, unavailable = self.locate(encoded, projected=projected)
        lexical, unavailable_words = self.lexical_head.rate(encoded, self.backbone.tokenizer)
        sentence_text = [text[span.start:span.end] for span in ratings.sentences]
        distributions = self.ordinal_head.score(encoded.states[list(ratings.sentence_tokens)], sentence_text)
        reference = (self.reference_head.cache(encoded, projected, self.backbone.tokenizer, lexical)
                     if self.reference_head is not None else None)
        cues = self.cue_head.rate(encoded, self.backbone.tokenizer, spans, lexical) if self.cue_head is not None else ()
        return CombinedArticleRatings(ratings.text, ratings.sentences, ratings.sentence_numbers,
                                      ratings.unavailable_sentences, ratings.sentence_tokens, anchors, unavailable,
                                      lexical, unavailable_words, distributions, reference, cues)
