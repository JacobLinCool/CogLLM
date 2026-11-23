"""Load a sealed token readout and produce its whole-document app trace."""

from contextlib import nullcontext
from dataclasses import replace
from math import isclose, isfinite, log
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import torch

from research.backbone_transfer.extract import digest
from research.token_readout.extraction import QwenRuntime
from research.token_readout.model import Scores, TokenReadout, question_mass


def trace_rows(text, offsets, scores, priors, anchor_width=1):
    """Map the causal mass distribution to exact Unicode spans for display."""
    n = len(offsets)
    if (not n or scores.opening.shape != (n,) or scores.resolution.shape != (n, n)
            or scores.continuity.shape != (n,)):
        raise ValueError('Token offsets and model score dimensions disagree')

    device = scores.opening.device
    candidate = torch.tensor([any(ch.isalnum() for ch in text[start:end]) for start, end in offsets],
                             dtype=torch.bool, device=device)
    opened, unresolved, resolved = question_mass(scores.opening + priors['opening'],
                                                 scores.resolution + priors['resolution'], candidate)
    previous = torch.cat((torch.zeros_like(unresolved[:1]), unresolved[:-1]), dim=0)
    discontinuity = (0.5 * (1 - scores.continuity)).clamp(0.0, 1.0)

    # Repeated Unicode offsets can represent several pieces of one character.
    # A displayed pointer always refers to a completed earlier source span.
    bounds = torch.tensor(offsets, device=device)
    positions = torch.arange(n, device=device)
    eligible = ((positions[:, None] > positions[None, :])
                & (bounds[:, :1] >= bounds[None, :, 1])
                & candidate[None, :])
    associated = resolved + discontinuity[:, None] * previous
    strength, pointer = associated.masked_fill(~eligible, -1).max(dim=1)

    values = torch.stack((unresolved.sum(dim=1), opened, resolved.sum(dim=1),
                          previous.sum(dim=1) * discontinuity), dim=1)
    if not torch.isfinite(values).all() or not torch.isfinite(strength).all():
        raise ArithmeticError('Nonfinite token trace')
    values = values.tolist()
    references = zip(pointer.tolist(), strength.tolist(), strict=True)
    rows = []
    for (start, end), (mass, onset, closure, transition), (anchor, association) in zip(offsets, values, references, strict=True):
        if association > 0:
            pointer_start = offsets[max(0, anchor + 1 - anchor_width)][0]
            pointer_end = offsets[anchor][1]
        else:
            pointer_start, pointer_end = None, None
        rows.append({'start': start, 'end': end, 'unresolved': mass, 'opened': onset,
                     'resolved': closure, 'transition': transition,
                     'pointer_start': pointer_start, 'pointer_end': pointer_end})
    return rows


class TokenRunner:
    def __init__(self, runtime, reader, adapter, priors, checkpoint, *, scale=None, anchor_width=1,
                 english_head=None, transition_head=None, narrative_head=None, terminology_head=None):
        self.runtime = runtime
        self.reader = reader
        self.adapter = adapter
        self.priors = priors
        self.checkpoint = checkpoint
        self.scale = scale
        self.anchor_width = anchor_width
        self.english_head = english_head
        self.transition_head = transition_head
        self.narrative_head = narrative_head
        self.terminology_head = terminology_head
        self._word_report = None
        self._reference_head = None
        self._physical_head = None
        self._population_head = None
        self._summary_head = None
        self._forwards = 0
        self._lock = Lock()

    def _observe_forward(self, _encoder, _inputs):
        self._forwards += 1

    def forward_count(self):
        return self._forwards

    @torch.no_grad()
    def __call__(self, text):
        return self._analyze(text, english=False)

    @torch.no_grad()
    def review_english(self, text, *, narrative=False, terminology=False, references=False, expectations=False, summary=False):
        if self.english_head is None or self.transition_head is None:
            raise ValueError('English writing review requires the validated wording and coherence heads.')
        if terminology and self.terminology_head is None:
            raise ValueError("The scientific terminology readout has not been loaded.")
        if expectations and not 2 <= len(text.split()) <= 64:
            raise ValueError("The next-word view accepts short English passages of 2 to 64 words.")
        return self._analyze(text, english=True, narrative=narrative, terminology=terminology,
                             references=references, expectations=expectations, summary=summary)

    @torch.inference_mode()
    def physical_story(self, text):
        """Read a whole short physical narrative once and locate a possible conflicting pair."""
        from research.reading_validation.argument_data import sentences
        from research.reading_validation.state_inference import PhysicalContinuity

        if not 5 <= len(sentences(text)) <= 7:
            raise ValueError('Enter a short narrative of five to seven sentences.')
        if len(self.runtime.tokenizer.encode(text, add_special_tokens=False).ids) > 1024:
            raise ValueError('Physical continuity accepts up to 1,024 model tokens.')
        with self._lock, self.runtime.encoder.register_forward_pre_hook(self._observe_forward):
            if self._physical_head is None:
                self._physical_head = PhysicalContinuity.load()
            before = self._forwards
            result = self._physical_head.from_features(text, self.runtime, self.runtime.encode(text))
            if self._forwards - before != 1:
                raise RuntimeError('Physical continuity requires one full-article forward.')
            return result

    @torch.inference_mode()
    def learner_sentence(self, text):
        """Compute calibrated probabilities for one sentence on the writer's existing backbone."""
        from research.reading_validation.local_inference import REVISION
        from research.token_readout.learner_view import validate_sentence

        validate_sentence(text)
        if self.english_head is None or self.runtime.revision != REVISION:
            raise ValueError('The English learner check requires the verified writing model.')
        if len(self.runtime.tokenizer.encode(text, add_special_tokens=False).ids) > 1024:
            raise ValueError('The learner check accepts sentences up to 1,024 model tokens.')
        with self._lock, self.runtime.encoder.register_forward_pre_hook(self._observe_forward):
            if self._word_report is None:
                from research.reading_validation.nns_inference import WordReport
                self._word_report = WordReport.load(native=self.english_head)
            before = self._forwards
            features = self.runtime.encode(text)
            result = self._word_report.score_levels(text, features, self.runtime.tokenizer, self.reader)
            if self._forwards - before != 1:
                raise RuntimeError('Each learner vocabulary check requires one encoder forward.')
        return result

    def _analyze(self, text, *, english, narrative=False, terminology=False, references=False, expectations=False, summary=False):
        with self._lock, self.runtime.encoder.register_forward_pre_hook(self._observe_forward):
            before = self._forwards
            if references and self._reference_head is None:
                from research.reading_validation.reference_inference import ReferenceInference
                self._reference_head = ReferenceInference.conditional()
            capture = self._reference_head.capture(text, self.runtime) if references else nullcontext()
            with capture as observed:
                features = self.runtime.encode(text) if self.adapter is None else self.runtime.encode(text, self.adapter)
                if references:
                    ref = dict(scope=observed.scope, pronouns=[list(observed.pointer.words[i][:2]) for i in observed.pointer.queries],
                               links=observed.links(features))
            if self._forwards - before != 1:
                raise RuntimeError('Each document trace requires exactly one encoder forward')
            scores = self.reader(features)
            if self.scale is not None:
                scores = replace(scores, opening=scores.opening * self.scale['opening_gain'],
                                 resolution=scores.resolution * self.scale['resolution_gain']
                                            + self.scale['resolution_bias'])
            rows = trace_rows(text, features.offsets, scores, self.priors, self.anchor_width)
            result = {'text': text, 'checkpoint': self.checkpoint, 'tokens': rows}
            if english:
                result['english'] = self.english_head.score(text, features, self.runtime.tokenizer, self.reader)
                spans = [(s['start'], s['end']) for s in result['english']['sentences']]
                from research.reading_validation.transition_external_forecast import token_surprisals

                surprisal = token_surprisals(self.runtime, features)
                common = dict(checkpoint_sha256=self.checkpoint, token_surprisal=surprisal)
                inputs = (text, features, self.runtime.tokenizer, self.reader, self.runtime, spans)
                result['english']['transitions'] = self.transition_head.score(*inputs, token_trace=rows, **common)
                if summary:
                    from research.reading_validation.summary_understanding import SummaryUnderstanding
                    if self._summary_head is None:
                        self._summary_head = SummaryUnderstanding.load()
                    result['english']['summary'] = self._summary_head.score(text, features, self.runtime, token_surprisal=surprisal)
                if references:
                    result['english']['references'] = ref
                if expectations:
                    from research.reading_validation.population_inference import PopulationWordInference
                    if self._population_head is None:
                        self._population_head = PopulationWordInference.load()
                    result['english']['expectations'] = self._population_head.analyze(text, features, self.runtime)
                if terminology:
                    result['english']['terminology'] = self.terminology_head.score(text,features,self.runtime)
                if narrative:
                    if self.narrative_head is None:
                        raise ValueError("The English narrative word model has not been loaded.")
                    result['english']['narrative_words'] = self.narrative_head.score(*inputs, **common)
                if self._forwards - before != 1:
                    raise RuntimeError('English writing review requires one encoder forward for the whole passage')
        return result


def load(*, model: Path, checkpoint: Path):
    """Gradio factory for a candidate emitted by token_readout.fit."""
    identity = digest(checkpoint)
    packet = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (not isinstance(packet, dict) or set(packet) != {'reader', 'adapter', 'priors', 'mode'}
            or packet['mode'] not in ('frozen', 'lora')
            or (packet['adapter'] is not None) != (packet['mode'] == 'lora')):
        raise ValueError('Expected a frozen or LoRA token readout checkpoint from fit.py')
    prior = packet['priors']
    if not isinstance(prior, dict) or any(
        type(prior.get(name)) not in (float, int) or not isfinite(prior[name])
        for name in ('opening', 'resolution')
    ):
        raise ValueError('Checkpoint lacks finite opening and resolution priors')

    runtime = QwenRuntime.load(model)
    reader = TokenReadout().to(runtime.device)
    reader.load_state_dict(packet['reader'], strict=True)
    reader.eval().requires_grad_(False)
    adapter = None
    if packet['adapter'] is not None:
        adapter = runtime.new_adapter(rank=4)
        adapter.load_state_dict(packet['adapter'], strict=True)
        adapter.eval().requires_grad_(False)
    return TokenRunner(runtime, reader, adapter, prior, identity)


def verify_mock_runtime():
    """Check known opening/closure arithmetic and exact app dispatch on CPU."""
    text = '甲 ? e\u0301'
    offsets = ((0, 1), (2, 3), (4, 6))
    runtime = SimpleNamespace(encoder=torch.nn.Identity())
    adapter = object()

    def encode(source, active_adapter):
        if source != text or active_adapter is not adapter:
            raise AssertionError('The runner changed the source or selected adapter')
        runtime.encoder(torch.zeros(1))
        return SimpleNamespace(offsets=offsets)

    def reader(features):
        if features.offsets != offsets:
            raise AssertionError('Encoded offsets changed before inference')
        return Scores(torch.zeros(3), torch.full((3, 3), -log(3.0)), torch.tensor([1.0, 0.0, -1.0]))

    runtime.encode = encode
    run = TokenRunner(runtime, reader, adapter, {'opening': 0.0, 'resolution': 0.0}, 'cpu-mock')
    rows = run(text)['tokens']
    expected = ((0.5, 0.5, 0.0, 0.0), (0.375, 0.0, 0.125, 0.25), (0.78125, 0.5, 0.09375, 0.375))
    for row, coordinates, values in zip(rows, offsets, expected, strict=True):
        if (row['start'], row['end']) != coordinates or any(
            not isclose(row[field], value, abs_tol=1e-6)
            for field, value in zip(('unresolved', 'opened', 'resolved', 'transition'), values, strict=True)
        ):
            raise AssertionError('Token trace disagrees with the independent event arithmetic')
    if [(row['pointer_start'], row['pointer_end']) for row in rows] != [(None, None), (0, 1), (0, 1)]:
        raise AssertionError('Pointer must refer to the earlier eligible span')
    if run.forward_count() != 1:
        raise AssertionError('Mock encoder was not counted exactly once')
    return {'mock_encoder_forwards': 1, 'tokens': 3, 'event_arithmetic_verified': True,
            'punctuation_candidate_mask_verified': True, 'pointers_verified': True}
