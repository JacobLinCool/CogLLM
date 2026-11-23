"""Causal expected count of article terms awaiting explanatory evidence.

Offsets refer to the original string. A value changes only at a certified word end.
The supervised unit is an annotated term–definition relation; no person-specific
psychological unit is measured by this estimator.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest, read_json
from research.reading_validation.definition_data import OUT
from research.reading_validation.definition_inference import ConceptLedger, causal_page
from research.reading_validation.stock_probability import predict


def render_distribution(page, trace, offsets, scale, include_source_history=False):
    active = trace.unresolved * scale
    closing = trace.resolved * scale
    opening = trace.occurrence * scale
    word_ends = {int(word[3]): i for i, word in enumerate(page.words)}
    counts = active.sum(axis=1)
    release = closing.sum(axis=1)
    mentions = []
    for right, left in enumerate(trace.starts):
        a, b = int(page.words[left, 0]), int(page.words[right, 1])
        mentions.append({'id': right, 'span': [a, b], 'text': page.text[a:b],
                         'token': int(page.words[right, 3]), 'opening_mass': float(opening[right])})
    tokens = []
    level = 0.0
    for t, (a, b) in enumerate(offsets):
        row = {'token': t, 'span': [a, b], 'expected_pending': level, 'opening_mass': 0.0,
               'resolved_mass': 0.0, 'opened_source': None, 'largest_resolved_source': None,
               'largest_resolved_mass': 0.0}
        word = word_ends.get(t)
        if word is not None:
            level = float(counts[word])
            source = int(np.argmax(closing[word, :word])) if word else None
            row.update(expected_pending=level, opening_mass=float(opening[word]), resolved_mass=float(release[word]),
                       opened_source=word, largest_resolved_source=source,
                       largest_resolved_mass=float(closing[word, source]) if source is not None else 0.0)
        tokens.append(row)
    result = {'unit': 'expected pending annotated terminology mentions', 'tokens': tokens, 'sources': mentions,
              'updates': 'word ends, indexed by their final native token', 'count_calibration': scale}
    if include_source_history:
        # [time, source] arrays use word-end units; source token offsets are in result["sources"].
        result['expected_pending_by_word_and_source'] = active.tolist()
        result['resolved_mass_by_word_and_source'] = closing.tolist()
    return result


@dataclass
class ExpectedConceptLedger:
    ledger: ConceptLedger
    fit: dict
    scale: float

    @classmethod
    def load(cls, root=OUT):
        root = Path(root)
        base = ConceptLedger.load(root, 'semantic')
        fitted = read_json(root / 'stock-probability-weights.json')
        calibration = read_json(root / 'stock-count-scales.json')
        if (fitted['base_weights'] != digest(root / 'weights.json') or
                calibration['weights'] != digest(root / 'stock-probability-weights.json')):
            raise ValueError('The fitted occurrence, explanation, and count estimators have different sources')
        return cls(base, fitted['fits']['semantic'], calibration['scales']['semantic']['scale'])

    def from_features(self, text, features, runtime, include_source_history=False):
        page = causal_page(text, features, runtime, self.ledger.model_revision)
        trace = predict(page, 'semantic', self.fit, (self.ledger.term, self.ledger.link))
        return render_distribution(page, trace, features.offsets, self.scale, include_source_history)

    def analyze(self, text, runtime, include_source_history=False):
        count = 0

        def seen(_module, _arguments):
            nonlocal count
            count += 1

        with runtime.encoder.register_forward_pre_hook(seen):
            result = self.from_features(text, runtime.encode(text), runtime, include_source_history)
        if count != 1:
            raise RuntimeError('Expected exactly one fresh backbone forward for the original input')
        result['encoder_forwards'] = count
        return result
