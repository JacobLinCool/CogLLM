"""Score sealed ContractNLI forecasts and the frozen evidence-arrival readouts."""

import argparse
from collections import Counter, defaultdict
import json
from math import exp, fsum, log, log1p
from pathlib import Path
import re
import unicodedata

import numpy as np

from research.semantic_tracking import contractnli_forecast as forecast
from research.semantic_tracking.contractnli_score import evaluate
from research.semantic_tracking.evidence_tail_calibrate import PathLayout, probability_path
from research.semantic_tracking.expmrc_forecast import HEADS_SHA256, METHODS, digest


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(__file__).with_name('contractnli-evidence-protocol.md')
CALIBRATION = ROOT / 'results' / 'evidence-calibration-comparison.json'
CALIBRATION_SHA = 'efab56f7379a00b6b468a82c5a0ac8ad34aa8fe9d1189f434e995bd27bf1f944'
TAIL_FIT = ROOT / 'results' / 'qasper-evidence-tail-l4' / 'fit.json'
TAIL_FIT_SHA = '1f38624d2059ab8dcee585c007b0efea72bb443471174aaa46546c9da4345f4a'
SOURCE_AUDIT_SHA = '38d43ddede0c67526e93ae8eaddc5de45292383efbb5a92f396a6da94cd3f14e'
WORD = re.compile(r'[^\W_]+', re.UNICODE)


def lexical_terms(text):
    return WORD.findall(unicodedata.normalize('NFKC', text).casefold())


def lexical_baseline(stimuli, layout, maps, hypotheses):
    """Use within-document BM25 over only the same certified source intervals."""
    terms = {key: set(lexical_terms(stimuli[key]['text'])) for key in hypotheses}
    predictions = {}
    for row in layout:
        key, spans = row['id'], row['spans']
        text = stimuli[key]['text']
        documents = {
            i: Counter(lexical_terms(text[start:end]))
            for i, ((start, end), stop) in enumerate(zip(spans, maps[key], strict=True))
            if stop is not None
        }
        sizes = {i: counts.total() for i, counts in documents.items()}
        length = fsum(sizes.values()) / len(sizes)
        if length <= 0:
            raise ValueError('Every contract needs at least one lexical candidate with a term')
        frequency = Counter(term for counts in documents.values() for term in counts)
        weights = {word: log1p((len(documents) - n + .5) / (n + .5)) for word, n in frequency.items()}
        norms = {i: 1.2 * (.25 + .75 * size / length) for i, size in sizes.items()}
        for query, words in terms.items():
            scores = [None] * len(spans)
            for i, counts in documents.items():
                scores[i] = fsum(weights[word] * count * 2.2 / (count + norms[i])
                                for word, count in counts.items() if word in words)
            predictions[key, query] = scores
    return predictions


def frozen_parameters():
    if (digest(CALIBRATION), digest(TAIL_FIT)) != (CALIBRATION_SHA, TAIL_FIT_SHA):
        raise ValueError('The independent experiment requires the exact previous calibration fits')
    expmrc = forecast.read_json(CALIBRATION)['frozen_exp_mrc_transfer_weights']['methods']
    qasper = forecast.read_json(TAIL_FIT)
    if qasper['source_hashes']['heads_sha256'] != HEADS_SHA256:
        raise ValueError('QASPER calibration belongs to a different set of heads')
    return {
        method: {
            'raw': (0, 0),
            'expmrc': (expmrc[method]['a'], log(expmrc[method]['b'])),
            'qasper_clock': qasper['fitted_methods'][method]['clock']['estimate'],
            'qasper_clock_plateau': qasper['fitted_methods'][method]['clock_plateau']['estimate'],
        }
        for method in METHODS
    }


def average(values):
    return fsum(values) / len(values)


def hypothesis_rows(path, fields):
    with path.open(encoding='utf-8') as handle:
        rows = [json.loads(line) for line in handle]
    if (not rows or any(not isinstance(row, dict) or row.keys() != fields
                        or type(row['id']) is not str or type(row['query_id']) is not str for row in rows)
            or len({(row['id'], row['query_id']) for row in rows}) != len(rows)):
        raise ValueError(f'Expected unique document and hypothesis pairs in {path.name}')
    return rows


def by_family(values, case_family):
    groups = defaultdict(lambda: defaultdict(list))
    for key, value in values.items():
        family, doc = case_family[key], key[0]
        groups[family][doc].append(value)
    if not groups:
        return None
    return average([average([average(rows) for rows in docs.values()]) for docs in groups.values()])


def arrival_summary(gold, forecasts, maps):
    params = frozen_parameters()
    truths = {(row['id'], row['query_id']): row for row in gold}
    records = defaultdict(dict)
    families, uses = {}, {}
    excluded = Counter()
    for row in forecasts:
        key = row['id'], row['query_id']
        observed = truths[key]
        selected = [maps[key[0]][i] for i in observed['evidence_indices']]
        if None in selected:
            excluded[observed['label']] += 1
            continue
        stop = max(selected) if selected else None
        families[key], uses[key] = observed['group_id'], observed['label']
        occurred = (np.arange(len(row['stops'])) >= stop) if stop is not None else 0
        for method, hazards in row['methods'].items():
            layout = PathLayout.from_hazards(hazards)
            for mode, estimate in params[method].items():
                path = probability_path(estimate, layout)
                nll = -path.log_tail if stop is None else -float(path.log_event[stop])
                records[method, mode][key] = (nll, float(np.mean((path.cdf - occurred) ** 2)), exp(path.log_tail))
    methods = defaultdict(dict)
    for (method, mode), items in records.items():
        for_name = dict(zip(('nll_nats', 'cumulative_brier', 'terminal_survival'), zip(*items.values(), strict=True), strict=True))
        methods[method][mode] = {
            **{name: by_family(dict(zip(items, vals, strict=True)), families) for name, vals in for_name.items()},
            'nll_by_label': {
                label: by_family({key: values[0] for key, values in items.items() if uses[key] == label}, families)
                for label in ('Entailment', 'Contradiction', 'NotMentioned')
            },
        }
    return {'methods': dict(methods), 'included_by_label': dict(Counter(uses.values())),
            'excluded_for_missing_evidence_boundary': dict(excluded)}


def score(prepared, packet, forecasts_dir, output):
    audit_path = prepared / 'source-audit.json'
    if digest(audit_path) != SOURCE_AUDIT_SHA:
        raise ValueError('The archived ContractNLI assay differs from the registered EDGAR test subset')
    audit = forecast.read_json(audit_path)
    for name, expected in audit['output_sha256'].items():
        if digest(prepared / name) != expected:
            raise ValueError(f'Audited input changed: {name}')
    stimuli, articles, hypotheses, preflight = forecast.load_packet(packet)
    if preflight['source_sha256'] != {name: audit['output_sha256'][name]
                                     for name in ('stimuli.jsonl', 'layout.jsonl', 'queries.jsonl')}:
        raise ValueError('Model packet and human outcomes must share the same public source')
    seal_path, path = (forecasts_dir / name for name in ('seal.json', 'predictions.jsonl'))
    seal = forecast.read_json(seal_path)
    expected = {
        'assay_sha256': SOURCE_AUDIT_SHA, 'protocol_sha256': digest(PROTOCOL), 'heads_sha256': HEADS_SHA256,
        'predictions_sha256': digest(path), 'preflight_sha256': digest(packet / 'preflight.json'),
        'stimuli_sha256': digest(packet / 'stimuli.jsonl'), 'articles_sha256': digest(packet / 'articles.jsonl'),
        'candidate_id': preflight['candidate_id'], 'model_revision': preflight['model_revision'],
        'source_code_sha256': digest(Path(forecast.__file__)),
        'encoded_hypotheses': 17, 'content_forward_per_document': 1, 'forecast_rows': 799,
    }
    if (any(seal.get(key) != value for key, value in expected.items())
            or len(articles) != 47 or len(hypotheses) != 17):
        raise ValueError('Forecasts must be frozen for the registered assay and single-forward readouts')
    layout = list(forecast.read_rows(
        prepared / 'layout.jsonl', {'id', 'official_id', 'group_id', 'document_type', 'spans'},
    ).values())
    gold = hypothesis_rows(prepared / 'gold.jsonl', {'id', 'group_id', 'official_id', 'query_id', 'label', 'evidence_indices'})
    predictions = hypothesis_rows(path, {'id', 'query_id', 'stops', 'methods'})
    if (len(predictions) != 799 or len(gold) != len(predictions)
            or any(row['stops'] != stimuli[row['id']]['stops'] for row in predictions)
            or {row['query_id'] for row in predictions} != set(hypotheses)):
        raise ValueError('Expected complete aligned predictions at every article–hypothesis pair')
    maps = {row['id']: row['candidate_stops']
            for row in forecast.read_rows(packet / 'articles.jsonl', {'id', 'candidate_stops'}).values()}
    if (set(maps) != set(articles) or sum(len(v) for v in maps.values()) != 4318
            or sum(x is not None for v in maps.values() for x in v) != 4207):
        raise ValueError('Evidence maps differ from the frozen candidate coordinates')
    lexical = lexical_baseline(stimuli, layout, maps, hypotheses)
    report = evaluate(layout, gold, predictions, maps, lexical)
    forecast.write_json(output, {
        **report, 'evidence_availability': arrival_summary(gold, predictions, maps),
        'input_sha256': {'source_audit': SOURCE_AUDIT_SHA, 'forecast_seal': digest(seal_path),
                         'protocol': digest(PROTOCOL), 'exp_mrc_calibrators': CALIBRATION_SHA,
                         'qasper_calibrators': TAIL_FIT_SHA, 'reporter_source': digest(Path(__file__))},
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('prepared', 'packet', 'forecasts-dir', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    score(**vars(parser.parse_args()))
