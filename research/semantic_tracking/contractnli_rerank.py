"""Fit a causal lexical-and-Qwen evidence readout, then score sealed held-out predictions."""

import argparse
from collections import Counter
from math import fsum, log1p
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import logit

from research.semantic_tracking import contractnli_forecast as archived
from research.semantic_tracking import contractnli_rerank_forecast as fresh
from research.semantic_tracking import contractnli_rerank_score as readout
from research.semantic_tracking.contractnli_report import hypothesis_rows, lexical_terms
from research.semantic_tracking.expmrc_forecast import HEADS_SHA256, digest


PROTOCOL = Path(__file__).with_name('contractnli-rerank-protocol.md')
TRAIN_AUDIT = '38d43ddede0c67526e93ae8eaddc5de45292383efbb5a92f396a6da94cd3f14e'
EVAL_AUDIT = '725a5414111bb4a703f261e4a45b478c6ec877e1478d05d3066690609c242699'
LAYOUT_FIELDS = {'id', 'official_id', 'group_id', 'document_type', 'spans'}
GOLD_FIELDS = {'id', 'group_id', 'official_id', 'query_id', 'label', 'evidence_indices'}
HAZARD_FIELDS = {'id', 'query_id', 'stops', 'methods'}
MAP_FIELDS = {'id', 'candidate_stops'}
CASE_FIELDS = {'id', 'query_id', 'group_id', 'candidate_indices', 'methods'}


def audit_sources(prepared, anchor, names):
    path = prepared / 'source-audit.json'
    if digest(path) != anchor:
        raise ValueError('The prepared source must match the prospectively registered split')
    audit = archived.read_json(path)
    for name in names:
        if digest(prepared / name) != audit['output_sha256'][name]:
            raise ValueError(f'Registered source file differs: {name}')
    return audit


def frozen_inputs(prepared, packet, root, exporter, anchor, protocol):
    sources = audit_sources(prepared, anchor, ('stimuli.jsonl', 'layout.jsonl', 'queries.jsonl'))
    stimuli, articles, hypotheses, preflight = exporter.load_packet(packet)
    scope = {
        'source_audit_sha256': anchor, 'assay_sha256': anchor, 'protocol_sha256': digest(protocol),
        'predictions_sha256': digest(root / 'predictions.jsonl'), 'heads_sha256': HEADS_SHA256,
        'preflight_sha256': digest(packet / 'preflight.json'), 'stimuli_sha256': digest(packet / 'stimuli.jsonl'),
        'articles_sha256': digest(packet / 'articles.jsonl'), 'candidate_id': archived.CANDIDATE,
        'model_revision': archived.REVISION, 'source_code_sha256': digest(Path(exporter.__file__)),
        'encoded_hypotheses': 17, 'content_forward_per_document': 1,
    }
    seal = archived.read_json(root / 'seal.json')
    if (preflight['source_sha256'] != {name: sources['output_sha256'][name]
                                       for name in ('stimuli.jsonl', 'layout.jsonl', 'queries.jsonl')}
            or any(seal.get(key) != value for key, value in scope.items())
            or set(seal['query_ids']) != set(hypotheses)):
        raise ValueError('Sources, tokenizer, model forecasts and registered split are inconsistent')
    layout = archived.read_rows(prepared / 'layout.jsonl', LAYOUT_FIELDS)
    maps = archived.read_rows(packet / 'articles.jsonl', MAP_FIELDS)
    paths = hypothesis_rows(root / 'predictions.jsonl', HAZARD_FIELDS)
    if (layout.keys() != maps.keys() or layout.keys() != set(articles)
            or {(row['id'], row['query_id']) for row in paths} != {(key, q) for key in articles for q in hypotheses}
            or any(row['stops'] != stimuli[row['id']]['stops'] for row in paths)):
        raise ValueError('Expected one complete source path for each document and hypothesis')
    return stimuli, hypotheses, layout, maps, paths


def candidate_counts(stimulus, layout, mapping):
    text, stops = stimulus['text'], stimulus['stops']
    indices, counts, selected = [], [], []
    for index, ((start, end), stop_index) in enumerate(zip(layout['spans'], mapping['candidate_stops'], strict=True)):
        if stop_index is None:
            continue
        if not 0 <= stop_index < len(stops) or not 0 <= start < stops[stop_index] <= end <= len(text):
            raise ValueError('A certified candidate must fall within the visible source span')
        indices.append(index)
        selected.append(stop_index)
        counts.append(Counter(lexical_terms(text[start:stops[stop_index]])))
    if not counts or selected != sorted(set(selected)):
        raise ValueError('Expected an ordered nonempty set of certified candidates')
    return indices, selected, counts


def corpus_features(stimuli, hypotheses, layout, maps):
    counts = {key: candidate_counts(stimuli[key], value, maps[key]) for key, value in layout.items()}
    terms = {key: set(lexical_terms(stimuli[key]['text'])) for key in hypotheses}
    return counts, terms


def lexical_fit(candidates):
    frequency = Counter()
    spans, words = 0, 0
    for _, _, rows in candidates.values():
        for terms in rows:
            frequency.update(terms.keys())
            words += terms.total()
            spans += 1
    if not words or not spans:
        raise ValueError('Cannot fit BM25 without training candidate words')
    return {'idf': {term: log1p((spans - n + 0.5) / (n + 0.5)) for term, n in sorted(frequency.items())},
            'mean_candidate_words': words / spans, 'training_candidates': spans, 'training_words': words,
            'k1': 1.2, 'b': 0.75}


def lexical_score(candidate, terms, fitted):
    k1, b = fitted['k1'], fitted['b']
    denominator = k1 * (1 - b + b * candidate.total() / fitted['mean_candidate_words'])
    return log1p(fsum(fitted['idf'].get(term, 0) * frequency * (k1 + 1) / (frequency + denominator)
                     for term, frequency in candidate.items() if term in terms))


def cases_for(paths, hypotheses, layout, candidates, terms, lexical):
    indexed = {(row['id'], row['query_id']): row['methods']['token_rank'] for row in paths}
    rows = []
    for key, (indices, stops, counts) in candidates.items():
        hazards = np.stack([np.asarray(indexed[key, query], dtype=np.float64)[stops] for query in hypotheses])
        if not np.all(np.isfinite(hazards) & (hazards > 0) & (hazards < 1)):
            raise ValueError('Readouts must provide finite interior hazards at every scoreable candidate')
        raw = logit(hazards)
        centered = raw - raw.mean(axis=0)
        for index, query in enumerate(hypotheses):
            bm25 = [lexical_score(candidate, terms[query], lexical) for candidate in counts]
            rows.append({'id': key, 'query_id': query, 'group_id': layout[key]['group_id'],
                         'candidate_indices': indices, 'features': np.column_stack((bm25, raw[index], centered[index])).tolist()})
    return rows


def fit(prepared, packet, source_forecasts, output):
    previous_protocol = PROTOCOL.with_name('contractnli-evidence-protocol.md')
    inputs = frozen_inputs(prepared, packet, source_forecasts, archived, TRAIN_AUDIT, previous_protocol)
    stimuli, hypotheses, layout, maps, paths = inputs
    candidates, terms = corpus_features(stimuli, hypotheses, layout, maps)
    lexical = lexical_fit(candidates)
    cases = cases_for(paths, hypotheses, layout, candidates, terms, lexical)
    audit_sources(prepared, TRAIN_AUDIT, ('gold.jsonl',))
    gold = hypothesis_rows(prepared / 'gold.jsonl', GOLD_FIELDS)
    fitted = readout.fit(cases, gold)
    archived.write_json(output, {
        'schema': 1, 'lexical': lexical, 'readout': fitted, 'training_source_audit_sha256': TRAIN_AUDIT,
        'training_forecast_seal_sha256': digest(source_forecasts / 'seal.json'),
        'training_groups': sorted({row['group_id'] for row in layout.values()}),
        'training_document_ids': sorted(layout),
        'hypotheses': {key: stimuli[key]['text'] for key in hypotheses},
        'protocol_sha256': digest(PROTOCOL), 'heads_sha256': HEADS_SHA256,
        'code_sha256': digest(Path(__file__)), 'readout_code_sha256': digest(Path(readout.__file__)),
    })


def load_fit(path):
    fitted = archived.read_json(path)
    if (fitted['schema'] != 1 or fitted['protocol_sha256'] != digest(PROTOCOL)
            or fitted['training_source_audit_sha256'] != TRAIN_AUDIT or fitted['heads_sha256'] != HEADS_SHA256
            or fitted['code_sha256'] != digest(Path(__file__))
            or fitted['readout_code_sha256'] != digest(Path(readout.__file__))):
        raise ValueError('Expected reranking weights with exact training and code provenance')
    return fitted


def predict(prepared, packet, source_forecasts, fitted_path, output):
    if output.exists():
        raise FileExistsError(output)
    fitted = load_fit(fitted_path)
    inputs = frozen_inputs(prepared, packet, source_forecasts, fresh, EVAL_AUDIT, PROTOCOL)
    stimuli, hypotheses, layout, maps, paths = inputs
    if (set(fitted['training_document_ids']) & layout.keys()
            or set(fitted['training_groups']) & {row['group_id'] for row in layout.values()}
            or fitted['hypotheses'] != {key: stimuli[key]['text'] for key in hypotheses}):
        raise ValueError('Evaluation documents must be untouched and the hypothesis bank identical')
    candidates, terms = corpus_features(stimuli, hypotheses, layout, maps)
    cases = cases_for(paths, hypotheses, layout, candidates, terms, fitted['lexical'])
    forecasts = readout.forecast(cases, fitted['readout'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.contract-rerank-') as scratch:
        stage = Path(scratch) / 'forecasts'
        stage.mkdir()
        path = stage / 'predictions.jsonl'
        archived.write_rows(path, forecasts)
        archived.write_json(stage / 'seal.json', {
            'schema': 1, 'fitted_sha256': digest(fitted_path), 'predictions_sha256': digest(path),
            'source_audit_sha256': EVAL_AUDIT, 'original_forecast_seal_sha256': digest(source_forecasts / 'seal.json'),
            'protocol_sha256': digest(PROTOCOL), 'code_sha256': digest(Path(__file__)),
            'documents': len(layout), 'predictions': len(forecasts),
        })
        stage.rename(output)


def score(prepared, sealed, fitted_path, output):
    fitted = load_fit(fitted_path)
    audit_sources(prepared, EVAL_AUDIT, ('gold.jsonl',))
    prediction_path, seal_path = sealed / 'predictions.jsonl', sealed / 'seal.json'
    seal = archived.read_json(seal_path)
    required = {'fitted_sha256': digest(fitted_path), 'predictions_sha256': digest(prediction_path),
                'source_audit_sha256': EVAL_AUDIT, 'protocol_sha256': digest(PROTOCOL),
                'code_sha256': digest(Path(__file__)), 'documents': 23, 'predictions': 391}
    if any(seal.get(key) != value for key, value in required.items()):
        raise ValueError('Scoring needs the exact frozen reranker and held-out forecasts')
    forecasts = hypothesis_rows(prediction_path, CASE_FIELDS)
    gold = hypothesis_rows(prepared / 'gold.jsonl', GOLD_FIELDS)
    result = readout.evaluate(forecasts, gold)
    archived.write_json(output, {
        **result, 'forecast_seal_sha256': digest(seal_path), 'fitted_sha256': digest(fitted_path),
        'source_audit_sha256': EVAL_AUDIT, 'protocol_sha256': fitted['protocol_sha256'],
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training, prediction, scoring = (commands.add_parser(name) for name in ('fit', 'predict', 'score'))
    for command in (training, prediction, scoring):
        for name in ('prepared', 'output'):
            command.add_argument(f'--{name}', type=Path, required=True)
    for command in (training, prediction):
        for name in ('packet', 'source-forecasts'):
            command.add_argument(f'--{name}', type=Path, required=True)
    for command in (prediction, scoring):
        command.add_argument('--fitted-path', type=Path, required=True)
    scoring.add_argument('--sealed', type=Path, required=True)
    args = vars(parser.parse_args())
    {'fit': fit, 'predict': predict, 'score': score}[args.pop('command')](**args)
