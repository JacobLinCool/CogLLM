"""Apply the frozen ContractNLI readout and score the sealed Evidence Inference transfer."""

import argparse
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from research.semantic_tracking import contractnli_forecast as frozen
from research.semantic_tracking import contractnli_rerank as rerank
from research.semantic_tracking import contractnli_rerank_score as readout
from research.semantic_tracking.contractnli_report import hypothesis_rows, lexical_terms


PROTOCOL = Path(__file__).with_name('evidence-inference-transfer-protocol.md')
FITTED_SHA256 = '8dbd9e3fdc1a538f90736d55e7b793835692ef7276a4908a4562aab5516160fe'
PACKET_FILES = ('stimuli.jsonl', 'articles.jsonl', 'preflight.json')
SCORER = Path(__file__).with_name('evidence_inference_transfer_score.py')
ARTICLE_FIELDS = {'id', 'text'}
QUERY_FIELDS = {'id', 'article_id', 'text'}
LAYOUT_FIELDS = {'id', 'spans', 'candidate_stops', 'query_ids'}
AUDIT_OUTPUTS = 'output_sha256'


def bindings(prepared, packet, fitted_path):
    if frozen.digest(fitted_path) != FITTED_SHA256:
        raise ValueError('Expected the exact prospectively fixed ContractNLI fit')
    audit = prepared / 'source-audit.json'
    preflight = frozen.read_json(packet / 'preflight.json')
    sources = frozen.read_json(audit)[AUDIT_OUTPUTS]
    if (preflight['source_audit_sha256'] != frozen.digest(audit)
            or any(frozen.digest(prepared / name) != sources[name] for name in ('articles.jsonl', 'queries.jsonl'))):
        raise ValueError('Evidence Inference prepared texts and preflight disagree')
    return {
        'candidate_id': frozen.CANDIDATE, 'heads_sha256': frozen.HEADS_SHA256,
        'model_revision': frozen.REVISION, 'assay_sha256': frozen.digest(audit),
        'protocol_sha256': frozen.digest(PROTOCOL),
        'packet_sha256': {name: frozen.digest(packet / name) for name in PACKET_FILES},
    }


def evidence_paths(root, expected):
    seal = frozen.read_json(root / 'seal.json')
    for key in ('candidate_id', 'heads_sha256', 'model_revision', 'assay_sha256', 'protocol_sha256'):
        if seal[key] != expected[key]:
            raise ValueError(f'Unexpected upstream forecast identity: {key}')
    expected_files = {**expected['packet_sha256'],
                      **{name: frozen.digest(root / name) for name in ('predictions.jsonl', 'extraction.json',
                                                                     'causality.json', 'telemetry.jsonl')}}
    if any(seal[f'{Path(name).stem}_sha256'] != value for name, value in expected_files.items()):
        raise ValueError('The upstream forecast files fail their seal')
    extraction = frozen.read_json(root / 'extraction.json')
    checks = frozen.read_json(root / 'causality.json')['checks']
    if (extraction['gpu'] != 'NVIDIA L4' or extraction['revision'] != frozen.REVISION
            or seal['content_forward_per_document'] != 1 or len(checks) != 12
            or any(not check['passed'] or not check['projected_tokens']['passed'] for check in checks)):
        raise ValueError('The forecast must pass the frozen L4 causal extraction checks')
    return hypothesis_rows(root / 'predictions.jsonl', rerank.HAZARD_FIELDS)


def frozen_scores(stimuli, layout, paths, fitted):
    by_article = defaultdict(list)
    for row in paths:
        by_article[row['id']].append(row)
    if by_article.keys() != layout.keys():
        raise ValueError('The forecast must cover every selected article')
    cases, unit = [], {}
    for key, mapping in layout.items():
        bank = mapping['query_ids']
        hypotheses = {query: stimuli[query] for query in bank}
        article = stimuli[key]
        if ({row['query_id'] for row in by_article[key]} != set(bank) or len(by_article[key]) != len(bank)
                or any(row['stops'] != article['stops'] for row in by_article[key])):
            raise ValueError(f'Incomplete or misaligned causal paths for {key}')
        candidates, terms = rerank.corpus_features(stimuli, hypotheses, {key: mapping}, {key: mapping})
        cases.extend(rerank.cases_for(by_article[key], hypotheses, {key: {'group_id': key}}, candidates, terms,
                                      fitted['lexical']))
        for query, words in terms.items():
            lexical = {**fitted['lexical'], 'idf': dict.fromkeys(words, 1.0)}
            unit[key, query] = [rerank.lexical_score(counts, words, lexical) for counts in candidates[key][2]]
    rows = readout.forecast(cases, fitted['readout'])
    for row in rows:
        row.pop('group_id')
        scores = row['methods']
        scores['bm25_causal'] = scores.pop('bm25')
        scores['bm25_unit'] = unit[row['id'], row['query_id']]
    return rows


def predict(prepared, packet, root, fitted_path, output):
    if output.exists():
        raise FileExistsError(output)
    expected = bindings(prepared, packet, fitted_path)
    fitted = rerank.load_fit(fitted_path)
    stimuli = frozen.read_rows(packet / 'stimuli.jsonl', {'id', 'text', 'stops'})
    layout = frozen.read_rows(packet / 'articles.jsonl', LAYOUT_FIELDS)
    articles = frozen.read_rows(prepared / 'articles.jsonl', ARTICLE_FIELDS)
    queries = frozen.read_rows(prepared / 'queries.jsonl', QUERY_FIELDS)
    if (layout.keys() != articles.keys() or articles.keys() & queries.keys()
            or stimuli.keys() != articles.keys() | queries.keys()
            or any(stimuli[key]['text'] != row['text'] for material in (articles, queries) for key, row in material.items())
            or {(key, query) for key, row in layout.items() for query in row['query_ids']}
            != {(row['article_id'], key) for key, row in queries.items()}
            or layout.keys() & (set(fitted['training_document_ids']) | set(fitted['training_groups']))):
        raise ValueError('The packet must preserve the article-specific questions and unique original sources')
    forecasts = frozen_scores(stimuli, layout, evidence_paths(root, expected), fitted)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.evidence-inference-') as scratch:
        stage = Path(scratch) / 'forecasts'
        stage.mkdir()
        path = stage / 'predictions.jsonl'
        frozen.write_rows(path, forecasts)
        frozen.write_json(stage / 'seal.json', {
            **expected, 'schema': 1, 'fitted_sha256': FITTED_SHA256,
            'original_forecast_seal_sha256': frozen.digest(root / 'seal.json'), 'predictions_sha256': frozen.digest(path),
            'code_sha256': frozen.digest(Path(__file__)), 'scoring_code_sha256': frozen.digest(SCORER),
            'documents': len(articles), 'forecasts': len(forecasts),
        })
        stage.rename(output)


def score(prepared, packet, sealed, fitted_path, output):
    from research.semantic_tracking import evidence_inference_transfer_score as scoring

    expected = bindings(prepared, packet, fitted_path)
    prediction_path, seal_path = sealed / 'predictions.jsonl', sealed / 'seal.json'
    seal = frozen.read_json(seal_path)
    pins = {**expected, 'schema': 1, 'fitted_sha256': FITTED_SHA256,
            'predictions_sha256': frozen.digest(prediction_path), 'code_sha256': frozen.digest(Path(__file__)),
            'scoring_code_sha256': frozen.digest(SCORER)}
    if any(seal[key] != value for key, value in pins.items()):
        raise ValueError('Scoring requires prospectively bound, sealed rankings')
    outcomes = prepared / 'outcomes.jsonl'
    if frozen.digest(outcomes) != frozen.read_json(prepared / 'source-audit.json')[AUDIT_OUTPUTS][outcomes.name]:
        raise ValueError('Human evidence outcomes differ from public source intake')
    articles = frozen.read_rows(prepared / 'articles.jsonl', ARTICLE_FIELDS)
    layout = frozen.read_rows(packet / 'articles.jsonl', LAYOUT_FIELDS)
    truth = frozen.read_rows(outcomes, {'id', 'article_id', 'references'})
    rankings = hypothesis_rows(prediction_path, {'id', 'query_id', 'candidate_indices', 'methods'})
    if len(articles) != seal['documents'] or len(rankings) != seal['forecasts']:
        raise ValueError('Unexpected denominator in sealed rankings')
    result = scoring.evaluate(list(articles.values()), list(layout.values()), list(truth.values()), rankings)
    frozen.write_json(output, {**result, **expected, 'fitted_sha256': FITTED_SHA256,
                               'forecast_seal_sha256': frozen.digest(seal_path),
                               'scoring_code_sha256': pins['scoring_code_sha256']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prediction = commands.add_parser('predict')
    scoring = commands.add_parser('score')
    for command in (prediction, scoring):
        for name in ('prepared', 'packet', 'fitted', 'output'):
            command.add_argument(f'--{name}', type=Path, required=True)
    prediction.add_argument('--source-forecasts', type=Path, required=True)
    scoring.add_argument('--sealed', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'predict':
        predict(args.prepared, args.packet, args.source_forecasts, args.fitted, args.output)
    else:
        score(args.prepared, args.packet, args.sealed, args.fitted, args.output)
