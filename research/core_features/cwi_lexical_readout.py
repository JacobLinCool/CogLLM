"""Fit and evaluate fixed Gradio token and surface readouts of CWI votes."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import rankdata

from research.core_features import cwi_news_prepare as sources
from research.core_features.cwi_news_score import DRAWS, SEED
from research.gradio_app.forecast import read_stimuli
from research.semantic_tracking.contractnli_forecast import read_json, read_rows, write_json, write_rows
from research.semantic_tracking.iirc_anchor_fit import WORD
from research.semantic_tracking.iirc_prepare import digest
from research.stage6 import medreadme_lexical_forecast
from research.stage6.medreadme_lexical_score import unique_predictions


PROTOCOL = Path(__file__).with_name('cwi-lexical-readout-protocol.md')
RIDGE = 0.05
SPLITS = ('train', 'dev', 'test')
COLUMNS = {'surface_fit': (0, 1), 'complex_fit': (2,), 'fused_fit': (0, 1, 2)}
METHODS = ('served', 'combined_surface', *COLUMNS)
WIKI_OUTCOME_FIELDS = {'id', 'group_id', 'split', 'words'}
NEWS_OUTCOME_FIELDS = {'id', 'hit_id', 'words'}


def inputs(prepared, forecast):
    plan = read_json(prepared / 'assay.json')
    seal = read_json(forecast / 'seal.json')
    documents = dict(read_stimuli(prepared / 'stimuli.jsonl', plan['stimuli_sha256']).documents)
    predictions = forecast / 'predictions.jsonl'
    if (seal['forecasts_sha256'] != digest(predictions)
            or seal['forecast_code_sha256'] != digest(Path(medreadme_lexical_forecast.__file__))
            or seal['candidate_id'] != sources.CANDIDATE or seal['lexical_weights_sha256'] != sources.LEXICAL_SHA256
            or seal['stimuli_sha256'] != plan['stimuli_sha256']
            or not seal['model_forwards'] == len(documents) == plan['retained_sentences']):
        raise ValueError('Expected the exact lexical forecast sealed against the fixed candidate and materials')
    scores = read_rows(predictions, {'id', 'words'})
    if scores.keys() != documents.keys():
        raise ValueError('Forecasts must cover each source sentence once')
    return plan, documents, scores


def checked_outcomes(path, sha256, fields):
    if digest(path) != sha256:
        raise ValueError(f'Original CWI outcomes differ from their source receipt: {path.name}')
    return read_rows(path, fields)


def positions(row, text):
    legal = {match.span() for match in WORD.finditer(text)}
    spans, frequencies, hits = [], [], []
    for word in row['words']:
        if word.keys() != {'start', 'end', 'form_count', 'native_votes', 'nonnative_votes'}:
            raise ValueError('Expected CWI singleton candidates with source votes')
        start, end, count = (word[key] for key in ('start', 'end', 'form_count'))
        native, nonnative = (word[key] for key in ('native_votes', 'nonnative_votes'))
        if (any(type(value) is not int for value in (start, end, count, native, nonnative))
                or (start, end) not in legal or count < 0 or not 0 <= native <= 10 or not 0 <= nonnative <= 10):
            raise ValueError('Invalid CWI source alignment, vote count or training frequency')
        spans.append((start, end))
        frequencies.append(count)
        hits.append(native + nonnative)
    if not spans or spans != sorted(set(spans)):
        raise ValueError('CWI source candidates must be nonempty, distinct and ordered')
    return spans, np.asarray(frequencies), np.asarray(hits)


def feature_rows(spans, counts, served):
    lengths = np.asarray([end - start for start, end in spans], dtype=np.float64)
    return np.column_stack((lengths, -np.log1p(counts), served))


def training_rows(documents, forecasts, outcomes):
    buckets = defaultdict(list)
    for key, row in outcomes.items():
        if row['split'] != 'train' or not row['group_id']:
            raise ValueError('Only prospectively registered WikiNews training groups may fit the readout')
        text = documents[key]
        spans, counts, hits = positions(row, text)
        served = unique_predictions(forecasts[key], {match.span() for match in WORD.finditer(text)})
        selected = np.asarray([span in served for span in spans])
        if selected.any():
            values = feature_rows(spans, counts, [served.get(span, -1) for span in spans])
            buckets[row['group_id']].append((values[selected], hits[selected] / 20))
    if len(buckets) < 2:
        raise ValueError('Expected distinct WikiNews training families')
    matrices, rates, weights = [], [], []
    for key in sorted(buckets):
        for design, y in buckets[key]:
            matrices.append(design)
            rates.append(y)
            weights.append(np.full(len(y), 1 / (len(buckets) * len(buckets[key]) * len(y))))
    return np.vstack(matrices), np.concatenate(rates), np.concatenate(weights), buckets


def fit_models(features, rates, weights):
    center = weights @ features
    scale = np.sqrt(weights @ ((features - center) ** 2))
    if not np.all(np.isfinite(scale) & (scale > 0)) or not 0 < weights @ rates < 1:
        raise ValueError('Every planned term requires positive feature variability and both vote outcomes')
    standardized = (features - center) / scale
    fits = {}
    for name, columns in COLUMNS.items():
        x = standardized[:, columns]

        def loss(parameters):
            logits = parameters[0] + x @ parameters[1:]
            residual = weights * (expit(logits) - rates)
            penalized = parameters[1:]
            value = weights @ (np.logaddexp(0, logits) - rates * logits) + RIDGE * (penalized @ penalized) / 2
            return value, np.r_[residual.sum(), x.T @ residual + RIDGE * penalized]

        guess = np.r_[logit(weights @ rates), np.zeros(x.shape[1])]
        solution = minimize(loss, guess, method='L-BFGS-B', jac=True, options={'ftol': 1e-13, 'gtol': 1e-9})
        if not solution.success or not np.all(np.isfinite(solution.x)):
            raise ValueError(f'Fit failed for {name}: {solution.message}')
        fits[name] = {'intercept': float(solution.x[0]), 'weights': solution.x[1:].tolist(),
                      'columns': list(columns), 'objective': float(solution.fun)}
    return {'center': center.tolist(), 'scale': scale.tolist(), 'methods': fits, 'ridge': RIDGE}


def apply_models(design, fitted):
    standardized = (design - np.asarray(fitted['center'])) / np.asarray(fitted['scale'])
    return {name: model['intercept'] + standardized[:, COLUMNS[name]] @ np.asarray(model['weights'])
            for name, model in fitted['methods'].items()}


def forecast_rows(documents, forecasts, frequencies, fitted):
    for key, text in documents.items():
        served = unique_predictions(forecasts[key], {match.span() for match in WORD.finditer(text)})
        spans = list(served)
        normalized = [sources.words(text[left:right]) for left, right in spans]
        counts = np.asarray([frequencies[values[0]] if len(values) == 1 else 0 for values in normalized])
        design = feature_rows(spans, counts, list(served.values()))
        logits = apply_models(design, fitted)
        yield {'id': key, 'words': [[left, right, served[left, right], *values]
                                    for (left, right), values in zip(spans, zip(*(logits[name] for name in COLUMNS)), strict=True)]}


def predict(wiki, wiki_forecast, news, news_forecast, complex_source, output):
    if output.exists():
        raise FileExistsError(output)
    plan, documents, forecasts = inputs(wiki, wiki_forecast)
    if plan['protocol_sha256'] != digest(PROTOCOL):
        raise ValueError('WikiNews must use the prospectively fixed experiment protocol')
    train = checked_outcomes(wiki / 'train.jsonl', plan['file_sha256']['train.jsonl'], WIKI_OUTCOME_FIELDS)
    layout = read_rows(wiki / 'layout.jsonl', {'id', 'group_id', 'split', 'words'})
    if (digest(wiki / 'layout.jsonl') != plan['file_sha256']['layout.jsonl'] or layout.keys() != documents.keys()
            or train.keys() != {key for key, row in layout.items() if row['split'] == 'train'}
            or any(layout[key]['group_id'] != row['group_id'] for key, row in train.items())):
        raise ValueError('Training identities and family split differ from the fixed WikiNews layout')
    values, rates, weight, groups = training_rows(documents, forecasts, train)
    fitted = fit_models(values, rates, weight)
    fitted.update(training_words=len(values), training_group_ids=sorted(groups),
                  training_sentences=sum(map(len, groups.values())))
    _, frequencies = sources.read_complex(complex_source)
    _, news_documents, news_scores = inputs(news, news_forecast)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.cwi-readout-') as temporary:
        stage = Path(temporary) / 'publish'
        stage.mkdir()
        write_json(stage / 'fit.json', fitted)
        write_rows(stage / 'wikinews.jsonl', forecast_rows(documents, forecasts, frequencies, fitted))
        write_rows(stage / 'news.jsonl', forecast_rows(news_documents, news_scores, frequencies, fitted))
        write_json(stage / 'seal.json', {
            'schema': 1, 'protocol_sha256': digest(PROTOCOL), 'code_sha256': digest(Path(__file__)),
            'wiki_assay_sha256': digest(wiki / 'assay.json'), 'news_assay_sha256': digest(news / 'assay.json'),
            'wiki_base_seal_sha256': digest(wiki_forecast / 'seal.json'), 'news_base_seal_sha256': digest(news_forecast / 'seal.json'),
            'candidate_id': sources.CANDIDATE, 'file_sha256': {name: digest(stage / name)
                                                             for name in ('fit.json', 'wikinews.jsonl', 'news.jsonl')},
        })
        stage.rename(output)


def compare_sentence(text, row, prediction):
    spans, counts, votes = positions(row, text)
    scored = {(left, right): (served, *logits) for left, right, served, *logits in prediction['words']}
    values = np.asarray([scored.get(span, (-np.inf,) * (len(METHODS) - 1)) for span in spans])
    present = np.isfinite(values).all(axis=1)
    paired_counts, rank_scores = 0, None
    earlier, later = np.triu_indices(len(spans), k=1)
    unequal = votes[earlier] != votes[later]
    if unequal.any():
        earlier, later = earlier[unequal], later[unequal]
        combined = rankdata([right - left for left, right in spans]) + rankdata(-counts)
        methods = np.column_stack((values[:, 0], combined, values[:, 1:]))
        # Comparisons also handle two unavailable words tied at negative infinity.
        direction = ((methods[later] > methods[earlier]).astype(int) - (methods[later] < methods[earlier]))
        agreement = direction * np.sign(votes[later] - votes[earlier])[:, None]
        rank_scores = ((agreement + 1) / 2).mean(axis=0)
        paired_counts = len(earlier)
    likelihood, bias = None, None
    if present.any():
        z = values[present, 1:]
        shares = votes[present, None] / 20
        likelihood = (np.logaddexp(0, z) - shares * z).mean(axis=0)
        bias = (expit(z) - shares).mean(axis=0)
    return len(spans), int(present.sum()), paired_counts, rank_scores, likelihood, bias


def aggregate(by_group, methods):
    if len(by_group) < 2:
        raise ValueError('Need more than one independent group to estimate the paired interval')
    matrix = np.asarray([np.mean(by_group[key], axis=0) for key in sorted(by_group)])
    estimate = matrix.mean(axis=0)
    rng = np.random.default_rng(SEED)
    draws = rng.multinomial(len(matrix), np.full(len(matrix), 1 / len(matrix)), size=DRAWS)
    sampled = draws @ matrix / len(matrix)
    fused = methods.index('fused_fit')
    return {'groups': len(matrix), 'sentences': sum(map(len, by_group.values())),
            'methods': dict(zip(methods, estimate.tolist(), strict=True)),
            'fused_minus': {name: {'difference': float(estimate[fused] - estimate[index]),
                                   'ci95': np.quantile(sampled[:, fused] - sampled[:, index], (.025, .975)).tolist()}
                            for index, name in enumerate(methods) if name != 'fused_fit'}}


def evaluate(documents, outcomes, predictions, group_key):
    rank, nll, errors = defaultdict(list), defaultdict(list), defaultdict(list)
    totals = Counter()
    for identifier, row in outcomes.items():
        count, covered, pairs, ordering, loss, bias = compare_sentence(documents[identifier], row, predictions[identifier])
        totals.update(candidates=count, aligned=covered, pairs=pairs)
        group = row[group_key]
        if ordering is not None:
            rank[group].append(ordering)
        if loss is not None:
            nll[group].append(loss)
            errors[group].append(bias)
    return {**totals, 'coverage': totals['aligned'] / totals['candidates'], 'source_sentences': len(outcomes),
            'rank': aggregate(rank, METHODS), 'vote_nll': aggregate(nll, tuple(COLUMNS)),
            'mean_vote_bias': dict(zip(COLUMNS, np.mean([np.mean(group, axis=0) for group in errors.values()], axis=0).tolist(), strict=True))}


def score(wiki, news, sealed, output):
    seal = read_json(sealed / 'seal.json')
    if (seal['protocol_sha256'] != digest(PROTOCOL) or seal['code_sha256'] != digest(Path(__file__))
            or seal['wiki_assay_sha256'] != digest(wiki / 'assay.json')
            or seal['news_assay_sha256'] != digest(news / 'assay.json')
            or any(digest(sealed / name) != value for name, value in seal['file_sha256'].items())):
        raise ValueError('Expected the exact weights and forecasts frozen under the CWI protocol')
    fit = read_json(sealed / 'fit.json')
    evaluations = {}
    for prepared, source in ((wiki, 'wikinews'), (news, 'news')):
        plan = read_json(prepared / 'assay.json')
        documents = dict(read_stimuli(prepared / 'stimuli.jsonl', plan['stimuli_sha256']).documents)
        forecasts = read_rows(sealed / f'{source}.jsonl', {'id', 'words'})
        if forecasts.keys() != documents.keys():
            raise ValueError('Sealed projections must cover every original sentence')
        if source == 'wikinews':
            for split in SPLITS[1:]:
                path = prepared / f'{split}.jsonl'
                heldout = checked_outcomes(path, plan['file_sha256'][path.name], WIKI_OUTCOME_FIELDS)
                if any(row['split'] != split or row['group_id'] in fit['training_group_ids'] for row in heldout.values()):
                    raise ValueError('WikiNews holdout contains a training group or mismatched split')
                evaluations[split] = evaluate(documents, heldout, forecasts, 'group_id')
        else:
            heldout = checked_outcomes(prepared / 'outcomes.jsonl', plan['outcomes_sha256'], NEWS_OUTCOME_FIELDS)
            evaluations['news_retro'] = evaluate(documents, heldout, forecasts, 'hit_id')
    primary = evaluations['test']
    gates = {'coverage': primary['coverage'] >= .95,
             'fused_rank_over_surface_fit': primary['rank']['fused_minus']['surface_fit']['ci95'][0] > 0,
             'fused_rank_over_served': primary['rank']['fused_minus']['served']['ci95'][0] > 0,
             'fused_vote_nll_under_surface_fit': primary['vote_nll']['fused_minus']['surface_fit']['ci95'][1] < 0}
    write_json(output, {'gates': gates, 'pass': all(gates.values()), 'evaluations': evaluations,
                        'forecast_seal_sha256': digest(sealed / 'seal.json'), 'bootstrap_seed': SEED,
                        'bootstrap_draws': DRAWS})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    make, grading = (commands.add_parser(name) for name in ('predict', 'score'))
    for command in (make, grading):
        for field in ('wiki', 'news', 'output'):
            command.add_argument(f'--{field}', type=Path, required=True)
    for field in ('wiki-forecast', 'news-forecast', 'complex-source'):
        make.add_argument(f'--{field}', type=Path, required=True)
    grading.add_argument('--sealed', type=Path, required=True)
    args = vars(parser.parse_args())
    (predict if args.pop('command') == 'predict' else score)(**args)
