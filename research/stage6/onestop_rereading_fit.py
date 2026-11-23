"""Fit and score body-only and exposure-conditioned OneStop readouts."""

import argparse
from collections import defaultdict
from hashlib import file_digest, sha256
import json
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .onestop_rereading_baseline import DRAWS, SEED, prior


WIDTH = 2048
PENALTY = 1.0
EFFECT_MARGIN = 0.03
DEVELOPMENT_ARTICLES = set(range(1, 21))
HOLDOUT_ARTICLES = set(range(21, 31))
METHODS = ('pooled', 'status_only', 'qwen', 'qwen_additive', 'qwen_interaction')
TEXT_METHODS = METHODS[2:]
REFERENCE_METHODS = ('qwen_additive', 'status_only', 'qwen')
FORECAST_FILE = 'forecasts.npz'
HEADS_FILE = 'heads.npz'
SEAL_FILE = 'fit-seal.json'


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def read_frozen_rows(path, assay, name):
    contents = path.read_bytes()
    if sha256(contents).hexdigest() != assay['file_sha256'][name]:
        raise ValueError(f'Unexpected bytes in sealed OneStop {name}')
    return [json.loads(line) for line in contents.splitlines()]


def stimulus_keys(rows):
    documents = {}
    for item in rows:
        if set(item) != {'id', 'text', 'stops'}:
            raise ValueError('Stimuli must contain only article text, its ID and stop offsets')
        case, text, stops = item['id'], item['text'], item['stops']
        if (not isinstance(text, str) or not text.strip() or case != sha256(text.encode('utf-8')).hexdigest()
                or case in documents or not isinstance(stops, list) or not stops
                or any(type(stop) is not int or not 0 < stop <= len(text) for stop in stops)
                or sorted(set(stops)) != stops):
            raise ValueError('Invalid source document or stop registry')
        documents[case] = stops
    if not documents:
        raise ValueError('Expected at least one source document')
    return {(case, stop) for case, stops in documents.items() for stop in stops}


def feature_keys(ids, stops):
    if ids.ndim != 1 or ids.dtype.kind != 'U' or stops.shape != ids.shape or stops.dtype.kind not in 'iu':
        raise ValueError('Expected one string ID and integer stop per feature row')
    keys = list(zip(ids.tolist(), stops.tolist(), strict=True))
    if not keys or len(set(keys)) != len(keys):
        raise ValueError('Expected nonempty, unique feature keys')
    return keys


def load_features(path, expected):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {'ids', 'stops', 'tokens', 'states'}:
            raise ValueError('Unexpected feature array inventory')
        ids, stops, tokens, raw = (archive[name] for name in ('ids', 'stops', 'tokens', 'states'))
    keys = feature_keys(ids, stops)
    if (set(keys) != expected or tokens.shape != stops.shape or tokens.dtype.kind not in 'iu'
            or (tokens < 0).any() or raw.shape != (len(keys), WIDTH) or raw.dtype != np.float32):
        raise ValueError('Extracted states or terminal tokens do not cover the sealed stops')
    by_document = defaultdict(list)
    for (case, stop), token in zip(keys, tokens.tolist(), strict=True):
        by_document[case].append((stop, token))
    for positions in by_document.values():
        sorted_tokens = [token for _, token in sorted(positions)]
        if any(a >= b for a, b in zip(sorted_tokens, sorted_tokens[1:])):
            raise ValueError('Terminal token indices must advance within the article')
    states = raw.astype(np.float64)
    norms = np.linalg.norm(states, axis=1)
    if not np.all(np.isfinite(norms) & (norms > 0)):
        raise ValueError('A feature row has a nonfinite or zero state')
    states /= norms[:, None]
    return ids, stops, keys, states


def observations(rows, keys, expected_articles):
    index = {key: i for i, key in enumerate(keys)}
    articles = {}
    result = []
    for item in rows:
        if set(item) != {'id', 'stop', 'article', 'cell', 'repeat', 'correct'}:
            raise ValueError('Expected exactly the registered anonymous outcome fields')
        case, stop, article, cell, status, outcome = (item[key] for key in (
            'id', 'stop', 'article', 'cell', 'repeat', 'correct'))
        if (type(stop) is not int or type(article) is not int or article not in expected_articles
                or type(status) is not int or status not in (0, 1) or type(outcome) is not int or outcome not in (0, 1)
                or not isinstance(cell, list) or len(cell) != 3 or type(cell[0]) is not int or not 1 <= cell[0] <= 7
                or cell[1] not in ('Adv', 'Ele') or type(cell[2]) is not int or cell[2] not in (0, 1, 2)
                or (case, stop) not in index):
            raise ValueError('Outcome is outside the certified stimuli, split or response contract')
        if articles.setdefault(case, article) != article:
            raise ValueError('One source instance belongs to multiple articles')
        result.append((index[case, stop], article, tuple(cell), status, outcome))
    if set(articles.values()) != expected_articles:
        raise ValueError('The source articles do not match the fixed split')
    return result


def logits(weights, states, method):
    base = weights[0] + states @ weights[1:WIDTH + 1]
    if method == 'qwen':
        return np.column_stack((base, base))
    exposed = base + weights[WIDTH + 1]
    if method == 'qwen_interaction':
        exposed += states @ weights[WIDTH + 2:]
    return np.column_stack((base, exposed))


def logistic_objective(weights, states, totals, positives, method):
    values = logits(weights, states, method)
    data_loss = (totals * np.logaddexp(0.0, values) - positives * values).sum()
    residual = totals * expit(values) - positives
    base_residual = residual.sum(axis=1)
    gradient = np.empty_like(weights)
    gradient[0] = base_residual.sum()
    gradient[1:WIDTH + 1] = states.T @ base_residual
    if method != 'qwen':
        gradient[WIDTH + 1] = residual[:, 1].sum()
    if method == 'qwen_interaction':
        gradient[WIDTH + 2:] = states.T @ residual[:, 1]
    gradient[1:] += PENALTY * weights[1:]
    return float(data_loss + 0.5 * PENALTY * (weights[1:] @ weights[1:])), gradient


def fit_head(method, states, totals, positives):
    sizes = {'qwen': WIDTH + 1, 'qwen_additive': WIDTH + 2, 'qwen_interaction': 2 * WIDTH + 2}
    initial = np.zeros(sizes[method])
    initial[0] = np.log(positives.sum() / (totals.sum() - positives.sum()))
    result = minimize(
        logistic_objective, initial, args=(states, totals, positives, method), jac=True, method='L-BFGS-B',
        options={'maxiter': 1000, 'gtol': 1e-8, 'ftol': 1e-14},
    )
    gradient_max = float(np.max(np.abs(result.jac)))
    if (not result.success or not np.all(np.isfinite(result.x)) or not np.isfinite(result.fun)
            or not np.isfinite(gradient_max) or gradient_max > 1e-4):
        raise RuntimeError(f'{method} logistic fit did not converge: {result.message}; gradient {gradient_max}')
    return result.x, {'iterations': int(result.nit), 'loss': float(result.fun), 'max_abs_gradient': gradient_max}


def fit(stimuli, features, development, protocol, assay, output):
    if output.exists():
        raise FileExistsError(output)
    receipt = json.loads(assay.read_text(encoding='utf-8'))
    keys_expected = stimulus_keys(read_frozen_rows(stimuli, receipt, 'stimuli.jsonl'))
    ids, stops, keys, states = load_features(features, keys_expected)
    rows = observations(read_frozen_rows(development, receipt, 'development.jsonl'), keys, DEVELOPMENT_ARTICLES)
    totals = np.zeros((len(keys), 2), dtype=np.int64)
    positives = np.zeros_like(totals)
    for index, _, _, status, outcome in rows:
        totals[index, status] += 1
        positives[index, status] += outcome
    exposure_totals, exposure_positives = totals.sum(axis=0), positives.sum(axis=0)
    if np.any(exposure_positives <= 0) or np.any(exposure_positives >= exposure_totals):
        raise ValueError('Training needs correct and incorrect outcomes at both exposure states')
    train = totals.sum(axis=1) > 0
    training_states = states[train]
    training_totals = totals[train]
    training_positives = positives[train]
    pooled = prior(positives.sum(), totals.sum())
    weights = {'pooled': np.array([pooled]), 'status_only': prior(exposure_positives, exposure_totals)}
    forecasts = {'pooled': np.full((len(keys), 2), pooled),
                 'status_only': np.broadcast_to(weights['status_only'], (len(keys), 2))}
    diagnostics = {}
    for method in TEXT_METHODS:
        weights[method], diagnostics[method] = fit_head(method, training_states, training_totals, training_positives)
        forecasts[method] = expit(logits(weights[method], states, method))
    if not all(np.all(np.isfinite(phase) & (0 < phase) & (phase < 1)) for phase in forecasts.values()):
        raise ValueError('All forecasts must be finite probabilities inside (0, 1)')

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.onestop-fit-', dir=output.parent) as directory:
        stage = Path(directory) / 'complete'
        stage.mkdir()
        np.savez_compressed(stage / HEADS_FILE, **weights)
        np.savez_compressed(stage / FORECAST_FILE, ids=ids, stops=stops, development=train, **forecasts)
        write_json(stage / SEAL_FILE, {
            'assay_sha256': digest(assay), 'fit_code_sha256': digest(Path(__file__)), 'protocol_sha256': digest(protocol),
            'stimuli_sha256': receipt['file_sha256']['stimuli.jsonl'],
            'features_sha256': digest(features), 'development_sha256': receipt['file_sha256']['development.jsonl'],
            'heads_sha256': digest(stage / HEADS_FILE), 'forecasts_sha256': digest(stage / FORECAST_FILE),
            'development_trials': len(rows), 'development_stops': int(train.sum()),
            'all_certified_stops': len(keys), 'methods': list(METHODS), 'optimizer': diagnostics,
            'ridge_penalty': PENALTY, 'state_dimensions': WIDTH,
        })
        stage.rename(output)


def score_article_cells(rows, probabilities):
    cells = defaultdict(lambda: ([], []))
    for index, article, cell, status, outcome in rows:
        cells[article, cell][status].append((index, outcome))
    by_article = defaultdict(list)
    vote_losses = []
    for (article, _), outcomes in sorted(cells.items()):
        if not all(outcomes):
            continue
        phase_metrics = []
        for status, entries in enumerate(outcomes):
            indices, labels = np.array(entries).T
            predictions = probabilities[:, indices, status]
            brier = np.square(predictions - labels).mean(axis=1)
            nll = -np.log(np.where(labels == 1, predictions, 1 - predictions))
            toggled = (probabilities[:, indices, 1] - probabilities[:, indices, 0]).mean(axis=1)
            phase_metrics.append((brier, nll.mean(axis=1), predictions.mean(axis=1), labels.mean(), toggled))
            vote_losses.append(nll)
        first, repeated = phase_metrics
        by_article[article].append(((first[0] + repeated[0]) / 2, (first[1] + repeated[1]) / 2,
                                    repeated[2] - first[2], repeated[3] - first[3], (first[4] + repeated[4]) / 2))
    if set(by_article) != HOLDOUT_ARTICLES:
        raise ValueError('Each held-out article needs cells with responses in both phases')
    articles = {}
    for article, article_cells in by_article.items():
        brier, nll, effects, human, toggled = (np.mean(values, axis=0) for values in zip(*article_cells, strict=True))
        articles[article] = {'cells': len(article_cells), 'brier': brier, 'nll': nll, 'effect': effects,
                             'observed_effect': float(human), 'same_prefix_effect': toggled}
    return articles, np.concatenate(vote_losses, axis=1)


def evaluate(rows, probabilities):
    by_article, vote_losses = score_article_cells(rows, probabilities)
    article_ids = sorted(by_article)
    brier, nll, effects, human, same_prefix = (np.array([by_article[article][key] for article in article_ids])
                                              for key in ('brier', 'nll', 'effect', 'observed_effect',
                                                          'same_prefix_effect'))
    n_articles = len(article_ids)
    rng = Random(SEED)
    sample = np.array([rng.choices(range(n_articles), k=n_articles) for _ in range(DRAWS)])
    method_results = {}
    per_vote_nll = vote_losses.mean(axis=1)
    for index, method in enumerate(METHODS):
        error = effects[:, index] - human
        interval = np.quantile(error[sample].mean(axis=1), (0.05, 0.95)).tolist()
        signed = float(effects[:, index].mean())
        method_results[method] = {
            'article_equal_brier': float(brier[:, index].mean()),
            'article_equal_nll_nats': float(nll[:, index].mean()),
            'per_vote_nll_nats': float(per_vote_nll[index]),
            'article_equal_effect': signed, 'article_equal_effect_error': float(error.mean()),
            'effect_error_ci90': interval, 'within_effect_margin': -EFFECT_MARGIN <= interval[0] <= interval[1] <= EFFECT_MARGIN,
            'effect_direction_matches': bool(np.sign(signed) == np.sign(human.mean())),
            'article_equal_same_prefix_status_effect': float(same_prefix[:, index].mean()),
        }
    comparisons = {}
    interaction = brier[:, METHODS.index('qwen_interaction')]
    for method in REFERENCE_METHODS:
        differences = interaction - brier[:, METHODS.index(method)]
        interval = np.quantile(differences[sample].mean(axis=1), (0.025, 0.975)).tolist()
        comparisons[method] = {'effect': float(differences.mean()), 'ci95': interval, 'improves_brier': interval[1] < 0}
    interaction_effect = method_results['qwen_interaction']
    passed = (all(comparison['improves_brier'] for comparison in comparisons.values())
              and interaction_effect['effect_direction_matches'] and interaction_effect['within_effect_margin'])
    return {
        'articles': n_articles, 'common_question_version_cells': sum(row['cells'] for row in by_article.values()),
        'votes_on_common_support': vote_losses.shape[1], 'article_equal_observed_effect': float(human.mean()),
        'methods': method_results, 'interaction_minus_reference_brier': comparisons,
        'interaction_acceptance_passed': passed,
        'article_details': {article: {
            'common_cells': by_article[article]['cells'], 'observed_effect': float(human[index]),
            'brier': dict(zip(METHODS, brier[index].tolist(), strict=True)),
            'effect': dict(zip(METHODS, effects[index].tolist(), strict=True)),
        } for index, article in enumerate(article_ids)},
    }


def score(forecast_dir, holdout, assay, output):
    seal_path = forecast_dir / SEAL_FILE
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    forecast_path = forecast_dir / FORECAST_FILE
    if (seal['assay_sha256'] != digest(assay) or seal['fit_code_sha256'] != digest(Path(__file__))
            or seal['forecasts_sha256'] != digest(forecast_path) or seal['heads_sha256'] != digest(forecast_dir / HEADS_FILE)
            or seal['methods'] != list(METHODS)):
        raise ValueError('The frozen forecasts do not match their fit receipt')
    with np.load(forecast_path, allow_pickle=False) as archive:
        if set(archive.files) != {'ids', 'stops', 'development', *METHODS}:
            raise ValueError('Unexpected forecast archive inventory')
        ids = archive['ids']
        keys = feature_keys(ids, archive['stops'])
        development = archive['development']
        probabilities = np.stack([archive[method] for method in METHODS])
    if (development.shape != ids.shape or development.dtype != np.bool_
            or probabilities.shape != (len(METHODS), len(keys), 2)
            or not np.all(np.isfinite(probabilities) & (0 < probabilities) & (probabilities < 1))):
        raise ValueError('Invalid forecast coverage or probabilities')
    receipt = json.loads(assay.read_text(encoding='utf-8'))
    rows = observations(read_frozen_rows(holdout, receipt, 'holdout.jsonl'), keys, HOLDOUT_ARTICLES)
    used = {index for index, *_ in rows}
    if (used != set(np.flatnonzero(~development).tolist())
            or not set(ids[~development]).isdisjoint(ids[development])):
        raise ValueError('Held-out responses overlap development, or do not cover the frozen test stops')
    result = evaluate(rows, probabilities)
    result.update({
        'fit_sha256': digest(seal_path), 'holdout_sha256': receipt['file_sha256']['holdout.jsonl'],
        'holdout_votes_before_common_support': len(rows), 'bootstrap_draws': DRAWS,
        'bootstrap_seed': SEED, 'effect_equivalence_margin': EFFECT_MARGIN,
        'by_level': {level: evaluate([row for row in rows if row[2][1] == level], probabilities)
                     for level in ('Adv', 'Ele')},
    })
    write_json(output, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    for command, options in (
        ('fit', ('stimuli', 'features', 'development', 'protocol', 'assay', 'output')),
        ('score', ('forecast-dir', 'holdout', 'assay', 'output')),
    ):
        action = actions.add_parser(command)
        for option in options:
            action.add_argument(f'--{option}', type=Path, required=True)
    args = vars(parser.parse_args())
    action = args.pop('action')
    (fit if action == 'fit' else score)(**args)


if __name__ == '__main__':
    main()
