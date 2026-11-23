"""Seal reader-disjoint OneStopQA-to-RACE education predictions, then score them."""

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unicodedata

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from research.joint.onestop_probe import RIDGE, align_features, row_losses
from research.joint.prepare import open_book_inputs, prompt_id
from research.stage6.race_audit import RACE_ID, SOURCE_SHA256, audit, visible_trials
from research.stage6.race_predict import METHODS as OLD_METHODS, ONESTOP_DIGESTS, digest, race_rows, softmax
from research.stage6.race_prepare import released_answers
from research.stage6.race_score import load_outcomes


PROTOCOL = Path(__file__).with_name('race-education-holdout-protocol.md')
PROTOCOL_SHA256 = 'ba4256006b14ce1e9a20b2ce86b46c7bfa26ab3aca1ee2165a9591f7f7a099e9'
CATEGORIES = {'Graduated College': 406, 'Graduated High School': 276, 'Higher Degree': 256, '': 32}
METHODS = ('historical', 'surface', 'text', 'text_null', 'education_additive', 'education_interaction')
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
SEED = 771304
DRAWS = 10000
CALIBRATION_RIDGE = 0.01


@dataclass(frozen=True)
class Trial:
    id: str
    reader: str
    fold: int
    group: str
    full: str
    question: str
    order: tuple[int, ...]
    surface: np.ndarray


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def corpus_design(source, evidence):
    source_audit = audit(source)
    if source_audit['overlap']['distinct_reader_keys'] != 970 or any(
        source_audit['overlap'][field] for field in ('item_id', 'question', 'paragraph')
    ):
        raise ValueError('Expected the fixed, disjoint OneStopQA and RACE materials')
    profile = {}
    observed = {}
    with source.open(encoding='utf-8', newline='') as stream:
        for index, row in enumerate(csv.DictReader(stream, delimiter='\t', strict=True), 1):
            if row['trial_type'] != 'reading_comprehension':
                continue
            reader = row['submission_ind']
            group = row['education']
            if group not in CATEGORIES or profile.setdefault(reader, group) != group:
                raise ValueError('An education value is unknown or inconsistent within a reader')
            if row['source'] == 'Onestop':
                observed[index] = int(row['answer_response'])
    if Counter(profile.values()) != CATEGORIES:
        raise ValueError('Expected the declared education categories and 970 readers')
    folds = {}
    for category in CATEGORIES:
        readers = sorted((key for key, value in profile.items() if value == category),
                         key=lambda key: sha256(f'race-education-v1:{key}'.encode()).digest())
        folds.update({key: j % 5 for j, key in enumerate(readers)})

    one, race, outcomes = [], [], []
    for index, row in visible_trials(source):
        reader = row['submission_ind']
        order = tuple(map(int, row['new_options'].split('|')))
        options = row['options'].split('|')
        displayed = [options[position] for position in order]
        full, question = open_book_inputs(row['paragraph'], row['question'], displayed)
        entry = Trial(f'race-row-{index:05d}', reader, folds[reader], profile[reader],
                      prompt_id('full', full), prompt_id('question', question), order,
                      surface_features(row['paragraph'], row['question'], displayed))
        if row['source'] == 'Onestop':
            one.append(entry)
            outcomes.append(order.index(observed[index]))
        else:
            race.append(entry)
    published_design = race_rows(evidence)
    reconstructed = [{'id': row.id, 'full_id': row.full, 'question_id': row.question, 'displayed_order': list(row.order)}
                     for row in race]
    if reconstructed != published_design or len(one) != 1940:
        raise ValueError('The Prolific records do not recover the original trial and display orders')
    return one, race, np.eye(4, dtype=np.float64)[outcomes], source_audit


def surface_features(body, question, options):
    def words(text):
        return WORD.findall(unicodedata.normalize('NFKC', text).casefold())

    body_words = set(words(body))
    question_words = set(words(question))
    values = []
    for option in options:
        terms = words(option)
        values.append([np.log1p(len(option.split())),
                       sum(word in question_words for word in terms) / max(len(terms), 1),
                       sum(word in body_words for word in terms) / max(len(terms), 1)])
    return np.array(values)


def checks(paths):
    if digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('RACE education protocol digest changed')
    actions = {
        'source': SOURCE_SHA256, 'onestop_design': ONESTOP_DIGESTS['design'],
        'onestop_features': ONESTOP_DIGESTS['features'],
    }
    original_race = read_json(paths['historical_seal'])
    reference = original_race['input_sha256']
    actions |= {'race_design': reference['outcome_free_design'], 'race_features': reference['race_l4_features'],
                'historical_weights': reference['onestop_weights'],
                'historical_predictions': original_race['predictions_sha256']}
    actual = {name: digest(path) for name, path in paths.items()}
    if any(actual[name] != expected for name, expected in actions.items()) or (
        len(read_json(paths['race_causality'])['checks']) != 12
        or not all(check['passed'] is True for kind in ('race_causality', 'onestop_causality')
                   for check in read_json(paths[kind])['checks'])
    ):
        raise ValueError('Expected the released causal L4 features and verified historical baseline')
    return {**actual, 'protocol': PROTOCOL_SHA256, 'code': digest(Path(__file__))}


def load_one(path, one, target):
    with np.load(path, allow_pickle=False) as archive:
        published = archive['onestop_ids'].tolist()
        count = archive['onestop_counts']
        gold = archive['onestop_correct']
        questions = archive['onestop_question_only_ids'].tolist()
    positions = {key: i for i, key in enumerate(published)}
    index = [positions[row.full] for row in one]
    computed = np.zeros_like(count)
    np.add.at(computed, index, target.astype(computed.dtype))
    if (not np.array_equal(computed, count) or count.sum() != 1940
            or any(row.question != questions[pos] or row.order.index(0) != gold[pos]
                   for row, pos in zip(one, index, strict=True))):
        raise ValueError('Original OneStopQA response counts, option positions or prompt joins differ')


def features(path, rows):
    with np.load(path, allow_pickle=False) as archive:
        ids, hidden = archive['ids'], archive['hidden']
    if hidden.dtype != np.float32 or set(ids) != {item for row in rows for item in (row.full, row.question)}:
        raise ValueError('Unexpected frozen FP32 prompt inventory')
    return align_features([row.full for row in rows], ids, hidden).astype(np.float64)


def historical(weights_path, predictions_path, states, trials):
    method = OLD_METHODS[2]
    with np.load(weights_path, allow_pickle=False) as archive:
        weights = archive[method]
        if weights.shape != (5, 2049, 4) or not np.array_equal(archive['folds'], np.arange(5)):
            raise ValueError('Expected the exact historical OneStopQA prompt weights')
    replay = np.mean([softmax(states @ value[:-1] + value[-1]) for value in weights], axis=0)
    with np.load(predictions_path, allow_pickle=False) as archive:
        if archive['ids'].tolist() != [row.id for row in trials] or tuple(archive['methods']) != OLD_METHODS:
            raise ValueError('Unexpected original RACE historical forecast order')
        deviation = float(np.max(np.abs(replay - archive['predictions'][2])))
    if deviation > 1e-12:
        raise ValueError('Historical RACE probabilities do not replay from archived states')
    return replay, deviation


def nll(logits, target):
    log_probability = logits - logsumexp(logits, axis=-1, keepdims=True)
    return float(-np.sum(target * log_probability) / len(target)), (np.exp(log_probability) - target) / len(target)


def optimize(objective, initial, bounds=None):
    result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 2e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise ArithmeticError(f'OneStopQA penalized fit did not converge: {result.message}')
    return result.x


def fit_text(states, target):
    def objective(parameters):
        weights = parameters.reshape(-1, 4)
        loss, residual = nll(states @ weights[:-1] + weights[-1], target)
        gradient = np.vstack((states.T @ residual + RIDGE * weights[:-1], residual.sum(axis=0)))
        return loss + RIDGE / 2 * np.square(weights[:-1]).sum(), gradient.ravel()

    parameters = optimize(objective, np.zeros((states.shape[1] + 1) * 4))
    _, gradient = objective(parameters)
    if np.max(np.abs(gradient)) > 1e-6:
        raise ArithmeticError('The refit did not reach the original OneStopQA gradient tolerance')
    return parameters.reshape(-1, 4)


def fit_surface(train, target, test):
    def objective(parameters):
        loss, residual = nll(train @ parameters[:3] + parameters[3:], target)
        grad = np.sum(train * residual[:, :, None], axis=(0, 1)) + RIDGE * parameters[:3]
        return loss + RIDGE / 2 * np.square(parameters[:3]).sum(), np.r_[grad, residual.sum(axis=0)]

    fitted = optimize(objective, np.zeros(7))
    return softmax(test @ fitted[:3] + fitted[3:])


def calibrated_logits(logits, parameters):
    return parameters[0] * logits + np.r_[parameters[1:], 0.0]


def fit_calibration(logits, target, center, *, free_temperature):
    def objective(parameters):
        loss, residual = nll(calibrated_logits(logits, parameters), target)
        difference = parameters - center
        gradient = np.r_[np.sum(residual * logits), residual.sum(axis=0)[:3]]
        return loss + CALIBRATION_RIDGE / 2 * np.square(difference).sum(), gradient + CALIBRATION_RIDGE * difference

    limit = (0, 4) if free_temperature else (center[0], center[0])
    return optimize(objective, center, [limit] + [(None, None)] * 3)


def forecast_folds(one, race, train, test, target):
    train_folds = np.array([row.fold for row in one])
    test_folds = np.array([row.fold for row in race])
    train_group = np.array([row.group for row in one])
    test_group = np.array([row.group for row in race])
    train_surface = np.stack([row.surface for row in one])
    test_surface = np.stack([row.surface for row in race])
    forecasts = np.full((len(METHODS) - 1, len(race), 4), np.nan)
    coverage = []
    for fold in range(5):
        training = train_folds != fold
        evaluation = test_folds == fold
        fit_readers = {row.reader for row, include in zip(one, training, strict=True) if include}
        test_readers = {row.reader for row, include in zip(race, evaluation, strict=True) if include}
        if fit_readers & test_readers:
            raise ValueError('Disjoint OneStopQA and RACE readers required in every evaluation fold')
        inner_logits = np.full((len(one), 4), np.nan)
        for inner in sorted(set(range(5)) - {fold}):
            fit_inner = training & (train_folds != inner)
            validation = train_folds == inner
            weights = fit_text(train[fit_inner], target[fit_inner])
            inner_logits[validation] = train[validation] @ weights[:-1] + weights[-1]
        weights = fit_text(train[training], target[training])
        test_logits = test[evaluation] @ weights[:-1] + weights[-1]
        null = fit_calibration(inner_logits[training], target[training], np.array([1., 0., 0., 0.]),
                               free_temperature=True)
        blank = softmax(calibrated_logits(test_logits, null))
        forecasts[0, evaluation] = fit_surface(train_surface[training], target[training], test_surface[evaluation])
        forecasts[1, evaluation] = softmax(test_logits)
        forecasts[2:, evaluation] = blank
        for group in (category for category in CATEGORIES if category):
            train_for_group = training & (train_group == group)
            eligible = evaluation & (test_group == group)
            visible = test_group[evaluation] == group
            for method, free_t in ((3, False), (4, True)):
                conditional = fit_calibration(inner_logits[train_for_group], target[train_for_group], null,
                                              free_temperature=free_t)
                forecasts[method, eligible] = softmax(calibrated_logits(test_logits[visible], conditional))
        coverage.append({'fold': fold, 'fit_readers': len(fit_readers), 'test_readers': len(test_readers),
                         'fit_trials': int(training.sum()), 'test_trials': int(evaluation.sum()),
                         'null_temperature': float(null[0])})
    if (not np.all(np.isfinite(forecasts) & (forecasts > 0)) or not np.allclose(forecasts.sum(axis=-1), 1)
            or not np.array_equal(forecasts[3:, test_group == ''], np.broadcast_to(
                forecasts[2, test_group == ''], forecasts[3:, test_group == ''].shape))):
        raise ValueError('Incomplete probability or empty-background invariance checks failed')
    return forecasts, coverage


def bootstrap(delta, articles, readers):
    _, a = np.unique(articles, return_inverse=True)
    _, r = np.unique(readers, return_inverse=True)
    n_articles, n_readers = max(a) + 1, max(r) + 1
    rng = np.random.default_rng(SEED)
    result = np.empty((DRAWS, delta.shape[1]))
    for left in range(0, DRAWS, 100):
        n = min(100, DRAWS - left)
        article_draw = rng.multinomial(n_articles, np.ones(n_articles) / n_articles, size=n)
        reader_draw = rng.multinomial(n_readers, np.ones(n_readers) / n_readers, size=n)
        weights = article_draw[:, a] * reader_draw[:, r]
        result[left:left + n] = (weights @ delta) / weights.sum(axis=1, keepdims=True)
    return np.quantile(result, (0.025, 0.975), axis=0).T.tolist()


def score(source, outcomes_path, outcomes_seal, race_design_sha, race, forecasts):
    source_seal = read_json(outcomes_seal)
    if (digest(outcomes_path) != source_seal['outcomes_sha256'] or source_seal['source_sha256'] != SOURCE_SHA256
            or source_seal['design_sha256'] != race_design_sha
            or source_seal['reference_items_matched'] != 648):
        raise ValueError('Expected RACE truth with independently verified option keys')
    rows = load_outcomes(outcomes_path, [row.id for row in race])
    visible = {f'race-row-{i:05d}': row for i, row in visible_trials(source) if row['source'] == 'RACE'}
    indices = {int(row.id.rsplit('-', 1)[1]) for row in race}
    released = released_answers(source, indices)
    for trial, outcome in zip(race, rows, strict=True):
        row = visible[trial.id]
        article = RACE_ID.fullmatch(row['item_id'])[1]
        if (outcome['reader_id'] != f'prolific-{trial.reader}'
                or outcome['article_id'] != f"{row['difficulty'].lower()}-{article}"
                or outcome['selected'] != trial.order.index(released[int(trial.id.rsplit('-', 1)[1])])
                or outcome['correct'] != trial.order.index(0)):
            raise ValueError('RACE reader key, item group or chosen display position disagrees with release')
    selected = np.eye(4, dtype=np.int64)[[row['selected'] for row in rows]]
    correct = np.array([row['correct'] for row in rows])
    all_losses = [row_losses(values, selected, correct) for values in forecasts]
    contrasts = ('null_minus_interaction', 'null_minus_additive', 'additive_minus_interaction')
    nll_matrix = np.column_stack([loss['choice_nll'] for loss in all_losses])
    independent = -np.log(forecasts[:, np.arange(len(rows)), np.argmax(selected, axis=1)]).T
    if not np.array_equal(nll_matrix, independent):
        raise ArithmeticError('Historical and independently indexed NLL computations differ')
    differences = np.column_stack([nll_matrix[:, a] - nll_matrix[:, b] for a, b in ((3, 5), (3, 4), (4, 5))])
    groups = np.array([trial.group for trial in race])
    articles = np.array([row['article_id'] for row in rows])
    readers = np.array([trial.reader for trial in race])

    def report(mask):
        ci = bootstrap(differences[mask], articles[mask], readers[mask])
        means = differences[mask].mean(axis=0)
        return {'trials': int(mask.sum()), 'articles': len(set(articles[mask])), 'readers': len(set(readers[mask])),
                'methods': {name: {key: float(values[mask].mean()) for key, values in loss.items()}
                            for name, loss in zip(METHODS, all_losses, strict=True)},
                'choice_nll_improvement': {name: {'point': float(point), 'ci95': interval}
                                           for name, point, interval in zip(contrasts, means, ci, strict=True)}}

    main = report(np.ones(len(rows), dtype=bool))
    return {'evaluation': main, 'known_education_only': report(groups != ''),
            'by_group': {group or 'unknown': report(groups == group) for group in CATEGORIES},
            'passes': main['choice_nll_improvement'][contrasts[0]]['ci95'][0] > 0,
            'bootstrap': {'seed': SEED, 'draws': DRAWS, 'groups': ['RACE_article', 'reader'],
                          'fixed_predictions': True}, 'truth_sha256': digest(outcomes_path)}


def run(cache, output):
    if output.exists():
        raise FileExistsError(output)
    joint = cache / 'joint/cogllm-joint-20261120T162205Z-e984cdd0/outputs/onestop_features'
    stage = cache / 'stage6/race-runs/cogllm-stage6-race-20261120T181401Z-d6e92f78'
    paths = {'source': cache / 'raw/onestop/qa.tsv', 'onestop_design': cache / 'joint/data/design.npz',
             'onestop_features': joint / 'features.npz', 'onestop_causality': joint / 'causality.json',
             'race_design': cache / 'stage6/race/design.jsonl', 'race_features': stage / 'outputs/extract/features.npz',
             'race_causality': stage / 'outputs/extract/causality.json',
             'historical_weights': cache / 'stage6/onestop-rebuild/weights.npz',
             'historical_predictions': stage / 'forecast/predictions.npz',
             'historical_seal': stage / 'forecast/verification.json'}
    receipt = checks(paths)
    one, race, target, audit_record = corpus_design(paths['source'], paths['race_design'])
    load_one(paths['onestop_design'], one, target)
    train = features(paths['onestop_features'], one)
    test = features(paths['race_features'], race)
    historic, deviation = historical(paths['historical_weights'], paths['historical_predictions'], test, race)
    estimates, folds = forecast_folds(one, race, train, test, target)
    forecasts = np.concatenate((historic[None], estimates))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.race-education-') as temporary:
        stage_out = Path(temporary) / 'results'
        stage_out.mkdir()
        forecast_path = stage_out / 'private-forecasts.npy'
        np.save(forecast_path, forecasts, allow_pickle=False)
        seal = {'input_sha256': receipt, 'predictions_sha256': digest(forecast_path),
                'methods': list(METHODS), 'folds': folds, 'historical_max_replay_error': deviation,
                'material_overlap': {key: audit_record['overlap'][key]
                                     for key in ('item_id', 'paragraph', 'question')},
                'readers_by_declared_group': {group or 'unknown': n for group, n in CATEGORIES.items()}}
        seal_path = stage_out / 'forecast-seal.json'
        write_json(seal_path, seal)
        sealed = np.load(forecast_path, allow_pickle=False)
        forecasts_sha = digest(forecast_path)
        if forecasts_sha != read_json(seal_path)['predictions_sha256']:
            raise ValueError('RACE predictions changed after sealing')
        result = score(paths['source'], stage / 'local-outcomes/outcomes.jsonl',
                       stage / 'local-outcomes/outcomes-audit.json', receipt['race_design'], race, sealed)
        write_json(stage_out / 'scores.json', {**result, 'forecast_seal_sha256': digest(seal_path),
                                             'forecasts_sha256': forecasts_sha})
        forecast_path.unlink()
        stage_out.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('.cache/cogllm'))
    parser.add_argument('--output', type=Path, required=True)
    run(**vars(parser.parse_args()))
