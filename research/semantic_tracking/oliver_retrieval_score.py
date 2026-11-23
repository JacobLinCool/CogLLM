"""Evaluate population retrieval paths from a single, definition-only Qwen forward."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from research.semantic_tracking import beukers_extract as extractor
from research.semantic_tracking import frick_consistency_score as records


PROTOCOL = Path(__file__).with_name('oliver_retrieval_protocol.md')
SOURCE = PROTOCOL.parents[1] / 'results/oliver-retrieval-source-audit.json'
STATES = ('know', 'dont_know', 'tot')
ARMS = ('constant', 'surface', 'qwen')
PATHS = tuple((state, delay, confirm) for state in STATES
              for delay in ((None,) if state == 'know' else (False, True))
              for confirm in (False, True))
WORD = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")
RIDGE = 0.01
DRAWS = 10_000
SEED = 608315


def load_vectors(work, extraction):
    packet = work / 'packet'
    audit = records.read_json(SOURCE)
    if (any(records.digest(packet / name) != value for name, value in audit['packet_sha256'].items())
            or any(records.digest(work / 'prepared' / (stem + '.jsonl')) != value
                   for stem, value in audit['prepared_sha256'].items())):
        raise ValueError('The audited Oliver packet or response crosswalk has changed')
    corpus = records.indexed(records.read_rows(packet / 'stimuli.jsonl'), 'Oliver certified definitions')
    metadata = records.indexed(records.read_rows(work / 'prepared/layout.jsonl'), 'Oliver definition layout')
    certification = records.read_json(packet / 'preflight.json')
    archive = records.read_json(extraction / 'seal.json')
    required = {'candidate_id': extractor.CANDIDATE, 'tokenizer_sha256': extractor.TOKENIZER_SHA256,
                'stimuli_sha256': records.digest(packet / 'stimuli.jsonl'), 'gpu': 'NVIDIA L4',
                'protocol_sha256': records.digest(PROTOCOL), 'states_dtype': 'float32',
                'assay_sha256': records.digest(packet / 'preflight.json'),
                'preflight_sha256': records.digest(packet / 'extractor-preflight.json'), 'raw_stops': True,
                'content_forward_per_document': 1, 'independent_causal_checks': 12}
    names = ('states.npy', 'coordinates.jsonl', 'causality.json')
    if (any(archive.get(key) != value for key, value in required.items())
            or any(archive['file_sha256'][name] != records.digest(extraction / name) for name in names)):
        raise ValueError('The Oliver L4 archive was not sealed against this protocol and exact corpus')
    coordinates = records.read_rows(extraction / 'coordinates.jsonl')
    if coordinates != [{'id': key, 'stops': [len(row['text'])]} for key, row in corpus.items()]:
        raise ValueError('An extracted state does not match its entire visible definition')
    if (set(corpus) != set(metadata) or set(corpus) != set(certification['documents'])
            or any(row['definition_stop'] != len(corpus[key]['text']) for key, row in metadata.items())):
        raise ValueError('The certified, extracted, and experimental definitions must be identical')
    checks = records.read_json(extraction / 'causality.json')['checks']
    if len(checks) != 12 or not all(check['passed'] and check['projected_tokens']['passed'] for check in checks):
        raise ValueError('Independent prefix consistency checks did not pass')
    hidden = np.load(extraction / 'states.npy', allow_pickle=False)
    if (hidden.shape != (len(corpus), 2048) or list(hidden.shape) != archive['states_shape']
            or hidden.dtype != np.float32 or not np.isfinite(hidden).all()):
        raise ValueError('A definition was not paired with a finite FP32 Qwen state')
    return corpus, metadata, hidden.astype(np.float64)


def target_split(work):
    targets = records.read_rows(work / 'prepared/targets.jsonl')
    names = {row['family']: row['target'].strip().casefold() for row in targets}
    if len(targets) != 44 or len(names) != len(targets) or len(set(names.values())) != len(targets):
        raise ValueError('The locked split requires all 44 unique author targets before exclusion')
    signatures = {key: sha256(('cogllm-oliver-e1-v1:' + value).encode()).hexdigest() for key, value in names.items()}
    keys = sorted(signatures, key=signatures.__getitem__)
    return {'train': keys[:30], 'test': keys[30:], 'signature': signatures}


def trials(work, metadata):
    rows = records.read_rows(work / 'prepared/outcomes.jsonl')
    reader_items = {(row['reader_id'], row['family']) for row in rows}
    fit = [row for row in rows if row['first']['id'] is not None]
    if len(rows) != 2288 or len(reader_items) != len(rows) or len(fit) != 2236:
        raise ValueError('The fit, holdout and quarantined reader × target panels do not reconcile')
    for row in fit:
        first = row['first']
        if (metadata[first['id']]['family'] != row['family']
                or (first['state'], first['recall_signaled_before_reveal'], first['same_target_confirmed']) not in PATHS
                or type(first['same_target_confirmed']) is not bool
                or (first['state'] != 'know' and type(first['recall_signaled_before_reveal']) is not bool)):
            raise ValueError('The first test has an invalid protocol branch or definition family')
    return fit


def designs(corpus, hidden, train_ids):
    text = [row['text'] for row in corpus.values()]
    words = [WORD.findall(value.casefold()) for value in text]
    frequency = Counter(word for key, terms in zip(corpus, words, strict=True) if key in train_ids for word in terms)
    surface = np.array([[np.log1p(len(line)), np.log1p(len(terms)), np.mean([len(word) for word in terms]),
                         sum(ch.isdigit() for ch in line) / len(line),
                         sum(not (ch.isalnum() or ch.isspace()) for ch in line) / len(line),
                         np.mean([frequency[word] <= 1 for word in terms])]
                        for line, terms in zip(text, words, strict=True)])
    learn = [key in train_ids for key in corpus]
    center_surface, center_hidden = surface[learn].mean(axis=0), hidden[learn].mean(axis=0)
    scale_surface = surface[learn].std(axis=0)
    scale_hidden = float(np.sqrt(np.square(hidden[learn] - center_hidden).sum(axis=1).mean()))
    if not np.isfinite(surface).all() or not 0 < scale_hidden:
        raise ValueError('Definition features cannot be standardized on the training families')
    surface = (surface - center_surface) / np.where(scale_surface > 1e-12, scale_surface, 1.)
    state = (hidden - center_hidden) / scale_hidden
    return {'constant': np.empty((len(corpus), 0)), 'surface': surface, 'qwen': np.column_stack((surface, state))}, {
        'definition_ids': list(corpus), 'surface_center': center_surface.tolist(), 'surface_scale': scale_surface.tolist(),
        'state_center': center_hidden.tolist(), 'state_scale': scale_hidden,
    }


def fit_factor(x, branch, definition, response, branch_count, class_count):
    indicators = np.eye(branch_count)
    design = np.column_stack((indicators[branch], x[definition]))
    target = np.eye(class_count)[response]
    cells = indicators[branch].T @ target
    if np.any(cells == 0):
        raise ValueError(f'A predeclared retrieval route lacks variation in training: {cells.tolist()}')

    def objective(flat):
        weights = flat.reshape(design.shape[1], class_count - 1)
        logits = np.column_stack((np.zeros(len(design)), design @ weights))
        log_prob = logits - logsumexp(logits, axis=1, keepdims=True)
        gradient = design.T @ (np.exp(log_prob) - target)[:, 1:] / len(design)
        gradient[branch_count:] += RIDGE * weights[branch_count:]
        loss = -np.sum(log_prob * target) / len(design) + RIDGE / 2 * np.square(weights[branch_count:]).sum()
        return loss, gradient.ravel()

    result = minimize(objective, np.zeros(design.shape[1] * (class_count - 1)), jac=True, method='L-BFGS-B',
                      options={'maxiter': 500, 'maxcor': 20, 'ftol': 1e-15, 'gtol': 1e-10})
    loss, gradient = objective(result.x)
    if not np.isfinite(loss) or np.max(np.abs(gradient)) > 3e-7:
        raise ArithmeticError(f'Retrieval factor failed to converge: {result.message}')
    weights = result.x.reshape(design.shape[1], class_count - 1)
    logits = x @ weights[branch_count:]
    logits = logits[:, None, :] + weights[None, :branch_count, :]
    probability = softmax(np.concatenate((np.zeros((*logits.shape[:2], 1)), logits), axis=2), axis=2)
    return probability, {'weights': weights.tolist(), 'training_counts': cells.tolist(),
                         'maximum_gradient': float(np.max(np.abs(gradient))), 'iterations': int(result.nit)}


def paths(x, document, observed):
    states = np.array([STATES.index(row[0]) for row in observed])
    delays = np.array([row[1] for row in observed], dtype=object)
    confirms = np.array([row[2] for row in observed], dtype=np.int64)
    reacted = states > 0
    routes = tuple(dict.fromkeys((state, delay) for state, delay, _ in PATHS))
    branch = np.array([routes.index(row[:2]) for row in observed])
    first, first_fit = fit_factor(x, np.zeros(len(states), dtype=np.int64), document, states, 1, 3)
    delay, delay_fit = fit_factor(x, states[reacted] - 1, document[reacted], delays[reacted].astype(np.int64), 2, 2)
    answer, answer_fit = fit_factor(x, branch, document, confirms, len(routes), 2)
    joint = []
    for state, signaled, confirmed in PATHS:
        rank = STATES.index(state)
        probability = first[:, 0, rank] * answer[:, routes.index((state, signaled)), int(confirmed)]
        if signaled is not None:
            probability *= delay[:, rank - 1, int(signaled)]
        joint.append(probability)
    forecast = np.column_stack(joint)
    if not np.allclose(forecast.sum(axis=1), 1, atol=1e-12, rtol=0) or np.any(forecast <= 0):
        raise ArithmeticError('The exhaustive protocol paths do not form distributions')
    return forecast, {'initial': first_fit, 'delay': delay_fit, 'confirmation': answer_fit}


def observed_path(row):
    first = row['first']
    return first['state'], first['recall_signaled_before_reveal'], first['same_target_confirmed']


def resampled_means(rows, observations):
    axes = [np.unique([row[axis] for row in rows], return_inverse=True) for axis in ('reader_id', 'family')]
    rngs = [np.random.default_rng(seed) for seed in np.random.SeedSequence(SEED).spawn(len(axes))]
    means = np.empty((DRAWS + 1, observations.shape[1]))
    means[0] = observations.mean(axis=0)
    for start in range(1, DRAWS + 1, 128):
        end = min(start + 128, DRAWS + 1)
        left, right = (rng.multinomial(len(keys), np.full(len(keys), 1 / len(keys)), size=end - start)[:, inv]
                       for rng, (keys, inv) in zip(rngs, axes, strict=True))
        weights = left * right
        means[start:end] = (weights @ observations) / weights.sum(axis=1, keepdims=True)
    return means


def summary(draws):
    if not np.isfinite(draws).all():
        raise ArithmeticError('A resampled retrieval endpoint has an empty denominator')
    return {'estimate': float(draws[0]), 'ci90': np.quantile(draws[1:], [.05, .95]).tolist(),
            'ci95': np.quantile(draws[1:], [.025, .975]).tolist()}


def endpoints(mass):
    groups = {state: [i for i, path in enumerate(PATHS) if path[0] == state] for state in STATES}
    initial = {state: mass[:, indices].sum(axis=1) for state, indices in groups.items()}
    values = {'initial_' + state: rate for state, rate in initial.items()}
    for name, axis in (('delay', 1), ('confirmation', 2)):
        event = {state: mass[:, [i for i in group if PATHS[i][axis] is True]].sum(axis=1)
                 for state, group in groups.items()}
        risk = initial['dont_know'] + initial['tot'] if axis == 1 else 1.
        values[name + '_marginal'] = sum(event.values()) / risk
        values[name + '_tot_minus_dont_know'] = event['tot'] / initial['tot'] - event['dont_know'] / initial['dont_know']
    return values


def score(work, extraction, output):
    corpus, metadata, hidden = load_vectors(work, extraction)
    split = target_split(work)
    indexed_ids = {key: i for i, key in enumerate(corpus)}
    train_ids = {key for key, row in metadata.items() if row['family'] in split['train']}
    feature, standardization = designs(corpus, hidden, train_ids)
    rows = trials(work, metadata)
    train = [row for row in rows if row['first']['id'] in train_ids]
    test = [row for row in rows if row['first']['id'] not in train_ids]
    train_doc, test_doc = (np.array([indexed_ids[row['first']['id']] for row in part]) for part in (train, test))
    train_path = [observed_path(row) for row in train]
    heads, measures, predictions, test_mass = {}, {}, {}, {}
    for arm in ARMS:
        forecast, heads[arm] = paths(feature[arm], train_doc, train_path)
        predictions[arm] = {key: forecast[indexed_ids[key]].tolist() for key in corpus if key not in train_ids}
        test_mass[arm] = forecast[test_doc]
    given = np.array([PATHS.index(observed_path(row)) for row in test])
    test_loss = [-np.log(test_mass[arm][np.arange(len(test)), given]) for arm in ARMS]
    samples = resampled_means(test, np.column_stack((np.eye(len(PATHS))[given], *test_mass.values(), *test_loss)))
    human = endpoints(samples[:, :len(PATHS)])
    nll = dict(zip(ARMS, samples[:, -len(ARMS):].T, strict=True))
    for i, arm in enumerate(ARMS, start=1):
        observed = endpoints(samples[:, i * len(PATHS):(i + 1) * len(PATHS)])
        measures[arm] = {'path_nll': summary(nll[arm]),
                         'endpoint': {name: summary(value) for name, value in observed.items()},
                         'forecast_minus_human': {name: summary(value - human[name]) for name, value in observed.items()}}
    comparisons = {arm: summary(nll[arm] - nll['qwen']) for arm in ARMS[:-1]}
    effects, passes = {}, {'nll_gain': all(item['ci95'][0] > 0 for item in comparisons.values())}
    for name in ('delay_tot_minus_dont_know', 'confirmation_tot_minus_dont_know'):
        actual, estimated = summary(human[name]), measures['qwen']['endpoint'][name]
        gap = measures['qwen']['forecast_minus_human'][name]['ci90']
        direction_observed = actual['ci95'][0] > 0 or actual['ci95'][1] < 0
        direction_matches = bool(actual['estimate'] * estimated['estimate'] > 0) if direction_observed else None
        effects[name] = {'human_direction_resolved': direction_observed, 'direction_matches': direction_matches,
                         'amplitude_passed': gap[0] >= -.05 and gap[1] <= .05}
        passes[name] = effects[name]['amplitude_passed'] and direction_matches is True
    report = {'split': split, 'training_trials': len(train), 'test_trials': len(test),
              'training_families': len({row['family'] for row in train}), 'test_families': len({row['family'] for row in test}),
              'test_readers': len({row['reader_id'] for row in test}),
              'paths': PATHS, 'human_test_counts': np.bincount(given, minlength=len(PATHS)).tolist(),
              'human': {name: summary(value) for name, value in human.items()}, 'arms': measures,
              'qwen_nll_gain_over_baseline': comparisons, 'effects': effects, 'pass': passes,
              'bootstrap': {'draws': DRAWS, 'seed': SEED}}
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with TemporaryDirectory(dir=output.parent, prefix='.oliver-score-') as temporary:
        stage = Path(temporary) / 'results'
        stage.mkdir()
        for name, contents in (('metrics.json', report), ('forecast.json', {'paths': PATHS, 'arms': predictions}),
                               ('coefficients.json', {'standardization': standardization, 'heads': heads})):
            (stage / name).write_text(json.dumps(contents, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
        (stage / 'receipt.json').write_text(json.dumps({
            'implementation_sha256': records.digest(Path(__file__)), 'protocol_sha256': records.digest(PROTOCOL),
            'source_audit_sha256': records.digest(SOURCE),
            'extraction_seal_sha256': records.digest(extraction / 'seal.json'),
            'payload_sha256': {name: records.digest(work / 'prepared' / name)
                               for name in ('stimuli.jsonl', 'layout.jsonl', 'targets.jsonl', 'outcomes.jsonl')},
            'file_sha256': {item.name: records.digest(item) for item in sorted(stage.iterdir())},
        }, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('work', 'extraction', 'output'):
        parser.add_argument('--' + argument, required=True, type=Path)
    score(**vars(parser.parse_args()))
