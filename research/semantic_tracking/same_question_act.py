"""Reevaluate prompted first-correct responses from archived ACT cue states."""

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from research.semantic_tracking.act_transfer_score import (
    DOCUMENTS, FOLDS, FULL_HORIZON, MATERIAL_SHA256, OUTCOMES_SHA256, SPLIT_SHA256,
    checked, log_masses, observations, read_json, read_jsonl, sha256, write_json,
)


ARCHIVE_SEAL_SHA256 = 'dca1735627761f7da38078a254c4d6259984554a1f387e580a51012f59faf6d7'
MODELS = {'step': 0, 'surface': 4, 'qwen': 7}
SURFACE = ('current_cue_characters', 'cumulative_displayed_characters',
           'added_native_tokens', 'cumulative_native_tokens')
SEMANTIC = ('cue_state_delta_l2', 'delta_dot_first_cue_state', 'delta_dot_prior_mean_state')
L2 = 0.10
MAX_ITERATIONS = 400
BOOTSTRAP_SEED = 735972
BOOTSTRAP_DRAWS = 10_000


def _prefix_counts(tokenizer, text, stops):
    complete = tokenizer.encode(text, add_special_tokens=False)
    counts = []
    for stop in stops:
        prefix = tokenizer.encode(text[:stop], add_special_tokens=False)
        if (not prefix.ids or complete.ids[:len(prefix.ids)] != prefix.ids
                or complete.offsets[len(prefix.ids) - 1][1] != stop):
            raise ValueError('ACT cue stop is not a certified native tokenizer prefix')
        counts.append(len(prefix.ids))
    counts = np.asarray(counts)
    if np.any(np.diff(counts, prepend=0) <= 0):
        raise ValueError('An ACT cue does not advance the native token axis')
    return counts


def _features(text, stops, vectors, tokenizer):
    cues = text.split('\n')
    lengths = np.array([len(cue) for cue in cues])
    if (len(cues) != FULL_HORIZON or any(len(cue.split()) != 1 for cue in cues)
            or list(np.cumsum(lengths) + np.arange(FULL_HORIZON)) != stops):
        raise ValueError('Expected 15 exact, one-word display cues in the released order')
    tokens = _prefix_counts(tokenizer, text, stops)
    increments = np.diff(tokens, prepend=0)
    surface = np.log1p(np.column_stack((lengths, np.cumsum(lengths), increments, tokens)))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError('Archived ACT cue states are nonfinite or have zero norm')
    unit = vectors / norms
    delta = np.diff(unit, axis=0)
    prior_mean = np.cumsum(unit, axis=0)[:-1] / np.arange(1, len(unit))[:, None]
    semantic = np.vstack((np.zeros(3), np.column_stack((
        np.linalg.norm(delta, axis=1), delta @ unit[0], np.sum(delta * prior_mean, axis=1),
    ))))
    return np.column_stack((surface, semantic)), increments, tokens - increments


def _load(material, assessment, archive, tokenizer_path):
    from tokenizers import Tokenizer

    checked(material / 'preflight.json', MATERIAL_SHA256)
    checked(material / 'split.json', SPLIT_SHA256)
    material_receipt = read_json(material / 'preflight.json')
    for name, digest in material_receipt['file_sha256'].items():
        checked(material / name, digest)
    seal_path = archive / 'extraction-seal.json'
    checked(seal_path, ARCHIVE_SEAL_SHA256)
    seal = read_json(seal_path)
    for name, digest in seal['file_sha256'].items():
        checked(archive / name, digest)
    checked(tokenizer_path, seal['tokenizer_sha256'])
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if (seal['assay_sha256'] != MATERIAL_SHA256 or seal['content_forward_per_document'] != 1
            or seal['stimuli_sha256'] != material_receipt['file_sha256']['model_input/stimuli.jsonl']):
        raise ValueError('ACT feature archive uses a different source or forward contract')
    stimuli = read_jsonl(material / 'model_input/stimuli.jsonl')
    coords = read_jsonl(archive / 'coordinates.jsonl')
    families = read_json(material / 'split.json')['material']
    if (len(stimuli) != DOCUMENTS or len(families) != DOCUMENTS or len(coords) != DOCUMENTS
            or Counter(row['fold'] for row in families) != {i: DOCUMENTS // FOLDS for i in range(FOLDS)}):
        raise ValueError('Unexpected ACT material, coordinate or holdout inventory')
    vectors = np.load(archive / 'vectors.npy', allow_pickle=False)
    if list(vectors.shape) != seal['vectors_shape'] or vectors.dtype != np.float16:
        raise ValueError('Unexpected archived ACT representation shape or precision')
    vectors = vectors.astype(np.float64).reshape(DOCUMENTS, FULL_HORIZON, seal['projection_width'])
    geometry = []
    for row, coord, family, state in zip(stimuli, coords, families, vectors, strict=True):
        if coord != {'id': row['id'], 'stops': row['stops']} or family['id'] != row['id']:
            raise ValueError('ACT response family or extraction rows are out of stimulus order')
        geometry.append(_features(row['text'], row['stops'], state, tokenizer))
    features, added, prior = map(np.stack, zip(*geometry, strict=True))
    _, full = observations(assessment, families)
    return families, features, added, prior, full.sum(axis=0)


def _standardize(features, train):
    """Remove step means and population-scale residuals using training lists only."""
    centered = features - features[train].mean(axis=0)
    scale = np.sqrt(np.square(centered[train]).mean(axis=(0, 1)))
    return centered / np.where(scale > 0, scale, 1.0)


def _fit(features, count):
    n = count.sum(axis=1)
    events = count[:, :-1]
    risk = np.cumsum(count[:, ::-1], axis=1)[:, ::-1][:, :-1]
    not_yet = risk - events
    weight = 1 / (len(n) * n[:, None])
    width = features.shape[-1]
    initial = np.r_[np.log((events.sum(axis=0) + .5) / (not_yet.sum(axis=0) + .5)), np.zeros(width)]

    def objective(parameters):
        intercept = parameters[:FULL_HORIZON]
        slopes = parameters[FULL_HORIZON:]
        logits = intercept + features @ slopes
        negative_nll = np.logaddexp(0, logits)
        positive_nll = np.logaddexp(0, -logits)
        nll = np.sum(weight * (not_yet * negative_nll + events * positive_nll))
        pseudo = np.logaddexp(0, intercept).sum() - .5 * intercept.sum()
        loss = nll + pseudo / n.sum() + L2 / 2 * (slopes @ slopes)
        derivative = weight * (risk * expit(logits) - events)
        intercept_grad = derivative.sum(axis=0) + (expit(intercept) - .5) / n.sum()
        slope_grad = np.einsum('li,lik->k', derivative, features) + L2 * slopes
        return float(loss), np.r_[intercept_grad, slope_grad]

    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      options={'maxiter': MAX_ITERATIONS, 'ftol': 1e-13, 'gtol': 1e-8})
    if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.jac).all():
        raise RuntimeError(f'ACT fold optimizer did not converge: {result.message}')
    return result.x, {'iterations': int(result.nit), 'max_abs_gradient': float(np.abs(result.jac).max()),
                      'objective': float(result.fun)}


def _forecast(features, count, folds):
    logits = {model: np.empty(count[:, :-1].shape) for model in MODELS}
    fit_audit = {}
    for fold in range(FOLDS):
        train = folds != fold
        standardized = _standardize(features, train)
        diagnostics = {}
        for model, width in MODELS.items():
            predictors = standardized[..., :width]
            parameters, diagnostics[model] = _fit(predictors[train], count[train])
            logits[model][~train] = parameters[:FULL_HORIZON] + predictors[~train] @ parameters[FULL_HORIZON:]
        fit_audit[str(fold)] = diagnostics
    return logits, fit_audit


def _contrast(values, draws):
    return {'mean': float(values.mean()), 'paired_family_bootstrap_ci95':
            np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist()}


def _matches(logits, count, folds, increments, prior, strict):
    n = np.cumsum(count[:, ::-1], axis=1)[:, ::-1][:, :-1]
    k = count[:, :-1]
    groups = defaultdict(list)
    for family, fold in enumerate(folds):
        for step, inc in enumerate(increments[family]):
            key = (fold, step, inc, prior[family, step] if strict else None)
            if n[family, step] > 0:
                groups[key].append(family)
    comparable, decisions, audit = 0, defaultdict(list), set()
    for (_, step, _, _), families in groups.items():
        for left, right in combinations(families, 2):
            comparable += 1
            true_sign = np.sign(k[left, step] * n[right, step] - k[right, step] * n[left, step])
            if true_sign:
                audit.add((left, right))
                for model, values in logits.items():
                    model_sign = np.sign(values[left, step] - values[right, step])
                    decisions[model].append((1 + model_sign * true_sign) / 2)
    return {
        'same_prior_native_token_count': strict, 'matched_opportunity_pairs': comparable,
        'human_non_ties': len(decisions['step']), 'distinct_non_tied_list_pairs': len(audit),
        'conditional_hazard_concordance': {model: float(np.mean(values)) for model, values in decisions.items()},
    }


def analyze(material, assessment, archive, tokenizer_path, output):
    if output.exists():
        raise FileExistsError(output)
    families, features, increments, prior, counts = _load(material, assessment, archive, tokenizer_path)
    folds = np.array([row['fold'] for row in families])
    logits, fit_audit = _forecast(features, counts, folds)
    n = counts.sum(axis=1)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(DOCUMENTS, size=(BOOTSTRAP_DRAWS, DOCUMENTS))
    losses, cumulative = {}, {}
    for model, values in logits.items():
        log_probability = log_masses(values)
        if not np.allclose(logsumexp(log_probability, axis=1), 0, atol=1e-12, rtol=0):
            raise ValueError('First-correct distributions do not normalize')
        losses[model] = -(counts * log_probability).sum(axis=1) / n
        cumulative[model] = np.cumsum(np.exp(log_probability[:, :-1]), axis=1)
    human_cumulative = np.cumsum(counts[:, :-1], axis=1) / n[:, None]
    pairwise = [('surface', 'step'), ('qwen', 'step'), ('qwen', 'surface')]
    report = {
        'endpoint': 'Prompted first correct with the original ordered ACT clues',
        'scope': 'Historical material holdout within one reader pool. Conditional hazard ordering '
                 'compares different lists at the same step and native token increment. '
                 'The design has no aligned randomized same-prefix clue replacements.',
        'human': {'retained_trials': int(n.sum()), 'first_correct': int(counts[:, :-1].sum()),
                  'right_censored': int(counts[:, -1].sum()), 'excluded_unknown': DOCUMENTS * 100 - int(n.sum()),
                  'cumulative_first_correct_equal_list': human_cumulative.mean(axis=0).tolist()},
        'design': {'horizon': FULL_HORIZON, 'outer_folds': FOLDS, 'heldout_lists_per_fold': DOCUMENTS // FOLDS,
                   'sentence_source': 'One original clue word per display; newline is model serialization',
                   'surface_log1p': list(SURFACE), 'semantic': list(SEMANTIC),
                   'scale': 'Train-only step centering and train-only residual RMS by dimension',
                   'slope_l2': L2, 'bootstrap_unit': 'Held-out clue list, conditional on the observed reader pool',
                   'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_draws': BOOTSTRAP_DRAWS},
        'nll_nats_per_trial_equal_list': {model: float(loss.mean()) for model, loss in losses.items()},
        'nll_improvement_nats_per_trial': {f'{model}_over_{baseline}': _contrast(losses[baseline] - losses[model], draws)
                                             for model, baseline in pairwise},
        'calibration': {
            str(step): {'human': float(human_cumulative[:, step - 1].mean()),
                        'by_model': {model: {'estimate': float(cdf[:, step - 1].mean()),
                                             'model_minus_human': _contrast(
                                                 cdf[:, step - 1] - human_cumulative[:, step - 1], draws)}
                                     for model, cdf in cumulative.items()}}
            for step in (1, 3, 6, 10, 15)
        },
        'matched_hazard': {
            'same_step_and_increment': _matches(logits, counts, folds, increments, prior, False),
            'also_same_prior_tokens': _matches(logits, counts, folds, increments, prior, True),
        },
        'fit_diagnostics': fit_audit,
        'folds': [{'family': item['family'], 'fold': item['fold'],
                   'first_correct_counts': counts[i, :-1].astype(int).tolist(), 'right_censored': int(counts[i, -1]),
                   'conditional_hazard': {model: expit(values[i]).tolist() for model, values in logits.items()}}
                  for i, item in enumerate(families)],
        'binding': {'scoring_code_sha256': sha256(Path(__file__)), 'material_preflight_sha256': MATERIAL_SHA256,
                    'outcomes_sha256': OUTCOMES_SHA256, 'split_sha256': SPLIT_SHA256,
                    'archive_extraction_seal_sha256': ARCHIVE_SEAL_SHA256},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('material', 'assessment', 'archive', 'tokenizer-path', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    analyze(**vars(parser.parse_args()))
