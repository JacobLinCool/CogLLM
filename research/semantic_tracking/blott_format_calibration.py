"""Freeze binary Sense readouts using only the archived Blott 2021 coherent trials."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from research.stage6 import blott_score as archived


PROTOCOL = Path(__file__).with_name('blott_format_transfer_protocol.md')
PROTOCOL_SHA256 = '3f3b011900c3f63b81d42a3ed88ad88366fb37c271175aab84a427bff3386079'
RIDGE = 0.05
FEATURES = ('log_characters', 'log_words', 'fixed_understanding')
METHODS = {'surface': 2, 'qwen': 3}


def inputs(text, understanding):
    if not text.split() or not 1 <= understanding <= 5:
        raise ValueError('Expected visible text and a valid fixed understanding expectation')
    return np.array([np.log(len(text)), np.log(len(text.split())), understanding])


def log_loss(coefficients, design, positive, observed):
    logits = design @ coefficients
    n = observed.sum()
    regularized = coefficients.copy()
    regularized[0] = 0
    value = ((observed @ np.logaddexp(0, logits) - positive @ logits) / n
             + RIDGE / 2 * (regularized @ regularized))
    gradient = design.T @ (observed * expit(logits) - positive) / n + RIDGE * regularized
    return value, gradient


def fit_statistics(matrix, positive, observed):
    center, scale = matrix.mean(axis=0), matrix.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError('Every Blott readout feature must vary in the registered training sample')
    design = np.column_stack((np.ones(len(matrix)), (matrix - center) / scale))
    fitted = {}
    for method, width in METHODS.items():
        x = design[:, :width + 1]
        bounds = [(None, None)] * x.shape[1]
        if method == 'qwen':
            bounds[-1] = (0, None)
        result = minimize(log_loss, np.zeros(x.shape[1]), args=(x, positive, observed),
                          jac=True, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-10})
        value, gradient = log_loss(result.x, x, positive, observed)
        if method == 'qwen' and result.x[-1] <= 1e-10:
            gradient[-1] = min(gradient[-1], 0.)
        if not result.success or not np.isfinite(value) or np.max(np.abs(gradient)) >= 1e-7:
            raise ArithmeticError(f'Blott calibration did not reach the required constrained optimum for {method}')
        logits = x @ result.x
        fitted[method] = {'coefficients': result.x.tolist(), 'objective': float(value),
                          'training_nll': float((observed @ np.logaddexp(0, logits) - positive @ logits) / observed.sum()),
                          'stationarity_max_error': float(np.max(np.abs(gradient)))}
    return {'center': center.tolist(), 'scale': scale.tolist(), 'methods': fitted}


def fit(stimuli, design, human_trials, archive, output):
    if output.exists():
        raise FileExistsError(output)
    if archived.sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError('The registered Blott transfer protocol changed')
    sources = {'stimuli': stimuli, 'design': design, 'human_trials': human_trials}
    if {key: archived.sha256(path) for key, path in sources.items()} != archived.INPUTS:
        raise ValueError('Blott calibration only accepts the exact archived 2021 source')
    docs, cases, _ = archived.read_inputs(stimuli, design)
    forecasts = archived.signed_forecasts(stimuli, archive / 'predictions.jsonl', archive / 'seal.json', docs)
    observed, positive = archived.human_matrices(human_trials, cases)
    coherent = np.arange(cases.size).reshape(cases.shape)[:, :2].ravel()
    keys = cases.ravel()[coherent]
    matrix = np.vstack([inputs(docs[key], forecasts[key][0]) for key in keys])
    positive = positive[:, coherent].sum(axis=0)
    observed = observed[:, coherent].sum(axis=0)
    if (len(keys), observed.sum(), positive.sum()) != (94, 2256, 1756):
        raise ValueError('Coherent training denominator disagrees with the existing human audit')
    record = fit_statistics(matrix, positive, observed)
    archived.write_json(output, {
        **record, 'candidate_id': archived.CANDIDATE_ID, 'protocol_sha256': PROTOCOL_SHA256,
        'scorer_sha256': archived.sha256(Path(__file__)), 'archive_seal_sha256': archived.sha256(archive / 'seal.json'),
        'input_sha256': archived.INPUTS, 'features': FEATURES, 'ridge': RIDGE,
        'training_documents': len(keys), 'training_families': len(cases), 'training_responses': int(observed.sum()),
        'training_positive_responses': int(positive.sum()),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stimuli', 'design', 'human-trials', 'archive', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    fit(**vars(parser.parse_args()))
