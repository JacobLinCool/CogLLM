"""Evaluate first causal alerts at pre-calibrated family-level false-alarm thresholds."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import beta

from research.semantic_tracking import blott_word_metrics as metrics
from research.semantic_tracking import blott_word_score as word_score
from research.semantic_tracking import blott_word_states as word_states
from research.semantic_tracking import frick_consistency_score as records


PROTOCOL = Path(__file__).with_name('blott_word_stream_protocol.md')
PROTOCOL_SHA256 = '24bb7d3df92a9bb8d73389a1bc5dba298bbcdc69fabb4804765e82e7d9430528'
RANK_RESULTS = PROTOCOL.parents[1] / 'results' / 'blott-word-l4'
RANK_SEAL_SHA256 = '8b376f2866658622efef581485e997f5ddbceb12b91aadb183f77743b3f61b02'
ARMS = ('surface', 'magnitude', 'state')
ALPHA, DRAWS, SEED = 0.10, 10_000, 735972


def calibrate(fitted, calibration, targets, updates):
    maxima = []
    for _, keys in calibration:
        early_logits = [fitted.score(updates[key].features)[:targets[key]] for key in keys]
        maxima.append(max(float(path.max()) for path in early_logits if len(path)))
    rank = int(np.ceil((len(maxima) + 1) * (1 - ALPHA)))
    return float(sorted(maxima)[rank - 1]) if rank <= len(maxima) else None


def first_alert(logits, limit):
    crossed = np.flatnonzero(logits > limit) if limit is not None else np.empty(0, dtype=int)
    return int(crossed[0]) if len(crossed) else None


def predict(heldout, fits, thresholds, updates, fold):
    predictions = []
    for family, keys in heldout:
        for key in keys:
            candidate = updates[key]
            alerts = {}
            for arm in ARMS:
                fit, threshold = fits[arm], thresholds[arm]
                full = fit.score(candidate.features)
                decision = first_alert(full, threshold)
                for count in range(1, len(full) + 1):
                    shorter = {name: rows[:count] for name, rows in candidate.features.items()}
                    known_alert = first_alert(fit.score(shorter), threshold)
                    expected = decision if decision is not None and decision < count else None
                    if known_alert != expected:
                        raise ValueError('A first alert changes after an unread suffix is deleted')
                alerts[arm] = {'candidate_row': decision, 'token_index': candidate.token_indices[decision],
                               'span': candidate.spans[decision]} if decision is not None else None
            predictions.append({'id': key, 'family': family, 'fold': fold, 'alerts': alerts})
    return predictions


def forecasts(train, calibration, test, train_targets, updates, fold):
    names = [[name for name, _ in group] for group in (train, calibration, test)]
    if sum(map(len, names)) != len(set().union(*map(set, names))):
        raise ValueError('Cue threshold estimation requires disjoint fit, calibration and test families')
    all_fits = word_score.train_models(train, train_targets, updates)
    thresholds = {arm: calibrate(all_fits[arm], calibration, train_targets, updates) for arm in ARMS}
    rows = predict(test, all_fits, thresholds, updates, fold)
    return rows, {'fold': fold, 'training_families': names[0], 'calibration_families': names[1],
                  'test_families': names[2], 'thresholds': thresholds,
                  'fit_diagnostics': {arm: all_fits[arm].diagnostics for arm in ARMS}}


def evaluate(rows, targets):
    order = sorted({row['family'] for row in rows})
    family_index = {name: i for i, name in enumerate(order)}
    # Columns are before the marked cue, exact, after the cue, and no alert.
    values = np.zeros((len(order), len(ARMS), 4))
    counts = np.zeros(len(order))
    for row in rows:
        family, target = family_index[row['family']], targets[row['id']]
        counts[family] += 1
        for a, arm in enumerate(ARMS):
            alert = row['alerts'][arm]
            decision = alert['candidate_row'] if alert is not None else None
            category = 3 if decision is None else 0 if decision < target else 1 if decision == target else 2
            values[family, a, category] += 1
    if len(order) < 2 or not np.all(counts == 2):
        raise ValueError('Every streaming case requires both study versions')
    family_early = (values[:, :, 0] > 0).sum(axis=0)
    per_version = values.mean(axis=0) / 2
    estimates = {}
    for i, arm in enumerate(ARMS):
        n, count = int(family_early[i]), len(order)
        interval = [float(beta.ppf(0.025, n, count - n + 1)) if n else 0.,
                    float(beta.ppf(0.975, n + 1, count - n)) if n < count else 1.]
        estimates[arm] = {'family_early_count': n, 'family_early_rate_ci95': interval,
                          **dict(zip(('early', 'exact', 'late', 'miss'), per_version[i].tolist(), strict=True))}
    drawn = np.random.default_rng(SEED).integers(0, len(order), (DRAWS, len(order)))
    gains = {}
    for i, arm in enumerate(ARMS[:-1]):
        difference = (values[:, -1, 1] - values[:, i, 1]) / 2
        interval = np.quantile(difference[drawn].mean(axis=1), [.025, .975]).tolist()
        gains[arm] = {'estimate': float(difference.mean()), 'ci95': interval}
    gates = {'family_early_upper_within_10_percent': estimates['state']['family_early_rate_ci95'][1] <= ALPHA,
             **{f'exact_rate_gain_over_{arm}': stats['ci95'][0] > 0 for arm, stats in gains.items()}}
    return {'families': len(order), 'sentences': len(rows), 'methods': estimates,
            'state_exact_rate_gain': gains, 'gates': gates, 'pass': all(gates.values())}


def run(work, extraction, tokenizer, early, late, output):
    if output.exists():
        raise FileExistsError(output)
    base_seal = RANK_RESULTS / 'forecast-seal.json'
    if records.digest(PROTOCOL) != PROTOCOL_SHA256 or records.digest(base_seal) != RANK_SEAL_SHA256:
        raise ValueError('The streaming protocol or underlying localization study changed')
    updates, provenance = word_states.load_updates(work, extraction, tokenizer, word_score.PROTOCOL_SHA256)
    if records.read_json(base_seal)['extraction_sha256'] != provenance['extraction_sha256']:
        raise ValueError('The streaming assay must reuse the exact localization representations')
    (earlier, targets), (later, later_targets) = word_score.sources(early, late, updates)
    all_rows, receipts = [], []
    for fold in range(word_score.FOLDS):
        calibration_fold = (fold + 1) % word_score.FOLDS
        train = [pair for i, pair in enumerate(earlier) if i % word_score.FOLDS not in {fold, calibration_fold}]
        rows, receipt = forecasts(train, earlier[calibration_fold::word_score.FOLDS], earlier[fold::word_score.FOLDS],
                                  targets, updates, fold)
        all_rows.extend(rows)
        receipts.append(receipt)
    transfer, transfer_receipt = forecasts([pair for i, pair in enumerate(earlier) if i % word_score.FOLDS],
                                             earlier[::word_score.FOLDS], later, targets, updates, '2023')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.blott-stream-') as temporary:
        stage = Path(temporary) / 'scored'
        stage.mkdir()
        files = {'2021-forecasts.json': all_rows, '2023-forecasts.json': transfer,
                 'fits.json': {'2021': receipts, '2023': transfer_receipt}}
        for name, data in files.items():
            records.write_json(stage / name, data)
        seal = stage / 'forecast-seal.json'
        records.write_json(seal, {**provenance, 'protocol_sha256': PROTOCOL_SHA256,
                                  'rank_study_seal_sha256': RANK_SEAL_SHA256,
                                  'scorer_sha256': records.digest(Path(__file__)),
                                  'ranker_sha256': records.digest(Path(word_score.ranking.__file__)),
                                  'file_sha256': {name: records.digest(stage / name) for name in files}})
        result = {year: evaluate(records.read_json(stage / f'{year}-forecasts.json'), gold)
                  for year, gold in (('2021', targets), ('2023', later_targets))}
        records.write_json(stage / 'results.json', {**result, 'forecast_seal_sha256': records.digest(seal),
                           '2023_scope': 'descriptive transfer with study-specific cue placement'})
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('work', 'extraction', 'tokenizer', 'early', 'late', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
