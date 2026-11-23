"""Score sealed AmbiStory effect-supervision forecasts on the author's validation split."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.stage6 import ambistory_effect_fit as effect_fit
from research.stage6.ambistory_score import Holdout, digest, interval, jsonl, read_json, write_json


FAMILIES, HOMONYMS, STORIES, SENSES, VOTES, FLAGGED = 98, 54, 294, 588, 2952, 44
DRAWS, SEED = 10_000, 43
FIT_ITEMS = 2268
COMPARED = ('candidate_ending', 'candidate_common_prefix', 'baseline_ending')
FORECASTS = {'baseline': 'baseline-predictions.jsonl', 'effect': 'effect-predictions.jsonl'}
FIT_FILES = {*FORECASTS.values(), 'heads.npz', 'train-diagnostics.json'}
METHODS, GRADES = effect_fit.baseline.METHODS, effect_fit.baseline.GRADES
DEV_DESIGN = f'dev/{effect_fit.baseline.DESIGN}'
DEV_OUTCOMES = f'dev/{effect_fit.baseline.OUTCOMES}'


def score_protocol(fit_protocol):
    fitted_plan = read_json(fit_protocol)
    if fitted_plan != effect_fit.protocol():
        raise ValueError('The effect fitting rules or implementation differ from the preseal')
    return {
        'status': 'scoring_rules_locked_before_author_validation_scoring',
        'design_sha256': fitted_plan['design_sha256'], 'fit_protocol_sha256': digest(fit_protocol),
        'fit_code_sha256': fitted_plan['fit_code_sha256'], 'scorer_sha256': digest(Path(__file__)),
        'preparation_sha256': fitted_plan['preparation_sha256'], 'fit_split': 'train', 'forecast_split': 'dev',
        'counts': {'families': FAMILIES, 'homonym_groups': HOMONYMS, 'stories': STORIES, 'sense_items': SENSES,
                   'votes': VOTES, 'flagged_votes': FLAGGED, 'ending_pairs': 2 * FAMILIES},
        'holdout_scope': 'author-public validation split held out by this project until forecast sealing',
        'file_contract': {'staged_members': sorted(effect_fit.STAGED_MEMBERS), 'successful_fit_files': sorted(FIT_FILES),
                          'forecast_methods': list(METHODS), 'grades': list(GRADES),
                          'no_candidate': 'validation responses remain unopened'},
        'primary_nll': 'all validation votes across all three story variants, in nats',
        'signed_update': 'later minus earlier ending by opaque story ID, matched family and sense',
        'update_pools': 'equal weight per family-sense pair',
        'bootstrap': {'unit': 'casefolded homonym', 'draws': DRAWS, 'seed': SEED, 'paired': True},
        'gates': fitted_plan['dev_gates'],
    }


def verify_selection(diagnostics, beta):
    folds = diagnostics['folds']
    groups = diagnostics['fold_homonyms']
    if (groups != effect_fit.fold_map(groups) or len(folds) != effect_fit.FOLDS
            or any(row['fold'] != index or row['fit_items'] != 3 * row['fit_ending_pairs']
                   or row['fit_items'] + row['held_out_items'] != FIT_ITEMS for index, row in enumerate(folds))
            or sum(row['held_out_items'] for row in folds) != FIT_ITEMS):
        raise ValueError('Training diagnostics differ from the frozen five-fold design')
    grid = set(map(str, effect_fit.BETAS))
    if any(set(row['optimizer']) != grid for row in folds):
        raise ValueError('Missing preregistered beta fits')
    converged = {key for key in grid if all(row['optimizer'][key]['success'] is True for row in folds)}
    metrics = diagnostics['oof']
    if '0' not in converged or set(metrics) != converged:
        raise ValueError('Only complete converged fold fits may enter beta selection')
    mae, nll = 'signed_update_mae_grades', 'nll_nats_per_vote'
    if any(not np.isfinite(metric[field]) for metric in metrics.values() for field in (mae, nll)):
        raise ValueError('Out-of-fold selection requires finite aggregate metrics')
    reference = metrics['0']
    eligible = [value for value in effect_fit.BETAS[1:] if str(value) in metrics
                and metrics[str(value)][mae] < reference[mae]
                and metrics[str(value)][nll] - reference[nll] <= effect_fit.VOTE_NLL_MARGIN]
    if (not eligible or diagnostics['eligible_betas'] != eligible or diagnostics['selected_beta'] != beta
            or min(eligible, key=lambda value: (metrics[str(value)][mae], value)) != beta):
        raise ValueError('Selected beta differs from the frozen training-only eligibility and tie-break rules')


def verify_fit(args, plan):
    if read_json(args.score_protocol) != plan:
        raise ValueError('The scoring code or preregistered rules have changed')
    seal_path = args.fit / 'fit-seal.json'
    seal = read_json(seal_path)
    if seal['status'] == 'no_candidate':
        raise ValueError('Training selected no eligible positive beta. Validation responses remain unopened.')
    receipt = effect_fit.check_preparation(args.preparation)
    if effect_fit.check_staged(args.staged) != receipt:
        raise ValueError('The staged release differs from the prepared author release')
    inputs = {name: receipt['files_sha256'][name] for name in effect_fit.SOURCE_MEMBERS}
    extraction = {key: digest(args.extraction / name) for key, name in effect_fit.baseline.FILES.items()}
    beta = seal['selected_beta']
    if (seal['status'] != 'effect_and_baseline_forecasts_locked_before_dev_scoring'
            or seal['protocol_sha256'] != plan['fit_protocol_sha256']
            or seal['source_preparation_sha256'] != plan['preparation_sha256']
            or seal['input_sha256'] != inputs or seal['extraction_sha256'] != extraction
            or seal['runtime'] != effect_fit.RUNTIME or seal['fit_split'] != 'train' or seal['forecast_split'] != 'dev'
            or seal['fit_samples'] != FIT_ITEMS or seal['forecast_samples'] != SENSES
            or tuple(seal['methods']) != METHODS or tuple(seal['grades']) != GRADES
            or type(beta) is not int or beta not in effect_fit.BETAS[1:]
            or set(seal['file_sha256']) != FIT_FILES):
        raise ValueError('Unexpected effect fit identity, partition, feature, or forecast receipt')
    if any(digest(args.staged / 'prepared' / name) != check for name, check in inputs.items()):
        raise ValueError('Staged inference or training bytes have changed')
    if any(digest(args.fit / name) != check for name, check in seal['file_sha256'].items()):
        raise ValueError('A sealed fit product changed before scoring')
    verify_selection(read_json(args.fit / 'train-diagnostics.json'), beta)
    return receipt, seal, digest(seal_path)


def read_holdout(path):
    families, seen = defaultdict(list), set()
    for row in jsonl(path):
        if row['id'] in seen or type(row['has_ending']) is not bool or len(row['senses']) != 2:
            raise ValueError('Unexpected validation story, sense count, or repeated story ID')
        seen.add(row['id'])
        families[row['family']].append(row)
    homonyms = {row['homonym'].casefold() for variants in families.values() for row in variants}
    if (len(families), len(homonyms), len(seen)) != (FAMILIES, HOMONYMS, STORIES):
        raise ValueError('Unexpected grade-free author validation inventory')
    cluster = {name: i for i, name in enumerate(sorted(homonyms))}
    keys, family_ids, groups = [], [], []
    for family, variants in sorted(families.items()):
        base = [row for row in variants if not row['has_ending']]
        endings = sorted((row for row in variants if row['has_ending']), key=lambda row: row['id'])
        if len(base) != 1 or len(endings) != 2 or len({row['homonym'].casefold() for row in variants}) != 1:
            raise ValueError('Each held-out family needs one base, two endings and one casefolded homonym')
        senses = {sense['id'] for sense in base[0]['senses']}
        if len(senses) != 2 or any({sense['id'] for sense in row['senses']} != senses for row in endings):
            raise ValueError('The original two judged senses must match across all variants')
        for sense in sorted(senses):
            keys.extend((row['id'], sense) for row in (*base, *endings))
            family_ids.append(family)
            groups.append(cluster[base[0]['homonym'].casefold()])
    return Holdout(tuple(keys), tuple(family_ids), np.asarray(groups))


def read_forecasts(path, holdout):
    by_key = {}
    for row in jsonl(path):
        key = row['id'], row['sense']
        if row.keys() != {'id', 'sense', *METHODS} or key in by_key:
            raise ValueError('Unexpected prediction fields or duplicate sense item')
        by_key[key] = [row[method] for method in METHODS]
    if by_key.keys() != set(holdout.keys):
        raise ValueError('Expected one forecast distribution from each method for every validation sense item')
    p = np.asarray([by_key[key] for key in holdout.keys], dtype=np.float64)
    if (p.shape != (len(holdout.keys), len(METHODS), len(GRADES)) or not np.isfinite(p).all()
            or np.any(p <= 0) or np.any(p > 1) or not np.allclose(p.sum(axis=-1), 1, atol=1e-7, rtol=0)):
        raise ValueError('Expected finite, positive probability vectors on the ordered five-grade scale')
    variants = p.reshape(-1, 3, len(METHODS), len(GRADES))
    if (not np.all(variants[:, 1:, :2] == variants[:, :1, :2])
            or not np.array_equal(variants[:, 0, 2], variants[:, 0, 1])):
        raise ValueError('The form and prefix must be invariant, and the base ending must equal its prefix')
    return p


def read_outcomes(path, holdout, expected):
    if digest(path) != expected:
        raise ValueError('Validation response bytes differ from the frozen author release')
    ratings, samples, flags = {}, set(), 0
    for row in jsonl(path):
        key = row['id'], row['sense_id']
        grades, nonsensical = row['ratings'], row['nonsensical']
        if (key in ratings or row['author_sample'] in samples or len(grades) < 5 or len(grades) != len(nonsensical)
                or any(type(grade) is not int or grade not in GRADES for grade in grades)
                or any(type(mark) is not bool for mark in nonsensical)):
            raise ValueError('Unexpected released vote, flag or rating-item identity')
        samples.add(row['author_sample'])
        flags += sum(nonsensical)
        ratings[key] = np.bincount(grades, minlength=len(GRADES) + 1)[1:]
    if ratings.keys() != set(holdout.keys) or flags != FLAGGED:
        raise ValueError('Validation ratings do not cover the frozen whole-family split')
    counts = np.stack([ratings[key] for key in holdout.keys])
    if counts.sum() != VOTES:
        raise ValueError('Unexpected total number of released validation votes')
    return counts


def clustered_mean(values, counts, cluster, weights):
    n = weights.shape[1]
    numerator = np.stack([np.bincount(cluster, weights=column, minlength=n) for column in values.T], axis=1)
    denominator = np.bincount(cluster, weights=counts, minlength=n)
    if np.any(denominator == 0):
        raise ValueError('Every casefolded homonym needs observed material for the paired comparisons')
    return (weights @ numerator) / (weights @ denominator)[:, None]


def summarize(holdout, baseline, effect, counts):
    if not np.array_equal(baseline[:, 0], effect[:, 0]):
        raise ValueError('Both fits must use the same frozen form-control predictions')
    clusters = holdout.pair_clusters
    n = len(np.unique(clusters))
    boot = np.random.default_rng(SEED).multinomial(n, np.full(n, 1 / n), size=DRAWS)
    weights = np.vstack((np.ones(n, dtype=np.int64), boot))
    probabilities = np.stack((effect[:, 2], effect[:, 1], baseline[:, 2]), axis=1)
    vote_counts = counts.sum(axis=1)
    loss = -np.einsum('ig,icg->ic', counts, np.log(probabilities))
    nll = clustered_mean(loss, vote_counts, np.repeat(clusters, 3), weights)
    scores = np.column_stack((counts @ GRADES / vote_counts, probabilities @ GRADES)).reshape(-1, 3, 1 + len(COMPARED))
    update = scores[:, 2] - scores[:, 1]
    pair_weights = np.ones(len(clusters))
    update_mae = clustered_mean(np.abs(update[:, 1:] - update[:, :1]), pair_weights, clusters, weights)
    amplitude = clustered_mean(np.abs(update), pair_weights, clusters, weights)
    prefix = {'vote_nll_nats': interval(nll[:, 0] - nll[:, 1]),
              'signed_update_mae_grades': interval(update_mae[:, 0] - update_mae[:, 1])}
    reference = {'vote_nll_nats': interval(nll[:, 0] - nll[:, 2]),
                 'signed_update_mae_grades': interval(update_mae[:, 0] - update_mae[:, 2])}
    amplitude_gap = interval(amplitude[:, 1] - amplitude[:, 0], 90)
    lower, upper = amplitude_gap['ci90']
    margin = effect_fit.AMPLITUDE_MARGIN
    gates = {
        'candidate_ending_improves_nll_vs_own_common_prefix': prefix['vote_nll_nats']['ci95'][1] < 0,
        'candidate_ending_improves_update_mae_vs_own_common_prefix': prefix['signed_update_mae_grades']['ci95'][1] < 0,
        'candidate_ending_improves_update_mae_vs_beta_zero_ending': reference['signed_update_mae_grades']['ci95'][1] < 0,
        'candidate_nll_increase_vs_beta_zero_ending_below_margin': reference['vote_nll_nats']['ci95'][1] < effect_fit.VOTE_NLL_MARGIN,
        'candidate_update_amplitude_equivalent_to_human': -margin < lower <= upper < margin,
    }
    return {
        'methods': {name: {'vote_nll_nats': interval(nll[:, i]), 'signed_update_mae_grades': interval(update_mae[:, i]),
                           'mean_absolute_update_grades': interval(amplitude[:, i + 1])}
                    for i, name in enumerate(COMPARED)},
        'human_mean_absolute_update_grades': interval(amplitude[:, 0]),
        'contrasts': {'candidate_ending_minus_own_common_prefix': prefix,
                      'candidate_ending_minus_beta_zero_ending': reference,
                      'candidate_minus_human_mean_absolute_update_grades': amplitude_gap},
        'acceptance': {'conditions': gates, 'all_conditions_passed': all(gates.values())},
    }


def score(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = score_protocol(args.fit_protocol)
    receipt, seal, seal_digest = verify_fit(args, plan)
    holdout = read_holdout(args.staged / 'prepared' / DEV_DESIGN)
    baseline, effect = (read_forecasts(args.fit / filename, holdout) for filename in FORECASTS.values())
    if not np.array_equal(baseline[:, 0], effect[:, 0]):
        raise ValueError('The form control changed between the two sealed fits')
    votes = read_outcomes(args.prepared / DEV_OUTCOMES, holdout, receipt['files_sha256'][DEV_OUTCOMES])
    write_json(args.output, {
        'status': 'scored_against_author_validation_after_forecast_seal', 'selected_beta': seal['selected_beta'],
        'score_protocol_sha256': digest(args.score_protocol), 'fit_seal_sha256': seal_digest,
        'forecasts_sha256': {name: seal['file_sha256'][filename] for name, filename in FORECASTS.items()},
        'validation_outcomes_sha256': receipt['files_sha256'][DEV_OUTCOMES],
        'counts': plan['counts'], **summarize(holdout, baseline, effect, votes),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command, paths in (('preseal', ('fit-protocol', 'output')),
                           ('score', ('prepared', 'preparation', 'staged', 'extraction', 'fit-protocol',
                                      'score-protocol', 'fit', 'output'))):
        current = commands.add_parser(command)
        for name in paths:
            current.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'preseal':
        write_json(args.output, score_protocol(args.fit_protocol))
    else:
        score(args)
