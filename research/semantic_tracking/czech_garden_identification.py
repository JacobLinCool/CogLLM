"""Bound joint Czech comprehension endorsements using the two observed probe marginals."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest, sha256
import json
from pathlib import Path

import numpy as np


SOURCE_RECEIPT = Path(__file__).resolve().parents[1] / 'results' / 'czech-garden-prepared-audit.json'
SOURCE_SHA256 = 'f74718a86ebe511511b6661013dd5d3855f22e2fb86a7e980313966531822929'
CONDITIONS = ('gp', 'nongp')
ANIMACIES = ('animate', 'inanimate')
PROBES = ('qcor', 'qmis')
RESPONSE_TYPES = ('yes', 'no', 'unsure', 'missing')
EXPERIMENTS = (2, 3)
DRAWS = 10_000
SEED = 735972
BATCH = 128


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def indexed(values):
    by_id = {row['id']: row for row in values}
    if len(by_id) != len(values):
        raise ValueError('Duplicate Czech stimulus ID')
    return by_id


def design(prepared):
    if any(digest(path) != SOURCE_SHA256 for path in (SOURCE_RECEIPT, prepared / 'source-audit.json')):
        raise ValueError('Expected the exact sealed Czech prepared-source receipt')
    source = json.loads(SOURCE_RECEIPT.read_text(encoding='utf-8'))
    for name in ('layout.jsonl', 'stimuli.jsonl', 'outcomes.jsonl'):
        if digest(prepared / name) != source['file_sha256'][f'prepared/{name}']:
            raise ValueError(f'Prepared Czech input differs from the source seal: {name}')
    stimuli, layout = (indexed(rows(prepared / name)) for name in ('stimuli.jsonl', 'layout.jsonl'))
    if len(layout) != 192 or stimuli.keys() != layout.keys():
        raise ValueError('Expected the exact 192 registered Czech stimuli')
    designs, verbs = defaultdict(set), defaultdict(set)
    for identifier, row in layout.items():
        family = row['family']
        match = row['animacy'], row['condition']
        if match in designs[family] or len(row['words']) != 11:
            raise ValueError('Expected one row per Czech analysis stratum and eleven author words')
        designs[family].add(match)
        text = stimuli[identifier]['text']
        left, right = row['words'][2]
        if not (0 <= left < right <= len(text)) or not text[left:right].isalpha():
            raise ValueError('Expected the original word-3 verb in every Czech material version')
        verbs[family].add(text[left:right])
    expected_cells = {(animacy, condition) for animacy in ANIMACIES for condition in CONDITIONS}
    if len(designs) != 48 or any(cells != expected_cells for cells in designs.values()):
        raise ValueError('Expected 48 complete item families')
    if any(len(values) != 1 for values in verbs.values()):
        raise ValueError('A Czech family crosses original word-3 verb groups')
    families = {family: i for i, family in enumerate(sorted(designs))}
    by_family = {key: sha256(('cogllm-czech-garden-v1:' + next(iter(values))).encode()).hexdigest()
                 for key, values in verbs.items()}
    by_group = Counter(by_family.values())
    if len(by_group) != 24 or set(by_group.values()) != {2}:
        raise ValueError('Expected 24 sealed repeated-verb clusters, two families each')
    group_index = {key: i for i, key in enumerate(sorted(by_group))}
    groups = np.array([group_index[by_family[family]] for family in families])
    coordinates = {key: (families[row['family']], ANIMACIES.index(row['animacy']), CONDITIONS.index(row['condition']))
                   for key, row in layout.items()}
    return source, coordinates, groups


def observed_counts(outcomes, coordinates, family_count, source, experiment):
    records = [row for row in outcomes if row['experiment'] == experiment]
    expected = source[f'experiment_{experiment}']
    readers = {name: i for i, name in enumerate(sorted({row['reader'] for row in records}))}
    if len(records) != expected['trials'] or len(readers) != expected['readers']:
        raise ValueError('Czech cohort differs from its sealed experiment totals')
    shape = (len(readers), family_count, len(ANIMACIES), len(CONDITIONS), len(PROBES))
    yes, answers = np.zeros(shape), np.zeros(shape)
    raw_counts = defaultdict(Counter)
    seen = set()
    for row in records:
        family, animacy, condition = coordinates[row['id']]
        identity = (row['reader'], family)
        if identity in seen or row['question'] not in (*PROBES, 'qpos') or row['response'] not in RESPONSE_TYPES:
            raise ValueError('The source must supply one recognized probe and response per Czech reader/family')
        seen.add(identity)
        if row['question'] == 'qpos':
            continue
        probe = PROBES.index(row['question'])
        outcome = row['response']
        raw_counts[condition, probe][outcome] += 1
        index = (readers[row['reader']], family, animacy, condition, probe)
        answers[index] = outcome != 'missing'
        yes[index] = outcome == 'yes'
    return yes, answers, raw_counts


def frechet(yes, answers):
    rates = np.divide(yes, answers, out=np.zeros_like(yes), where=answers != 0)
    return np.stack((np.maximum(0.0, rates.sum(axis=-1) - 1.0), rates.min(axis=-1)), axis=-1), rates


def crossed_draws(yes, answers, groups, draws):
    random = np.random.default_rng(SEED)
    reader_count, family_count = yes.shape[:2]
    cluster_count = len(set(groups))
    reader_probability = np.full(reader_count, 1 / reader_count)
    cluster_probability = np.full(cluster_count, 1 / cluster_count)
    yes_flat, answers_flat = yes.reshape(reader_count, -1), answers.reshape(reader_count, -1)
    shape = yes.shape[1:]
    results = []
    accepted, rejected = 0, 0
    while accepted < draws:
        size = min(BATCH, draws - accepted)
        cluster_weights = random.multinomial(cluster_count, cluster_probability, size=size)
        reader_weights = random.multinomial(reader_count, reader_probability, size=size).astype(np.float64)
        family_weights = cluster_weights[:, groups]
        observed = (reader_weights @ answers_flat).reshape(size, *shape)
        positives = (reader_weights @ yes_flat).reshape(size, *shape)
        populated = (observed > 0).all(axis=(2, 3, 4))
        valid = ((family_weights == 0) | populated).all(axis=1)
        bounds, _ = frechet(positives[valid], observed[valid])
        values = np.einsum('bf,bfce->bce', family_weights[valid], bounds.mean(axis=2)) / family_count
        results.append(values)
        accepted += int(valid.sum())
        rejected += int((~valid).sum())
    return np.concatenate(results), rejected


def summarize(yes, answers, raw_counts, groups, draws):
    present = answers.sum(axis=0)
    if np.any(present <= 0):
        raise ValueError('A source stratum lacks an observed marginal')
    cell_bounds, cell_rates = frechet(yes.sum(axis=0), present)
    weights = (0, 1)
    marginal = cell_rates.mean(axis=weights)
    lower_upper = cell_bounds.mean(axis=weights)
    if not np.all((0 <= lower_upper) & (lower_upper <= 1)) or np.any(lower_upper[:, 0] > lower_upper[:, 1]):
        raise ValueError('Invalid outer bounds on joint Czech endorsements')
    draws_by_arm, rejected = crossed_draws(yes, answers, groups, draws)
    endpoint_ci = np.quantile(draws_by_arm, [0.025, 0.975], axis=0)
    widths_ci = np.quantile(draws_by_arm[:, :, 1] - draws_by_arm[:, :, 0], [0.025, 0.975], axis=0)
    results = {}
    for i, condition in enumerate(CONDITIONS):
        lower, upper = map(float, lower_upper[i])
        results[condition] = {
            'strata': len(groups) * len(ANIMACIES),
            'marginals': {name: {'condition_family_animacy_macro_endorsement': float(marginal[i, j]),
                                 'counts': {key: raw_counts[i, j][key] for key in RESPONSE_TYPES},
                                 'nonmissing_answers': int(present[:, :, i, j].sum()),
                                 'observed_answers_per_stratum_min_max': [int(present[:, :, i, j].min()),
                                                                         int(present[:, :, i, j].max())]}
                          for j, name in enumerate(PROBES)},
            'lower': lower, 'upper': upper, 'width': upper - lower,
            'lower_endpoint_ci95': endpoint_ci[:, i, 0].tolist(),
            'upper_endpoint_ci95': endpoint_ci[:, i, 1].tolist(),
            'width_ci95': widths_ci[:, i].tolist(),
        }
    return {'readers': len(yes), 'item_families': len(groups), 'verb_clusters': len(set(groups)),
            'bootstrap_draws': len(draws_by_arm), 'bootstrap_draws_rejected_empty_sampled_stratum': rejected,
            'by_condition': results}


def analyze(prepared):
    source, coordinates, groups = design(prepared)
    outcomes = rows(prepared / 'outcomes.jsonl')
    if set(row['experiment'] for row in outcomes) != set(EXPERIMENTS):
        raise ValueError('Expected the two registered yes/no experiments')
    results = {str(experiment): summarize(*observed_counts(outcomes, coordinates, len(groups), source, experiment),
                                         groups, DRAWS) for experiment in EXPERIMENTS}
    return {
        'study_type': 'Exploratory partial identification from separate text comprehension probes',
        'paper': source['paper'], 'source': source['project'], 'prepared_source_sha256': SOURCE_SHA256,
        'prepared_artifact_sha256': {key: value for key, value in source['file_sha256'].items()
                                    if key.startswith('prepared/')},
        'analysis_code_sha256': digest(Path(__file__)),
        'estimand': 'Probability that one reader would endorse both qcor and qmis at the sentence end under the same item, object animacy and ambiguity condition.',
        'probes': {'qcor': 'Yes endorses the intended direct object.',
                   'qmis': 'Yes endorses the initially plausible but incorrect direct object.'},
        'identification_assumptions': [
            'Potential response invariance: choosing which single probe to ask does not alter its respective counterfactual endorsement marginal in the joint endpoint interpretation.',
            'Within each item, animacy and condition, the separately prompted nonmissing marginals represent a common reader population. Selective missing responses could violate this assumption.',
        ],
        'aggregation': 'Compute max(0, p_qcor + p_qmis - 1) and min(p_qcor, p_qmis) separately within each stratum; then weight 48 families equally and both animacy levels equally within family. Missing excluded per marginal; explicit unsure remains in the denominator as nonyes.',
        'uncertainty': {'seed': SEED, 'draws_per_experiment': DRAWS,
                        'unit': 'Independently resample original word-3 verb clusters and experiment-scoped readers with replacement; use identical weights across conditions and probes.',
                        'empty_cell_rule': 'Discard and replace a draw if any resampled verb cluster has a stratum with no sampled answer to either probe.',
                        'interval_scope': 'Separate percentile intervals for identified endpoints and their width, without claiming simultaneous coverage of the full joint identified set.'},
        'interpretation': 'Potential joint endpoint co-endorsement remains partially identified. Each reader saw one probe per item; co-endorsement under the assumptions does not establish simultaneous representations during reading.',
        'experiments': results,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = analyze(args.prepared)
    with args.output.open('x', encoding='utf-8') as stream:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), file=stream)
    for key, experiment in report['experiments'].items():
        print(key, {condition: [values['lower'], values['upper']]
                    for condition, values in experiment['by_condition'].items()})
