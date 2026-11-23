"""Score sentence-final Czech garden-path responses against fixed forecasts."""

import numpy as np


RESPONSES = ('yes', 'no', 'unsure')
ANIMACIES = ('animate', 'inanimate')
CONDITIONS = ('control', 'gp')
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 735972
EQUIVALENCE = 0.05
BATCH_SIZE = 128


def cell_key(row):
    return row['family'], row['animacy'], row['condition']


def design(trials, forecasts, methods):
    """Build the crossed reader/family counts and an ordered forecast cube."""
    source = tuple(trials)
    identities = {(row['reader_id'], row['family']) for row in source}
    if len(identities) != len(source):
        raise ValueError('Expected at most one qmis trial per reader and source family')
    for row in source:
        if (row['response'] not in (*RESPONSES, None) or row['animacy'] not in ANIMACIES
                or row['condition'] not in CONDITIONS or not row['reader_id'] or not row['family']):
            raise ValueError('Invalid qmis response, condition, animacy or identity')
    observed = [row for row in source if row['response'] is not None]
    families = sorted({row['family'] for row in source})
    readers = sorted({row['reader_id'] for row in observed})
    cells = [(family, animacy, condition) for family in families for animacy in ANIMACIES for condition in CONDITIONS]
    by_cell = {}
    for row in forecasts:
        key = cell_key(row)
        if key in by_cell or set(row['probabilities']) != set(methods):
            raise ValueError('Expected one forecast per cell and exactly the requested methods')
        by_cell[key] = [row['probabilities'][method] for method in methods]
    if len(families) < 2 or len(readers) < 2 or set(by_cell) != set(cells):
        raise ValueError('Expected complete paired forecasts over crossed reader and source families')
    cell_design = (len(families), len(ANIMACIES), len(CONDITIONS), len(RESPONSES))
    reader_index = {value: index for index, value in enumerate(readers)}
    cell_index = {value: index for index, value in enumerate(cells)}
    response_index = {value: index for index, value in enumerate(RESPONSES)}
    counts = np.zeros((len(readers), len(cells), len(RESPONSES)), dtype=np.float64)
    for row in observed:
        counts[reader_index[row['reader_id']], cell_index[cell_key(row)], response_index[row['response']]] += 1
    if np.any(counts.sum(axis=(0, 2)) == 0):
        raise ValueError('Human outcomes do not cover both conditions in every family and animacy stratum')
    probabilities = np.asarray([by_cell[key] for key in cells], dtype=np.float64)
    if (probabilities.shape != (len(cells), len(methods), len(RESPONSES)) or not np.isfinite(probabilities).all()
            or np.any(probabilities <= 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=-1), 1, rtol=0, atol=1e-8)):
        raise ValueError('Every model must assign positive, normalized probability to yes, no and unsure')
    return (counts.reshape((len(readers), *cell_design)),
            probabilities.swapaxes(1, 2).reshape((*cell_design, len(methods))), families)


def sample_metrics(counts, family_weights, log_losses, forecast_effects):
    cell_n = counts.sum(axis=-1)
    weights = family_weights[:, :, None, None, None]
    counts_weighted = (counts * weights).reshape(len(counts), len(log_losses))
    nll = (counts_weighted @ log_losses) / counts_weighted.sum(axis=1, keepdims=True)
    rates = counts[..., 0] / np.maximum(cell_n, 1)
    human = rates[..., 1] - rates[..., 0]
    contrasts = np.concatenate((human[..., None], np.broadcast_to(forecast_effects, (*human.shape, nll.shape[1]))), axis=-1)
    by_animacy = (contrasts * weights[..., 0]).sum(axis=1) / family_weights.sum(axis=1)[:, None, None]
    return nll, np.concatenate((by_animacy.mean(axis=1, keepdims=True), by_animacy), axis=1)


def sample_distribution(counts, probabilities, family_to_cluster, draws, seed):
    """Use shared crossed multinomial weights and redraw incomplete selected families."""
    readers, families = counts.shape[:2]
    log_losses = -np.log(probabilities).reshape(-1, probabilities.shape[-1])
    forecast_effects = probabilities[:, :, 1, 0, :] - probabilities[:, :, 0, 0, :]
    point = sample_metrics(counts.sum(axis=0, keepdims=True), np.ones((1, families)), log_losses, forecast_effects)
    rng = np.random.default_rng(seed)
    reader_probabilities = np.full(readers, 1 / readers)
    clusters = int(family_to_cluster.max()) + 1
    cluster_probabilities = np.full(clusters, 1 / clusters)
    reader_cells = counts.reshape(readers, -1)
    retained = rejected = 0
    losses, effects = [], []
    while retained < draws:
        batch = min(BATCH_SIZE, draws - retained)
        reader_weights = rng.multinomial(readers, reader_probabilities, size=batch)
        cluster_weights = rng.multinomial(clusters, cluster_probabilities, size=batch)
        family_weights = cluster_weights[:, family_to_cluster]
        resampled = (reader_weights @ reader_cells).reshape((batch, *counts.shape[1:]))
        cell_has_data = (resampled.sum(axis=-1) > 0).all(axis=(2, 3))
        usable = (cell_has_data | (family_weights == 0)).all(axis=1)
        batch_losses, batch_effects = sample_metrics(resampled[usable], family_weights[usable], log_losses, forecast_effects)
        losses.append(batch_losses)
        effects.append(batch_effects)
        retained += int(usable.sum())
        rejected += int((~usable).sum())
        if rejected > 100 * draws:
            raise ValueError('The crossed design cannot reliably produce complete matched resamples')
    return point, (np.concatenate(losses), np.concatenate(effects)), rejected


def interval(point, draws):
    return {'estimate': float(point), 'ci95': np.quantile(draws, [0.025, 0.975]).tolist()}


def family_clusters(families, groups):
    if groups is None:
        return np.arange(len(families))
    if not set(families).issubset(groups) or not all(isinstance(groups[family], str) and groups[family] for family in families):
        raise ValueError('Each evaluated family requires a named verb cluster')
    cluster_names = sorted({groups[family] for family in families})
    if len(cluster_names) < 2:
        raise ValueError('Expected at least two verb clusters for crossed uncertainty')
    ordered_clusters = {cluster: index for index, cluster in enumerate(cluster_names)}
    return np.array([ordered_clusters[groups[family]] for family in families])


def summarize_cells(families, counts, probabilities, methods):
    observed = counts.sum(axis=0).astype(np.int64)
    marginal = observed.sum(axis=-1)
    observed_nll = -(observed[..., None] * np.log(probabilities)).sum(axis=-2) / marginal[..., None]
    return [
        {'family': family, 'animacy': animacy, 'condition': condition,
         'observed_counts': dict(zip(RESPONSES, observed[i, j, k].tolist(), strict=True)),
         'predicted_probability': {
             method: dict(zip(RESPONSES, probabilities[i, j, k, :, h].tolist(), strict=True))
             for h, method in enumerate(methods)
         },
         'nll_nats': dict(zip(methods, observed_nll[i, j, k].tolist(), strict=True))}
        for i, family in enumerate(families)
        for j, animacy in enumerate(ANIMACIES)
        for k, condition in enumerate(CONDITIONS)
    ]


def evaluate(trials, forecasts, *, primary, comparators, bootstrap_draws=BOOTSTRAP_DRAWS,
             seed=BOOTSTRAP_SEED, equivalence=EQUIVALENCE, family_groups=None):
    """Evaluate one study's qmis forecasts; caller is responsible for family-held-out fitting."""
    methods = (primary, *comparators)
    if (len(methods) < 2 or len(set(methods)) != len(methods) or 'human' in methods
            or not all(isinstance(method, str) and method for method in methods)):
        raise ValueError('Expected distinct named primary and comparator methods')
    if type(bootstrap_draws) is not int or bootstrap_draws < 1 or not 0 < equivalence < 1:
        raise ValueError('Expected a positive number of draws and a probability-scale equivalence margin')
    trials = tuple(trials)
    counts, probabilities, families = design(trials, forecasts, methods)
    family_to_cluster = family_clusters(families, family_groups)
    (point_loss, point_effect), (draw_loss, draw_effect), rejected = sample_distribution(
        counts, probabilities, family_to_cluster, bootstrap_draws, seed,
    )
    point_loss, point_effect = point_loss[0], point_effect[0]
    nll = {method: interval(point_loss[i], draw_loss[:, i]) for i, method in enumerate(methods)}
    nll_delta = {method: interval(point_loss[0] - point_loss[i], draw_loss[:, 0] - draw_loss[:, i])
                 for i, method in enumerate(methods[1:], 1)}
    yes_effects = {
        label: {method: interval(point_effect[i, j], draw_effect[:, i, j])
                for j, method in enumerate(('human', *methods))}
        for i, label in enumerate(('pooled', *ANIMACIES))
    }
    effect_error = interval(point_effect[0, 1] - point_effect[0, 0], draw_effect[:, 0, 1] - draw_effect[:, 0, 0])
    lower, upper = effect_error['ci95']
    response_counts = counts.sum(axis=(0, 1)).astype(np.int64)
    return {
        'counts': {'readers': len(counts), 'families': counts.shape[1], 'observed': int(counts.sum()),
                   'missing': len(trials) - int(counts.sum()),
                   'by_animacy_condition_response': {
                       animacy: {condition: dict(zip(RESPONSES, response_counts[i, j].tolist(), strict=True))
                                 for j, condition in enumerate(CONDITIONS)}
                       for i, animacy in enumerate(ANIMACIES)
                   }},
        'bootstrap': {'seed': seed, 'draws': bootstrap_draws, 'rejected_empty_cell_draws': rejected,
                      'unit': 'crossed ' + ('family' if family_groups is None else 'verb cluster') + ' and reader',
                      'families_per_cluster': np.bincount(family_to_cluster).tolist(),
                      'interval': 'percentile 95%', 'forecast_refit': False},
        'cells': summarize_cells(families, counts, probabilities, methods),
        'nll_nats': nll,
        'primary_nll_minus_comparator_nats': nll_delta,
        'yes_gp_minus_control': yes_effects,
        'primary_minus_human_yes_effect': effect_error,
        'gates': {'positive_effect': yes_effects['pooled'][primary]['ci95'][0] > 0,
                  'human_effect_positive': yes_effects['pooled']['human']['ci95'][0] > 0,
                  'effect_equivalent': -equivalence < lower <= upper < equivalence,
                  'effect_equivalence_margin': equivalence,
                  'nll_improved': {method: value['ci95'][1] < 0 for method, value in nll_delta.items()}},
    }
