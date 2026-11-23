"""Evaluate sealed Currie et al. passage-final consistency forecasts on CPU."""

from collections import defaultdict
from numbers import Real

import numpy as np


GENRES = ('narrative', 'expository')
CONDITIONS = ('near', 'far')
METHODS = ('global', 'target_weighted', 'paired_effect', 'constant')
LAYOUT_FIELDS = {'id', 'family', 'genre', 'item', 'condition', 'antecedent_sentence',
                 'target_sentence', 'same_sentence_inventory', 'same_target_sentence', 'sentences'}
DRAWS = 10_000
SEED = 735972
BATCH = 200
EFFECT_MARGIN = 0.10
NLL_MARGIN = 0.02


def identifier(value):
    return type(value) is int or (type(value) is str and bool(value))


def identity_order(value):
    return isinstance(value, str), value


def index_layout(layout):
    families, cells, items = defaultdict(dict), {}, {}
    for row in layout:
        if set(row) != LAYOUT_FIELDS:
            raise ValueError('Unexpected fields in the held-out LARRC layout')
        key, genre, family, item, condition = (row[field] for field in ('id', 'genre', 'family', 'item', 'condition'))
        antecedent, target, sentences = (row[field] for field in ('antecedent_sentence', 'target_sentence', 'sentences'))
        if (type(key) is not str or not key or key in cells or genre not in GENRES or condition not in CONDITIONS
                or not identifier(family) or not identifier(item) or type(row['same_sentence_inventory']) is not bool
                or type(row['same_target_sentence']) is not bool
                or not isinstance(sentences, (list, tuple)) or type(antecedent) is not int or type(target) is not int
                or not 1 <= antecedent < target <= len(sentences)):
            raise ValueError('Invalid or duplicate held-out LARRC story version')
        group = genre, family
        if condition in families[group] or items.get((genre, item), family) != family:
            raise ValueError('Repeated near/far version or split story pair in LARRC layout')
        cells[key] = row
        families[group][condition] = key
        items[genre, item] = family
    if {genre for genre, _ in families} != set(GENRES):
        raise ValueError('Both genres must be present in the LARRC holdout')
    groups = sorted(families, key=lambda group: (GENRES.index(group[0]), identity_order(group[1])))
    modes, matching, positions = [], [], {}
    for slot, group in enumerate(groups):
        pair = families[group]
        if pair.keys() != set(CONDITIONS):
            raise ValueError(f'Incomplete near/far pairing for {group}')
        near, far = (cells[pair[condition]] for condition in CONDITIONS)
        if near['item'] != far['item'] or len(near['sentences']) != len(far['sentences']):
            raise ValueError(f'Nearby and distant versions are not aligned for {group}')
        modes.append(GENRES.index(group[0]))
        matching.append(near['same_sentence_inventory'] and far['same_sentence_inventory'])
        positions.update({pair[condition]: (slot, c) for c, condition in enumerate(CONDITIONS)})
    return positions, np.asarray(modes), np.asarray(matching)


def index_outcomes(outcomes, positions, n_families):
    by_cell, readers_by_family, readers = {}, set(), set()
    for row in outcomes:
        if set(row) != {'id', 'reader', 'report'}:
            raise ValueError('Unexpected fields in the LARRC outcomes')
        key, reader, value = row['id'], row['reader'], row['report']
        if (type(key) is not str or key not in positions or not identifier(reader)
                or type(value) is not int or value not in (0, 1)):
            raise ValueError('Unknown LARRC story, invalid reader identifier or invalid binary outcome')
        family, condition = positions[key]
        if (family, reader) in readers_by_family:
            raise ValueError('A reader occurs more than once within a LARRC story family')
        readers.add(reader)
        readers_by_family.add((family, reader))
        by_cell[family, condition, reader] = value
    reader_slots = {reader: index for index, reader in enumerate(sorted(readers, key=identity_order))}
    present = np.zeros((n_families, len(CONDITIONS), len(readers)), dtype=np.float64)
    report = np.zeros_like(present)
    for (family, condition, reader), value in by_cell.items():
        slot = family, condition, reader_slots[reader]
        present[slot] = 1
        report[slot] = value
    if np.any(present.sum(axis=-1) == 0):
        raise ValueError('Every registered LARRC version needs at least one observed judgment')
    return present, report


def index_forecasts(forecasts, positions, n_families):
    table = np.full((n_families, len(CONDITIONS), len(METHODS)), np.nan)
    seen = set()
    for row in forecasts:
        if (set(row) != {'id', 'methods'} or type(row['id']) is not str
                or row['id'] not in positions or row['id'] in seen):
            raise ValueError('Unknown or duplicate LARRC forecast version')
        arms = row['methods']
        if (not isinstance(arms, dict) or arms.keys() != set(METHODS)
                or any(not isinstance(p, Real) or isinstance(p, bool) or not 0 < p < 1 for p in arms.values())):
            raise ValueError('Every forecast must contain four finite, strictly interior probabilities of No')
        seen.add(row['id'])
        table[positions[row['id']]] = [arms[method] for method in METHODS]
    if seen != positions.keys():
        raise ValueError('Forecasts do not cover every held-out LARRC story version')
    return table


def averaged_readers(present, report, weights):
    families, conditions, readers = present.shape
    totals = weights @ present.reshape(-1, readers).T
    if np.any(totals <= 0):
        raise ValueError('A reader bootstrap draw has no judgment for a registered version')
    hits = weights @ report.reshape(-1, readers).T
    return (hits / totals).reshape(len(weights), families, conditions)


def genre_cells(human, probabilities, family_groups, weights):
    """Aggregate version cells after averaging each version's available readers."""
    event = human[..., None]
    nll = -(event * np.log(probabilities) + (1 - event) * np.log1p(-probabilities))
    observed, predicted, losses = [], [], []
    for group, weight in zip(family_groups, weights, strict=True):
        weight = weight / weight.sum(axis=1, keepdims=True)
        observed.append(np.einsum('bf,bfc->bc', weight, human[:, group]))
        predicted.append(np.einsum('bf,fcm->bcm', weight, probabilities[group]))
        losses.append(np.einsum('bf,bfcm->bcm', weight, nll[:, group]))
    return tuple(np.stack(group, axis=1) for group in (observed, predicted, losses))


def contrasts(cells, effect):
    human, model, nll = cells
    if effect == 'narrative_minus_expository':
        observed = (human[:, 0] - human[:, 1]).mean(axis=1)
        forecast = (model[:, 0] - model[:, 1]).mean(axis=1)
    elif effect == 'far_minus_near':
        observed = (human[:, :, 1] - human[:, :, 0]).mean(axis=1)
        forecast = (model[:, :, 1] - model[:, :, 0]).mean(axis=1)
    else:
        raise ValueError(f'Unknown LARRC contrast: {effect}')
    return observed, forecast, nll.mean(axis=(1, 2))


def point(present, report, probabilities, groups):
    human = averaged_readers(present, report, np.ones((1, present.shape[-1])))
    return genre_cells(human, probabilities, groups, [np.ones((1, len(group))) for group in groups])


def bootstrap(present, report, probabilities, primary, secondary):
    rng = np.random.default_rng(SEED)
    readers = present.shape[-1]
    axes = [(primary, 'narrative_minus_expository')]
    if secondary is not None:
        axes.append((secondary, 'far_minus_near'))
    statistics = {effect: [] for _, effect in axes}
    for start in range(0, DRAWS, BATCH):
        batch = min(BATCH, DRAWS - start)
        reader_weight = rng.multinomial(readers, np.full(readers, 1 / readers), size=batch).astype(np.float64)
        human = averaged_readers(present, report, reader_weight)
        for groups, effect in axes:
            weights = [rng.multinomial(len(group), np.full(len(group), 1 / len(group)), size=batch)
                       for group in groups]
            statistics[effect].append(contrasts(genre_cells(human, probabilities, groups, weights), effect))
    return {key: tuple(np.concatenate(component) for component in zip(*value, strict=True))
            for key, value in statistics.items()}


def estimate(value, draws):
    return {'estimate': float(value), 'ci95': np.quantile(draws, (0.025, 0.975)).tolist()}


def summary(effect, cells, draws, present, report, groups):
    point_human, point_model, point_nll = contrasts(cells, effect)
    draw_human, draw_model, draw_nll = draws
    human, model, nll = (value[0] for value in cells)
    by_cell = {}
    for g, genre in enumerate(GENRES):
        by_cell[genre] = {}
        for c, condition in enumerate(CONDITIONS):
            by_cell[genre][condition] = {
                'families': len(groups[g]), 'judgments': int(present[groups[g], c].sum()),
                'no_reports': int(report[groups[g], c].sum()), 'human_report_rate': float(human[g, c]),
                'methods': {method: {'report_probability': float(model[g, c, m]), 'nll_nats': float(nll[g, c, m])}
                            for m, method in enumerate(METHODS)},
            }
    return {
        'effect': effect, 'human_effect': estimate(point_human[0], draw_human), 'by_genre_condition': by_cell,
        'methods': {method: {'nll_nats_per_judgment': estimate(point_nll[0, m], draw_nll[:, m]),
                             'effect': estimate(point_model[0, m], draw_model[:, m]),
                             'effect_minus_human': estimate(point_model[0, m] - point_human[0],
                                                            draw_model[:, m] - draw_human)}
                    for m, method in enumerate(METHODS)},
    }


def evaluate(layout, outcomes, forecasts):
    """Score complete near/far pairs using a shared-reader, genre-stratified bootstrap."""
    positions, genres, matching = index_layout(layout)
    present, report = index_outcomes(outcomes, positions, len(genres))
    probabilities = index_forecasts(forecasts, positions, len(genres))
    primary_groups = [np.flatnonzero(genres == g) for g in range(len(GENRES))]
    strict_groups = [np.flatnonzero((genres == g) & matching) for g in range(len(GENRES))]
    secondary_available = all(len(group) for group in strict_groups)
    samples = bootstrap(present, report, probabilities, primary_groups, strict_groups if secondary_available else None)

    primary_effect = 'narrative_minus_expository'
    primary_cells = point(present, report, probabilities, primary_groups)
    primary = summary(primary_effect, primary_cells, samples[primary_effect], present, report, primary_groups)
    point_nll = contrasts(primary_cells, primary_effect)[2][0]
    sampled_nll = samples[primary_effect][2]
    paired = METHODS.index('paired_effect')
    differences = {}
    for baseline in ('global', 'constant'):
        index = METHODS.index(baseline)
        differences[f'paired_effect_minus_{baseline}'] = estimate(point_nll[paired] - point_nll[index],
                                                                  sampled_nll[:, paired] - sampled_nll[:, index])
    primary['nll_differences'] = differences
    effect_ci = primary['methods']['paired_effect']['effect']['ci95']
    gap_ci = primary['methods']['paired_effect']['effect_minus_human']['ci95']
    gates = {'paired_effect_positive': effect_ci[0] > 0,
             'paired_effect_gap_inside_margin': -EFFECT_MARGIN < gap_ci[0] <= gap_ci[1] < EFFECT_MARGIN,
             'paired_effect_nll_noninferior_to_global': differences['paired_effect_minus_global']['ci95'][1] < NLL_MARGIN,
             'paired_effect_nll_noninferior_to_constant': differences['paired_effect_minus_constant']['ci95'][1] < NLL_MARGIN}
    primary.update(gates=gates, passed=all(gates.values()))

    secondary_effect = 'far_minus_near'
    secondary = {'available': secondary_available, 'effect': secondary_effect,
                 'families_by_genre': dict(zip(GENRES, map(len, strict_groups), strict=True))}
    if secondary_available:
        cells = point(present, report, probabilities, strict_groups)
        secondary.update(summary(secondary_effect, cells, samples[secondary_effect], present, report, strict_groups))
    volume = present.sum(axis=-1)
    return {
        'schema': 1, 'report_event': 'No', 'primary': primary, 'secondary': secondary,
        'inventory': {'families': len(genres), 'versions': len(positions), 'readers': present.shape[-1],
                      'judgments': int(volume.sum()), 'no_reports': int(report.sum()),
                      'min_judgments_per_version': int(volume.min()), 'max_judgments_per_version': int(volume.max()),
                      'families_by_genre': dict(zip(GENRES, map(len, primary_groups), strict=True)),
                      'readers_by_genre': {genre: int(np.count_nonzero(present[group].any(axis=(0, 1))))
                                          for genre, group in zip(GENRES, primary_groups, strict=True)}},
        'bootstrap': {'draws': DRAWS, 'seed': SEED, 'interval': 0.95, 'nll_margin_nats': NLL_MARGIN,
                      'probability_effect_margin': EFFECT_MARGIN, 'family_strata': list(GENRES),
                      'reader_pool': 'shared_across_genres_and_conditions'},
    }
