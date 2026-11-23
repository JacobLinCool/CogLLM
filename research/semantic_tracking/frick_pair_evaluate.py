"""Compare retrospective Frick readouts with frozen forecasts on paired bootstrap draws."""

from typing import NamedTuple

import numpy as np

from research.semantic_tracking.frick_consistency_score import (
    CONDITIONS,
    DRAWS,
    MARGIN,
    METHODS as REFERENCE_METHODS,
    SEED,
    TASKS,
    averaged_cells,
    interval,
    trial_statistics,
)


METHODS = ('global', 'target_weighted', 'paired_effect')
ALL_METHODS = (*METHODS, 'original_contextual')
PHASES = ('before_target', 'at_target', 'after_target', 'no_d')
NLL_MARGINS = {'current': 0.005, 'first': 0.05}


class DrawValues(NamedTuple):
    nll: np.ndarray
    human_target: np.ndarray
    model_target: np.ndarray
    human_phase: np.ndarray
    model_phase: np.ndarray


def reader_phases(families, metadata, trials, readers, present, phase_counts):
    """Keep the same reader units for the four-category onset distribution."""
    story_indices = {story: i for i, story in enumerate(sorted(families))}
    reader_indices = {reader: i for i, reader in enumerate(readers)}
    phases = np.zeros((*present.shape, len(PHASES)), dtype=np.int8)
    for key, reader, _, first in trials:
        info = metadata[key]
        story = story_indices[info['story']]
        condition = CONDITIONS.index(info['condition'])
        target = info['target_index'] - 1
        phase = 3 if first is None else int(first >= target) + int(first > target)
        phases[story, condition, reader_indices[reader], phase] = 1
    if not np.array_equal(phases.sum(axis=-1), present) or not np.array_equal(phases.sum(axis=2), phase_counts):
        raise ValueError('Reader onset categories do not match the original trial inventory')
    return phases


def reader_phase_means(reader_weights, present, phases):
    stories, conditions, readers = present.shape
    totals = reader_weights @ present.reshape(-1, readers).T
    flat_phases = phases.transpose(0, 1, 3, 2).reshape(-1, readers)
    counts = (reader_weights @ flat_phases.T).reshape(len(reader_weights), stories, conditions, len(PHASES))
    return counts / totals.reshape(len(reader_weights), stories, conditions, 1)


def draw_values(story_weights, nll, human_target, model_target, human_phase, model_phase):
    weights = story_weights / story_weights.sum(axis=1, keepdims=True)
    return DrawValues(
        np.einsum('bs,bstm->btm', weights, nll.mean(axis=2)),
        np.einsum('bs,bsct->bct', weights, human_target),
        np.einsum('bs,sctm->bctm', weights, model_target),
        np.einsum('bs,bscp->bcp', weights, human_phase),
        np.einsum('bs,scmp->bcmp', weights, model_phase),
    )


def bootstrap_values(logits, slots, families, metadata, trials, readers, reference_logits):
    expected_shape = (len(TASKS), len(METHODS), sum(slot.stop - slot.start for slot in slots.values()))
    logits, reference_logits = np.asarray(logits), np.asarray(reference_logits)
    if any(a.shape != expected_shape or not np.isfinite(a).all() for a in (logits, reference_logits)):
        raise ValueError('New logits and original forecasts must align at all scored stops for both tasks')
    present, nll, target, predicted, counts, phase_predicted = trial_statistics(
        logits, slots, families, metadata, trials, readers,
    )
    ref_present, ref_nll, ref_target, ref_predicted, ref_counts, ref_phase_predicted = trial_statistics(
        reference_logits, slots, families, metadata, trials, readers,
    )
    if any(not np.array_equal(first, second) for first, second in (
        (present, ref_present), (target, ref_target), (counts, ref_counts),
    )):
        raise ValueError('New and original logits do not share identical human trajectories')
    reference = REFERENCE_METHODS.index('contextual')
    predicted = np.concatenate((predicted, ref_predicted[..., reference:reference + 1]), axis=-1)
    phase_predicted = np.concatenate((phase_predicted, ref_phase_predicted[:, :, reference:reference + 1]), axis=2)
    phases = reader_phases(families, metadata, trials, readers, present, counts)

    def estimate(reader_weights, story_weights, fixed_weights=None):
        sampled_nll, sampled_target = averaged_cells(reader_weights, present, nll, target)
        original_nll, _ = averaged_cells(reader_weights, present, ref_nll, ref_target)
        sampled_nll = np.concatenate((sampled_nll, original_nll[..., reference:reference + 1]), axis=-1)
        sampled_phase = reader_phase_means(reader_weights, present, phases)
        estimates = (sampled_nll, sampled_target, predicted, sampled_phase, phase_predicted)
        crossed = draw_values(story_weights, *estimates)
        if fixed_weights is None:
            return crossed
        return crossed, draw_values(fixed_weights, *estimates)

    n_stories, _, n_readers = present.shape
    point = estimate(np.ones((1, n_readers)), np.ones((1, n_stories)))
    rng = np.random.default_rng(SEED)
    reader_draws = rng.multinomial(n_readers, np.full(n_readers, 1 / n_readers), size=DRAWS)
    story_draws = rng.multinomial(n_stories, np.full(n_stories, 1 / n_stories), size=DRAWS)
    crossed, fixed = [], []
    for start in range(0, DRAWS, 200):
        stop = min(start + 200, DRAWS)
        cross_batch, fixed_batch = estimate(
            reader_draws[start:stop], story_draws[start:stop], np.ones((stop - start, n_stories)),
        )
        crossed.append(cross_batch)
        fixed.append(fixed_batch)
    crossed_values = DrawValues(*(np.concatenate(values) for values in zip(*crossed, strict=True)))
    fixed_values = DrawValues(*(np.concatenate(values) for values in zip(*fixed, strict=True)))
    return (point, crossed_values, fixed_values), counts.sum(axis=0)


def summarize(draws, selector):
    point, crossed, fixed = (selector(value) for value in draws)
    return {
        'estimate': float(point.item()),
        'crossed_story_reader_ci95': interval(crossed),
        'fixed_story_reader_only_ci95': interval(fixed),
    }


def endpoint_summary(draws, task):
    t = TASKS.index(task)
    paired = METHODS.index('paired_effect')

    def human_effect(value):
        return value.human_target[:, 1, t] - value.human_target[:, 0, t]

    def model_effect(value, method):
        return value.model_target[:, 1, t, method] - value.model_target[:, 0, t, method]

    def effect_error(value, method):
        return model_effect(value, method) - human_effect(value)

    methods = {}
    for m, name in enumerate(ALL_METHODS):
        methods[name] = {
            'nll': summarize(draws, lambda v: v.nll[:, t, m]),
            'target_effect': summarize(draws, lambda v: model_effect(v, m)),
            'target_model_minus_human': summarize(draws, lambda v: effect_error(v, m)),
            'target_absolute_effect_error': summarize(draws, lambda v: np.abs(effect_error(v, m))),
            'target_probability_by_condition': {
                condition: summarize(draws, lambda v: v.model_target[:, c, t, m])
                for c, condition in enumerate(CONDITIONS)
            },
        }

    comparisons = {}
    for comparator in ('global', 'target_weighted', 'original_contextual'):
        m = ALL_METHODS.index(comparator)
        comparisons[comparator] = {
            'nll_paired_minus_comparator': summarize(draws, lambda v: v.nll[:, t, paired] - v.nll[:, t, m]),
            'target_effect_paired_minus_comparator': summarize(draws, lambda v: model_effect(v, paired) - model_effect(v, m)),
            'target_absolute_error_reduction': summarize(
                draws, lambda v: np.abs(effect_error(v, m)) - np.abs(effect_error(v, paired)),
            ),
        }

    direction = methods['paired_effect']['target_effect']['crossed_story_reader_ci95']
    gap = methods['paired_effect']['target_model_minus_human']['crossed_story_reader_ci95']
    nll_margin = NLL_MARGINS[task]
    gates = {
        'target_effect_positive': direction[0] > 0,
        'target_effect_equivalent_to_human': -MARGIN < gap[0] and gap[1] < MARGIN,
        'nll_noninferior_to_global': comparisons['global']['nll_paired_minus_comparator']['crossed_story_reader_ci95'][1] < nll_margin,
        'nll_noninferior_to_original_contextual': (
            comparisons['original_contextual']['nll_paired_minus_comparator']['crossed_story_reader_ci95'][1] < nll_margin
        ),
    }
    return {
        'nll_unit': 'nats_per_sentence' if task == 'current' else 'nats_per_story_trial',
        'human_target_effect': summarize(draws, human_effect),
        'human_target_probability_by_condition': {
            condition: summarize(draws, lambda v: v.human_target[:, c, t]) for c, condition in enumerate(CONDITIONS)
        },
        'methods': methods,
        'paired_effect_vs': comparisons,
        'gates': gates,
        'pass': all(gates.values()),
    }


def onset_summary(draws, counts):
    by_condition = {}
    for c, condition in enumerate(CONDITIONS):
        methods = {}
        for m, method in enumerate(ALL_METHODS):
            methods[method] = {
                'probabilities': {
                    phase: summarize(draws, lambda v: v.model_phase[:, c, m, p])
                    for p, phase in enumerate(PHASES)
                },
                'model_minus_human': {
                    phase: summarize(draws, lambda v: v.model_phase[:, c, m, p] - v.human_phase[:, c, p])
                    for p, phase in enumerate(PHASES)
                },
            }
        by_condition[condition] = {
            'human_counts': dict(zip(PHASES, counts[c].tolist(), strict=True)),
            'human_probabilities': {
                phase: summarize(draws, lambda v: v.human_phase[:, c, p]) for p, phase in enumerate(PHASES)
            },
            'methods': methods,
        }
    return by_condition


def evaluate(logits, slots, families, metadata, trials, readers, reference_logits):
    """Return scoring receipts for held-out logits and the original persisted forecasts."""
    draws, counts = bootstrap_values(logits, slots, families, metadata, trials, readers, reference_logits)
    return {
        'endpoints': {task: endpoint_summary(draws, task) for task in TASKS},
        'first_d_by_condition': onset_summary(draws, counts),
        'bootstrap': {'draws': DRAWS, 'seed': SEED, 'effect_margin': MARGIN, 'nll_margins': dict(NLL_MARGINS)},
    }
