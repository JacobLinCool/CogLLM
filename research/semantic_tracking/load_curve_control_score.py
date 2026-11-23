"""Score frozen trajectories on constructed continuation controls."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


CONDITIONS = ('early', 'abrupt', 'delayed')
ABSOLUTE_TOLERANCE = 1e-6
RELATIVE_TOLERANCE = 1e-5
RETENTION_FRACTION = 0.80
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 735972

EFFECTS = {
    'paragraph_end_delay_minus_early': 'Delayed minus early at the end of sentence 3',
    'final_abrupt_minus_delay': 'Abrupt minus delayed at the end of sentence 5',
    'delayed_cue_drop_minus_sham': 'Drop through the delayed answer cue minus drop through the last '
                                    'equally many tokens of its related, pre-answer bridge in sentence 4',
    'abrupt_shift_retention_margin': 'Minimum from the end of the switch phrase to the end of sentence 4, '
                                    'minus 80% of the mass at the preceding paragraph end',
}


def _labels(path):
    data = Path(path).read_bytes()
    families, identifiers = defaultdict(dict), set()
    for line in data.decode('utf-8').splitlines():
        row = json.loads(line)
        identifier = row['id']
        condition = row['condition']
        family = families[row['family_id']]
        if (identifier in identifiers or condition in family or condition not in CONDITIONS
                or row['source'] != 'constructed_control'):
            raise ValueError(f'Invalid or repeated continuation label {identifier}')
        family[condition] = row
        identifiers.add(identifier)
    if not families or any(set(family) != set(CONDITIONS) for family in families.values()):
        raise ValueError('Expected one complete authored triplet per family')
    return sorted(families.items()), identifiers, hashlib.sha256(data).hexdigest()


def _frontier(text, offsets, end):
    """Return the last fully visible token index at an exclusive character end."""
    if type(end) is not int or not 0 < end <= len(text):
        raise ValueError(f'Invalid document character end {end!r}')
    index = int(np.searchsorted(offsets[:, 1], end, side='right') - 1)
    content_end = len(text[:end].rstrip())
    if index < 0 or offsets[index, 1] < content_end:
        raise ValueError(f'Character end {end} cuts through a visible token')
    return index


def _curve(record, text, label):
    ids = np.asarray(record['token_ids'])
    offsets = np.asarray(record['offsets'])
    if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu' or np.any(ids < 0)
            or offsets.shape != (len(ids), 2) or offsets.dtype.kind not in 'iu'
            or np.any(offsets[:, 0] < 0) or np.any(offsets[:, 1] > len(text))
            or np.any(offsets[:, 0] >= offsets[:, 1]) or np.any(offsets[1:, 1] < offsets[:-1, 1])):
        raise ValueError(f'Invalid token lengths or offsets for {label["id"]}')
    channels = {'load': np.asarray(record['load'], dtype=np.float64)}
    if record.get('cohort') is not None:
        channels['cohort'] = np.asarray(record['cohort'], dtype=np.float64)
        if np.any(channels['cohort'] < 0):
            raise ValueError('The prefix-anchor cohort must be nonnegative')
    if any(values.shape != ids.shape or not np.isfinite(values).all() for values in channels.values()):
        raise ValueError(f'Missing or nonfinite trajectory values for {label["id"]}')

    meta = label['offsets']
    condition = label['condition']
    chars = meta['sentence_ends']
    opened = meta['opening']
    if (len(chars) != 5 or not 0 < chars[0] <= opened['start'] < opened['end'] <= chars[1]
            or not chars[1] < chars[2] < chars[3] < chars[4] == len(text)
            or meta['common_prefix_end'] != chars[1] or meta['paragraph_end'] != chars[2]
            or meta['late_comparison_end'] != chars[4]
            or text[chars[2]:meta['continuation_start']] != '\n\n'):
        raise ValueError(f'Invalid control sentence or paragraph geometry for {label["id"]}')
    stops = {f'sentence_{i}': _frontier(text, offsets, end) for i, end in enumerate(chars, start=1)}
    stops['shared_break'] = _frontier(text, offsets, meta['continuation_start'])
    if condition == 'abrupt':
        shift = meta['shift']
        if (meta['cue'] is not None or meta['before_cue_end'] is not None or shift is None
                or not meta['continuation_start'] <= shift['start'] < shift['end'] <= chars[3]):
            raise ValueError(f'Invalid switch geometry for {label["id"]}')
        stops['shift'] = _frontier(text, offsets, shift['end'])
    else:
        cue = meta['cue']
        before = meta['before_cue_end']
        cue_sentence = 2 if condition == 'early' else 4
        if (meta['shift'] is not None or cue is None
                or not chars[cue_sentence - 1] <= before <= cue['start'] < cue['end'] <= chars[cue_sentence]
                or before != len(text[:cue['start']].rstrip())):
            raise ValueError(f'Invalid cue geometry for {label["id"]}')
        stops['before_cue'] = _frontier(text, offsets, before)
        stops['cue'] = _frontier(text, offsets, cue['end'])
        if stops['before_cue'] >= stops['cue']:
            raise ValueError(f'Cue contains no newly visible token for {label["id"]}')
    return {'ids': ids, 'offsets': offsets, 'channels': channels, 'stops': stops, 'label': label}


def _prefix_check(left, right, stop):
    prefix_ends = [curve['stops'][stop] + 1 for curve in (left, right)]
    ids_match = np.array_equal(left['ids'][:prefix_ends[0]], right['ids'][:prefix_ends[1]])
    offsets_match = np.array_equal(left['offsets'][:prefix_ends[0]], right['offsets'][:prefix_ends[1]])
    checks = {}
    for channel in left['channels']:
        values_left = left['channels'][channel][:prefix_ends[0]]
        values_right = right['channels'][channel][:prefix_ends[1]]
        aligned = values_left.shape == values_right.shape
        checks[channel] = {
            'max_absolute_difference': float(np.max(np.abs(values_left - values_right))) if aligned else None,
            'passed': aligned and bool(np.allclose(values_left, values_right,
                                                  atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE)),
        }
    return {
        'conditions': [curve['label']['condition'] for curve in (left, right)],
        'through': stop,
        'tokens': prefix_ends,
        'token_ids_match': bool(ids_match),
        'character_offsets_match': bool(offsets_match),
        'channels': checks,
        'passed': bool(ids_match and offsets_match and all(row['passed'] for row in checks.values())),
    }


def _first_divergence(left, right, channel, stop, event):
    first = max(curve['stops'][stop] for curve in (left, right)) + 1
    limit = min(len(left['ids']), len(right['ids']))
    left_values, right_values = (curve['channels'][channel] for curve in (left, right))
    left_slice, right_slice = left_values[first:limit], right_values[first:limit]
    changed = np.flatnonzero(~np.isclose(left_slice, right_slice,
                                         atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE))
    if not len(changed):
        return {'compared_suffix_tokens': max(0, limit - first), 'first': None}
    index = first + int(changed[0])
    end = int(left['offsets'][index, 1])
    event_span = left['label']['offsets'][event]
    return {
        'compared_suffix_tokens': limit - first,
        'first': {
            'token_index': index,
            'left_character_end': end,
            'right_character_end': int(right['offsets'][index, 1]),
            'left_value_minus_right': float(left_values[index] - right_values[index]),
            'relative_to_left_event_end_characters': end - event_span['end'],
            'relative_to_left_event_end_tokens': index - left['stops'][event],
        },
    }


def _family_effects(curves, channel):
    early, abrupt, delayed = (curves[name] for name in CONDITIONS)
    early_values, abrupt_values, delayed_values = (curve['channels'][channel] for curve in (early, abrupt, delayed))
    early_stops, abrupt_stops, delayed_stops = (curve['stops'] for curve in (early, abrupt, delayed))

    before_cue, cue = delayed_stops['before_cue'], delayed_stops['cue']
    sham_end = delayed_stops['sentence_4']
    window_tokens = cue - before_cue
    sham_start = sham_end - window_tokens
    if sham_start < delayed_stops['shared_break']:
        raise ValueError(f'Cue window exceeds the matched bridge for {delayed["label"]["family_id"]}')
    cue_drop = float(delayed_values[before_cue] - delayed_values[cue])
    sham_drop = float(delayed_values[sham_start] - delayed_values[sham_end])

    before_shift = float(abrupt_values[abrupt_stops['sentence_3']])
    shift_min = float(abrupt_values[abrupt_stops['shift']:abrupt_stops['sentence_4'] + 1].min())
    return {
        'paragraph_end_delay_minus_early': float(delayed_values[delayed_stops['sentence_3']]
                                                 - early_values[early_stops['sentence_3']]),
        'final_abrupt_minus_delay': float(abrupt_values[abrupt_stops['sentence_5']]
                                          - delayed_values[delayed_stops['sentence_5']]),
        'delayed_cue_drop_minus_sham': cue_drop - sham_drop,
        'abrupt_shift_retention_margin': shift_min - RETENTION_FRACTION * before_shift,
        'delayed_cue_window': {'baseline_token': before_cue, 'end_token': cue, 'drop': cue_drop},
        'sham_window': {'baseline_token': sham_start, 'end_token': sham_end, 'drop': sham_drop},
        'matched_window_tokens': window_tokens,
        'abrupt_before_shift': before_shift,
        'abrupt_shift_window_minimum': shift_min,
        'abrupt_shift_retained_fraction': shift_min / before_shift if before_shift > 0 else None,
        'early_vs_delayed_first_divergence': _first_divergence(early, delayed, channel, 'sentence_2', 'cue'),
        'abrupt_vs_delayed_first_divergence': _first_divergence(abrupt, delayed, channel, 'shared_break', 'shift'),
    }


def _summarize(rows, samples, technical_pass):
    statistics = {}
    for effect, definition in EFFECTS.items():
        values = np.array([row[effect] for row in rows])
        interval = np.quantile(values[samples].mean(axis=1), (0.025, 0.975)).tolist()
        nonzero_baseline = (effect != 'abrupt_shift_retention_margin'
                            or all(row['abrupt_before_shift'] > 0 for row in rows))
        statistics[effect] = {
            'definition': definition,
            'mean': float(values.mean()),
            'paired_family_bootstrap_95_interval': interval,
            'families_positive': int(np.count_nonzero(values > 0)),
            'eligible': technical_pass and nonzero_baseline,
            'passed': technical_pass and nonzero_baseline and interval[0] > 0,
        }
    return {'effects': statistics, 'families': rows}


def evaluate(labels_path, traces, texts):
    """Evaluate sealed text-only forecasts without passing label geometry into a model.

    traces[method][id] must contain token_ids, offsets, and load. If supplied,
    cohort is the active mass from pre-branch anchors and must cover the same
    tokens as load. Extra non-control IDs in traces or texts are ignored.
    """
    families, identifiers, label_digest = _labels(labels_path)
    if not traces or not identifiers.issubset(texts):
        raise ValueError('No forecasts or incomplete control texts')
    samples = np.random.default_rng(BOOTSTRAP_SEED).integers(
        len(families), size=(BOOTSTRAP_DRAWS, len(families)),
    )
    methods = {}
    for method, records in sorted(traces.items()):
        if not identifiers.issubset(records):
            raise ValueError(f'Incomplete control forecasts for {method}')
        prefix_checks, raw = [], {}
        family_effects = defaultdict(list)
        for family_id, labels in families:
            cuts = [texts[row['id']][:row['offsets']['common_prefix_end']] for row in labels.values()]
            break_cuts = [texts[labels[name]['id']][:labels[name]['offsets']['continuation_start']]
                          for name in ('abrupt', 'delayed')]
            if len(set(cuts)) != 1 or len(set(break_cuts)) != 1:
                raise ValueError(f'Authored texts do not share their prescribed prefix in {family_id}')
            curves = {name: _curve(records[row['id']], texts[row['id']], row) for name, row in labels.items()}
            channels = curves['early']['channels'].keys()
            if any(curve['channels'].keys() != channels for curve in curves.values()):
                raise ValueError(f'Inconsistent channel coverage in {method}/{family_id}')
            for left, right, stop in (('early', 'delayed', 'sentence_2'), ('early', 'abrupt', 'sentence_2'),
                                      ('abrupt', 'delayed', 'shared_break')):
                prefix_checks.append({'family_id': family_id, **_prefix_check(curves[left], curves[right], stop)})
            for channel in channels:
                family_effects[channel].append({'family_id': family_id, **_family_effects(curves, channel)})
            for curve in curves.values():
                raw[curve['label']['id']] = {
                    'token_ids': curve['ids'].tolist(),
                    'offsets': curve['offsets'].tolist(),
                    **{channel: values.tolist() for channel, values in curve['channels'].items()},
                }

        if any(len(rows) != len(families) for rows in family_effects.values()):
            raise ValueError(f'Every channel of {method} must cover every family')
        technical_pass = all(row['passed'] for row in prefix_checks)
        methods[method] = {
            'technical_pass': technical_pass,
            'prefix_checks': prefix_checks,
            'channels': {channel: _summarize(rows, samples, technical_pass)
                         for channel, rows in family_effects.items()},
            'raw_curves': raw,
        }

    return {
        'scope': 'Constructed text comparisons without human outcomes',
        'controls_sha256': label_digest,
        'families': len(families),
        'stimuli': len(identifiers),
        'thresholds': {
            'prefix_token_ids_and_offsets_exact': True,
            'prefix_absolute_tolerance': ABSOLUTE_TOLERANCE,
            'prefix_relative_tolerance': RELATIVE_TOLERANCE,
            'mean_difference_interval_lower_bound_strictly_above': 0.0,
            'minimum_retained_fraction_through_abrupt_window': RETENTION_FRACTION,
            'retention_requires_positive_baseline_for_every_family': True,
            'bootstrap_unit': 'matched triplet family',
            'bootstrap_draws': BOOTSTRAP_DRAWS,
            'bootstrap_seed': BOOTSTRAP_SEED,
        },
        'methods': methods,
    }
