"""Audit public SALT 35 temporal interpretations and reading stops."""

import argparse
from collections import Counter, defaultdict
from csv import DictReader
from hashlib import sha256
from io import StringIO
from itertools import accumulate
import json
from math import isfinite
from pathlib import Path
from statistics import fmean


SOURCES = {
    'e1-materials.csv': ('ued98', 'b046354b6397ad3afdc73cb795666d0ce0f1a15a94a3558d5a15d7d55ba22e00'),
    'e1-key.csv': ('6756ecf2791cce30eef73c33', '110f3a9ce3e527436d254ce8bb48097acd6fa4fef01de86887d1d4261c3e0a1a'),
    'e1-ratings.csv': ('6756ed091426ab8053563bbb', '5c422c534cce1547e88a861895eb2eb01a134d7a3ec621442634e05b55bb862d'),
    'e2-materials.csv': ('9ef4x', '4cb87a5f6a071b6c33c4e779667514a22f5b0a6a970be2b0d72862993b891dd4'),
    'e2-regions.csv': ('6756ed3f0eb8dba7d7f73a57', 'c04c38dc4d6fa1e60f6ff241bb6c0ddd794354ebed6585a9f002802b8e7a5dd4'),
    'e2-all-critical.csv': ('6756ed673430cda16f661498', 'ee78e3510cf7ac3427fe12583e1b78a6a1912a92d226479080e1a34aa4e2f9ce'),
    'e2-correct-critical.csv': ('kv2ay', 'e6c3a09574048332ce526e5c2435e6a25fe28c2b2101ed533de95268acb96858'),
}
E1_CONDITIONS = {
    'a': 'ambiguous_explanation_biased', 'b': 'ambiguous_debiased',
    'c': 'late_backshift_cue', 'd': 'late_progression_cue',
    'e': 'explicit_backshift_connective', 'f': 'explicit_progression_connective',
}
E2_CONDITIONS = {'a': 'late_cue', 'b': 'earlier_adverb', 'c': 'earliest_connective'}
E2_STOPS = ('S1_2', 'S1_3', 'S2', 'S3_1', 'S3_2', 'S3_3', 'S4_1', 'S4_2', 'S5_1', 'S5_2', 'S5_3')
ORDER_QUESTIONS = {'S2 order', 'S3 order'}


def read_source(raw, name):
    data = (raw / name).read_bytes()
    if sha256(data).hexdigest() != SOURCES[name][1]:
        raise ValueError(f'Unexpected bytes in published source {name}')
    with StringIO(data.decode('utf-8-sig'), newline='') as stream:
        return list(DictReader(stream))


def display_text(marked):
    return ' '.join(marked.replace('_', ' ').split())


def e1_audit(raw):
    materials = read_source(raw, 'e1-materials.csv')
    stories = {(row['item'], row['condition']): tuple(map(display_text, row['story'].splitlines())) for row in materials}
    keys = {(row['Item'], row['Condition']): row for row in read_source(raw, 'e1-key.csv')}
    ratings = read_source(raw, 'e1-ratings.csv')
    inventory = Counter((row['Item'], row['Condition']) for row in ratings)
    items = {item for item, _ in stories}
    if (len(materials) != len(stories) or len(items) != 48 or set(stories) != set(keys)
            or inventory != Counter({(item, variant): 5 for item in items for variant in E1_CONDITIONS})
            or len({row['Subject'] for row in ratings}) != 30
            or len({(row['Subject'], row['Item']) for row in ratings}) != len(ratings)):
        raise ValueError('Unexpected Experiment 1 item and reader inventory')

    for item in items:
        intro, prior, before = stories[item, 'a']
        for variant in ('c', 'd'):
            extended_intro, extended_prior, boundary, cue = stories[item, variant]
            if ((intro, prior) != (extended_intro, extended_prior)
                    or not before.endswith('.') or boundary != before[:-1] + ',' or not cue
                    or any(keys[item, variant][event] != keys[item, 'a'][event] for event in ('E1', 'E2'))):
                raise ValueError('Experiment 1 continuation did not preserve the pre-cue words')

    choices = defaultdict(Counter)
    alternatives = defaultdict(Counter)
    ordinals = defaultdict(Counter)
    item_choices = defaultdict(Counter)
    for row in ratings:
        version = (row['Item'], row['Condition'])
        key = keys[version]
        backshift, likelihood, ordinal = (int(row[name])
                                          for name in ('BackshiftResponse', 'AlternativeLikelihood', 'CompoundResponse'))
        first_event = key['E2'] if backshift else key['E1']
        if (backshift not in (0, 1) or likelihood not in range(1, 5)
                or ordinal != (9 - likelihood if backshift else likelihood)
                or int(row['ExpResponse']) != (first_event == key['Expected'])):
            raise ValueError('Experiment 1 option coding or ordinal scoring differs from the author rules')
        label = E1_CONDITIONS[row['Condition']]
        choices[label][backshift] += 1
        alternatives[label][likelihood] += 1
        ordinals[label][ordinal] += 1
        item_choices[version][backshift] += 1

    cells = {}
    for label in E1_CONDITIONS.values():
        count = choices[label].total()
        cells[label] = {'trials': count, 'backshift_choices': choices[label][1],
                        'backshift_proportion': choices[label][1] / count,
                        'alternative_likelihood_counts_1_to_4': [alternatives[label][value] for value in range(1, 5)],
                        'compound_response_counts_1_to_8': [ordinals[label][value] for value in range(1, 9)]}
    reversal = [item_choices[item, 'd'][1] - item_choices[item, 'a'][1] for item in items]
    return stories, {
        'readers': 30, 'critical_items': len(items), 'trials': len(ratings),
        'paper_describes_candidate_items': 60, 'trials_per_item_version': 5,
        'matched_prefix_and_response_options_items': len(items),
        'matched_prefix_boundary_change': 'period to comma',
        'conditions': cells,
        'late_progression_minus_ambiguous_backshift_percentage_points':
            100 * sum(reversal) / (5 * len(items)),
        'items_decreased_tied_increased_backshift': [
            sum(value < 0 for value in reversal), reversal.count(0), sum(value > 0 for value in reversal)],
    }


def e2_audit(raw, e1_stories):
    materials = read_source(raw, 'e2-materials.csv')
    forms = {(row['group'], row['item'], row['condition']): row for row in materials}
    stories = {(item, variant): row['story'] for (_, item, variant), row in forms.items()}
    segments = {version: tuple(display_text(chunk) for chunk in story.split()) for version, story in stories.items()}
    published_regions = {row['Label']: row for row in read_source(raw, 'e2-regions.csv')}
    items = {item for item, _ in segments}
    if (len(materials) != len(forms) or len(items) != 24
            or len(segments) != 72 or len(published_regions) != 72
            or any(row['story'] != stories[item, variant] for (_, item, variant), row in forms.items())):
        raise ValueError('Unexpected Experiment 2 material inventory')
    word_offset_mismatches = Counter()
    for (item, variant), chunks in segments.items():
        region = published_regions[item + variant]
        cumulative_words = tuple(accumulate(len(chunk.split()) for chunk in chunks))[:-1]
        differences = [count - int(region[key]) for count, key in zip(cumulative_words, E2_STOPS, strict=True)]
        # Regions.csv retains positions from before the extra connective in the earliest-cue version.
        expected_differences = [0] * 4 + [1] * 7 if variant == 'c' else [0] * 11
        if len(chunks) != 12 or ' '.join(chunks) != region['Text'] or differences != expected_differences:
            raise ValueError('Experiment 2 display segments differ from the published regions')
        word_offset_mismatches[E2_CONDITIONS[variant]] += sum(bool(value) for value in differences)
        if variant == 'a':
            sections = tuple(map(display_text, stories[item, variant].splitlines()))
            if sections[:4] != e1_stories[item, 'd'] or len(sections) != 5:
                raise ValueError('The Experiment 2 late cue differs from the Experiment 1 progression suffix')
    for item in items:
        if (len({segments[item, variant][7:10] for variant in E2_CONDITIONS}) != 1
                or segments[item, 'b'][5] != segments[item, 'c'][5]):
            raise ValueError('Compared time regions must have identical displayed words within item')

    observations = read_source(raw, 'e2-all-critical.csv')
    correct = read_source(raw, 'e2-correct-critical.csv')
    if correct != [row for row in observations if row['isCorrect'] == 'TRUE']:
        raise ValueError('Published correct-answer time subset disagrees with all trials')
    reader_items = defaultdict(dict)
    for row in observations:
        trial = (row['Subject'], row['Item'])
        stop = int(row['Region'])
        source = forms[row['Form'], row['Item'], row['Condition']]
        expected = 'NA' if row['Response'] == 'NA' else str(row['Response'] == source['correct']).upper()
        if (stop not in range(1, 13) or stop in reader_items[trial]
                or row['Text'] != segments[row['Item'], row['Condition']][stop - 1]
                or row['isCorrect'] != expected or row['QType'] not in {'NA', source['qtype']}
                or not isfinite(float(row['RT'])) or float(row['RT']) <= 0):
            raise ValueError('Experiment 2 displayed region, response or recorded time disagrees with the source')
        reader_items[trial][stop] = row

    counts = Counter()
    temporal = defaultdict(Counter)
    for stops in reader_items.values():
        if stops.keys() != set(range(1, 13)):
            raise ValueError('Experiment 2 trial is missing a display region')
        first = stops[1]
        if any(row[key] != first[key] for row in stops.values()
               for key in ('Form', 'Trial', 'Condition', 'QType', 'Response', 'isCorrect')):
            raise ValueError('Experiment 2 regions disagree about the trial or post-reading answer')
        source = forms[first['Form'], first['Item'], first['Condition']]
        if source['qtype'] in ORDER_QUESTIONS:
            temporal[E2_CONDITIONS[first['Condition']]][first['isCorrect']] += 1
        counts[first['Item'], first['Condition']] += 1
    if (counts != Counter({(item, variant): 30 for item in items for variant in E2_CONDITIONS})
            or len({reader for reader, _ in reader_items}) != 90):
        raise ValueError('Unexpected Experiment 2 item and reader inventory')

    contrasts = (
        ('earlier_adverb', 6, 'b', 'c'),
        ('late_cue_first_chunk', 8, 'a', 'b'),
        ('late_cue_final_chunk', 9, 'a', 'b'),
        ('post_late_cue_first_chunk', 10, 'a', 'b'),
    )
    times = {}
    for label, region, condition, control in contrasts:
        target, reference = ([float(row['RT']) for row in correct
                              if int(row['Region']) == region and row['Condition'] == variant]
                             for variant in (condition, control))
        times[label] = {'region': region, 'target': E2_CONDITIONS[condition], 'control': E2_CONDITIONS[control],
                        'target_count': len(target), 'control_count': len(reference),
                        'target_mean_ms': fmean(target), 'control_mean_ms': fmean(reference),
                        'difference_ms': fmean(target) - fmean(reference)}
    answer_counts = {}
    for label, answers in temporal.items():
        answered = answers['TRUE'] + answers['FALSE']
        answer_counts[label] = {'assigned': answers.total(), 'answered': answered, 'correct': answers['TRUE'],
                                'missing': answers['NA'], 'proportion_correct': answers['TRUE'] / answered}
    return {
        'readers': 90, 'critical_items': len(items), 'trials': len(reader_items),
        'region_times': len(observations), 'regions_per_trial': 12,
        'matched_e1_progression_suffix_items': len(items), 'correct_answer_trials_for_time_comparison': len(correct) // 12,
        'published_word_offset_mismatches_by_condition': word_offset_mismatches,
        'temporal_order_probes_after_full_story': answer_counts,
        'descriptive_time_differences_for_correct_trials': times,
    }


def run(raw, output):
    stories, e1 = e1_audit(raw)
    report = {
        'paper_doi': '10.3765/ejyxjx09',
        'repository_doi': '10.17605/OSF.IO/C32K6',
        'sources': {name: {'url': f'https://osf.io/download/{key}/', 'sha256': digest}
                    for name, (key, digest) in SOURCES.items()},
        'experiment_1': e1, 'experiment_2': e2_audit(raw, stories),
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(**vars(parser.parse_args()))
