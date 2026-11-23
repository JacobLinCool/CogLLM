"""Audit Yadav et al.'s German and Dutch sentence-rating materials."""

import argparse
from ast import literal_eval
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


RELEASES = {
    'experiment1-slider.rda': ('xu5j9', '36548e061607eaf43f32b3c68ef57fa410e63f805bcb18564f4dccc2a5f9a601'),
    'experiment2-all-slider.rda': ('ers89', 'fddff9dedf8c8b5c7b3a1c609d3ba9dc22225f4849eaf7f5b326d53c1e364363'),
    'experiment1-stimuli.txt': ('t5d89', 'ad8ffb2e0601d15f76b3e6ec6ae7b09c1f16cf3ddb86a60acf3bf067187795ec'),
    'experiment2-stimuli.txt': ('kxvf4', '08527bc48fe583e63f6347064cb837f28c3ec061799a4a15611341ba54cc93ba'),
    'analysis.Rmd': ('tj2qd', '2e071870e17e347aef579e8c10c2684c0dd9442aa6a13a7420d2b130dfb3eca3'),
    'preparation.Rmd': ('qd4j7', 'f1438c2e425ba1165536aaf6491dfb14e90caec91f486d4df32a81d532dc06e0'),
    'readme.md': ('6mvkj', 'e2a136ef0b4b5500944545313fec9fb8a6a1f66d697b11898c75bcb45381bda8'),
}
LANGUAGES = ('Dutch', 'German')
CONDITIONS = {'a': ('Level 2', 'No'), 'b': ('Level 2', 'Yes'),
              'c': ('Level 3', 'No'), 'd': ('Level 3', 'Yes')}


def build_stimulus_lookup(path):
    lookup = {}
    with path.open(encoding='utf-8-sig', newline='') as source:
        for row in csv.DictReader(source, delimiter='\t'):
            key = int(row['Item']), f"Level {row['Level']}", row['Paraphrase'].title()
            for language in LANGUAGES:
                if (language, *key) in lookup:
                    raise ValueError(f'Duplicate {language} item and condition {key}')
                lookup[language, *key] = row[language]
    if len(lookup) != 192:
        raise ValueError('Expected 24 underlying items, 4 item versions and 2 native languages')
    return lookup


def paired_effects(item_means):
    complete, output = {}, {}
    for level in ('Level 2', 'Level 3'):
        cases = [item for item in range(1, 25) if all((language, level, paraphrase, item) in item_means
                 for language in LANGUAGES for paraphrase in ('Yes', 'No'))]
        complete[level] = cases

        def costs(items):
            return {language: np.array([item_means[language, level, 'No', item]
                                        - item_means[language, level, 'Yes', item] for item in items])
                    for language in LANGUAGES}

        by_language = costs(cases)
        output[level] = {
            'complete_items': len(cases),
            'unmatched_item_ids': sorted(set(range(1, 25)) - set(cases)),
            'embedded_minus_paraphrase': {key: float(values.mean()) for key, values in by_language.items()},
            'german_minus_dutch_embedded': float(np.mean([item_means['German', level, 'No', item]
                                                         - item_means['Dutch', level, 'No', item] for item in cases])),
            'german_minus_dutch_paraphrase': float(np.mean([item_means['German', level, 'Yes', item]
                                                           - item_means['Dutch', level, 'Yes', item] for item in cases])),
            'embedding_cost_german_minus_dutch': float((by_language['German'] - by_language['Dutch']).mean()),
        }
    common = sorted(set(complete['Level 2']) & set(complete['Level 3']))
    double_differences = np.array([
        [sum(sign * (item_means[language, level, 'No', item] - item_means[language, level, 'Yes', item])
             for language, sign in (('German', 1), ('Dutch', -1))) for item in common]
        for level in ('Level 2', 'Level 3')
    ])
    output['level_interaction'] = {'complete_items': len(common),
                                   'level3_minus_level2_german_minus_dutch_cost': float(np.diff(double_differences, axis=0).mean())}
    return output


def audit_experiment(path, stimuli_path, rsvp, rdata):
    table = rdata.read_rda(path)['dat.slider']
    lookup = build_stimulus_lookup(stimuli_path)
    data = table.loc[table['condition'] != 'F']
    readers = ['Language', 'subj']
    if 'Experiment' in table:
        readers.append('Experiment')
    if data.duplicated([*readers, 'item']).any() or not data['response'].between(1, 9).all():
        raise ValueError('Unexpected duplicate critical trial or response outside the stored 1–9 scale')
    by_cell, by_item = defaultdict(list), defaultdict(list)
    matches, trailing_whitespace_matches = set(), set()
    for row in data.itertuples(index=False):
        if CONDITIONS[row.condition] != (row.Level, row.Paraphrase) or row.trial_type != 'html-slider-response':
            raise ValueError('Trial fields disagree with the authored experimental condition')
        key = row.Language, int(row.item), row.Level, row.Paraphrase
        sentence = ' '.join(literal_eval(row.stimulus)) if rsvp else row.stimulus
        if sentence != lookup[key]:
            if rsvp or sentence.rstrip() != lookup[key].rstrip():
                raise ValueError(f'Observed stimulus differs from the published text for {key}')
            trailing_whitespace_matches.add(key)
        matches.add(key)
        cell = row.Language, row.Level, row.Paraphrase
        difficulty = (float(row.response) - 1) * 12.5
        by_cell[cell].append(difficulty)
        by_item[(*cell, row.item)].append(difficulty)
    cells = []
    for (language, level, paraphrase), values in sorted(by_cell.items()):
        cells.append({
            'language': language, 'embedding_level': int(level[-1]), 'paraphrase': paraphrase == 'Yes',
            'ratings': len(values), 'items': sum(key[:3] == (language, level, paraphrase) for key in by_item),
            'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1)),
            'quantiles_0_25_50_75_100': np.percentile(values, (0, 25, 50, 75, 100)).tolist(),
        })
    return {
        'all_ratings_in_author_table': len(table), 'critical_ratings': len(data),
        'readers_by_language': table[readers].drop_duplicates().groupby('Language', observed=True).size().to_dict(),
        'published_text_variants': len(lookup), 'matched_observed_text_variants': len(matches),
        'variants_matching_after_trailing_whitespace_only': len(trailing_whitespace_matches),
        'stimulus_join': 'native language × item × embedding level × paraphrase. The published and raw list '
                         'indices are not interchangeable. RSVP underscore display-group markers are untouched.',
        'cell_difficulty_0_easy_100_hard': cells,
        'item_equal_weight_complete_cell_contrasts': paired_effects({key: float(np.mean(v)) for key, v in by_item.items()}),
    }


def run(raw, output):
    import rdata

    provenance = {}
    for name, (file_id, expected) in RELEASES.items():
        digest = hashlib.sha256((raw / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f'Unrecognized OSF artifact bytes: {name}')
        provenance[name] = {'download': f'https://osf.io/download/{file_id}/', 'sha256': digest}
    result = {
        'study': 'Yadav et al. (2025), Revisiting processing complexity of nested and cross-serial dependencies',
        'paper': 'https://doi.org/10.5070/G601120916',
        'project': 'https://api.osf.io/v2/nodes/qbc5f/',
        'license': {'name': 'CC BY 4.0', 'source': 'https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/'},
        'files': provenance,
        'difficulty': 'Observed continuous slider; author analysis rescales stored 1–9 as (response−1)*12.5',
        'experiment_1_full_sentence': audit_experiment(raw / 'experiment1-slider.rda', raw / 'experiment1-stimuli.txt', False, rdata),
        'experiment_2_ab_timed_rsvp': audit_experiment(raw / 'experiment2-all-slider.rda', raw / 'experiment2-stimuli.txt', True, rdata),
        'scope': 'Item-matched directional transfer diagnostic only. Languages and native reader populations covary '
                 'with dependency type. Ratings follow isolated stimuli and different scales than PLABA-EVAL. '
                 'Experiment 2 shares materials with Experiment 1 and controls exposure time through RSVP. '
                 'Matched cell means describe the retained responses and do not reproduce the mixed-effects regression.',
        'model_scoring': 'none',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ('raw', 'output'):
        cli.add_argument(f'--{key}', type=Path, required=True)
    run(**vars(cli.parse_args()))
