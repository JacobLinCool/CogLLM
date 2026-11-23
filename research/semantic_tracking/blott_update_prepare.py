"""Certify exact 2021 Blott cue prefixes against the released interest-area labels."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.semantic_tracking import blott_format_packet
from research.stage6 import blott_audit


ORIGINAL_HASHES = {
    'model_input/stimuli.jsonl': '281fb2fd37ba881ae49390c39a4204187d66a325fd722002c7e9714428c5f1fe',
    'scoring/design.jsonl': '20c8c68285fcdb3bbc7ad610264e6caaf154e1e7ea299948ef4452b1c83f0fcf',
    'scoring/human_trials.jsonl': '9b8a517bc8c12a5189bfb339e67659a42c203da763a9c997c94f445962b7b047',
}
ARTICLE = 'https://doi.org/10.1037/xlm0000936'
MANUSCRIPT = 'https://discovery.ucl.ac.uk/id/eprint/10101418/1/BlottRoddFerreiraWarren_JEPLMC_2020_AcceptedManuscript.pdf'


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def read_observed_areas(path, materials):
    trial_geometry = defaultdict(dict)
    variant_geometry = {}
    choices = {}
    readers = set()
    area_rows = 0
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            material = materials[row['sentence_text']]
            identifier = material.stimulus_id
            region = int(row['IA_ID'])
            label = row['IA_LABEL']
            reader = row['RECORDING_SESSION_LABEL']
            trial = (reader, row['TRIAL_INDEX'])
            if (row['item_label'] != material.item_label or row['cond_label'] != material.condition
                    or row['practice'] != 'experiment' or not label.endswith(' ')):
                raise ValueError('Observed area and published stimulus identity disagree')
            if region in trial_geometry[trial]:
                raise ValueError('Repeated interest area in a published reader trial')
            trial_geometry[trial][region] = label
            regions = variant_geometry.setdefault(identifier, {})
            if regions.setdefault(region, label) != label:
                raise ValueError('Interest-area text differs between readings of one version')
            button = row['BUTTON_PRESSED']
            expected = '4' if material.condition.startswith('Plaus') else '2'
            if (button not in {'2', '4'} or row['expected.button.press'] != expected
                    or int(row['accuracy']) != int(button == expected)):
                raise ValueError('Trial does not match the published corrected button decoding')
            choice = (identifier, button == '4')
            if choices.setdefault(trial, choice) != choice:
                raise ValueError('Interest-area rows disagree on a sentence-end judgment')
            readers.add(reader)
            area_rows += 1
    outcomes = {}
    for trial, by_region in trial_geometry.items():
        identifier, sense = choices[trial]
        if by_region != variant_geometry[identifier]:
            raise ValueError('A reader trial is missing an authored interest area')
        outcome_key = trial[0], identifier
        if outcome_key in outcomes:
            raise ValueError('Duplicate reader and version in original observed outcomes')
        outcomes[outcome_key] = sense
    if len(readers) != 96 or len(outcomes) != 4512 or area_rows != 26976:
        raise ValueError('Unexpected reader, trial or interest-area denominator')
    return variant_geometry, outcomes, area_rows


def observed_spans(text, regions):
    end = 0
    spans = []
    for label in regions:
        # The export's separator belongs to the display layout, outside each source span.
        start = end
        end = start + len(label)
        if text[start:min(end, len(text))] != label[:min(end, len(text)) - start]:
            raise ValueError('Interest-area substring disagrees with exact input text')
        spans.append([start, end - 1])
    if end != len(text) + 1 or ''.join(regions)[:-1] != text:
        raise ValueError('Interest-area labels do not partition the complete sentence')
    return spans


def prepare_rows(inputs, original_design, materials, areas):
    identifiers = {row['id'] for row in inputs}
    if (len(identifiers) != 188 or len(inputs) != 188 or identifiers != areas.keys()
            or identifiers != original_design.keys() or identifiers != materials.keys()):
        raise ValueError('Existing text IDs and observed source variants do not agree')
    grade_free, layout, by_family = [], [], defaultdict(list)
    for row in inputs:
        identifier, text = row['id'], row['text']
        material = materials[identifier]
        existing = original_design[identifier]
        if text != material.text or existing['family'] != material.family or existing['condition'] != material.condition:
            raise ValueError('Reconstructed source variant disagrees with an existing frozen pairing')
        by_region = areas[identifier]
        count = 5 if material.item_label == 'log' else 6
        if set(by_region) != set(range(1, count + 1)):
            raise ValueError('Unsupported experimental interest-area layout')
        spans = observed_spans(text, [by_region[i] for i in range(1, count + 1)])
        noun, cue, spillover = spans[1], None, None
        if count == 6:
            cue, spillover = spans[3:5]
        if (noun != existing['main_noun_span'] or cue != existing['coherence_cue_span']
                or spillover != existing['spillover_span']):
            raise ValueError('Reconstructed raw interest areas differ from the frozen design')
        sentence_stop = len(text)
        before_cue = len(text[:cue[0]].rstrip()) if cue is not None else None
        stops = [before_cue, cue[1], sentence_stop] if cue is not None else [sentence_stop]
        grade_free.append({'id': identifier, 'text': text, 'stops': stops})
        layout.append({'id': identifier, 'family': material.family, 'condition': material.condition,
                       'ambiguity': 'Amb' if material.condition.endswith('Amb') else 'UA',
                       'noun_span': noun, 'pre_noun_stop': len(text[:noun[0]].rstrip()),
                       'before_cue_stop': before_cue, 'cue_span': cue, 'spillover_span': spillover,
                       'sentence_stop': sentence_stop})
        by_family[material.family].append((material.condition, text[:noun[0]], text[noun[1]:],
                                           text[slice(*cue)] if cue is not None else None))
    for family in by_family.values():
        if (len(family) != 4 or {row[0] for row in family} != {key for key, _ in blott_audit.CONDITIONS}
                or len({row[1:] for row in family}) != 1):
            raise ValueError('Expected four word-substituted versions of the same sentence frame')
    return grade_free, layout


def report_counts(grade_free, layout, outcomes, packet):
    certificate = json.loads((packet / 'preflight.json').read_text())
    certified = {key: set(value) for key, value in certificate['documents'].items()}
    locations = {
        row['id']: {'before_cue': row['before_cue_stop'],
                    'cue_end': row['cue_span'][1] if row['cue_span'] is not None else None,
                    'sentence': row['sentence_stop']}
        for row in layout
    }
    counts = Counter()
    families, complete_families = defaultdict(set), defaultdict(set)
    for row in layout:
        identifier = row['id']
        for name, stop in locations[identifier].items():
            if stop is not None:
                counts['listed_' + name] += 1
                counts['certified_' + name] += stop in certified[identifier]
        families[row['family']].add(identifier)
        if None not in locations[identifier].values() and set(locations[identifier].values()) <= certified[identifier]:
            complete_families[row['family']].add(identifier)
    entirely_certified = [family for family, ids in families.items() if complete_families[family] == ids]
    cue_word_counts = Counter()
    cue_with_punctuation = 0
    for row, stimulus in zip(layout, grade_free, strict=True):
        if row['cue_span'] is not None:
            cue = stimulus['text'][slice(*row['cue_span'])]
            cue_word_counts[len(cue.split())] += 1
            cue_with_punctuation += any(mark in cue for mark in '.,;:!?')
    return {
        'families': len(families), 'sentences': len(layout), 'readers': len({reader for reader, _ in outcomes}),
        'sentence_end_judgments': len(outcomes), 'experiment_cue_word_counts': dict(cue_word_counts),
        'cues_with_sentence_punctuation': cue_with_punctuation, 'candidate_offsets': dict(counts),
        'source_offsets_requested': sum(len(row['stops']) for row in grade_free),
        'certified_offsets': sum(len(stops) for stops in certified.values()),
        'all_four_cue_pairs_certified_families': sorted(entirely_certified),
        'certifier_exclusion_counts': certificate['exclusion_counts_by_reason'],
    }


def prepare(source, tokenizer, output, report):
    if output.exists() or report.exists():
        raise FileExistsError('Blott cue update output or receipt already exists')
    old = source / 'blott-holdout'
    for relative, expected in ORIGINAL_HASHES.items():
        if blott_audit.sha256(old / relative) != expected:
            raise ValueError('The current source differs from the frozen Blott 2021 artifacts')
    material_file, trial_file = source / 'blott-materials.docx', source / 'blott-exp-trials.csv'
    blott_audit.verify(material_file, blott_audit.MATERIALS)
    blott_audit.verify(trial_file, blott_audit.TRIALS)
    materials = blott_audit.stimulus_table(material_file)
    areas, outcomes, area_rows = read_observed_areas(trial_file, materials)
    previous_outcomes = {(row['reader_id'], row['id']): row['coherent']
                         for row in read_jsonl(old / 'scoring/human_trials.jsonl')}
    if outcomes != previous_outcomes:
        raise ValueError('Exact original binary choices do not match the protected previous outcomes')
    original = {row['id']: row for row in read_jsonl(old / 'scoring/design.jsonl')}
    grade_free, layout = prepare_rows(read_jsonl(old / 'model_input/stimuli.jsonl'), original,
                                      {value.stimulus_id: value for value in materials.values()}, areas)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.blott-update-', dir=output.parent) as temporary:
        build = Path(temporary) / 'build'
        local = build / 'prepared'
        packet = build / 'packet'
        local.mkdir(parents=True)
        blott_audit.write_jsonl(local / 'stimuli.jsonl', grade_free)
        blott_audit.write_jsonl(local / 'layout.jsonl', layout)
        blott_format_packet.prepare(local / 'stimuli.jsonl', tokenizer, packet)
        outputs = ('prepared/stimuli.jsonl', 'prepared/layout.jsonl', 'packet/stimuli.jsonl',
                   'packet/preflight.json', 'packet/extractor-preflight.json')
        receipt = {
            'article': ARTICLE, 'author_manuscript': MANUSCRIPT,
            'project': blott_audit.PROJECT, 'license': blott_audit.LICENSE,
            'published_sources': {'materials': dict(zip(('url', 'sha256'), blott_audit.MATERIALS)),
                                  'trial_interest_areas': dict(zip(('url', 'sha256'), blott_audit.TRIALS))},
            'prior_frozen_sha256': ORIGINAL_HASHES, 'tokenizer_sha256': blott_audit.sha256(tokenizer),
            'artifact_sha256': {name: blott_audit.sha256(build / name) for name in outputs},
            'coordinates': 'Unicode code point offsets; region spans are half open, stops are exact prefix lengths.',
            'geometry': 'Reconstructed display regions from stable IA_ID and IA_LABEL in every original trial. The final layout separator is excluded. IA 2 is the main noun, IA 4 is the coherence cue, IA 5 is spillover.',
            'cue_interpretation': 'Internal disambiguating/coherence word documenting stimulus design. Explicit Sense/Nonsense was assessed at the whole-sentence endpoint only.',
            'published_exception': 'The log quartet has five regions and is excluded from the original specific cue/spillover analysis. It is retained here for sentence-end analysis; no cue location is assigned.',
            'family_match': 'Within each quartet only the noun region is replaced, and the remainder of the literal sentence, including the cue where defined, is identical.',
            'author_interest_area_rows_verified': area_rows,
            'inventory': report_counts(grade_free, layout, outcomes, packet),
        }
        serialized = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n'
        (local / 'source-audit.json').write_text(serialized, encoding='utf-8')
        build.rename(output)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x', encoding='utf-8') as stream:
        stream.write(serialized)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('source', 'tokenizer', 'output', 'report'):
        parser.add_argument(f'--{argument}', type=Path, required=True)
    result = prepare(**vars(parser.parse_args()))
    print(json.dumps(result['inventory'], sort_keys=True))
