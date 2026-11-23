"""Reconstruct Ceháková and Chromý's published Czech Experiments 2 and 3."""

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
from hashlib import file_digest, sha256
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from unicodedata import normalize

from research.semantic_tracking.blott_format_prepare import TOKEN


SOURCE_AUDIT = Path(__file__).resolve().parents[1] / 'results' / 'czech-garden-path-source-audit.json'
SOURCE_AUDIT_SHA256 = '6fdefc7d5c033e75dc214c10934285cf99662791a027c48011df6ce01a4e5698'
MATERIALS = 'experiments_2_and_3_items.csv'
RESPONSE_FILES = {2: 'results_exp2_whole_yes-no_analysis.csv', 3: 'results_exp3_wbw_yes-no_analysis.csv'}
CONDITIONS = ('gp', 'nongp')
ANIMACIES = ('animate', 'inanimate')
ANIMACY_ALIAS = {'anim': 'animate', 'inanim': 'inanimate'}
KEYS = {'qcor': 'yes', 'qmis': 'no', 'qpos': 'yes'}
RESPONSES = {'Ano': 'yes', 'Ne': 'no', 'Nevím': 'unsure', 'NA': 'missing'}
WORDS = 11
FAMILIES = 48


@dataclass(frozen=True)
class Material:
    family: str
    condition: str
    animacy: str
    text: str
    words: tuple[tuple[int, int], ...]
    document: str


@dataclass
class MovingWindowTrial:
    tags: tuple[str, str, str]
    question: str
    words: dict[int, str] = field(default_factory=dict)
    row_one: tuple[str, str] | None = None
    repeated_choices: set[tuple[str, str]] = field(default_factory=set)


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def source_files(source):
    if digest(SOURCE_AUDIT) != SOURCE_AUDIT_SHA256:
        raise ValueError('Expected the previously verified Czech source audit')
    audit = json.loads(SOURCE_AUDIT.read_text(encoding='utf-8'))
    expected = {name: audit['files'][name] for name in (MATERIALS, *RESPONSE_FILES.values())}
    for name, item in expected.items():
        path = source / name
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError(f'Unexpected bytes in the officially deposited source {name}')
    return audit, expected


def rows(path, delimiter=','):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream, delimiter=delimiter)


def tags(row):
    condition, animacy, question = (row[key] for key in ('SentType', 'ObjAnim', 'QuesType'))
    animacy = ANIMACY_ALIAS.get(animacy, animacy)
    if condition not in CONDITIONS or animacy not in ANIMACIES or question not in KEYS:
        raise ValueError('Unexpected sentence, animacy, or question category')
    return condition, animacy, question


def question_key(value):
    return ' '.join(normalize('NFC', value).split())


def load_materials(path):
    variants, index = {}, {}
    paired_types = defaultdict(set)
    for row in rows(path, ';'):
        condition, animacy, kind = tags(row)
        family, text, prompt = (row[name] for name in ('ItemNo', 'Sentence', 'Question'))
        words = tuple(match.span() for match in re.finditer(r'\S+', text))
        key = family, condition, animacy
        identifier = sha256(text.encode('utf-8')).hexdigest()
        material = Material(family, condition, animacy, text, words, identifier)
        signature = condition, animacy, kind, tuple(text.split()), question_key(prompt)
        if (not family.isdecimal() or len(words) != WORDS or row['CorrAnswer'] != KEYS[kind]
                or variants.setdefault(key, material) != material or signature in index
                or kind in paired_types[key]):
            raise ValueError('Each published variant must have one literal 11-word sentence and three distinct probes')
        index[signature] = material, prompt
        paired_types[key].add(kind)
    families = {material.family for material in variants.values()}
    if (len(families) != FAMILIES or set(variants) != {(family, condition, animacy) for family in families
                                                    for condition in CONDITIONS for animacy in ANIMACIES}
            or len({item.document for item in variants.values()}) != len(variants)
            or any(value != KEYS.keys() for value in paired_types.values())):
        raise ValueError('The literal source does not contain all 192 unique four-way variants')
    for family in families:
        for animacy in ANIMACIES:
            a, b = [variants[family, condition, animacy].text.split() for condition in CONDITIONS]
            if [i for i, (left, right) in enumerate(zip(a, b, strict=True), 1) if left != right] != [4]:
                raise ValueError('A published matched sentence changes more than the ambiguous fourth word')
    return tuple(variants[key] for key in sorted(variants)), index


def trial_identity(row):
    item = row['Item']
    if not item.startswith('item') or not item[4:].isdecimal() or not row['Participant']:
        raise ValueError('An export row lacks a stable reader and item key')
    return row['Participant'], item[4:]


def answer(row):
    return RESPONSES[row['Answer']], row['Correct']


def extract_trial_rows(path, experiment):
    evidence = Counter()
    if experiment == 2:
        for row in rows(path):
            evidence['literal_source_rows'] += 1
            yield trial_identity(row), tags(row), tuple(row['Sentence'].split()), row['Question'], answer(row)
    else:
        trials = {}
        for row in rows(path):
            identity, cell, question = trial_identity(row), tags(row), row['Question']
            position, word = int(row['WordNo']), row['Word']
            trial = trials.get(identity)
            if trial is None:
                trial = trials[identity] = MovingWindowTrial(cell, question)
            if (trial.tags != cell or trial.question != question or not 1 <= position <= WORDS
                    or position in trial.words or len(word.split()) != 1 or word.strip() != word):
                raise ValueError('An exported moving-window trial has inconsistent source words or question')
            trial.words[position] = word
            choice = answer(row)
            trial.repeated_choices.add(choice)
            if position == 1:
                trial.row_one = choice
            evidence['literal_source_rows'] += 1
            time = row['ReadTimes']
            if time not in ('', 'NA', 'NaN') and math.isfinite(float(time)):
                evidence['word_rows_with_numeric_rt'] += 1
                evidence['trials_with_numeric_disambiguating_word_rt'] += position == 7
        for key, trial in trials.items():
            if len(trial.words) != WORDS or trial.row_one is None:
                raise ValueError('A moving-window trial lacks a complete 11-word path or its WordNo=1 outcome')
            evidence['trials_with_all_11_word_positions'] += 1
            evidence['answer_fields_vary_across_word_rows'] += len(trial.repeated_choices) > 1
            rendered = tuple(trial.words[position] for position in range(1, WORDS + 1))
            yield key, trial.tags, rendered, trial.question, trial.row_one
    return evidence


def collect_experiment(source, experiment, materials, audited):
    collected, source_to_family, by_reader = {}, {}, defaultdict(set)
    response_counts, type_counts, qmis_counts = Counter(), Counter(), Counter()
    literal = 0
    stream = extract_trial_rows(source / RESPONSE_FILES[experiment], experiment)
    while True:
        try:
            identity, cell, words, prompt, (response, correct) = next(stream)
        except StopIteration as stop:
            provenance = stop.value
            break
        reader, item = identity
        condition, animacy, kind = cell
        matched = (*cell, words, question_key(prompt))
        if matched not in materials:
            raise ValueError('A released trial failed the exact word, question and condition crosswalk')
        variant, published_prompt = materials[matched]
        expected = ('NA' if response == 'missing' else ('yes' if response == KEYS[kind] else 'no'))
        if (identity in collected or variant.family in by_reader[reader] or correct != expected
                or source_to_family.setdefault(item, variant.family) != variant.family):
            raise ValueError('An author trial conflicts with the item bijection or published answer key')
        collected[identity] = variant, kind, response
        by_reader[reader].add(variant.family)
        response_counts[response] += 1
        type_counts[kind] += 1
        literal += prompt == published_prompt
        if kind == 'qmis' and response != 'missing':
            qmis_counts[condition, 'total'] += 1
            qmis_counts[condition, 'yes'] += response == 'yes'
    qmis = {key: [qmis_counts[key, 'yes'], qmis_counts[key, 'total']] for key in CONDITIONS}
    inventory = {
        'readers': len(by_reader), 'trials': len(collected), 'question_type_counts': dict(type_counts),
        'explicit_unsure': response_counts['unsure'], 'missing_answer': response_counts['missing'],
        'literal_word_sequence_and_question_join': literal, 'joins_after_question_normalization': len(collected) - literal,
        'qmis_yes_among_nonmissing': qmis,
    }
    if (any(inventory[key] != audited[key] for key in inventory)
            or any(len(values) != FAMILIES for values in by_reader.values())
            or len(source_to_family) != FAMILIES or len(set(source_to_family.values())) != FAMILIES):
        raise ValueError('The reconstructed cohort differs from the audited Czech public observations')
    if experiment == 3:
        counts = ('word_rows_with_numeric_rt', 'trials_with_numeric_disambiguating_word_rt',
                  'trials_with_all_11_word_positions')
        if (any(provenance[key] != audited[key] for key in counts)
                or provenance['literal_source_rows'] != audited['word_rows']
                or provenance['answer_fields_vary_across_word_rows'] != audited['trials_with_inconsistent_repeated_field']['Answer']):
            raise ValueError('Released word positions or WordNo=1 response audit differed')
    pseudonyms = {reader: f'e{experiment}-reader-{i:03d}' for i, reader in enumerate(sorted(by_reader), 1)}
    outcomes = [{'reader': pseudonyms[reader], 'id': variant.document, 'experiment': experiment, 'question': kind,
                 'response': response} for (reader, _), (variant, kind, response) in sorted(collected.items())]
    return outcomes, {**inventory, **provenance, 'response_counts': dict(response_counts)}, source_to_family


def encode_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def write_rows(path, records, *, private=False):
    if private:
        path.touch(mode=0o600, exist_ok=False)
    with path.open('w' if private else 'x', encoding='utf-8') as stream:
        for row in records:
            stream.write(encode_json(row) + '\n')


def write_materials(output, materials):
    layout, stimuli = [], []
    for item in materials:
        study_stops = {end for _, end in item.words}
        regex_stops = {match.end() for match in TOKEN.finditer(item.text)}
        stops = sorted(study_stops | regex_stops | {len(item.text)})
        stimuli.append({'id': item.document, 'text': item.text, 'stops': stops})
        layout.append({'id': item.document, 'family': item.family, 'condition': item.condition,
                       'animacy': item.animacy, 'words': item.words})
    write_rows(output / 'stimuli.jsonl', stimuli)
    write_rows(output / 'layout.jsonl', layout)
    return stimuli


def certify(prepared, packet, tokenizer, materials):
    from research.semantic_tracking.blott_format_packet import prepare

    prepare(prepared / 'stimuli.jsonl', tokenizer, packet)
    receipt = json.loads((packet / 'preflight.json').read_text(encoding='utf-8'))
    runtime = json.loads((packet / 'extractor-preflight.json').read_text(encoding='utf-8'))
    covered = Counter()
    for item in materials:
        stops = set(receipt['documents'][item.document])
        covered['eos'] += len(item.text) in stops
        for ordinal in (4, 6, 7, 8, 11):
            covered[f'word_{ordinal}'] += item.words[ordinal - 1][1] in stops
    return {'candidate_id': receipt['candidate_id'], 'tokenizer_sha256': runtime['tokenizer_sha256'],
            'documents': runtime['encoded_documents'], 'certified_stops': runtime['certified_stops'],
            'input_tokens': runtime['input_tokens'], 'max_input_tokens': runtime['max_input_tokens'],
            'exclusions_by_reason': receipt['exclusion_counts_by_reason'], 'study_stop_coverage': dict(covered)}


def prepare(source, tokenizer, output, receipt):
    if output.exists() or receipt.exists():
        raise FileExistsError('Use fresh Czech output and audit paths')
    previous, inputs = source_files(source)
    materials, index = load_materials(source / MATERIALS)
    exp2, audit2, map2 = collect_experiment(source, 2, index, previous['experiment_2'])
    exp3, audit3, map3 = collect_experiment(source, 3, index, previous['experiment_3'])
    if map2 != map3 or sum(source_id == material_id for source_id, material_id in map2.items()) != 2:
        raise ValueError('The two experiments do not share the audited 48-family material crosswalk')

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.czech-garden-') as temporary:
        stage = Path(temporary) / 'assay'
        prepared, packet = stage / 'prepared', stage / 'packet'
        prepared.mkdir(parents=True, mode=0o700)
        stimuli = write_materials(prepared, materials)
        write_rows(prepared / 'outcomes.jsonl', [*exp2, *exp3], private=True)
        certification = certify(prepared, packet, tokenizer, materials)
        result = {
            'schema': 1, 'project': previous['public_osf_project'], 'paper': previous['paper'],
            'source_audit_sha256': SOURCE_AUDIT_SHA256, 'source_files': inputs, 'code_sha256': digest(Path(__file__)),
            'license': {'paper': previous['paper_license'], 'osf_node': previous['osf_node_license'],
                        'osf_relationship': previous['osf_license_relationship']},
            'documents': len(materials), 'families': FAMILIES, 'words_per_sentence': WORDS,
            'material_text_changed': 0,
            'material_text_not_single_space_join': sum(item.text != ' '.join(item.text.split()) for item in materials),
            'requested_stops': sum(len(row['stops']) for row in stimuli), 'certification': certification,
            'family_crosswalk': {'pairs': len(map2), 'identity_pairs': 2,
                                 'sha256': sha256(json.dumps(map2, sort_keys=True).encode('utf-8')).hexdigest()},
            'experiment_2': audit2, 'experiment_3': audit3,
            'coordinates': 'Exact published material text, Unicode half-open spans for visible author words. Stops also include orthographic word ends and EOS.',
            'question_normalization': 'Only linkage keys use NFC and whitespace collapse. Display words and material text are unmodified.',
            'privacy': 'Only experiment-scoped reader aliases and categorical choices are retained. Layout, reader rows and raw OSF files are outside the inference packet.',
            'file_sha256': {str(path.relative_to(stage)): digest(path) for path in sorted(stage.glob('*/*'))},
        }
        audit_bytes = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        (prepared / 'source-audit.json').write_bytes(audit_bytes)
        stage.rename(output)
    with receipt.open('xb') as stream:
        stream.write(audit_bytes)
    return certification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'tokenizer', 'output', 'receipt'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    summary = prepare(**vars(parser.parse_args()))
    print(encode_json(summary))
