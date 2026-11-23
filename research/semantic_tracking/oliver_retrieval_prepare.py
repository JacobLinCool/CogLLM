"""Prepare exact written definitions and separate coded outcomes for Oliver Exp. 1."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

from research.semantic_tracking import blott_format_packet


PAPER = 'https://doi.org/10.1525/collabra.200'
DATASET = 'https://doi.org/10.5683/SP2/SP6BYO'
FILE = 'https://borealisdata.ca/api/access/datafile/75345'
FILE_BYTES = 1_167_967
FILE_SHA256 = '84632126243ff8b0229080247ce5e54ade20b67310d7ab26a8ef848f5ddf2fc2'
PHASE_FIELDS = (
    'Subject', 'Trial', 'item', 'Definition 1', 'Definition 2', 'target', 'T1', 'T2',
    'Defin_resp', 'Defin_RT', 'Delay_resp', 'Delay_RT', 'Answer_resp', 'Answer_RT',
    'Conf_resp', 'Conf_RT', 'Time1or2', 'ListNum', 'Def1or2',
)
DERIVED_FIELDS = ('Check', 'Response1', 'Response2', 'R1Valid', 'R2Valid', 'Condition', 'Summary Response')
STATES = {'1': 'know', '2': 'dont_know', '3': 'tot'}
QUARANTINED_ITEM = 'd33'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def text_sha256(text):
    return sha256(text.encode('utf-8'))


def source_bytes(local_source):
    source = local_source.open('rb') if local_source else urlopen(FILE, timeout=25)
    with source:
        raw = source.read(FILE_BYTES + 1)
    if len(raw) != FILE_BYTES or sha256(raw) != FILE_SHA256:
        raise ValueError('Expected the exact pinned Borealis Experiment 1 file')
    return raw


def positive_rt(value):
    result = int(value)
    if result <= 0:
        raise ValueError('Expected a positive observed reaction time')
    return result


def phase_outcome(phase, summary, valid, stimulus_id):
    state = phase['Defin_resp']
    confirmation = phase['Conf_resp']
    if state != summary or state not in STATES or confirmation not in {'4', '5'}:
        raise ValueError('Unexpected source state or confirmation code')
    if valid not in {'valid', 'invalid'}:
        raise ValueError('Unexpected source validity code')
    is_valid = valid == 'valid'
    confirmed = confirmation == '4'
    if state in {'1', '3'} and is_valid != confirmed:
        raise ValueError('Known/TOT confirmation contradicts the author validity rule')
    return {
        'id': stimulus_id,
        'state': STATES[state],
        'definition_rt': positive_rt(phase['Defin_RT']),
        'same_target_confirmed': confirmed,
        'confirmation_rt': positive_rt(phase['Conf_RT']),
        'source_valid': is_valid,
    }


def first_delay(phase, summary):
    state = phase['Defin_resp']
    code = phase['Delay_resp'].strip()
    timing = phase['Delay_RT'].strip()
    summary = summary.strip()
    if state == '1':
        if code or timing or summary != '1':
            raise ValueError('Initial KNOW unexpectedly entered the delay')
        return None, None
    if code not in {'0', '1'} or summary != state + code:
        raise ValueError('Delay key disagrees with the author summary')
    delay_rt = int(timing)
    pressed = code == '1'
    if (pressed and not 0 < delay_rt <= 15_000) or (not pressed and delay_rt != 0):
        raise ValueError('Delay timing disagrees with the 15-second source protocol')
    return pressed, delay_rt


def parse(raw):
    data = csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    width = len(PHASE_FIELDS)
    if tuple(next(data)) != PHASE_FIELDS * 2 + DERIVED_FIELDS:
        raise ValueError('Unexpected duplicate-phase Experiment 1 CSV schema')

    readers = {}
    targets = {}
    exact_versions = {}
    joined = defaultdict(set)
    pair_rows = []
    source_conditions = Counter()
    for row in data:
        if len(row) != width * 2 + len(DERIVED_FIELDS):
            raise ValueError('Unexpected row width')
        first = dict(zip(PHASE_FIELDS, row[:width], strict=True))
        second = dict(zip(PHASE_FIELDS, row[width:width * 2], strict=True))
        derived = dict(zip(DERIVED_FIELDS, row[width * 2:], strict=True))
        reader, item, target = (first[field] for field in ('Subject', 'item', 'target'))
        if not reader or not item or not target or any(first[field] != second[field] for field in ('Subject', 'item', 'target')):
            raise ValueError('Test 1/Test 2 reader/item join failed')
        if item in joined[reader] or targets.setdefault(item, target) != target:
            raise ValueError('Reader/item is duplicated or authored target changed')
        joined[reader].add(item)
        reader_id = readers.setdefault(reader, f'reader-{len(readers) + 1:03d}')
        family = text_sha256(f'oliver-2019-exp1-item:{item}:{target}')
        version_keys = first['Def1or2'], second['Def1or2']
        if version_keys != (first['T1'].title(), second['T2'].title()) or any(key not in {'Def1', 'Def2'} for key in version_keys):
            raise ValueError('Realized definition disagrees with test-specific source selector')
        if second['Delay_resp'].strip() or second['Delay_RT'].strip():
            raise ValueError('Unexpected Test 2 recall delay')
        condition = int(derived['Condition'])
        if condition not in {1, 2, 3, 4}:
            raise ValueError('Unexpected condition code')
        source_conditions[(condition, *version_keys)] += 1

        ids = []
        for phase, version_key in zip((first, second), version_keys, strict=True):
            version = int(version_key[-1])
            text = phase[f'Definition {version}']
            if item == QUARANTINED_ITEM:
                ids.append(None)
                continue
            if not text.strip() or '\t' in text or '\n' in text or '\r' in text:
                raise ValueError('Malformed literal definition outside the quarantined item')
            if exact_versions.setdefault((family, version), text) != text:
                raise ValueError('Author definition changes for one item/version')
            ids.append(text_sha256(text))

        first_outcome = phase_outcome(first, derived['Response1'], derived['R1Valid'], ids[0])
        second_outcome = phase_outcome(second, derived['Response2'], derived['R2Valid'], ids[1])
        pressed, delay_rt = first_delay(first, derived['Summary Response'])
        first_outcome.update({
            'recall_signaled_before_reveal': pressed,
            'delay_rt': delay_rt,
            'spoken_equals_target_before_reveal': None,
        })
        pair_rows.append({
            'reader_id': reader_id,
            'family': family,
            'same_written_definition': version_keys[0] == version_keys[1],
            'first': first_outcome,
            'second': second_outcome,
        })

    if len(readers) != 52 or len(targets) != 44 or any(items != set(targets) for items in joined.values()):
        raise ValueError('Expected a complete 52 × 44 grid with unique reader/item pairs')
    if len(exact_versions) != 86 or len(set(exact_versions.values())) != 86:
        raise ValueError('Expected 86 unique exact definitions across 43 usable item families')
    rows = sorted(pair_rows, key=lambda row: (row['reader_id'], row['family']))
    stimuli = []
    layout = []
    for (family, version), text in sorted(exact_versions.items()):
        stimulus_id = text_sha256(text)
        stimuli.append({'id': stimulus_id, 'text': text, 'stops': [len(text)]})
        layout.append({'id': stimulus_id, 'family': family, 'version': version, 'definition_stop': len(text)})
    target_rows = [
        {'family': text_sha256(f'oliver-2019-exp1-item:{item}:{target}'), 'target': target}
        for item, target in sorted(targets.items())
    ]
    conditions = [
        {'source_condition': condition, 'first_version': first, 'second_version': second, 'pairs': count}
        for (condition, first, second), count in sorted(source_conditions.items())
    ]
    return stimuli, layout, target_rows, rows, conditions


def summarize(rows):
    counts = {}
    for name in ('first', 'second'):
        phase = [row[name] for row in rows]
        counts[name] = {
            'states': dict(Counter(row['state'] for row in phase)),
            'explicit_target_confirmations': len(phase),
            'explicit_target_yes': sum(row['same_target_confirmed'] for row in phase),
            'author_valid': sum(row['source_valid'] for row in phase),
        }
    counts['first']['delay_participation'] = sum(row['first']['recall_signaled_before_reveal'] is not None for row in rows)
    counts['first']['delay_know_signals'] = sum(row['first']['recall_signaled_before_reveal'] is True for row in rows)
    counts['both_author_valid'] = sum(row['first']['source_valid'] and row['second']['source_valid'] for row in rows)
    counts['same_realized_definition'] = sum(row['same_written_definition'] for row in rows)
    counts['alternate_realized_definition'] = len(rows) - counts['same_realized_definition']
    counts['paired_reader_items'] = len(rows)
    return counts


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def prepare(output, tokenizer, report, local_source=None):
    if output.exists() or report.exists():
        raise FileExistsError('Expected fresh output and report paths')
    stimuli, layout, targets, rows, conditions = parse(source_bytes(local_source))
    literal_rows = [row for row in rows if row['first']['id'] is not None and row['second']['id'] is not None]
    if len(literal_rows) != 2236:
        raise ValueError('Unexpected exact literal pair count')
    quarantined = [row for row in rows if row['first']['id'] is None or row['second']['id'] is None]
    if len({row['family'] for row in quarantined}) != 1 or any(row['first']['id'] is not None or row['second']['id'] is not None for row in quarantined):
        raise ValueError('Expected one fully quarantined family, with both stage IDs missing')

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.oliver-prepare-') as temporary:
        staged = Path(temporary) / 'dataset'
        prepared = staged / 'prepared'
        prepared.mkdir(parents=True)
        files = {'stimuli': stimuli, 'layout': layout, 'targets': targets, 'outcomes': rows}
        for name, values in files.items():
            write_jsonl(prepared / f'{name}.jsonl', values)
        packet = staged / 'packet'
        blott_format_packet.prepare(prepared / 'stimuli.jsonl', tokenizer, packet)
        preflight = json.loads((packet / 'extractor-preflight.json').read_text(encoding='utf-8'))
        receipt = {
            'source': {
                'paper': PAPER,
                'dataset': DATASET,
                'version': '3.0',
                'repository_license': 'CC0-1.0',
                'file_url': FILE,
                'file_bytes': FILE_BYTES,
                'file_sha256': FILE_SHA256,
                'materials_appendix': 'https://doi.org/10.1525/collabra.200.s1',
                'definition_provenance': 'Authors credit Warriner and Humphreys (2008) for original definitions and WordNet for adapted definitions.',
            },
            'assay': {
                'model_input': 'One exact written author definition per document, full-source Unicode endpoint. No revealed target, assignments, instructions, timings, grades or speaker identifiers in packet.',
                'observed_reveal': 'Intended target revealed after first attempt. Reveal modality or exact screen transcript unreported. Confirmations are explicit post-reveal yes/no reports of sought-word identity.',
                'first_delay': 'Only initial TOT and do-not-know enter the 15-second delay. The exported delayed KNOW key is self-report. Recorded spoken guesses are absent, so equality with the target is missing.',
                'second_trial': 'Retest after five-minute break with same or alternative written definition, following prior intended-target exposure. The realized selector fields determine same versus alternate.',
                'timing': 'Integer raw reaction-time units. Positive delay values fit the 15-second programmed window on a millisecond scale. Zero denotes a timeout in first-delay RT.',
                'split_unit': 'Author item/target family, keeping the two definitions and both stages together.',
                'scope': 'Elicited lexical retrieval and target recognition from an English definition, with only whole-definition model alignment.',
            },
            'readers': 52,
            'author_families': 44,
            'literal_families': 43,
            'literal_definitions': len(stimuli),
            'exact_definitions_with_edge_whitespace': sum(row['text'] != row['text'].strip() for row in stimuli),
            'quarantine': {
                'item': QUARANTINED_ITEM,
                'target': 'chameleon',
                'pairs': len(quarantined),
                'reason': 'One definition field contains two prompts separated by a tab. The alternate field is blank and two paired Test 2 records have inconsistent fields. No reconstruction.',
            },
            'all_rows': summarize(rows),
            'literal_rows': summarize(literal_rows),
            'source_condition_by_realized_versions': conditions,
            'paper_table_one': {
                'paper_valid_trials': 1686,
                'strict_source_both_valid': summarize(rows)['both_author_valid'],
                'scope': 'Raw paired validity flags are retained. Printed Table 1 and exported flags give different denominators.',
            },
            'prepared_sha256': {
                name: sha256((prepared / f'{name}.jsonl').read_bytes()) for name in files
            },
            'packet_sha256': {path.name: sha256(path.read_bytes()) for path in sorted(packet.iterdir())},
            'tokenizer_sha256': preflight['tokenizer_sha256'],
            'preparer_sha256': sha256(Path(__file__).read_bytes()),
        }
        write_json(prepared / 'source-audit.json', receipt)
        staged.rename(output)
    report.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, receipt)
    print(json.dumps({'literal_pairs': len(literal_rows), 'documents': len(stimuli), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('output', 'tokenizer', 'report'):
        parser.add_argument(f'--{field}', type=Path, required=True)
    parser.add_argument('--local-source', type=Path)
    prepare(**vars(parser.parse_args()))
