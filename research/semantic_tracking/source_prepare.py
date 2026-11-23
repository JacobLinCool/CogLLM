"""Prepare prior-source localization for publicly released SCARECROW judgments."""

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata


RELEASE = 'f3f4fffc23596af6840b973ddb7fca8e03d992ab'
CSV_SHA256 = 'bc0867107dbefddfe43a6d8dd0ceea797ce225bb90b01c73416bf5e4e7ed054c'
TOKENIZER_SHA256 = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
REPO = 'https://github.com/Yao-Dou/scarecrow'
SOURCE = f'https://raw.githubusercontent.com/Yao-Dou/scarecrow/{RELEASE}/data/grouped_data.csv'
WORD = re.compile(r'\S+')
SEED = 43


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def utf16_to_characters(text):
    indices = {0: 0}
    offset = 0
    for i, char in enumerate(text, 1):
        offset += 1 + (ord(char) >= 0x10000)
        indices[offset] = i
    return indices


def convert_span(text, mapping, start, stop):
    if start not in mapping or stop not in mapping or start >= stop:
        return None
    a, b = mapping[start], mapping[stop]
    segment = text[a:b]
    trimmed_start = a + len(segment) - len(segment.lstrip())
    trimmed_stop = a + len(segment.rstrip())
    if trimmed_start >= trimmed_stop:
        return None
    return trimmed_start, trimmed_stop


def certified_ends(text, ends, tokenizer):
    full = tokenizer.encode(text, add_special_tokens=False)
    end_positions = {stop: i for i, (start, stop) in enumerate(full.offsets) if start < stop}
    return {
        end for end in ends if end in end_positions
        and full.ids[:end_positions[end] + 1] == tokenizer.encode(text[:end], add_special_tokens=False).ids
    }


def stage_material(raw, tokenizer, stats):
    text = raw['generation']
    words = [match.span() for match in WORD.finditer(text)]
    mapping = utf16_to_characters(text)
    observations = []
    for reader_index, response in enumerate(ast.literal_eval(raw['responses'])):
        for annotation_index, annotation in enumerate(response):
            category, _, severity, start, stop, source = annotation
            if category != 'Self-contradiction':
                continue
            stats['self_contradiction_rows'] += 1
            if len(source) != 2:
                stats['missing_or_discontinuous_source'] += 1
                continue
            marked = convert_span(text, mapping, start, stop)
            antecedent = convert_span(text, mapping, *source)
            if marked is None or antecedent is None:
                stats['invalid_released_character_offset'] += 1
                continue
            if antecedent[1] > marked[0]:
                stats['source_not_wholly_before_marked_issue'] += 1
                continue
            if not 1 <= severity <= 3:
                raise ValueError(f'Unexpected severity for release row {raw["id"]}')
            observations.append((reader_index, annotation_index, severity, marked, antecedent))
    if not observations:
        return None

    ends = {stop for _, stop in words} | {marked[1] for *_, marked, antecedent in observations}
    certified = certified_ends(text, ends, tokenizer)
    retained_words = [(start, stop) for start, stop in words if stop in certified]
    stats['initial_word_candidates'] += len(words)
    stats['word_candidates_at_exact_causal_ends'] += len(retained_words)
    outcomes = []
    stops = {stop for _, stop in retained_words}
    for reader, annotation, severity, marked, antecedent in observations:
        if marked[1] not in certified:
            stats['issue_end_failed_token_alignment'] += 1
            continue
        earlier = [(a, b) for a, b in retained_words if b <= marked[0]]
        if len(earlier) < 2:
            stats['fewer_than_two_earlier_words'] += 1
            continue
        target = [b for a, b in earlier if a < antecedent[1] and b > antecedent[0]]
        if not target:
            stats['no_certified_word_overlaps_source'] += 1
            continue
        if len(target) == len(earlier):
            stats['every_candidate_overlaps_source'] += 1
            continue
        stops.add(marked[1])
        outcomes.append({
            'id': f'scarecrow-{raw["id"]}', 'marked_start': marked[0], 'stop': marked[1],
            'source_start': antecedent[0], 'source_stop': antecedent[1],
            'source_candidates': [b for _, b in earlier], 'candidate_spans': earlier,
            'annotation_id': f'{raw["id"]}-{reader}-{annotation}', 'severity': severity,
        })
    if not outcomes:
        return None
    return {'id': f'scarecrow-{raw["id"]}', 'text': text, 'stops': sorted(stops)}, outcomes


def assign_groups(materials):
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            key = parent[key]
        return key

    seen = {}
    for raw, _, _ in materials:
        key = normalized(raw['prompt'])
        if not key:
            raise ValueError('Empty source-prompt group key')
        generation = normalized(raw['generation'])
        linked = find(seen.setdefault(generation, key))
        root = find(key)
        parent[max(root, linked)] = min(root, linked)

    groups = {raw['id']: 'scarecrow-prompt-' + hashlib.sha256(find(normalized(raw['prompt'])).encode()).hexdigest()[:16]
              for raw, _, _ in materials}
    universe = sorted(set(groups.values()), key=lambda group: hashlib.sha256(f'{SEED}:{group}'.encode()).digest())
    test_groups = set(universe[:max(1, len(universe) // 5)])
    return groups, test_groups


def dump_rows(path, records):
    with path.open('w', encoding='utf-8') as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')


def prepare(raw_path, tokenizer_path, output):
    from tokenizers import Tokenizer

    if digest(raw_path) != CSV_SHA256 or digest(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError('Unexpected author release or Qwen tokenizer hash')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    with raw_path.open(encoding='utf-8', newline='') as file:
        raw = list(csv.DictReader(file))
    if len(raw) != 1308 or len({row['id'] for row in raw}) != len(raw):
        raise ValueError('Unexpected SCARECROW publication inventory')
    counts, materials = Counter(), []
    for row in raw:
        if row['model'] == 'human':
            counts['excluded_human_news_continuations'] += 1
            continue
        result = stage_material(row, tokenizer, counts)
        if result is not None:
            materials.append((row, *result))
    groups, test_groups = assign_groups(materials)
    grouped = defaultdict(list)
    stimuli, outcomes = [], []
    for row, document, labels in materials:
        group = groups[row['id']]
        split = 'test' if group in test_groups else 'train'
        stimuli.append(document)
        grouped[split].append((group, document, labels))
        outcomes.extend({**label, 'group_id': group, 'split': split} for label in labels)
    if set(grouped) != {'train', 'test'}:
        raise ValueError('Expected both train and test prompt groups')

    output.mkdir(parents=True, exist_ok=False)
    stimuli_path, outcomes_path = output / 'stimuli.jsonl', output / 'outcomes.jsonl'
    dump_rows(stimuli_path, stimuli)
    dump_rows(outcomes_path, outcomes)
    inventory = {
        split: {
            'prompt_groups': len({group for group, _, _ in rows}),
            'stimuli': len(rows), 'annotation_links': sum(len(labels) for _, _, labels in rows),
            'unique_marked_cutoffs': sum(len({(label['marked_start'], label['stop']) for label in labels}) for _, _, labels in rows),
        }
        for split, rows in sorted(grouped.items())
    }
    audit = {
        'study': 'https://aclanthology.org/2022.acl-long.501.pdf#page=5',
        'author_release': 'https://yao-dou.github.io/scarecrow/',
        'pinned_csv': SOURCE, 'csv_sha256': CSV_SHA256, 'tokenizer_sha256': TOKENIZER_SHA256,
        'license_status': 'Explicit public research release by the authors, with no separate repository license stated.',
        'selection_scope': 'Only model-generated continuations. The original human news prompt is used locally for grouping.',
        'unit': 'Retrospective human Self-contradiction annotation with one identifiable prior contiguous source.',
        'target_scope': 'Potential earlier words come from the continuation up to the marked issue beginning, independent of its chosen source.',
        'alignment': 'Released UTF-16 offsets mapped to Python Unicode character intervals, trimming boundary whitespace. Word stops must be exact Qwen causal ends.',
        'split_rule': 'Split normalized prompt groups by fixed hash, coalescing identical normalized generated continuations.',
        'seed': SEED,
        'raw_inventory': len(raw), 'counts': dict(counts), 'split': inventory,
        'output_sha256': {path.name: digest(path) for path in (stimuli_path, outcomes_path)},
    }
    (output / 'audit.json').write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('csv', 'tokenizer', 'output'):
        parser.add_argument(f'--{key}', required=True, type=Path)
    args = parser.parse_args()
    prepare(args.csv, args.tokenizer, args.output)
