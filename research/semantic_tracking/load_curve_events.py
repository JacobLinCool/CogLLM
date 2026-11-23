"""Prepare archived human span and sentence observations for load-curve scoring."""

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.semantic_tracking import blott_format_prepare as blott
from research.semantic_tracking import scarecrow_detection_prepare as scarecrow
from research.semantic_tracking import source_prepare as source


RESULTS = Path(__file__).resolve().parents[1] / 'results'
SCARECROW_TASK = 'scarecrow_contradiction_coverage'
BLOTT_TASK = 'blott_sentence_sense'
SCARECROW_TRAIN = 275
SCARECROW_TEST = 85
SCARECROW_TOKEN_CAP = 45_000
BLOTT_TEST_FAMILIES = 11


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def index_rows(rows):
    indexed = {row['id']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate archived document ID')
    return indexed


def rank(namespace, value):
    return hashlib.sha256(f'{namespace}:{value}'.encode()).digest()


def read_prepared(directory, digests):
    data = {}
    for stem, digest in digests.items():
        path = directory / f'{stem}.jsonl'
        if source.digest(path) != digest:
            raise ValueError(f'Archived data does not match the reviewed checksum for {path.name}')
        data[stem] = read_jsonl(path)
    return data


def aligned_rows(stimuli, layout, outcomes):
    documents = index_rows(stimuli)
    geometry = index_rows(layout)
    if documents.keys() != geometry.keys() or documents.keys() != outcomes.keys():
        raise ValueError('Archived stimuli, layout and observed outcomes do not join exactly')
    return documents, geometry


def select_scarecrow(documents, layout, tokenizer):
    token_counts = {
        identifier: len(tokenizer.encode(row['text'], add_special_tokens=False).ids)
        for identifier, row in documents.items()
    }
    ranked = defaultdict(list)
    group_splits = {}
    for identifier, row in layout.items():
        split = row['split']
        if split not in {'train', 'test'} or group_splits.setdefault(row['group_id'], split) != split:
            raise ValueError('A SCARECROW prompt group changes archival split')
        ranked[split].append(identifier)
    for identifiers in ranked.values():
        identifiers.sort(key=lambda identifier: rank('load-curve-scarecrow-v1', identifier))
    selected = ranked['test'][:SCARECROW_TEST]
    if len(selected) != SCARECROW_TEST:
        raise ValueError('Too few held-out continuations in the approved archive')
    used = sum(token_counts[identifier] for identifier in selected)
    for identifier in ranked['train'][:SCARECROW_TRAIN]:
        size = token_counts[identifier]
        if used + size > SCARECROW_TOKEN_CAP:
            break
        selected.append(identifier)
        used += size
    if len(selected) < 350:
        raise ValueError('The fixed size and token limits no longer admit the archival cohort')
    return selected, used


def explicit_anchors(text, responses):
    mapping = source.utf16_to_characters(text)
    by_connection = defaultdict(Counter)
    for response in responses:
        resolved = {}
        uncertain = []
        unlocatable = False
        for category, _, severity, start, end, references in response:
            if category != scarecrow.CATEGORY:
                continue
            target = source.convert_span(text, mapping, start, end)
            if target is None:
                unlocatable = True
                break
            if not references or len(references) % 2:
                uncertain.append(target)
                continue
            antecedents = tuple(source.convert_span(text, mapping, left, right)
                                for left, right in zip(references[::2], references[1::2], strict=True))
            if any(span is None or span[1] > target[0] for span in antecedents):
                uncertain.append(target)
                continue
            key = target, tuple(sorted(set(antecedents)))
            resolved[key] = max(severity, resolved.get(key, 0))
        if unlocatable:
            continue
        for (target, antecedents), severity in resolved.items():
            if not any(scarecrow.overlaps(target, span) for span in uncertain):
                by_connection[target, antecedents][severity] += 1
    return [
        {'span': target, 'sources': sources, 'votes': grades.total(),
         'severity_counts': [grades[grade] for grade in (1, 2, 3)]}
        for (target, sources), grades in sorted(by_connection.items())
    ]


def scarecrow_events(raw_path, prepared, tokenizer):
    if source.digest(raw_path) != source.CSV_SHA256:
        raise ValueError('Expected the exact approved SCARECROW grouped release')
    receipt = scarecrow.read_json(RESULTS / 'scarecrow-detection-source-audit.json')
    if receipt['source_sha256'] != source.CSV_SHA256:
        raise ValueError('SCARECROW source ledger uses a different archive')
    data = read_prepared(prepared, receipt['prepared_sha256'])
    outcomes = index_rows(data['outcomes'])
    documents, layout = aligned_rows(data['stimuli'], data['layout'], outcomes)
    selected, token_count = select_scarecrow(documents, layout, tokenizer)
    selected_ids = set(selected)
    with raw_path.open(encoding='utf-8', newline='') as stream:
        rows = {f'scarecrow-{row["id"]}': row for row in csv.DictReader(stream)
                if f'scarecrow-{row["id"]}' in selected_ids}
    labels, packet = [], []
    for identifier in selected:
        body = documents[identifier]['text']
        geometry = layout[identifier]
        words = geometry['words']
        row = rows[identifier]
        if row['model'] == 'human' or row['generation'] != body:
            raise ValueError('Packet text must be an exact archived machine-generated continuation')
        responses = ast.literal_eval(row['responses'])
        counts, has_raw_marks = scarecrow.coverage(body, words, responses, Counter())
        if counts != outcomes[identifier]['counts']:
            raise ValueError('SCARECROW eligible word coverage has changed')
        anchors = explicit_anchors(body, responses)
        observations = [
            {'span': span, 'stop': span[1],
             'counts': [votes['total'] - votes['positive'], *votes['severity_counts']]}
            for span, votes in zip(words, counts, strict=True) if votes['total'] > 0
        ]
        packet.append({'id': identifier, 'text': body})
        labels.append({
            'id': identifier, 'group_id': geometry['group_id'], 'split': geometry['split'],
            'task': SCARECROW_TASK, 'observations': observations, 'anchors': anchors,
            'prior_pointer_group': geometry['prior_pointer_group'],
            'raw_self_contradiction_mark_present': has_raw_marks,
        })
    audit = {
        'paper': receipt['paper'], 'source_url': receipt['source_url'],
        'source_sha256': source.CSV_SHA256, 'prepared_sha256': receipt['prepared_sha256'],
        'release_scope': receipt['release_scope'], 'token_count': token_count,
        'cohort_rule': 'Keep the archived train/test prompt partition. Within each split rank '
                       'document IDs by SHA-256 of load-curve-scarecrow-v1:<id>. Keep the first '
                       '85 test and up to 275 train documents within the 45000-token cap. '
                       'Sampling does not use annotation outcomes or remove unmarked controls.',
        'endpoint': receipt['endpoint'], 'risk': receipt['risk'], 'severity': receipt['severity'],
        'anchor_scope': 'Retrospective explicit target and earlier antecedent span(s), counted '
                        'at most once per annotator and connection. Annotators saw the news prompt '
                        'plus the complete continuation. These marks localize reported text spans '
                        'without measuring when a reader first noticed a contradiction.',
        'trial_split_scope': 'Frozen archival prompt families from prior project analyses.',
    }
    return packet, labels, audit


def blott_events(prepared):
    receipt = scarecrow.read_json(RESULTS / 'blott-format-source-audit.json')
    if receipt['license'] != blott.LICENSE or receipt['project'] != blott.PROJECT:
        raise ValueError('Expected the cleared Blott Experiment 1 release')
    data = read_prepared(prepared, {Path(name).stem: digest for name, digest in receipt['prepared_sha256'].items()})
    counts = defaultdict(Counter)
    seen = set()
    for trial in data['outcomes']:
        key = trial['id'], trial['reader_id']
        if type(trial['sense']) is not bool or key in seen:
            raise ValueError('Blott must contain one verified explicit response per reader and sentence')
        seen.add(key)
        counts[trial['id']][trial['sense']] += 1
    documents, layout = aligned_rows(data['stimuli'], data['layout'], counts)
    paired = defaultdict(list)
    for row in layout.values():
        paired[row['family']].append(row['ambiguity'])
    if len(paired) != 44 or any(sorted(arms) != ['Amb', 'UA'] for arms in paired.values()):
        raise ValueError('Expected the published, overlap-screened 44 Blott version pairs')
    held = set(sorted(paired, key=lambda family: rank('load-curve-blott-2023-v1', family))[:BLOTT_TEST_FAMILIES])
    labels, packet = [], []
    for identifier, row in documents.items():
        text = row['text']
        family = layout[identifier]['family']
        cue = layout[identifier]['cue_span']
        if layout[identifier]['sentence_stop'] != len(text) or not 0 <= cue[0] < cue[1] <= len(text):
            raise ValueError('Blott cue or boundary does not align to the displayed sentence')
        packet.append({'id': identifier, 'text': text})
        labels.append({
            'id': identifier, 'group_id': f'blott-2023-{family}',
            'split': 'test' if family in held else 'train', 'task': BLOTT_TASK,
            'observations': [{'span': [0, len(text)], 'stop': len(text),
                              'counts': [counts[identifier][False], counts[identifier][True]]}],
            'anchors': [], 'cues': [{'span': cue, 'origin': 'author_defined_stimulus'}],
            'ambiguity': layout[identifier]['ambiguity'],
        })
    audit = {
        'paper': receipt['paper'], 'project': blott.PROJECT, 'license': blott.LICENSE,
        'prepared_sha256': receipt['prepared_sha256'], 'source_files': receipt['source_files'],
        'endpoint': 'Explicit Sense/Nonsense response after viewing the complete written sentence. '
                    'Counts include only eligible, explicit published trials. The authored final cue '
                    'is a stimulus coordinate, not a report during reading.',
        'split': 'The first 11 families by SHA-256 of load-curve-blott-2023-v1:<family> are test. '
                 'Each family keeps both versions together. The archive was previously analyzed '
                 'in this project. This is an item-family split, with reader overlap allowed.',
    }
    return packet, labels, audit


def inventory(labels):
    totals = defaultdict(Counter)
    groups = defaultdict(set)
    for row in labels:
        key = row['task'], row['split']
        totals[key]['documents'] += 1
        totals[key]['observations'] += len(row['observations'])
        totals[key]['eligible_responses'] += sum(sum(obs['counts']) for obs in row['observations'])
        totals[key]['anchors'] += len(row['anchors'])
        totals[key]['anchor_votes'] += sum(anchor['votes'] for anchor in row['anchors'])
        if row['task'] == SCARECROW_TASK:
            totals[key]['positive_word_votes'] += sum(sum(obs['counts'][1:]) for obs in row['observations'])
            totals[key]['documents_without_raw_mark'] += not row['raw_self_contradiction_mark_present']
        groups[key].add(row['group_id'])
    return {
        task: {split: {'groups': len(groups[task, split]), **totals[task, split]} for split in ('train', 'test')}
        for task in sorted({task for task, _ in totals})
    }


def prepare(scarecrow_raw, scarecrow_prepared, blott_prepared, tokenizer_path, output):
    from tokenizers import Tokenizer

    if output.exists():
        raise FileExistsError(output)
    if source.digest(tokenizer_path) != source.TOKENIZER_SHA256:
        raise ValueError('Expected the pinned Qwen tokenizer used for the extraction budget')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    stimuli, labels, scarecrow_audit = scarecrow_events(scarecrow_raw, scarecrow_prepared, tokenizer)
    blott_stimuli, blott_labels, blott_audit = blott_events(blott_prepared)
    stimuli.extend(blott_stimuli)
    labels.extend(blott_labels)
    if index_rows(stimuli).keys() != index_rows(labels).keys():
        raise ValueError('Grade-free packet and local labels must share exactly the same IDs')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.load-curve-events-') as temporary:
        stage = Path(temporary) / 'events'
        packet = stage / 'packet'
        packet.mkdir(parents=True)
        files = {
            'packet/stimuli.jsonl': sorted(stimuli, key=lambda row: row['id']),
            'labels.jsonl': sorted(labels, key=lambda row: row['id']),
        }
        for name, rows in files.items():
            scarecrow.write_jsonl(stage / name, rows)
        scarecrow.write_json(stage / 'source-audit.json', {
            'schema_version': 'load-curve-events-v1',
            'tokenizer_sha256': source.TOKENIZER_SHA256,
            'files_sha256': {name: source.digest(stage / name) for name in files},
            'inventory': inventory(labels),
            'coordinates': 'Half-open Unicode code-point spans in literal packet text. Stop is a '
                           'character boundary. SCARECROW word stops locate coverage, not response time.',
            'counts': {SCARECROW_TASK: ['eligible_unmarked', 'severity_1', 'severity_2', 'severity_3'],
                       BLOTT_TASK: ['Nonsense', 'Sense']},
            'missingness': 'No unobserved count is imputed. An explicit eligible grade-zero word '
                           'response means no observed mark at that word. Missing anchors are unobserved '
                           'connections. A Blott negative comes from an explicit Nonsense response.',
            'sources': {SCARECROW_TASK: scarecrow_audit, BLOTT_TASK: blott_audit},
            'excluded_sources': {
                'Frick': 'Source archive has no declared reuse license and contains adapted third-party text.',
                'Hippocorpus V3': 'Reuse license for upgraded annotations is unverified.',
                'INQUISITIVE': 'No cleared, complete source article prefixes are available.',
            },
        })
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('scarecrow-raw', 'scarecrow-prepared', 'blott-prepared', 'tokenizer-path', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    prepare(**vars(parser.parse_args()))
