"""Prepare exact continuation-only Scarecrow spans with separate ordinal vote counts."""

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.semantic_tracking import blott_format_packet
from research.semantic_tracking import source_prepare as prior


PAPER = 'https://aclanthology.org/2022.acl-long.501/'
RELEASE_PAGE = 'https://yao-dou.github.io/scarecrow/'
CATEGORY = 'Self-contradiction'
HISTORICAL_GROUP_DOCUMENTS = {'train': 442, 'test': 112}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def frozen_assignments(previous):
    audit = read_json(previous / 'audit.json')
    for name, digest in audit['output_sha256'].items():
        if prior.digest(previous / name) != digest:
            raise ValueError('The historical SCARECROW split does not match its audit')
    if (audit['csv_sha256'] != prior.CSV_SHA256 or audit['tokenizer_sha256'] != prior.TOKENIZER_SHA256
            or audit['seed'] != prior.SEED):
        raise ValueError('Unexpected historical SCARECROW source, split or tokenizer')
    associations = {}
    with (previous / 'outcomes.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            key = row['id'].removeprefix('scarecrow-')
            value = row['group_id'], row['split']
            if associations.setdefault(key, value) != value or row['split'] not in HISTORICAL_GROUP_DOCUMENTS:
                raise ValueError('One historical continuation belongs to several groups or folds')
    if len(associations) != 526 or len(set(associations.values())) != 515:
        raise ValueError('Expected the 526 historical documents in 515 frozen groups')
    return associations


def assign_documents(raw, historical):
    if len({prior.normalized(row['generation']) for row in raw}) != len(raw):
        raise ValueError('Duplicate generated continuations require group coalescing')
    groups = {}
    known_folds = {}
    for row in raw:
        prompt = prior.normalized(row['prompt'])
        if not prompt:
            raise ValueError('Empty source prompt in the grouped release')
        group = 'scarecrow-prompt-' + hashlib.sha256(prompt.encode()).hexdigest()[:16]
        if groups.setdefault(group, prompt) != prompt:
            raise ValueError('Different original prompts have the same group identifier')
        if row['id'] in historical:
            old_group, old_fold = historical[row['id']]
            if group != old_group or known_folds.setdefault(group, old_fold) != old_fold:
                raise ValueError('A generated continuation changes old prompt group or old fold')
    novel = sorted(groups.keys() - known_folds.keys(), key=lambda group:
                   hashlib.sha256(f'scarecrow-detection-v1:{group}'.encode()).digest())
    novel_test = set(novel[:len(novel) // 5])
    prompt_groups = {prompt: group for group, prompt in groups.items()}
    assignments = []
    for row in raw:
        group = prompt_groups[prior.normalized(row['prompt'])]
        known = group in known_folds
        split = known_folds[group] if known else ('test' if group in novel_test else 'train')
        assignments.append((row, group, split, known))
    if Counter(split for _, _, split, known in assignments if known) != HISTORICAL_GROUP_DOCUMENTS:
        raise ValueError('The set of generated continuations in these prior groups changed')
    return assignments


def overlaps(first, second):
    return first[0] < second[1] and second[0] < first[1]


def coverage(text, words, responses, stats):
    utf16 = prior.utf16_to_characters(text)
    counts = [[0, 0, 0, 0] for _ in words]
    marked = False
    for response in responses:
        known = []
        uncertain = []
        mask_whole = False
        for category, _, severity, start, end, source in response:
            if category != CATEGORY:
                continue
            marked = True
            if type(severity) is not int or not 1 <= severity <= 3:
                raise ValueError('Expected source severity 1, 2, or 3')
            span = prior.convert_span(text, utf16, start, end)
            if span is None:
                mask_whole = True
                stats['mark_unlocatable'] += 1
                continue
            if not source or len(source) % 2:
                uncertain.append(span)
                stats['mark_source_missing_or_incomplete'] += 1
                continue
            references = [prior.convert_span(text, utf16, start, end)
                          for start, end in zip(source[::2], source[1::2], strict=True)]
            if any(reference is None or reference[1] > span[0] for reference in references):
                uncertain.append(span)
                stats['mark_source_outside_visible_prior_continuation'] += 1
                continue
            known.append((span, severity))
            stats['mark_retained'] += 1
            stats['mark_discontinuous_source_retained'] += len(references) > 1
        if mask_whole:
            stats['annotator_word_masked'] += len(words)
            continue
        for index, word in enumerate(words):
            if any(overlaps(word, span) for span in uncertain):
                stats['annotator_word_masked'] += 1
                continue
            grade = max((severity for span, severity in known if overlaps(word, span)), default=0)
            counts[index][grade] += 1
    return [
        {'positive': sum(by_grade[1:]), 'total': sum(by_grade), 'severity_counts': by_grade[1:]}
        for by_grade in counts
    ], marked


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8') as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def write_json(path, value):
    with path.open('x', encoding='utf-8') as output:
        output.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')


def prepare(raw_path, tokenizer_path, previous, output, report):
    from tokenizers import Tokenizer

    if output.exists() or report.exists():
        raise FileExistsError('Expected new SCARECROW output and report paths')
    if prior.digest(raw_path) != prior.CSV_SHA256 or prior.digest(tokenizer_path) != prior.TOKENIZER_SHA256:
        raise ValueError('Expected the frozen public SCARECROW source and Qwen tokenizer')
    historical = frozen_assignments(previous)
    with raw_path.open(encoding='utf-8', newline='') as source:
        raw = list(csv.DictReader(source))
    if len(raw) != 1308 or len({row['id'] for row in raw}) != len(raw):
        raise ValueError('Unexpected published SCARECROW source inventory')
    generated = [row for row in raw if row['model'] != 'human']
    if len(generated) != 1229:
        raise ValueError('Unexpected generated continuation inventory')
    assignments = assign_documents(generated, historical)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    stimuli, layout, outcomes = [], [], []
    inventories = defaultdict(Counter)
    group_set = defaultdict(set)
    text_seen = set()
    prior_groups = defaultdict(set)
    novel_groups = defaultdict(set)
    for row, group, split, known in assignments:
        stats = inventories[split]
        body = row['generation']
        if not body.strip() or body in text_seen:
            raise ValueError('Missing or duplicate literal generated continuation')
        text_seen.add(body)
        identifier = f'scarecrow-{row["id"]}'
        words = [match.span() for match in prior.WORD.finditer(body)
                 if any(char.isalnum() for char in match.group())]
        endings = {end for _, end in words} | {len(body)}
        certificate = prior.certified_ends(body, endings, tokenizer)
        aligned = [word for word in words if word[1] in certificate]
        if not aligned or len(body) not in certificate:
            raise ValueError('Expected an exact full continuation and at least one lexical candidate')
        responses = ast.literal_eval(row['responses'])
        if not 8 <= len(responses) <= 10:
            raise ValueError('Unexpected numbers of published annotator submissions')
        counts, marked = coverage(body, aligned, responses, stats)
        stimuli.append({'id': identifier, 'text': body, 'stops': sorted(certificate)})
        layout.append({'id': identifier, 'group_id': group, 'split': split,
                       'prior_pointer_group': known, 'words': aligned})
        outcomes.append({'id': identifier, 'counts': counts})
        group_set[split].add(group)
        (prior_groups if known else novel_groups)[split].add(group)
        stats['documents'] += 1
        stats['documents_in_prior_pointer_groups'] += known
        stats['documents_in_novel_groups'] += not known
        stats['historical_documents'] += row['id'] in historical
        stats['documents_with_no_raw_contradiction_marks'] += not marked
        stats['annotator_submissions'] += len(responses)
        stats['lexical_words'] += len(words)
        stats['aligned_lexical_words'] += len(aligned)
        stats['words_with_any_positive_vote'] += sum(cell['positive'] > 0 for cell in counts)
        stats['words_unobserved_after_masking'] += sum(cell['total'] == 0 for cell in counts)
        stats['annotator_word_at_risk'] += sum(cell['total'] for cell in counts)
        stats['annotator_word_positive'] += sum(cell['positive'] for cell in counts)
        for grade in range(3):
            stats[f'annotator_word_severity_{grade + 1}'] += sum(cell['severity_counts'][grade] for cell in counts)
    for split, groups in group_set.items():
        inventories[split]['prompt_groups'] = len(groups)
        inventories[split]['frozen_prompt_groups'] = len(prior_groups[split])
        inventories[split]['novel_prompt_groups'] = len(novel_groups[split])
    if group_set['train'] & group_set['test'] or sum(map(len, prior_groups.values())) != 515:
        raise ValueError('Historical group partition changed')

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.scarecrow-detection-') as temporary:
        staged = Path(temporary) / 'assay'
        prepared = staged / 'prepared'
        prepared.mkdir(parents=True)
        files = {'stimuli': stimuli, 'layout': layout, 'outcomes': outcomes}
        for name, content in files.items():
            write_jsonl(prepared / f'{name}.jsonl', content)
        packet = staged / 'packet'
        blott_format_packet.prepare(prepared / 'stimuli.jsonl', tokenizer_path, packet)
        extraction = read_json(packet / 'extractor-preflight.json')
        receipt = {
            'paper': PAPER,
            'author_release': RELEASE_PAGE,
            'source_url': prior.SOURCE,
            'source_sha256': prior.CSV_SHA256,
            'prior_audit_sha256': prior.digest(previous / 'audit.json'),
            'release_scope': 'Authors explicitly release data for research use. No separate repository license is stated. Use only released machine-generated continuation text in model input, following the historical project assay. Third-party news prompt stays out of the inference packet.',
            'endpoint': 'Retrospective annotator word coverage by a self-contradiction mark with recorded antecedent span(s) wholly in preceding continuation. Human annotators saw complete text with original news context.',
            'candidate': 'Every Unicode non-whitespace substring with at least one alphanumeric character and an exactly certified Qwen prefix endpoint. A terminal stop may supplement the lattice and is not scored unless it equals a candidate word endpoint.',
            'risk': 'One count per eligible submitted annotator for each word. Mutually exclusive grades 0–3. Grade zero records no observed mark and does not establish absence of contradiction. A possible mark relying on unavailable text or missing/future reference masks its covered words for that annotator. A mark with no recoverable range masks that annotator for the full continuation.',
            'severity': 'Section 4.5 gives three ordinal ratings of the impact of a marked error on comprehension. Overlapping allowed marks from one annotator contribute the maximum severity, once per word.',
            'split': 'Preserve the fit/test assignments of 515 prompt groups from the prior known-mark pointer experiment. Retain their additional rated continuations. For the remaining groups, sort sha256("scarecrow-detection-v1:" + group_id) and assign the first floor(n/5) to test. No generated continuations are shared even after Unicode/case/whitespace normalization.',
            'excluded_human_written_continuations': len(raw) - len(generated),
            'inventory': {split: dict(counts) for split, counts in sorted(inventories.items())},
            'packet_inventory': {key: extraction[key] for key in ('input_documents', 'input_tokens', 'max_input_tokens', 'certified_stops')},
            'projected_fp16_bytes_estimate': extraction['certified_stops'] * 128 * 2,
            'prepared_sha256': {name: prior.digest(prepared / f'{name}.jsonl') for name in files},
            'packet_sha256': {path.name: prior.digest(path) for path in sorted(packet.iterdir())},
            'preparer_sha256': prior.digest(Path(__file__)),
        }
        write_json(prepared / 'source-audit.json', receipt)
        staged.rename(output)
    report.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw-path', 'tokenizer-path', 'previous', 'output', 'report'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    prepare(**vars(parser.parse_args()))
