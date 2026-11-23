"""Freeze the linked-document evidence-arrival assay from the IIRC author releases."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path

from research.semantic_tracking.iirc_linked_evidence_audit import (
    SOURCE_SHA256,
    SOURCE_URL,
    exact_span,
    family,
    fingerprint,
    lookup,
    source_articles,
)
from research.semantic_tracking.iirc_prepare import (
    RELEASE_SHA256,
    RELEASE_URL,
    digest,
    normalized,
    read_release,
    verify_split_separation,
    visible_anchors,
)
from research.stage2.records import write_jsonl


CAP = 8_000
STRIDE = 128
SEPARATOR = '\n\n'
TOKENIZER_FILES = {
    'tokenizer.json': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42',
    'tokenizer_config.json': '49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c',
    'config.json': 'ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4',
}


@dataclass(frozen=True)
class Observation:
    paragraph_id: str
    seed_title: str
    seed_text: str
    question_id: str
    anchor_start: int
    anchor_end: int
    linked_title: str
    linked_text: str
    evidence_start: int
    evidence_end: int
    answer_type: str
    context_index: int


def load_tokenizer(snapshot):
    from transformers import AutoTokenizer, __version__

    from research.stage3.run import TRANSFORMERS_VERSION

    if __version__ != TRANSFORMERS_VERSION:
        raise ValueError(f'Expected Transformers {TRANSFORMERS_VERSION}, found {__version__}')
    for name, expected in TOKENIZER_FILES.items():
        if digest(snapshot / name) != expected:
            raise ValueError(f'Unexpected tokenizer snapshot file: {name}')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError('Expected the same fast AutoTokenizer used by the causal extractor')
    return tokenizer


def observations(passages, articles, counts):
    for paragraph in passages:
        anchors = defaultdict(list)
        for anchor in visible_anchors(paragraph):
            anchors[anchor['target']].append(anchor)
        authored = Counter(target for question in paragraph['questions'] for target in set(question['question_links']))
        for question in paragraph['questions']:
            counts['authored_questions'] += 1
            links = question['question_links']
            if len(links) != 1 or len(anchors[links[0]]) != 1:
                counts['no_single_uniquely_positioned_writer_link'] += 1
                continue
            target, = links
            if authored[target] != 1:
                counts['multiple_authored_questions_for_writer_link'] += 1
                continue
            if not normalized(question['question']):
                counts['empty_question'] += 1
                continue
            answer_type = question['answer']['type']
            if answer_type == 'none':
                counts['answerer_reported_unanswerable'] += 1
                continue
            external = [(index, excerpt) for index, excerpt in enumerate(question['context'])
                        if excerpt['passage'] != 'main']
            if not external:
                counts['answerable_without_linked_support'] += 1
                continue
            if len(external) != 1 or external[0][1]['passage'] != target:
                counts['multiple_or_differently_linked_supports'] += 1
                continue
            (index, evidence), = external
            text = lookup(target, articles)
            if text is None:
                counts['linked_source_unavailable'] += 1
                continue
            start, end = evidence['indices']
            if not exact_span(text, start, end, evidence['text']):
                counts['source_excerpt_not_exactly_aligned'] += 1
                continue
            anchor, = anchors[target]
            yield Observation(paragraph['pid'], paragraph['title'], paragraph['text'], question['qid'],
                              anchor['start'], anchor['end'], target, text, start, end, answer_type, index)


def source_keys(observation):
    return frozenset((('title', family(observation.seed_title)), ('title', family(observation.linked_title)),
                      ('sha256', fingerprint(observation.seed_text)), ('sha256', fingerprint(observation.linked_text))))


def blocked_families(dev, articles):
    titles = {paragraph['title'] for paragraph in dev}
    titles.update(link['target'] for paragraph in dev for link in paragraph['links'])
    hashes = {fingerprint(paragraph['text']) for paragraph in dev}
    hashes.update(fingerprint(text) for title in titles if (text := lookup(title, articles)) is not None)
    return {('title', family(title)) for title in titles} | {('sha256', sha) for sha in hashes}


def certified_path(tokenizer, seed, raw_document, anchor_end):
    """Compute prefix stops using only the displayed path and the supplied anchor."""
    prefix = raw_document[:CAP]
    text = seed + SEPARATOR + prefix
    linked_start = len(seed) + len(SEPARATOR)
    if not prefix.strip():
        return None, 'linked_prefix_blank'
    full = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    positions = {end: pos + 1 for pos, (start, end) in enumerate(full['offset_mapping']) if start < end}
    ids = full['input_ids']
    if positions.get(len(text)) != len(ids):
        return None, 'uncertified_terminal_character_boundary'
    if anchor_end not in positions or tokenizer.encode(text[:anchor_end], add_special_tokens=False) != ids[:positions[anchor_end]]:
        return None, 'uncertified_anchor_character_boundary'

    ends = sorted(end for end in positions if linked_start < end < len(text))
    indices = [bisect_right(ends, linked_start + target) - 1 for target in range(STRIDE, len(prefix), STRIDE)]
    candidates = sorted({ends[index] for index in indices if index >= 0})
    if candidates:
        prefix_ids = tokenizer([text[:end] for end in candidates], add_special_tokens=False)['input_ids']
        certified = {end: expected == ids[:positions[end]] for end, expected in zip(candidates, prefix_ids, strict=True)}
    else:
        certified = {}

    stops = [anchor_end]
    for index in indices:
        while index >= 0 and ends[index] > stops[-1]:
            end = ends[index]
            if end not in certified:
                certified[end] = tokenizer.encode(text[:end], add_special_tokens=False) == ids[:positions[end]]
            if certified[end]:
                stops.append(end)
                break
            index -= 1
    stops.append(len(text))
    return (text, stops, linked_start, len(ids)), 'retained'


def connected_articles(keys_by_case):
    cases_by_key = defaultdict(list)
    for case, keys in keys_by_case.items():
        for key in keys:
            cases_by_key[key].append(case)
    leaders = {}
    used_keys = set()
    for first in sorted(keys_by_case):
        if first in leaders:
            continue
        leaders[first] = first
        pending = [first]
        while pending:
            case = pending.pop()
            for key in keys_by_case[case] - used_keys:
                used_keys.add(key)
                for other in cases_by_key[key]:
                    if other not in leaders:
                        leaders[other] = first
                        pending.append(other)
    return leaders


def matched_swaps(labels, schedules):
    by_document = defaultdict(list)
    for row in labels:
        if row['upper'] is not None:
            by_document[row['linked_text_sha256']].append(row)
    for rows in by_document.values():
        for a, b in combinations(rows, 2):
            case_a, case_b = a['case_id'], b['case_id']
            if (a['seed_family'] == b['seed_family'] or a['upper'] == b['upper']
                    or schedules[case_a] != schedules[case_b]
                    or max(a['evidence_start'], b['evidence_start']) < min(a['evidence_end'], b['evidence_end'])):
                continue
            yield {'case_ids': [case_a, case_b], 'article_cluster': a['article_cluster']}


def prepare_split(passages, articles, tokenizer, blocked):
    counts = Counter()
    texts, queries, seeds, documents, labels = [], [], {}, {}, []
    keys_by_case, schedules = {}, {}
    token_total = token_max = 0
    for item in observations(passages, articles, counts):
        counts['exact_one_authored_question_support'] += 1
        keys = source_keys(item)
        if keys & blocked:
            counts['removed_by_dev_article_family_guard'] += 1
            continue
        counts['eligible_exact_after_family_guard'] += 1
        counts['support_end_at_or_before_requested_cap'] += item.evidence_end <= CAP
        rendered, status = certified_path(tokenizer, item.seed_text, item.linked_text, item.anchor_end)
        counts[status] += 1
        if rendered is None:
            continue
        text, stops, linked_start, token_count = rendered
        case = f'iirc-linked-{item.question_id}'
        document_hash = fingerprint(item.linked_text)
        prefix = item.linked_text[:CAP]
        if text != item.seed_text + SEPARATOR + prefix or text[item.anchor_start:item.anchor_end] != item.seed_text[item.anchor_start:item.anchor_end]:
            raise ValueError(f'{case} is not an exact join of the original plain text and link prefix')
        if case in keys_by_case:
            raise ValueError(f'Repeated case ID: {case}')
        keys_by_case[case] = keys
        local_stops = [stop - linked_start for stop in stops[1:]]
        schedules[case] = local_stops
        arriving = bisect_left(local_stops, item.evidence_end)
        upper = arriving + 1 if arriving < len(local_stops) else None
        counts['observed_full_support'] += upper is not None
        counts['right_censored_full_support'] += upper is None
        counts['censored_source_starts_before_end_of_prefix'] += upper is None and item.evidence_start < len(prefix)
        counts['censored_source_starts_at_or_after_end_of_prefix'] += upper is None and item.evidence_start >= len(prefix)
        token_total += token_count
        token_max = max(token_max, token_count)
        texts.append({'id': case, 'text': text, 'stops': stops})
        queries.append({'case_id': case, 'anchor_start': item.anchor_start, 'anchor_end': item.anchor_end,
                        'linked_start': linked_start})
        seeds[item.paragraph_id] = {'id': item.paragraph_id, 'text': item.seed_text}
        documents[document_hash] = {'id': document_hash, 'text': prefix}
        labels.append({'case_id': case, 'question_id': item.question_id, 'seed_paragraph_id': item.paragraph_id,
                       'seed_title': item.seed_title, 'seed_family': family(item.seed_title),
                       'seed_text_sha256': fingerprint(item.seed_text), 'linked_title': item.linked_title,
                       'linked_family': family(item.linked_title), 'linked_text_sha256': document_hash,
                       'evidence_start': item.evidence_start, 'evidence_end': item.evidence_end,
                       'source_context_index': item.context_index, 'answer_type': item.answer_type,
                       'lower': arriving, 'upper': upper})

    if not labels or counts['observed_full_support'] + counts['right_censored_full_support'] != len(labels):
        raise ValueError('Experiment requires a nonempty, exhaustively classified linked-evidence split')
    clusters = connected_articles(keys_by_case)
    for label in labels:
        label['article_cluster'] = clusters[label['case_id']]
    pairs = list(matched_swaps(labels, schedules))
    counts.update({'seed_paragraphs': len(seeds), 'linked_documents': len(documents), 'tokens': token_total,
                   'max_tokens': token_max, 'linked_stops_excluding_anchor': sum(map(len, schedules.values())),
                   'article_components': len(set(clusters.values())), 'same_document_swaps': len(pairs)})
    tables = {'stimuli': texts, 'queries': queries, 'seed-text': list(seeds.values()),
              'linked-prefix-text': list(documents.values()), 'locked-labels': labels, 'locked-matched-swaps': pairs}
    return tables, dict(sorted(counts.items())), set().union(*keys_by_case.values())


def run(release, context, snapshot, output, audit_path):
    from research.stage3.run import MODEL, REVISION, TRANSFORMERS_VERSION

    data = read_release(release)
    raw_overlap = verify_split_separation(data)
    titles = {excerpt['passage'] for rows in data.values() for row in rows for question in row['questions']
              for excerpt in question['context'] if excerpt['passage'] != 'main'}
    titles.update(link['target'] for row in data['dev'] for link in row['links'])
    titles.update(row['title'] for row in data['dev'])
    articles, archive_count = source_articles(context, titles)
    blocked = blocked_families(data['dev'], articles)
    tokenizer = load_tokenizer(snapshot)
    train, tr_stats, train_keys = prepare_split(data['train'], articles, tokenizer, blocked)
    dev, dv_stats, dev_keys = prepare_split(data['dev'], articles, tokenizer, set())
    overlap = Counter(kind for kind, _ in train_keys & dev_keys)
    if overlap or train_keys & blocked:
        raise ValueError('Dev title or text hash survived the training isolation filter')
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for split, tables in (('train', train), ('dev', dev)):
        directory = output / split
        directory.mkdir()
        for name, rows in tables.items():
            path = directory / f'{name}.jsonl'
            write_jsonl(path, rows)
            files[str(path.relative_to(output))] = digest(path)
    audit = {
        'author_iirc': {'url': RELEASE_URL, 'sha256': RELEASE_SHA256},
        'author_context_articles': {'url': SOURCE_URL, 'sha256': SOURCE_SHA256, 'article_entries': archive_count},
        'model': MODEL, 'revision': REVISION, 'transformers': TRANSFORMERS_VERSION,
        'tokenizer_files_sha256': TOKENIZER_FILES,
        'annotation_protocol': 'https://aclanthology.org/2020.emnlp-main.86.pdf#page=4',
        'author_dataset_license': 'https://github.com/allenai/allennlp-website/blob/master/iirc.html',
        'source_text_terms': 'https://foundation.wikimedia.org/w/index.php?title=Policy:Terms_of_Use&oldid=98524#7._Licensing_of_Content',
        'source_overlap': raw_overlap,
        'training_identity_filter': 'Remove cases whose seed/linked titles or exact source text fingerprints occur among '
                                    'any official dev seed or visible linked title. Retain the full official dev split '
                                    'under the same per-case eligibility conditions.',
        'overlap_after_filter': {'article_title': overlap['title'], 'exact_source_text_sha256': overlap['sha256']},
        'primary_subset': 'One authored question across the original dataset paragraph for one writer-selected '
                          'visible link and one exactly aligned answer-annotator excerpt in that document.',
        'path': {'separator': SEPARATOR, 'linked_cutoff_unicode_characters': CAP, 'nominal_stop_stride_unicode_characters': STRIDE,
                 'coordinates': 'Half-open Unicode character offsets. Evidence uses original stripped linked-document coordinates. '
                                'Query offsets and stimulus stops use joined path coordinates.',
                 'stop_selection': 'Require exact independent retokenization at the anchor endpoint and each linked stop. '
                                   'Snap nominal checkpoints backwards and include the final character endpoint. '
                                   'The first stimulus stop is the supplied anchor and has no arrival label.'},
        'inference_inputs': ['dev/stimuli.jsonl', 'dev/queries.jsonl'],
        'fit_only_additional_inputs': ['train/stimuli.jsonl', 'train/queries.jsonl', 'train/locked-labels.jsonl'],
        'restricted_before_dev_forecast_seal': ['dev/locked-labels.jsonl', 'dev/locked-matched-swaps.jsonl'],
        'input_validation': 'Each joined model stimulus was verified character-for-character against the original seed '
                            'paragraph, a fixed two-newline separator and the character-zero linked article prefix. '
                            'No question, answer, evidence coordinates, link title or scoring label is provided as a feature.',
        'outcome': 'First certified linked-document stop that includes the entire selected context.indices excerpt, '
                   'including right-censored cases. An unanswerable response, missing support or bad source alignment is '
                   'counted separately and does not supply a time label. The endpoint is retrospective text coverage.',
        'splits': {'train': tr_stats, 'dev': dv_stats}, 'files_sha256': files,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('release', 'context', 'snapshot', 'output', 'audit'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    args = parser.parse_args()
    run(args.release, args.context, args.snapshot, args.output, args.audit)
