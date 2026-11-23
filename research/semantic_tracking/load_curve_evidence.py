"""Prepare text-only inputs and local human-marked support cues for load-curve research."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import re
import tarfile
from tempfile import TemporaryDirectory
import unicodedata

from research.backbone_transfer.extract import (
    MAX_INPUT_TOKENS, certify, digest, read_json, stream_rows, write_json, write_rows,
)
from research.semantic_tracking import expmrc_prepare, qasper_prepare
from research.semantic_tracking.expmrc_score import PRIOR_MATERIAL_OVERLAP


PREPARED_SHA256 = {
    'qasper': '4f8a5358704ca67af480bbc41575567ab8d0d4417903045f6ff38b56fb0b16c9',
    'expmrc': '0e177e3bfca4047dc48cb47cd9429daf469962b6b6d99899ce650b06fee67bb0',
}
LIMITS = {'train': (24, 30_000), 'holdout': (14, 15_000), 'expmrc': (100, 35_000)}
QASPER_KEY = re.compile(rb'  "([0-9.]+)": \{\r?\n')
WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
DISPLAY = re.compile(r"[^\W_]+(?:['’][^\W_]+)*|[^\w\s]|_")
CLAUSE_END = re.compile(r'[.!?;]+(?=\s|$)|\n')
FUNCTION_WORDS = frozenset('''
a an and are as at be been being but by can could did do does for from had has have how if in into is it its
may might of on or our per shall should so than that the their them then there these they this those through
to upon was were what when where whether which who whom why will with would you your
'''.split())


def identifier(kind, source, value):
    return sha256(f'cogllm-load-curve-evidence-v1:{kind}:{source}:{value}'.encode()).hexdigest()[:32]


def table(path):
    result = {}
    for row in stream_rows(path):
        key = row['id']
        if not key or key in result:
            raise ValueError(f'Duplicate or missing identity in {path.name}')
        result[key] = row
    return result


def verify_receipt(directory, source):
    path = directory / 'assay.json'
    if digest(path) != PREPARED_SHA256[source]:
        raise ValueError(f'Unexpected prepared {source} receipt')
    receipt = read_json(path)
    if any(digest(directory / name) != expected for name, expected in receipt['file_sha256'].items()):
        raise ValueError(f'{source} files differ from their prepared receipt')
    return receipt


def select(source, split, candidates, tokenizer):
    document_limit, token_limit = LIMITS[split]
    selected, inventory = [], Counter()
    total = 0
    for key, row in sorted(candidates.items(), key=lambda entry: identifier('article', source, entry[0])):
        if len(selected) == document_limit:
            break
        text = row['text']
        encoded = tokenizer.encode(text, add_special_tokens=False)
        counts = len(encoded.ids)
        if not 0 < counts <= MAX_INPUT_TOKENS:
            inventory['outside_article_token_limit'] += 1
            continue
        if total + counts > token_limit:
            inventory['first_article_over_budget'] += 1
            break
        ident = identifier('article', source, key)
        requested = sorted({len(text), *(match.end() for match in DISPLAY.finditer(text))})
        document = certify(ident, text, requested, encoded, tokenizer)
        if len(text) not in document.certified:
            raise ValueError('An eligible article lacks an exact causal terminal coordinate')
        selected.append((key, {'id': ident, 'text': text, 'stops': list(document.certified)}))
        total += counts
        inventory.update(document.missing.values())
        inventory['display_stops_requested'] += len(requested)
        inventory['display_stops_certified'] += len(document.certified)
    if not selected:
        raise ValueError(f'No selected documents in {source} {split}')
    return selected, {'input_articles': len(candidates), 'selected_articles': len(selected),
                      'input_tokens': total, **dict(inventory)}


def qasper_rows(raw, split, keys):
    """Stream only selected objects from the pinned, formatted Ai2 release."""
    archive, member = qasper_prepare.MEMBERS[split]
    path = raw / archive
    if digest(path) != qasper_prepare.ARCHIVES[archive]:
        raise ValueError(f'Unexpected QASPER archive: {archive}')
    pending = set(keys)
    with tarfile.open(path) as source, source.extractfile(member) as stream:
        for line in stream:
            match = QASPER_KEY.fullmatch(line)
            if match is None or (key := match[1].decode('ascii')) not in pending:
                continue
            payload = bytearray(b'{\n')
            for line in stream:
                if line in (b'  },\n', b'  }\n'):
                    payload.extend(b'}')
                    break
                payload.extend(line)
            paper = json.loads(payload)
            yield key, paper
            pending.remove(key)
            if not pending:
                return
    raise ValueError('One or more selected QASPER articles absent from the official split')


def render_qasper(paper):
    full, body, end = [], [], 0

    def append(block):
        nonlocal end
        value = (block or '').strip()
        if not value:
            return None
        start = end + 2 * bool(full)
        end = start + len(value)
        full.append(value)
        return start, end

    append(paper['title'])
    abstract = append(paper['abstract'])
    for section in paper['full_text']:
        append(section['section_name'])
        for block in section['paragraphs']:
            span = append(block)
            if span is not None:
                body.append(span)
    return '\n\n'.join(full), body, abstract


def qasper_spans(question, truth, lookup, body):
    answers = question['answers']
    if len(answers) != truth['annotation_count']:
        raise ValueError('QASPER original annotation denominator differs from the prepared one')
    if question['paper_read'] != 'no' or not answers:
        return None, 'question_writer_without_documented_title_abstract_only_exposure'
    references = []
    for record in answers:
        answer = record['answer']
        evidence = answer['evidence']
        if answer['unanswerable'] or len(evidence) != 1 or evidence[0].lstrip().startswith('FLOAT SELECTED'):
            return None, 'annotation_not_answerable_with_one_text_block'
        matches = lookup[qasper_prepare.normalized(evidence[0])]
        if len(matches) != 1 or matches[0] not in body:
            return None, 'not_unique_body_evidence'
        references.append(matches[0])
    if len(set(references)) != 1:
        return None, 'references_disagree_on_body_span'
    if truth['excluded'] or truth['refs'] != [{'end': end} for _, end in references]:
        raise ValueError('Reconstructed body evidence disagrees with sealed QASPER endpoints')
    return references, None


def word_key(word):
    return unicodedata.normalize('NFKC', word).casefold().replace('’', "'")


def anchor_proxy(text, question, body, cutoff):
    """Compare visible body clauses using content-word overlap with question text."""
    words = {word_key(word.group()) for word in WORD.finditer(question)}
    query = {word for word in words - FUNCTION_WORDS if len(word) >= 3 and word.isalpha()}
    best = None
    for start, stop in body:
        if start >= cutoff:
            break
        boundaries = list(CLAUSE_END.finditer(text, start, stop))
        starts = [start, *[mark.end() for mark in boundaries]]
        ends = [*[mark.start() for mark in boundaries], stop]
        for left, right in zip(starts, ends, strict=True):
            if left >= cutoff:
                break
            tokens = [(word, word_key(word.group())) for word in WORD.finditer(text, left, right)
                      if word.end() <= cutoff]
            unique = {key for _, key in tokens if key in query}
            if len(unique) < 2:
                continue
            hits = [word for word, key in tokens if key in unique]
            span = hits[0].start(), hits[-1].end()
            rank = (-len(unique), span[1] - span[0], span[0])
            if best is None or rank < best[0]:
                best = rank, {'span': list(span), 'distinct_matching_terms': len(unique)}
    return best[1] if best is not None else None


def label(source, row, query, refs, body):
    text, question = row['text'], query['text']
    first_start = min(start for start, _ in refs)
    return {
        'id': identifier('question', source, query['id']), 'article_id': row['id'], 'question': question,
        'reference_spans': refs, 'first_marked_start': first_start,
        'first_reference_end': min(end for _, end in refs), 'consensus_end': max(end for _, end in refs),
        'anchor_proxy': anchor_proxy(text, question, body, first_start),
    }


def controls(labels, stops):
    by_article = defaultdict(list)
    for row in labels:
        by_article[row['article_id']].append(row)
    for key, records in by_article.items():
        ordered = sorted(records, key=lambda row: (row['first_marked_start'], row['id']))
        for first, second in combinations(ordered, 2):
            lo, hi = first['consensus_end'], second['first_marked_start']
            index = bisect_left(stops[key], lo)
            if index < len(stops[key]) and stops[key][index] < hi:
                yield {'article_id': key, 'earlier_id': first['id'], 'later_id': second['id'], 'window': [lo, hi]}


def qasper_labels(directory, raw, selections, metadata, queries):
    labels, audits = [], {}
    for split, official, truth_name in (('train', 'train', 'development'), ('holdout', 'test', 'holdout')):
        rows = dict(selections[split])
        sources = {metadata[key]['source_paper_id']: key for key in rows}
        truth = table(directory / f'{truth_name}.jsonl')
        inventory = Counter()
        audits[split] = inventory
        for raw_id, paper in qasper_rows(raw, official, sources):
            key = sources[raw_id]
            text, body, abstract = render_qasper(paper)
            if text != rows[key]['text'] or len(body) != metadata[key]['body_paragraphs']:
                raise ValueError('QASPER official text did not reproduce the sealed full article')
            lookup = defaultdict(list)
            for start, end in [*body, *([abstract] if abstract is not None else [])]:
                lookup[qasper_prepare.normalized(text[start:end])].append((start, end))
            originals = {f"question-{item['question_id']}": item for item in paper['qas']}
            for query in queries[key]:
                item, reference = originals[query['id']], truth[query['id']]
                if item['question'] != query['text'] or reference['article_id'] != key:
                    raise ValueError('Invalid QASPER question/evidence identity')
                refs, reason = qasper_spans(item, reference, lookup, body)
                inventory[reason or 'accepted'] += 1
                if refs is not None:
                    labels.append(label('qasper', rows[key], query, refs, body))
    return labels, audits


def expmrc_labels(directory, raw, selections, queries):
    raw_articles, raw_queries, raw_outcomes = ({row['id']: row for row in entries}
                                              for entries in expmrc_prepare.load_release(raw))
    outcomes = table(directory / 'outcomes.jsonl')
    for key, row in selections:
        text = row['text']
        if text != raw_articles[key]['text']:
            raise ValueError('ExpMRC official text differs from prepared text')
        for query in queries[key]:
            qid = query['id']
            outcome = outcomes[qid]
            if query != raw_queries[qid] or outcome != raw_outcomes[qid] or outcome['article_id'] != key:
                raise ValueError('Invalid ExpMRC official question/evidence correspondence')
            refs = outcome['refs']
            if max(start for start, _ in refs) >= min(end for _, end in refs):
                yield None
            else:
                yield label('expmrc', row, query, refs, [(0, len(text))])


def queries_by_article(directory):
    result = defaultdict(list)
    for row in table(directory / 'queries.jsonl').values():
        result[row['article_id']].append(row)
    return result


def label_inventory(rows, stops):
    return {'questions': len(rows), 'annotations': sum(len(row['reference_spans']) for row in rows),
            'articles_with_labels': len({row['article_id'] for row in rows}),
            'matching_prior_anchor_proxy': sum(row['anchor_proxy'] is not None for row in rows),
            'no_prior_anchor_proxy': sum(row['anchor_proxy'] is None for row in rows),
            'first_end_is_certified': sum(row['first_reference_end'] in stops[row['article_id']] for row in rows),
            'consensus_end_is_certified': sum(row['consensus_end'] in stops[row['article_id']] for row in rows)}


def qasper_coverage(args, tokenizer):
    qasper = table(args.qasper / 'articles.jsonl')
    metadata = table(args.qasper / 'article-metadata.jsonl')
    if qasper.keys() != metadata.keys():
        raise ValueError('Invalid prepared QASPER article inventory')
    selections, report = {}, {}
    for split, official in (('train', 'train'), ('holdout', 'test')):
        keys = {key: row for key, row in qasper.items() if metadata[key]['official_split'] == official}
        selections[split], report[split] = select('qasper', split, keys, tokenizer)
    labels, audits = qasper_labels(args.qasper, args.qasper_raw, selections, metadata, queries_by_article(args.qasper))
    for split, documents in selections.items():
        stops = {row['id']: set(row['stops']) for _, row in documents}
        subset = [row for row in labels if row['article_id'] in stops]
        report[split].update(label_inventory(subset, stops), question_disposition=dict(audits[split]))
    return report


def prepare(args):
    from tokenizers import Tokenizer

    if args.output.exists():
        raise FileExistsError(args.output)
    if digest(args.tokenizer) != qasper_prepare.TOKENIZER_SHA:
        raise ValueError('Expected the pinned Qwen3.5-2B tokenizer')
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    receipts = {name: verify_receipt(getattr(args, name), name) for name in PREPARED_SHA256}
    expmrc = table(args.expmrc / 'articles.jsonl')
    eligible = {key: row for key, row in expmrc.items() if key not in PRIOR_MATERIAL_OVERLAP}
    selections, inventory = select('expmrc', 'expmrc', eligible, tokenizer)
    train_count = len(selections) * 70 // 100
    stimuli, layout = [], []
    by_article, normalized_texts = {}, set()
    for index, (key, row) in enumerate(selections):
        canonical = ' '.join(unicodedata.normalize('NFKC', row['text']).casefold().split())
        if canonical in normalized_texts or row['id'] in by_article:
            raise ValueError('Selected split contains a duplicate document')
        normalized_texts.add(canonical)
        split = 'train' if index < train_count else 'test'
        by_article[row['id']] = split
        stimuli.append(row)
        layout.append({'id': row['id'], 'group_id': row['id'], 'source': 'expmrc', 'source_id': key,
                       'split': split, 'official_split': 'dev'})
    queries = queries_by_article(args.expmrc)
    labels = [record for record in expmrc_labels(args.expmrc, args.expmrc_raw, selections, queries)
              if record is not None]
    labels.sort(key=lambda row: (row['article_id'], row['id']))
    if len({row['id'] for row in labels}) != len(labels):
        raise ValueError('Duplicate support labels')
    matched = list(controls(labels, {row['id']: row['stops'] for row in stimuli}))
    stops = {row['id']: set(row['stops']) for row in stimuli}
    report = {}
    for split in ('train', 'test'):
        subset = [row for row in labels if by_article[row['article_id']] == split]
        if not subset:
            raise ValueError('No human support labels in a split')
        documents = [row for row in layout if row['split'] == split]
        questions = sum(len(queries[row['source_id']]) for row in documents)
        report[split] = {**label_inventory(subset, stops), 'selected_articles': len(documents),
                         'questions_without_overlapping_support': questions - len(subset),
                         'same_text_controls': sum(by_article[row['article_id']] == split for row in matched)}
    qasper_report = qasper_coverage(args, tokenizer)
    files = {'model_input/stimuli.jsonl': stimuli, 'layout.jsonl': layout,
             'labels.jsonl': labels, 'controls.jsonl': matched}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        staged = Path(temporary) / 'assay'
        (staged / 'model_input').mkdir(parents=True)
        for name, rows in files.items():
            write_rows(staged / name, rows)
        write_json(staged / 'assay.json', {
            'schema': 1, 'endpoint': 'Retrospective human-marked textual support for supplied questions.',
            'offsets': 'Zero-based half-open Unicode code point spans in unchanged model-input text.',
            'first_marked_start': 'Earliest start among retained references. This is not a reading onset.',
            'first_reference_end': 'First complete retained reference.',
            'consensus_end': 'Position by which all retained references are complete.',
            'anchor_proxy': ('Label-side only. Before earliest marked support, maximize distinct question content-word '
                             'matches within a body clause. Require two matches, then prefer smallest matching hull '
                             'and earliest start. This is a weak lexical link, not a human-supplied anchor.'),
            'control_scope': ('Disjoint human reference hulls in the same article and at least one certified stop '
                              'within the half-open window. Only annotated evidence is compared.'),
            'evaluation_scope': ('Both ExpMRC arms come from the official dev release and have been used in prior '
                                 'CogLLM evaluation. This assay is retrospective at the study level.'),
            'selection': {'expmrc_document_and_token_limits': LIMITS['expmrc'], 'individual_token_limit': MAX_INPUT_TOKENS,
                          'order': ('Opaque SHA-256 article identity before outcomes. The first floor(70%) of selected '
                                    'articles form train, and the remainder form test.'),
                          'expmrc_prior_race_overlap_removed': len(PRIOR_MATERIAL_OVERLAP)},
            'sources': {name: {'receipt_sha256': PREPARED_SHA256[name], 'license': receipt['source_license'],
                               'method': receipt['paper']} for name, receipt in receipts.items()},
            'official_archive_sha256': {'qasper': qasper_prepare.ARCHIVES, 'expmrc': expmrc_prepare.SOURCE_SHA256},
            'tokenizer_sha256': qasper_prepare.TOKENIZER_SHA, 'prepare_sha256': digest(Path(__file__)),
            'file_sha256': {name: digest(staged / name) for name in files}, 'splits': report,
            'expmrc_input': inventory, 'qasper_coverage_only': qasper_report,
        })
        staged.rename(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('prepare')
    for name in ('qasper', 'qasper-raw', 'expmrc', 'expmrc-raw', 'tokenizer', 'output'):
        command.add_argument(f'--{name}', required=True, type=Path)
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()
