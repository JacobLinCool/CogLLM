"""Prepare a grade-free text packet and its separately stored research annotations."""

from argparse import ArgumentParser
from bisect import bisect_right
from collections import Counter, defaultdict
from itertools import islice, product
from pathlib import Path
from tempfile import TemporaryDirectory
import re

from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json, write_rows
from research.union_training.diagnose_frick_locations import read_stories
from research.union_training.verify_observations import require, text_digest, verified_member


REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
TOKENIZER_SHA256 = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
TOKENIZER = (Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots'
             / REVISION / 'tokenizer.json')
FRICK = 'semantic-tracking/frick-consistency/prepared'
OBSERVATIONS = 'union-training/observations-2'
EXPMRC = 'semantic-tracking/same-question-evidence/expmrc-v1'
CONTROL_SEPARATOR = '\n\n'

# Each fixture gives an identifiable question in the shared prefix and a direct answer.
# These are designed controls. The fields never enter the grade-free text packet.
CONTROL_TEXTS = (
    ('A replication team found that one instrument kept returning the same reading. '
     'The report left the cause of the repeated reading unspecified.',
     'the cause of the repeated reading', 'What caused the repeated reading?',
     'A stale cache caused the repeated reading.'),
    ('The engineering report describes a sensor installed inside the chamber. '
     'One missing detail is the position of the reference sensor.',
     'the position of the reference sensor', 'Where was the reference sensor?',
     'The reference sensor was mounted at the outlet.'),
    ('A study of archive entries distinguished draft records from approved ones. '
     'The description leaves open which group was included in the analysis.',
     'which group was included in the analysis', 'Which group was included in the analysis?',
     'The analysis included only the approved records.'),
    ('A field team recorded measurements in a fixed season. '
     'The introductory paragraph does not state the sampling month.',
     'the sampling month', 'In which month did sampling take place?',
     'The team collected the samples in October.'),
    ('The first analysis used a single outcome to summarize attendance. '
     'The account does not explain which unit the attendance measure used.',
     'which unit the attendance measure used', 'What unit did the attendance measure use?',
     'The attendance measure used minutes rather than sessions.'),
    ('The survey team compared two ways of sending notices. '
     'The summary leaves the distribution method in the reference group unspecified.',
     'the distribution method in the reference group', 'How were notices sent in the reference group?',
     'The reference group received the notices by email.'),
    ('The new annotation scheme was tried on two document formats. '
     'The method does not yet identify the format used for the pilot.',
     'the format used for the pilot', 'Which document format was used for the pilot?',
     'The team conducted the pilot on plain text.'),
    ('The assessment used one primary score for each submitted report. '
     'The opening paragraph leaves the score aggregation rule unspecified.',
     'the score aggregation rule', 'How were the scores aggregated?',
     'The team aggregated the scores by taking the median.'),
    ('A public records review used files drawn from a municipal directory. '
     'The introduction omits the file date used as the cutoff.',
     'the file date used as the cutoff', 'What file date was the cutoff?',
     'The file date cutoff was the first of March.'),
    ('The classroom pilot tested two arrangements of the same lesson. '
     'The description does not specify when the worked example appeared.',
     'when the worked example appeared', 'When did the worked example appear?',
     'The worked example appeared immediately after the introduction.'),
    ('The library project used a defined subset of its loan records. '
     'The opening does not say which branch contributed the records.',
     'which branch contributed the records', 'Which branch contributed the records?',
     'The north branch supplied all of the records.'),
    ('The statistics note describes a correction applied before plotting. '
     'It leaves the reason for the correction unexplained.',
     'the reason for the correction', 'Why was a correction applied?',
     'The correction compensated for the sampling bias.'),
)


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def certified_end(tokenizer, text, encoded, end):
    """Return how many full tokens were visible at an exclusive character stop."""
    require(type(end) is int and 0 < end <= len(text), f'Invalid visible end {end}')
    stops = [stop for _, stop in encoded.offsets]
    count = bisect_right(stops, end)
    require(count > 0 and stops[count - 1] == end and encode(tokenizer, text[:end]).ids == encoded.ids[:count],
            f'Uncertified tokenizer prefix ending at character {end}')
    return count


def certified_window(tokenizer, text, encoded, span):
    start, end = span
    require(0 <= start < end <= len(text), 'Invalid span')
    first = certified_end(tokenizer, text, encoded, start) if start else 0
    last = certified_end(tokenizer, text, encoded, end)
    require(first < last, 'The window contains no complete token')
    return first, last


def frick_examples(cache, tokenizer, stimuli):
    source = cache / FRICK
    stories, counts, reports, readers, hashes = read_stories(source)
    metadata = {row['id']: row for row in stream_rows(source / 'metadata.jsonl')}
    displayed = {row['id']: row['text'] for row in stream_rows(source / 'stimuli.jsonl') if row['id'] in stories}
    packet = cache / OBSERVATIONS
    receipt = read_json(packet / 'receipt.json')
    name = 'frick_first_report.jsonl'
    split_rows = list(stream_rows(verified_member(packet, Path(name), receipt['packet_sha256'][name])))
    splits = {row['coordinates'][0][0]: row for row in split_rows}
    require(len(split_rows) == len(splits) == len(stories) and splits.keys() == stories.keys(),
            'The original development split must cover each story version exactly once')
    rows = []
    group_splits = {}
    for key, story in sorted(stories.items()):
        old = splits[key]
        data = metadata[key]
        text = displayed[key]
        spans = [[sentence['start'], sentence['stop']] for sentence in data['sentences']]
        event_ends = [end for _, end in spans[1:]]
        outcome = [counts[key][i] for i in range(len(event_ends) + 1)]
        require(outcome == old['outcome'] and old['family'] == story.family and old['role'] in (0, 1)
                and [[key, stop] for stop in event_ends] == old['coordinates'], 'Original Frick path differs')
        split = {0: 'train', 1: 'test'}[old['role']]
        require(group_splits.setdefault(story.family, split) == split, 'Frick paired variants crossed the split')
        encoded = encode(tokenizer, text)
        for first, last in spans:
            certified_window(tokenizer, text, encoded, [first, last])
        stimuli[key] = text
        rows.append({'id': key, 'group_id': story.family, 'split': split, 'condition': story.condition,
                     'sentence_spans': spans, 'event_ends': event_ends, 'event_counts': outcome[:-1],
                     'censor_count': outcome[-1], 'target_sentence_index': story.target,
                     'target_span': spans[story.target], 'changed_span': spans[story.changed],
                     'all_report_counts': [reports[key][i] for i in range(len(event_ends))],
                     'observations': len(readers[key])})
    require(Counter(group_splits.values()) == {'train': 16, 'test': 4}, 'Unexpected prepared story split')
    provenance = {'path': FRICK, 'files_sha256': hashes, 'split_packet_sha256': digest(packet / name),
                  'split_origin': 'Prior joint-model official development families reused for diagnosis',
                  'source': 'https://doi.org/10.3758/s13421-026-01900-7',
                  'author_data': 'https://osf.io/pg67h/'}
    return rows, group_splits, provenance


def frick_position_baseline(rows):
    total_events = Counter()
    at_risk = Counter()
    train_groups = set()
    for row in rows:
        if row['split'] != 'train':
            continue
        train_groups.add(row['group_id'])
        remaining = row['censor_count'] + sum(row['event_counts'])
        for position, count in enumerate(row['event_counts']):
            total_events[position] += count
            at_risk[position] += remaining
            remaining -= count
        require(remaining == row['censor_count'], 'Inconsistent Frick risk path')
    require(all(len(row['event_ends']) <= len(at_risk) for row in rows), 'Unobserved test ordinal')
    return {'index': 'zero-based event ordinal, beginning with presented sentence 2',
            'method': 'Training-path-only pooled Bernoulli hazard, Jeffreys smoothing',
            'formula': '(events + 0.5) / (at_risk + 1)', 'training_groups': sorted(train_groups),
            'positions': [{'event_index': i, 'events': total_events[i], 'at_risk': at_risk[i],
                           'hazard': (total_events[i] + 0.5) / (at_risk[i] + 1)} for i in sorted(at_risk)]}


def evidence_examples(cache, tokenizer, stimuli):
    source = cache / EXPMRC
    assay = read_json(source / 'assay.json')
    require(assay['tokenizer_sha256'] == TOKENIZER_SHA256, 'ExpMRC tokenizer changed')
    files = {name: verified_member(source, Path(name), assay['file_sha256'][name])
             for name in ('inputs.jsonl', 'pairs.jsonl')}
    loaded = list(stream_rows(files['inputs.jsonl']))
    require(all(set(row) == {'id', 'text'} for row in loaded), 'Non-text field in the ExpMRC inputs')
    text_by_id = {row['id']: row['text'] for row in loaded}
    require(len(text_by_id) == len(loaded) == assay['documents'], 'Duplicate ExpMRC documents')
    rows = []
    group_splits = {}
    for row in stream_rows(files['pairs.jsonl']):
        key = row['id']
        require(key not in stimuli, 'Duplicate document across evaluation tasks')
        text = text_by_id[key]
        encoded = encode(tokenizer, text)
        require(len(encoded.ids) == row['input_tokens'], 'ExpMRC token count differs from its receipt')
        require(row['split'] in ('train', 'test') and row['group_id'] not in group_splits,
                'ExpMRC article family cannot occur twice')
        group_splits[row['group_id']] = row['split']
        windows = (row['earlier_window'], row['later_window'])
        require(all(window['start'] >= row['article_start'] for window in windows)
                and windows[0]['end'] <= windows[1]['start'], 'ExpMRC support windows overlap or precede article')

        def extract_window(window):
            if window is None:
                return None
            span = [window['start'], window['end']]
            require(certified_window(tokenizer, text, encoded, span) == (window['start_token'], window['end_token']),
                    'ExpMRC support window differs in tokenizer coordinates')
            return span

        by_query = {qid: extract_window(window) for qid, window in
                    zip((row['earlier_query'], row['later_query']), windows, strict=True)}
        require(len(row['queries']) == len(by_query) == 2, 'Two distinct evidence questions required')
        queries = []
        for query in row['queries']:
            qid = query['id']
            start, end = query['span']
            token_end = certified_end(tokenizer, text, encoded, end)
            require(token_end - 1 == query['end_token'] and 0 <= start < end <= row['article_start'],
                    'ExpMRC question must align to the visible prefix')
            lowest = next(i for i, (_, j) in enumerate(encoded.offsets) if j > start)
            anchor_start = 0 if lowest == 0 else encoded.offsets[lowest - 1][1]
            require(anchor_start <= start and certified_window(tokenizer, text, encoded, [anchor_start, end])
                    == (lowest, token_end), 'ExpMRC question anchor crosses a tokenizer boundary')
            window = by_query[qid]
            require(all(window[0] <= a < b <= window[1] for a, b in query['references']),
                    'The certified interval must cover each retained reference')
            queries.append({'id': qid, 'span': [start, end], 'anchor_span': [anchor_start, end],
                            'references': query['references'], 'evidence_window': window})
        earlier_sham = extract_window(row['earlier_sham'])
        if earlier_sham is not None:
            require(earlier_sham[1] <= windows[0]['start'] and earlier_sham[0] >= row['article_start']
                    and row['earlier_sham']['token_count'] == windows[0]['token_count'],
                    'ExpMRC preceding sham must be disjoint and token-width matched')
        stimuli[key] = text
        rows.append({'id': key, 'group_id': f"expmrc:{row['group_id']}", 'split': row['split'],
                     'article_start': row['article_start'], 'queries': queries, 'earlier_query': row['earlier_query'],
                     'later_query': row['later_query'], 'earlier_sham': earlier_sham,
                     'later_equal_window': extract_window(row['later_equal_window'])})
    require(len(rows) == len(text_by_id) and Counter(group_splits.values()) == {'train': 66, 'test': 27},
            'Unexpected ExpMRC split')
    provenance = {'path': EXPMRC, 'assay_sha256': digest(source / 'assay.json'), 'files_sha256': assay['file_sha256'],
                  'source': 'https://ymcui.com/expmrc/', 'source_license': assay['source_license'],
                  'split_origin': 'Existing article-family split reused for diagnosis'}
    return rows, provenance


def neutral_sentences():
    # All combinations are grammatical standalone sentences and contain no answers to the fixture questions.
    subjects = ('The editor', 'The reviewer', 'The writer', 'The lead editor', 'The assistant', 'The project assistant')
    actions = ('checked', 'reviewed', 'looked over', 'read', 'carefully checked', 'carefully looked over')
    objects = ('the notes', 'the layout', 'the margins', 'the wording', 'the section outline', 'the style guide')
    endings = ('', 'again', 'in the draft', 'in the next paragraph', 'before publication')
    for subject, action, item, suffix in product(subjects, actions, objects, endings):
        yield ' '.join(x for x in (subject, action, item, suffix) if x) + '.'


def control_examples(tokenizer, stimuli):
    neutral_by_length = defaultdict(list)
    for text in neutral_sentences():
        neutral_by_length[len(encode(tokenizer, text).ids)].append(text)
    used_neutrals = set()
    rows = []
    for i, (prefix, anchor, question, answer) in enumerate(CONTROL_TEXTS, start=1):
        answer_length = len(encode(tokenizer, answer).ids)
        neutral, replacement = islice((p for p in neutral_by_length[answer_length] if p not in used_neutrals), 2)
        used_neutrals.update((neutral, replacement))
        endings = {'early': (answer, neutral), 'late': (neutral, answer), 'none': (neutral, replacement)}
        variants = {}
        common_prefix_end = len(prefix)
        late_none_prefix_end = len(prefix + CONTROL_SEPARATOR + neutral)
        a = prefix.index(anchor)
        anchor_span = (a, a + len(anchor))
        shared_ids, late_none_ids, lengths = set(), set(), set()
        for variant, (first, second) in endings.items():
            text = CONTROL_SEPARATOR.join((prefix, first, second))
            encoded = encode(tokenizer, text)
            # Start includes the original token's leading whitespace whenever needed.
            anchor_first = next(j for j, (_, end) in enumerate(encoded.offsets) if end > anchor_span[0])
            anchor_start = 0 if anchor_first == 0 else encoded.offsets[anchor_first - 1][1]
            certified_window(tokenizer, text, encoded, (anchor_start, anchor_span[1]))
            p = certified_end(tokenizer, text, encoded, common_prefix_end)
            shared_ids.add(tuple(encoded.ids[:p]))
            if variant in ('late', 'none'):
                q = certified_end(tokenizer, text, encoded, late_none_prefix_end)
                late_none_ids.add(tuple(encoded.ids[:q]))
            lengths.add(len(encoded.ids))
            cue_span = None
            decisive_end = None
            decisive_tokens = None
            if variant != 'none':
                require(text.count(answer) == 1, 'Designed answer must occur once')
                start = text.index(answer)
                cue_span = [start, start + len(answer)]
                decisive_end = cue_span[1] - 1   # Exclude the full stop after the last answer word.
                certified_window(tokenizer, text, encoded, cue_span)
                decisive_tokens = certified_end(tokenizer, text, encoded, decisive_end)
            else:
                require(answer not in text, 'Unanswered control contains the direct answer')
            key = text_digest(text)
            require(key not in stimuli, 'Designed fixture duplicated a document')
            stimuli[key] = text
            variants[variant] = {'id': key, 'cue_span': cue_span, 'decisive_end': decisive_end,
                                 'decisive_tokens': decisive_tokens, 'tokens': len(encoded.ids)}
        require(len(shared_ids) == len(late_none_ids) == len(lengths) == 1, 'Designed prefix or token budget differs')
        middle_tokens = len(next(iter(late_none_ids)))
        require(variants['early']['decisive_tokens'] < middle_tokens < variants['late']['decisive_tokens'],
                'Designed answer was not delayed through the middle window')
        rows.append({'group_id': f'control:{i:02}', 'split': 'test', 'question': question,
                     'anchor_span': [anchor_start, anchor_span[1]], 'anchor_text_span': list(anchor_span),
                     'variants': variants, 'common_prefix_end': common_prefix_end,
                     'late_none_prefix_end': late_none_prefix_end})
    return rows


def twelve_word_windows(text):
    words = re.findall(r'\w+', text.casefold())
    return (tuple(words[offset:offset + 12]) for offset in range(len(words) - 11))


def remove_expmrc_training_copies(stimuli, rows):
    test_index = defaultdict(set)
    for row in rows:
        if row['split'] == 'test':
            for window in twelve_word_windows(stimuli[row['id']]):
                test_index[window].add(row['id'])
    kept, rejected = [], []
    for row in rows:
        if row['split'] == 'test':
            kept.append(row)
            continue
        matches = Counter()
        for window in set(twelve_word_windows(stimuli[row['id']])):
            matches.update(test_index.get(window, ()))
        if matches:
            rejected.append({'id': row['id'], 'group_id': row['group_id'],
                             'test_matches_unique_twelve_word_windows': dict(sorted(matches.items()))})
            del stimuli[row['id']]
        else:
            kept.append(row)
    return kept, rejected


def verify_split_overlap(stimuli, frick, evidence, controls):
    train, test = set(), set()
    for row in frick + evidence:
        (train if row['split'] == 'train' else test).add(row['id'])
    for row in controls:
        test.update(variant['id'] for variant in row['variants'].values())
    require(not (train & test) and train | test == stimuli.keys(), 'A document crossed the split or was unassigned')

    def ngrams(ids):
        for key in sorted(ids):
            yield from twelve_word_windows(stimuli[key])

    train_ngrams = set(ngrams(train))
    test_windows = 0
    for span in ngrams(test):
        require(span not in train_ngrams, 'A twelve-word text segment crossed the split')
        test_windows += 1
    return {'train_documents': len(train), 'test_documents': len(test), 'n': 12,
            'unique_train_windows': len(train_ngrams), 'test_windows': test_windows, 'overlap_windows': 0}


def prepare(cache, tokenizer_path, output):
    require(not output.exists(), f'Packet already exists at {output}')
    require(digest(tokenizer_path) == TOKENIZER_SHA256, 'Tokenizer does not match the fixed Qwen revision')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    stimuli = {}
    frick, frick_groups, frick_source = frick_examples(cache, tokenizer, stimuli)
    evidence, evidence_source = evidence_examples(cache, tokenizer, stimuli)
    evidence, rejected = remove_expmrc_training_copies(stimuli, evidence)
    require(Counter(row['split'] for row in evidence) == {'train': 64, 'test': 27},
            'Unexpected number of source copies crossing the split')
    evidence_source['training_copies_removed'] = rejected
    controls = control_examples(tokenizer, stimuli)
    isolation = verify_split_overlap(stimuli, frick, evidence, controls)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.token-readout-') as temporary:
        stage = Path(temporary) / 'prepared'
        stage.mkdir()
        write_rows(stage / 'stimuli.jsonl', [{'id': key, 'text': text} for key, text in sorted(stimuli.items())])
        write_json(stage / 'design.json', {
            'schema_version': 1, 'frick': frick, 'evidence': evidence, 'controls': controls,
            'frick_group_splits': frick_groups, 'frick_position_baseline': frick_position_baseline(frick),
            'sources': {'frick': frick_source, 'evidence': evidence_source, 'controls': 'Agent-authored test fixtures'},
            'positions': 'Unicode character offsets, half-open spans and exclusive prefix ends',
            'compatibility': {
                'frick': 'First explicit D after a complete sentence. Prescribed first sentence K omitted from events. '
                         'A later K is not evidence that an earlier doubt resolved. The test material was inspected historically.',
                'evidence': 'A supplied question appears before the article. Windows are retrospective human evidence annotations, '
                            'not human reports of when a doubt resolved. The test material was inspected historically.',
                'controls': 'Sealed engineering conditions with matched prefixes and budgets. No human responses.',
            },
        })
        write_json(stage / 'receipt.json', {'schema_version': 1, 'model_revision': REVISION,
                   'tokenizer_sha256': TOKENIZER_SHA256, 'documents': len(stimuli), 'frick_families': len(frick_groups),
                   'frick_trials': sum(row['observations'] for row in frick), 'evidence_articles': len(evidence),
                   'control_families': len(controls), 'split_isolation': isolation,
                   'file_sha256': {name: digest(stage / name) for name in ('stimuli.jsonl', 'design.json')}})
        stage.rename(output)


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default=Path('.cache/cogllm'), type=Path)
    parser.add_argument('--tokenizer', default=TOKENIZER, type=Path)
    parser.add_argument('--output', default=Path('.cache/cogllm/token-readout/prepared'), type=Path)
    args = parser.parse_args()
    prepare(args.cache, args.tokenizer, args.output)
