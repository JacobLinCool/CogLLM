"""Adapters for licensed, aligned human outcomes in the local research cache."""

from collections import defaultdict
import math

from research.backbone_transfer.extract import read_json


BINARY = (0, 1)
FIVE = tuple(range(1, 6))
CHOICES = tuple('ABCD')


def indexed(rows):
    result = {row['id']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Repeated stimulus identifier')
    return result


def observation(trial, key):
    field = trial['observations'].get(key)
    return field['value'] if field and field['status'] == 'observed' else None


def question_text(text, context, preserve_order):
    options = [option['text'] for option in context['options']]
    if not preserve_order:
        # The release does not preserve the PLABA presentation order.
        options.sort()
    return text + '\n\n' + context['question'] + '\n' + '\n'.join(options)


def plaba(union):
    source = 'plaba'
    materials = indexed(union.rows(source, 'stage2/plaba/stimuli.jsonl'))
    aligned = indexed(union.rows(source, 'expository-lora/prepared/plaba/ratings.jsonl'))
    direct = union.rows(source, 'stage2/plaba/trials.jsonl')
    tags = sorted({tag for row in direct for tag in (observation(row, 'issue_tags') or [])})
    keys = {
        key: union.document(source, aligned[key]['group_id'], row['text'])
        for key, row in materials.items()
    }
    stops = {segment['id']: segment['end'] for row in materials.values() for segment in row['segments']}
    ordinal = ('processing_ease', 'perceived_understanding')
    for trial in direct:
        record = trial['stimulus_id']
        key = keys[record]
        if trial['target_segment_id'] is not None:
            stop = stops[trial['target_segment_id']]
            for name in ordinal:
                value = observation(trial, name)
                if value is not None:
                    union.category(f'plaba_{name}', source, key, stop, FIVE, value)
            issues = observation(trial, 'issue_tags')
            if issues is not None:
                union.category('plaba_issue_report', source, key, stop, BINARY, bool(issues))
                for tag in tags:
                    union.category(f'plaba_issue_{tag}', source, key, stop, BINARY, tag in issues)
            continue
        context = trial['task_context']
        if trial['exposure']['task'] == 'document_rating':
            stop = trial['exposure']['visible_end']
            for name in ('overall_processing_ease', 'overall_perceived_understanding'):
                value = observation(trial, name)
                if value is not None:
                    union.category(f'plaba_{name}', source, key, stop, FIVE, value)
            continue
        if not context:
            raise ValueError('Expected observed open-book question')
        text = question_text(materials[record]['text'], context, preserve_order=False)
        question = union.document(source, aligned[record]['group_id'], text)
        union.category('plaba_open_book_accuracy', source, question, len(text), BINARY,
                       observation(trial, 'mcq_correct'))
        for name in ('quiz_effort', 'quiz_confidence'):
            value = observation(trial, name)
            if value is not None:
                union.category(f'plaba_{name}', source, question, len(text), FIVE, value)
    for row in union.rows(source, 'expository-lora/prepared/plaba/pairs.jsonl'):
        left = [(keys[key], stop) for key, stop in row['original']]
        right = [(keys[key], stop) for key, stop in row['plain']]
        for name in ordinal:
            union.contrast(f'plaba_{name}', 'plain_minus_original', left, right, 4.0)


def zhan(union):
    source = 'zhan'
    materials = indexed(union.rows(source, 'stage2/zhan/stimuli.jsonl'))
    keys = {key: union.document(source, row['family_id'], row['text']) for key, row in materials.items()}
    families = defaultdict(dict)
    for key, row in materials.items():
        families[row['family_id']][row['metadata']['prompt_type']] = (keys[key], len(row['text']))
    for row in union.rows(source, 'stage2/zhan/trials.jsonl'):
        key = row['stimulus_id']
        doc, stop = keys[key], row['exposure']['visible_end']
        union.category('zhan_remention', source, doc, stop, ('np1', 'np2'), observation(row, 'reMention'))
        form = observation(row, 'referenceForm')
        if form is not None:
            union.category('zhan_reference_form', source, doc, stop, ('name', 'overtPro', 'nullPro'), form)
    for prompts in families.values():
        other, = set(prompts) - {'free'}
        union.contrast('zhan_remention', 'pronoun_prompt_minus_free', [prompts['free']], [prompts[other]],
                       1.0, metric='category', category=0)


def onestop(union):
    source = 'onestop'
    materials = indexed(union.rows(source, 'stage2/onestop/stimuli.jsonl'))
    by_item = defaultdict(lambda: defaultdict(set))
    for trial in union.rows(source, 'stage2/onestop/trials.jsonl'):
        material = materials[trial['stimulus_id']]
        text = question_text(material['text'], trial['task_context'], preserve_order=True)
        key = union.document(source, material['family_id'], text)
        stop = len(text)
        union.category('onestop_answer_choice', source, key, stop, CHOICES,
                       observation(trial, 'mcq_selected_option'))
        union.category('onestop_answer_accuracy', source, key, stop, BINARY, observation(trial, 'mcq_correct'))
        meta = material['metadata']
        identity = (*[meta[k] for k in ('source_article_number', 'source_paragraph_number')],
                    observation(trial, 'mcq_correct_option'),
                    trial['task_context']['question'])
        by_item[identity][meta['difficulty']].add((key, stop))
    for variants in by_item.values():
        if variants.keys() == {'adv', 'ele'}:
            union.contrast('onestop_answer_accuracy', 'elementary_minus_advanced',
                           sorted(variants['adv']), sorted(variants['ele']), 1.0, metric='category', category=1)


def complex_words(union):
    source = 'complex'
    base = 'backbone-transfer/prepared/complex'
    materials = indexed(union.rows(source, f'{base}/stimuli.jsonl'))
    outcomes = {row['anchor_id']: row['value'] for row in union.rows(source, f'{base}/outcomes.jsonl')}
    for row in union.rows(source, f'{base}/layout.jsonl'):
        text = materials[row['id']]['text']
        key = union.document(source, row['group_id'], text)
        union.scalar('complex_word_difficulty', source, key, row['end'], outcomes[row['anchor_id']])


def cwi(union):
    source = 'cwi'
    materials = indexed(union.rows(source, 'core-features/cwi/stimuli.jsonl'))
    for row in union.rows(source, 'core-features/cwi/outcomes.jsonl'):
        key = union.document(source, row['group_id'], materials[row['id']]['text'])
        for population in ('native', 'nonnative'):
            positive = row[f'{population}_marked']
            negative = row[f'{population}_total'] - positive
            union.counts(f'cwi_{population}_marked', source, key, row['stop'], BINARY, [negative, positive])


def first_readback(union):
    source = 'first-readback'
    base = 'semantic-tracking/first-readback-v1'
    material = indexed(union.rows(source, f'{base}/stimuli.jsonl'))
    for part in ('train', 'test'):
        for row in union.rows(source, f'{base}/{part}.jsonl'):
            key = union.document(source, f"{row['study']}:{row['group_id']}", material[row['id']]['text'])
            union.counts('first_readback_risk', source, key, row['stop'], BINARY,
                         [row['negative'], row['positive']])
    # This overlapping release adds the linked acceptability response only.
    for row in union.rows(source, 'core-features/readback/trials.jsonl'):
        if row['acceptable'] is None:
            continue
        text = material[row['id']]['text']
        key = union.document(source, f"english:{row['group_id']}", text)
        union.category('first_readback_end_acceptability', source, key, len(text), BINARY, row['acceptable'])


def context_readback(union):
    from research.semantic_tracking.context_readback_prepare import design, observed_counts, read_regions

    source = 'context-readback'
    base = 'semantic-tracking/context-6uv37-source'
    authored = union.input(source, f'{base}/alt_stim.txt')
    events = union.input(source, f'{base}/fixations.Rda')
    published = read_json(union.input(source, 'semantic-tracking/context-readback-v1/assay.json'))
    if {path.name: union.sources[source][f'{base}/{path.name}'] for path in (authored, events)} != {
        path.name: published['source_sha256'][path.name] for path in (authored, events)
    }:
        raise ValueError('Author readback records fail the previously verified hashes')
    regions = read_regions(authored)
    texts, targets = design(regions)
    prior = indexed(union.rows(source, 'semantic-tracking/context-readback-v1/stimuli.jsonl'))
    if prior != indexed(list(texts.values())):
        raise ValueError('Context readback reconstruction differs from the registered archive')
    by_context = defaultdict(dict)
    for row in observed_counts(events, regions, texts, targets):
        doc = union.document(source, row['family'], prior[row['id']]['text'])
        point = union.counts('context_first_readback', source, doc, row['stop'], BINARY,
                             [row['negative'], row['positive']])
        if row['phase'] in {'critical', 'spill'}:
            by_context[row['family'], row['disamb'], row['phase']][row['context']] = point
    for (_, disambiguation, phase), pair in by_context.items():
        if pair.keys() == {'oneref', 'tworef'}:
            union.contrast('context_first_readback', f'{disambiguation}_{phase}_two_minus_one',
                           [pair['oneref']], [pair['tworef']], 1.0, metric='category', category=1)


def kerwer(union):
    source = 'kerwer'
    base = 'kerwer-2021/holdout'
    materials = indexed(union.rows(source, f'{base}/model_input/stimuli.jsonl'))
    design = indexed(union.rows(source, f'{base}/scoring/design.jsonl'))
    keys = {}
    by_article = defaultdict(dict)
    for identifier, material in materials.items():
        text = material['text']
        key = union.document(source, design[identifier]['doi'], text)
        keys[identifier] = key, len(text)
        by_article[design[identifier]['doi']][design[identifier]['arm']] = keys[identifier]
    endpoints = {'comprehensibility': tuple(range(1, 9)), 'confusion': FIVE, 'frustration': FIVE}
    for human in union.rows(source, f'{base}/scoring/public-study-ratings.jsonl'):
        key, stop = keys[human['id']]
        for name, levels in endpoints.items():
            union.category(f'kerwer_{name}', source, key, stop, levels, human[name])
    for point in by_article.values():
        for name, levels in endpoints.items():
            for label, left, right in [
                ('headings', 'plain_without_headings', 'plain_with_headings'),
                ('plain', 'scientific_abstract', 'plain_without_headings'),
            ]:
                union.contrast(f'kerwer_{name}', label, [point[left]], [point[right]], len(levels) - 1)


def currie(union):
    source = 'currie'
    base = 'semantic-tracking/larrc-consistency/prepared'
    materials = indexed(union.rows(source, f'{base}/stimuli.jsonl'))
    layout = indexed(union.rows(source, f'{base}/layout.jsonl'))
    paired = defaultdict(dict)
    for row in union.rows(source, f'{base}/outcomes.jsonl'):
        identifier = row['id']
        meta = layout[identifier]
        text = materials[identifier]['text']
        key = union.document(source, meta['family'], text)
        coordinate = union.category('currie_end_coherence_report', source, key, len(text), BINARY, row['report'])
        paired[meta['family'], meta['genre']][meta['condition']] = coordinate
    for (_, genre), conditions in paired.items():
        if conditions.keys() == {'near', 'far'}:
            union.contrast('currie_end_coherence_report', f'{genre}_far_minus_near', [conditions['near']],
                           [conditions['far']], 1.0, metric='category', category=1)


def clear(union):
    source = 'clear'
    base = 'clear-reading-ease/prepared-v1'
    materials = indexed(union.rows(source, f'{base}/stimuli.jsonl'))
    design = indexed(union.rows(source, f'{base}/materials.jsonl'))
    for row in union.rows(source, f'{base}/outcomes.jsonl'):
        identifier = row['id']
        text = materials[identifier]['text']
        key = union.document(source, design[identifier]['family'], text)
        union.scalar('clear_excerpt_easiness', source, key, len(text), row['bt_easiness'])


def rodger(union):
    source = 'rodger'
    base = 'semantic-tracking/rodger-belief/prepared'
    texts = indexed(union.rows(source, f'{base}/stimuli.jsonl'))
    design = indexed(union.rows(source, f'{base}/stimulus_metadata.jsonl'))
    # Policy topic has only six levels. Variant families remain together, with
    # matched pre/post ratings available on the same reader exposure.
    for row in union.rows(source, f'{base}/outcomes.jsonl'):
        policy = row['policy']
        post = texts[row['id']]['text']
        baseline_id, = (identifier for identifier, meta in design.items()
                        if meta['policy'] == policy and meta['condition'] is None)
        before = texts[baseline_id]['text']
        earlier = union.document(source, policy, before)
        after = union.document(source, policy, post)
        for endpoint in ('effectiveness', 'support'):
            before_score = row[f'prior_{endpoint}']
            after_score = row[f'posterior_{endpoint}']
            union.scalar(f'rodger_rating_{endpoint}', source, earlier, len(before), before_score)
            union.scalar(f'rodger_rating_{endpoint}', source, after, len(post), after_score)
            union.scalar(f'rodger_change_{endpoint}', source, after, len(post), after_score - before_score)


def frick(union):
    source = 'frick'
    base = 'semantic-tracking/frick-consistency/prepared'
    materials = indexed(union.rows(source, f'{base}/stimuli.jsonl'))
    layout = indexed(union.rows(source, f'{base}/metadata.jsonl'))
    by_story = defaultdict(dict)
    keys = {identifier: union.document(source, row['story'], materials[identifier]['text'])
            for identifier, row in layout.items()}
    for trial in union.rows(source, f'{base}/outcomes.jsonl'):
        key = trial['id']
        desc = layout[key]
        first = True
        for rank, (segment, keypress, rt) in enumerate(zip(desc['sentences'], trial['keys'], trial['rt_seconds'], strict=True), 1):
            if rank == 1:
                continue  # The task instructions prescribe 'k' on the first sentence.
            doc, stop = keys[key], segment['stop']
            observed = keypress == 'd'
            if keypress not in {'k', 'd'}:
                raise ValueError('Unexpected Frick decision key')
            point = union.category('frick_sentence_inconsistent', source, doc, stop, BINARY, observed)
            if rt > 0.0:
                union.scalar('frick_log_keypress_seconds', source, doc, stop, math.log(rt))
            if first:
                union.category('frick_first_report_hazard', source, doc, stop, BINARY, observed)
            if rank == desc['target_index'] + 1:
                union.category('frick_target_first_report', source, doc, stop, BINARY, first and observed)
                by_story[desc['story']][desc['condition']] = point
            if observed:
                first = False
    for pair in by_story.values():
        if set(pair) != {'01', '02'}:
            raise ValueError('Incomplete matched Frick contrast')
        for name in ('frick_sentence_inconsistent', 'frick_target_first_report'):
            union.contrast(name, 'inconsistent_minus_consistent', [pair['01']], [pair['02']], 1.0,
                           metric='category', category=1)


def add_sources(union):
    from research.union_training.extra_sources import add_extra_sources
    from research.union_training.more_sources import add_more_sources

    for reader in (plaba, zhan, onestop, complex_words, cwi, first_readback,
                   context_readback, kerwer, currie, clear, rodger, frick, add_more_sources, add_extra_sources):
        reader(union)
