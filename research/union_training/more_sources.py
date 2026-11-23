"""Human outcome adapters for prior lexical, interpretation, and quality assays."""

from collections import defaultdict
import csv
import math

from research.backbone_transfer.extract import read_json

BINARY = (0, 1)
FIVE = (1, 2, 3, 4, 5)


def oliver(union, indexed):
    source = 'oliver'
    base = 'semantic-tracking/oliver-retrieval/prepared'
    stimuli = indexed(union.rows(source, f'{base}/stimuli.jsonl'))
    layout = indexed(union.rows(source, f'{base}/layout.jsonl'))
    if stimuli.keys() != layout.keys():
        raise ValueError('Oliver definitions and lexical target families differ')
    keys = {
        item: union.document(source, layout[item]['family'], row['text'])
        for item, row in stimuli.items()
    }
    states = ('know', 'dont_know', 'tot')
    paths = tuple(
        (state, delay, confirmed) for state in states
        for delay in ((None,) if state == 'know' else (False, True))
        for confirmed in (False, True)
    )
    labels = {
        (state, delay, confirmed): f'{state}|delay_{delay}|confirmation_{confirmed}'
        for state, delay, confirmed in paths
    }
    for trial in union.rows(source, f'{base}/outcomes.jsonl'):
        first = trial['first']
        if first['id'] is None or trial['second']['id'] is None:
            union.filters['oliver_definition_without_exact_version'] += 1
            continue
        item = first['id']
        key, stop = keys[item], layout[item]['definition_stop']
        outcome = (first['state'], first['recall_signaled_before_reveal'], first['same_target_confirmed'])
        if layout[item]['family'] != trial['family'] or type(first['same_target_confirmed']) is not bool:
            raise ValueError('Oliver initial retrieval identity or confirmation differs from its published definition')
        union.category('oliver_retrieval_protocol_path', source, key, stop, tuple(labels.values()), labels[outcome])
        for phase in ('first', 'second'):
            row = trial[phase]
            item = row['id']
            key, stop = keys[item], layout[item]['definition_stop']
            if layout[item]['family'] != trial['family']:
                raise ValueError('Oliver stages must share their authored target family')
            if phase == 'second':
                union.category('oliver_retest_initial_state', source, key, stop, states, row['state'])
                union.category('oliver_retest_target_confirmation', source, key, stop, BINARY,
                               row['same_target_confirmed'])
            for field in ('definition_rt', 'confirmation_rt', 'delay_rt'):
                time = row.get(field)
                if time is not None and time > 0:
                    union.scalar(f'oliver_{phase}_{field}_log_milliseconds', source, key, stop, math.log(time))


def scarecrow(union, indexed):
    source = 'scarecrow'
    base = 'semantic-tracking/scarecrow-detection/prepared'
    stimuli = indexed(union.rows(source, f'{base}/stimuli.jsonl'))
    layout = indexed(union.rows(source, f'{base}/layout.jsonl'))
    outcomes = indexed(union.rows(source, f'{base}/outcomes.jsonl'))
    if stimuli.keys() != layout.keys() or outcomes.keys() != layout.keys():
        raise ValueError('Scarecrow continuation, annotation and group inventories differ')
    for item, row in stimuli.items():
        key = union.document(source, layout[item]['group_id'], row['text'])
        for (_, stop), result in zip(layout[item]['words'], outcomes[item]['counts'], strict=True):
            count, positive, grades = result['total'], result['positive'], result['severity_counts']
            if not 0 <= positive <= count or positive != sum(grades) or len(grades) != 3:
                raise ValueError('Scarecrow word outcome denominators and marked severity differ')
            if not count:
                union.filters['scarecrow_no_at_risk_vote'] += 1
                continue
            union.counts('scarecrow_retrospective_contradiction_mark_ordinal', source, key, stop,
                         ('unmarked', 'severity_1', 'severity_2', 'severity_3'), [count - positive, *grades])


def czech(union, indexed):
    source = 'czech'
    base = 'semantic-tracking/czech-garden-path/prepared'
    stimuli = indexed(union.rows(source, f'{base}/stimuli.jsonl'))
    layout = indexed(union.rows(source, f'{base}/layout.jsonl'))
    if stimuli.keys() != layout.keys():
        raise ValueError('Czech texts and lexical conditions differ')
    keys = {
        item: union.document(source, layout[item]['family'], row['text'])
        for item, row in stimuli.items()
    }
    pairs = defaultdict(dict)
    labels = ('no', 'yes', 'unsure')
    for row in union.rows(source, f'{base}/outcomes.jsonl'):
        item, response = row['id'], row['response']
        if response == 'missing':
            union.filters['czech_unrecorded_answer'] += 1
            continue
        name = f"czech_exp{row['experiment']}_{row['question']}"
        condition = layout[item]
        observed = union.category(name, source, keys[item], len(stimuli[item]['text']), labels, response)
        pairs[name, condition['family'], condition['animacy']][condition['condition']] = observed
    for (name, _, _), pair in pairs.items():
        if pair.keys() == {'gp', 'nongp'}:
            union.contrast(name, 'garden_path_minus_unambiguous_yes', [pair['nongp']], [pair['gp']],
                           1.0, metric='category', category=1)


def blott(union, indexed):
    source = 'blott'
    cohorts = (
        ('2021', 'reading-trace-source-audit/blott-holdout/model_input/stimuli.jsonl',
         'reading-trace-source-audit/blott-holdout/scoring/design.jsonl',
         'reading-trace-source-audit/blott-holdout/scoring/human_trials.jsonl', 'coherent', 'PlausAmb', 'PlausUA'),
        ('2023', 'semantic-tracking/blott-ambiguity/prepared/stimuli.jsonl',
         'semantic-tracking/blott-ambiguity/prepared/layout.jsonl',
         'semantic-tracking/blott-ambiguity/prepared/outcomes.jsonl', 'sense', 'Amb', 'UA'),
    )
    for year, corpus, design, outcomes, response, ambiguous, unambiguous in cohorts:
        stimuli = indexed(union.rows(source, corpus))
        layout = indexed(union.rows(source, design))
        if stimuli.keys() != layout.keys():
            raise ValueError('Blott sentence stimuli and design do not reconcile')
        keys = {
            item: union.document(source, f"{year}:{layout[item]['family']}", row['text'])
            for item, row in stimuli.items()
        }
        name = f'blott_{year}_sentence_end_sense'
        conditions = defaultdict(dict)
        for trial in union.rows(source, outcomes):
            item = trial['id']
            key, stop = keys[item], len(stimuli[item]['text'])
            anchor = union.category(name, source, key, stop, BINARY, trial[response])
            meta = layout[item]
            condition = meta['condition'] if year == '2021' else meta['ambiguity']
            if condition in (ambiguous, unambiguous):
                conditions[meta['family']][condition] = anchor
            if trial.get('rt_ms', 0) > 0:
                union.scalar(f'blott_{year}_sense_response_log_milliseconds', source, key, stop,
                             math.log(trial['rt_ms']))
        for pair in conditions.values():
            if pair.keys() == {ambiguous, unambiguous}:
                union.contrast(name, 'ambiguity_minus_unambiguous_sense', [pair[unambiguous]], [pair[ambiguous]],
                               1.0, metric='category', category=1)


def ambistory(union, indexed):
    source = 'ambistory'
    base = 'ambistory/causal-v3'
    manifest = read_json(union.input(source, f'{base}/manifest.json'))
    for member, expected in manifest['files_sha256'].items():
        relative = f'{base}/{member}'
        union.input(source, relative)
        if union.sources[source][relative] != expected:
            raise ValueError(f'AmbiStory input differs from original preparation: {member}')
    name = 'ambistory_given_sense_plausibility'
    pairs = defaultdict(lambda: defaultdict(set))
    for portion in ('train', 'dev', 'test'):
        stimuli = indexed(union.rows(source, f'{base}/{portion}/model_input/stimuli.jsonl'))
        layout = indexed(union.rows(source, f'{base}/{portion}/design/stops.jsonl'))
        if stimuli.keys() != layout.keys():
            raise ValueError('AmbiStory ratings need the released matching story layouts')
        for row in union.rows(source, f'{base}/{portion}/scoring/ordinal-ratings.jsonl'):
            item, sense_id = row['id'], row['sense_id']
            metadata = layout[item]
            [sense] = [item for item in metadata['senses'] if item['id'] == sense_id]
            text = stimuli[item]['text']
            if len(text) != metadata['post_stop']:
                raise ValueError('Expected the entire author-rated story version')
            # Rating task supplies the intended interpretation, so the complete
            # input is the visible story and the supplied definition.
            prompt = f"{text}\n\nMeaning of {metadata['homonym']}:\n{sense['definition']}"
            key = union.document(source, metadata['homonym'].casefold(), prompt)
            anchor = union.counts(name, source, key, len(prompt), FIVE,
                                  [row['ratings'].count(grade) for grade in FIVE])
            if len(row['ratings']) != len(row['nonsensical']):
                raise ValueError('AmbiStory rating and plausibility flag denominators differ')
            union.counts('ambistory_given_sense_nonsensical_flag', source, key, len(prompt),
                         BINARY, [row['nonsensical'].count(False), row['nonsensical'].count(True)])
            pairs[metadata['family'], sense_id][metadata['has_ending']].add(anchor)
    for variants in pairs.values():
        if variants.keys() != {False, True} or len(variants[False]) != 1 or len(variants[True]) != 2:
            raise ValueError('AmbiStory must retain one shared prefix and two endings per sense')
        union.contrast(name, 'mean_ending_minus_base_for_given_sense',
                       sorted(variants[False]), sorted(variants[True]), 4.0)


def race(union, indexed):
    source = 'race'
    base = 'stage6/race'
    stimuli = indexed(union.rows(source, f'{base}/inputs.jsonl'))
    design = indexed(union.rows(source, f'{base}/design.jsonl'))
    result = 'stage6/race-runs/cogllm-stage6-race-20261120T181401Z-d6e92f78/local-outcomes/outcomes.jsonl'
    outcomes = indexed(union.rows(source, result))
    if outcomes.keys() != design.keys():
        raise ValueError('Human RACE choices must align with the exact displayed question and options')
    for item, row in outcomes.items():
        text = stimuli[design[item]['full_id']]['text']
        key = union.document(source, row['article_id'], text)
        stop = len(text)
        union.category('race_answer_choice', source, key, stop, (0, 1, 2, 3), row['selected'])
        union.category('race_answer_accuracy', source, key, stop, BINARY, row['selected'] == row['correct'])


def textcomplexityde(union, indexed):
    source = 'textcomplexityde'
    base = 'textcomplexityde'
    assay = read_json(union.input(source, f'{base}/transfer-v2/assay.json'))
    text_name = f'{base}/transfer-v2/stimuli.jsonl'
    stimuli = indexed(union.rows(source, text_name))
    if (len(stimuli) != assay['cases']
            or union.sources[source][text_name] != assay['model_input_hashes']['stimuli.jsonl']):
        raise ValueError('TextComplexityDE exact attributed input inventory differs from its receipt')
    for name, expected in assay['author_file_sha256'].items():
        relative = f'{base}/raw/{name}'
        union.input(source, relative)
        if union.sources[source][relative] != expected:
            raise ValueError(f'TextComplexityDE release file changed: {name}')
    with union.input(source, f'{base}/raw/ratings.csv').open(encoding='cp1252', newline='') as stream:
        ratings = {f'textde-{row["ID"]}': row for row in csv.DictReader(stream)}
    fields = (
        ('Complexity', 'Votes_Complexity'),
        ('Understandability', 'Votes_Understandability'),
        ('Lexical_difficulty', 'Vote_Lexical_difficulty'),
    )
    for item, row in stimuli.items():
        observed = ratings[item]
        text = row['text']
        if observed['Sentence'] != text:
            raise ValueError('TextComplexityDE written sentence differs from the rated sentence')
        key = union.document(source, f"wikipedia-de-{observed['Article_ID']}", text)
        for field, votes in fields:
            count = int(observed[votes])
            mean, sample_sd = (float(observed[f'{prefix}_{field}']) for prefix in ('MOS', 'Std'))
            total = count * mean
            squared_total = (count - 1) * sample_sd ** 2 + count * mean ** 2
            if (
                count < 2 or not 1 <= mean <= 7 or sample_sd < 0
                or not math.isclose(total, round(total), abs_tol=1e-5)
                or not math.isclose(squared_total, round(squared_total), abs_tol=1e-5)
            ):
                raise ValueError('Rounded TextComplexityDE moments do not recover integral rating statistics')
            mean = round(total) / count
            variance = max(0.0, round(squared_total) / count - mean ** 2)
            union.scalar(f'textcomplexityde_{field.casefold()}', source, key, len(text), mean, count, variance)


def add_more_sources(union):
    from research.union_training.sources import indexed

    for prepare in (oliver, scarecrow, czech, blott, ambistory, race, textcomplexityde):
        prepare(union, indexed)
