"""Recover joint rating and preference observations from the registered studies."""

from research.backbone_transfer.extract import digest, read_json, stream_rows
from research.union_training.sources import CHOICES, indexed, observation, question_text


def read_rodger(base, coordinates, add):
    materials = indexed(list(stream_rows(base / 'stimuli.jsonl')))
    design = indexed(list(stream_rows(base / 'stimulus_metadata.jsonl')))
    if materials.keys() != design.keys():
        raise ValueError('Rodger stimuli and presentation layout differ')

    baselines = {}
    for identifier, row in design.items():
        if row['condition'] is None:
            if row['policy'] in baselines:
                raise ValueError('Repeated Rodger baseline for one policy')
            text = materials[identifier]['text']
            baseline, = row['pages']
            if baseline['start'] != 0 or baseline['stop'] != len(text):
                raise ValueError('Rodger baseline boundary differs from the released text')
            baselines[row['policy']] = text

    seen_trials = set()
    for row in stream_rows(base / 'outcomes.jsonl'):
        trial = row['reader_id'], row['policy']
        if trial in seen_trials:
            raise ValueError('Repeated Rodger reader-policy exposure')
        seen_trials.add(trial)
        identifier = row['id']
        layout = design[identifier]
        if (layout['policy'], layout['condition']) != (row['policy'], row['condition']):
            raise ValueError('Rodger trial does not match its assigned evidence')
        before = baselines[row['policy']]
        after = materials[identifier]['text']
        baseline, evidence = layout['pages']
        if (baseline['start'] != 0 or baseline['stop'] != len(before)
                or not after.startswith(before + '\n\n') or evidence['stop'] != len(after)):
            raise ValueError('Rodger pre/post material boundary is inconsistent')
        document = coordinates(after)
        points = [(document, len(before)), (document, len(after))]
        for endpoint in ('effectiveness', 'support'):
            pair = [row[f'prior_{endpoint}'], row[f'posterior_{endpoint}']]
            if any(type(value) is not int or not 0 <= value <= 100 for value in pair):
                raise ValueError('Rodger rating is outside the observed slider scale')
            add(f'rodger_paired_{endpoint}', points, pair)
    if len(seen_trials) != 1806:
        raise ValueError('Rodger registered trial count changed')


def read_turkish(base, coordinates, add):
    receipt = read_json(base / 'assay.json')
    hashes = receipt['file_sha256']
    for member in ('model_input/stimuli.jsonl', 'sentence-metadata.jsonl',
                   'fit.jsonl', 'fit-truth.jsonl', 'test.jsonl', 'test-truth.jsonl'):
        if digest(base / member) != hashes[member]:
            raise ValueError(f'Turkish released preference file changed: {member}')
    texts = indexed(list(stream_rows(base / 'model_input/stimuli.jsonl')))
    metadata = indexed(list(stream_rows(base / 'sentence-metadata.jsonl')))
    if texts.keys() != metadata.keys():
        raise ValueError('Turkish sentences and connected component identifiers differ')

    seen_trials = set()
    retained = 0
    for split in ('fit', 'test'):
        design = indexed(list(stream_rows(base / f'{split}.jsonl')))
        outcomes = indexed(list(stream_rows(base / f'{split}-truth.jsonl')))
        if design.keys() != outcomes.keys() or seen_trials.intersection(design):
            raise ValueError('Turkish comparison design must join to one outcome per trial')
        seen_trials.update(design)
        for identifier, row in design.items():
            earlier, later = row['earlier_id'], row['later_id']
            first, second = metadata[earlier], metadata[later]
            if (first['component_id'], first['word_count']) != (second['component_id'], second['word_count']):
                raise ValueError('Turkish trial violates the fixed same-length comparison set')
            result = outcomes[identifier]
            if type(result['quality_retained']) is not bool or type(result['earlier_easier']) is not bool:
                raise ValueError('Turkish outcomes must record Boolean quality and preference')
            if not result['quality_retained']:
                continue
            first_text = texts[earlier]['text']
            presentation = first_text + '\n\n' + texts[later]['text']
            document = coordinates(presentation)
            counts = [0, 0]
            counts[int(result['earlier_easier'])] = 1
            add('turkish_sentence_comparison', [(document, len(first_text)), (document, len(presentation))], counts)
            retained += 1
    if retained != 2183:
        raise ValueError('Turkish registered preference count changed')


def read_pairs(cache, coordinates, add):
    """Emit original ordered ratings and choices at the evidence and response boundaries."""
    read_rodger(cache / 'semantic-tracking/rodger-belief/prepared', coordinates, add)
    read_turkish(cache / 'turkish-v5/assay-v1', coordinates, add)


def choice_keys(cache, coordinates):
    """Yield one correct option index per verified displayed OneStopQA or RACE input."""
    onestop = cache / 'stage2/onestop'
    materials = indexed(list(stream_rows(onestop / 'stimuli.jsonl')))
    answers = {}
    for trial in stream_rows(onestop / 'trials.jsonl'):
        text = question_text(materials[trial['stimulus_id']]['text'], trial['task_context'], preserve_order=True)
        point = (coordinates(text), len(text))
        key = CHOICES.index(observation(trial, 'mcq_correct_option'))
        choice = CHOICES.index(observation(trial, 'mcq_selected_option'))
        if observation(trial, 'mcq_correct') != (key == choice) or answers.setdefault(point, key) != key:
            raise ValueError('OneStopQA key conflicts with the displayed item or scored choice')
    yield from (('onestop_answer_choice', point, key) for point, key in sorted(answers.items()))

    race = cache / 'stage6/race'
    version = 'cogllm-stage6-race-20261120T181401Z-d6e92f78'
    outcome_path = cache / 'stage6/race-runs' / version / 'local-outcomes/outcomes.jsonl'
    stimuli = indexed(list(stream_rows(race / 'inputs.jsonl')))
    design = indexed(list(stream_rows(race / 'design.jsonl')))
    outcomes = indexed(list(stream_rows(outcome_path)))
    if outcomes.keys() != design.keys():
        raise ValueError('RACE trial keys disagree with the displayed item manifest')
    answers = {}
    for identifier, row in outcomes.items():
        text = stimuli[design[identifier]['full_id']]['text']
        point = (coordinates(text), len(text))
        key = row['correct']
        if type(key) is not int or not 0 <= key < 4 or answers.setdefault(point, key) != key:
            raise ValueError('RACE key conflicts with the displayed item')
    yield from (('race_answer_choice', point, key) for point, key in sorted(answers.items()))
