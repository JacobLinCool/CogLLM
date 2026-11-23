"""Rebuild linked human observations from the preserved source outcomes."""

from research.backbone_transfer.extract import digest, read_json, stream_rows
from research.union_training.sources import indexed


def read_frick(base, coordinates, add):
    materials = indexed(list(stream_rows(base / 'stimuli.jsonl')))
    design = indexed(list(stream_rows(base / 'metadata.jsonl')))
    paths = {}
    for identifier, row in design.items():
        text = materials[identifier]['text']
        stops = [segment['stop'] for segment in row['sentences']]
        if len(stops) < 2 or stops[-1] != len(text) or stops != sorted(set(stops)):
            raise ValueError(f'Invalid Frick sentence boundaries for {identifier}')
        document = coordinates(text)
        path = [(document, stop) for stop in stops[1:]]
        paths[identifier] = (path, [0] * (len(path) + 1), [[0, 0] for _ in path])

    seen_trials = set()
    for trial in stream_rows(base / 'outcomes.jsonl'):
        identifier = trial['id']
        trial_key = identifier, trial['reader_id']
        if trial_key in seen_trials:
            raise ValueError(f'Duplicate Frick reader and story version for {identifier}')
        seen_trials.add(trial_key)
        path, first_counts, subsequent_counts = paths[identifier]
        keys = trial['keys']
        if len(keys) != len(path) + 1 or not set(keys) <= {'k', 'd'}:
            raise ValueError(f'Frick decisions disagree with the displayed sentences for {identifier}')
        # The instructions prescribe K on the first sentence. The task's risk set starts on sentence two.
        observed_keys = keys[1:]
        first = next((index for index, key in enumerate(observed_keys) if key == 'd'), len(path))
        first_counts[first] += 1
        for index in range(first + 1, len(path)):
            subsequent_counts[index][int(observed_keys[index] == 'd')] += 1

    for path, first_counts, subsequent_counts in paths.values():
        if not sum(first_counts):
            raise ValueError('Frick story version is missing its decision paths')
        add('frick_first_report', path, first_counts)
        for point, counts in zip(path, subsequent_counts, strict=True):
            if sum(counts):
                add('frick_after_first_report', [point], counts)


def read_ambistory(base, coordinates, add):
    manifest = read_json(base / 'manifest.json')
    for member, expected in manifest['files_sha256'].items():
        if digest(base / member) != expected:
            raise ValueError(f'AmbiStory prepared source differs from its receipt: {member}')
    for portion in ('train', 'dev', 'test'):
        root = base / portion
        materials = indexed(list(stream_rows(root / 'model_input/stimuli.jsonl')))
        design = indexed(list(stream_rows(root / 'design/stops.jsonl')))
        if materials.keys() != design.keys():
            raise ValueError('AmbiStory ratings lack a matching displayed story')
        seen = set()
        for row in stream_rows(root / 'scoring/ordinal-ratings.jsonl'):
            identifier, sense_id = row['id'], row['sense_id']
            source_key = identifier, sense_id
            if source_key in seen:
                raise ValueError(f'Duplicate AmbiStory story and supplied sense for {identifier}')
            seen.add(source_key)
            item = design[identifier]
            sense, = (sense for sense in item['senses'] if sense['id'] == sense_id)
            text = materials[identifier]['text']
            if len(text) != item['post_stop']:
                raise ValueError(f'AmbiStory input is not the entire rated story for {identifier}')
            prompt = f"{text}\n\nMeaning of {item['homonym']}:\n{sense['definition']}"
            point = (coordinates(prompt), len(prompt))
            counts = [0] * 10
            for rating, flag in zip(row['ratings'], row['nonsensical'], strict=True):
                if type(rating) is not int or not 1 <= rating <= 5 or type(flag) is not bool:
                    raise ValueError(f'Invalid AmbiStory paired rating and flag for {identifier}')
                counts[2 * (rating - 1) + int(flag)] += 1
            if not sum(counts):
                raise ValueError(f'Missing AmbiStory rating and flag for {identifier}')
            add('ambistory_joint', [point], counts)


def read_paths(cache, coordinates, add):
    """Emit source-faithful observation support and counts on archived text."""
    read_frick(cache / 'semantic-tracking/frick-consistency/prepared', coordinates, add)
    read_ambistory(cache / 'ambistory/causal-v3', coordinates, add)
