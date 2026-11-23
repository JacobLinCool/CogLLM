"""Freeze a shared text split and certify across the three native tokenizers."""

from argparse import ArgumentParser
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from tokenizers import Tokenizer

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json, write_rows
from research.token_readout.data import TOKENIZER, twelve_word_windows


MODELS = ('qwen', 'lfm3', 'gemma4')
MODEL_REGISTRY = Path(__file__).parents[1] / 'backbone_transfer' / 'models.json'
TOKENIZER_ROOT = Path('.cache/cogllm/backbone-transfer/source')
SPLIT_SALT = 'CogLLM portable readout development v1'
ANNOTATED = ('frick', 'evidence')


@dataclass(frozen=True)
class EncodedText:
    text: str
    tokenizer: Tokenizer
    ids: tuple
    offsets: tuple
    ends: tuple

    @classmethod
    def encode(cls, text, tokenizer):
        value = tokenizer.encode(text, add_special_tokens=False)
        offsets = tuple(map(tuple, value.offsets))
        ends = tuple(end for _, end in offsets)
        if (not offsets or len(value.ids) != len(offsets)
                or any(not 0 <= start <= end <= len(text) for start, end in offsets)
                or any(left > right for left, right in zip(ends, ends[1:]))):
            raise ValueError('Invalid original Unicode tokenizer offsets')
        return cls(text, tokenizer, tuple(value.ids), offsets, ends)

    def prefix_count(self, endpoint):
        count = bisect_right(self.ends, endpoint)
        prefix = self.tokenizer.encode(self.text[:endpoint], add_special_tokens=False)
        return count if tuple(prefix.ids) == self.ids[:count] else None

    def conservative_count(self, endpoint):
        count = bisect_right(self.ends, endpoint)
        while count:
            stop = self.ends[count - 1]
            stable = self.prefix_count(stop)
            if stable is not None:
                return stable
            count = bisect_right(self.ends, stop - 1)
        return 0

    def overlap_window(self, span):
        start, end = span
        if not 0 <= start < end <= len(self.text):
            raise ValueError(f'Invalid original character interval {span}')
        first = bisect_right(self.ends, start)
        last = self.conservative_count(end)
        if first >= last:
            raise ValueError(f'No complete visible token for original character interval {span}')
        return [first, last]


def shared_stop(text, endpoint, encodings):
    """Certify the same prefix in every tokenizer while allowing consumed delimiters."""
    end = endpoint
    while end < len(text) and text[end].isspace():
        end += 1
    for stop in range(endpoint, end + 1):
        counts = [encoded.prefix_count(stop) for encoded in encodings]
        if all(count is not None and count > 0 and not text[encoded.ends[count - 1]:endpoint].strip()
               for encoded, count in zip(encodings, counts, strict=True)):
            return stop, counts
    raise ValueError(f'No common causal tokenizer stop at character {endpoint}')


def verify_family_prefix(texts, encodings, endpoint):
    if len({text[:endpoint] for text in texts}) != 1:
        raise ValueError('Designed common prefix text differs between variants')
    counts = [entry.prefix_count(endpoint) for entry in encodings]
    if (any(count is None or count == 0 for count in counts)
            or len({entry.ids[:count] for entry, count in zip(encodings, counts, strict=True)}) != 1):
        raise ValueError(f'Designed prefix is not invariant at character {endpoint}')
    return counts[0]


def common_family_stop(stimuli, enc, keys, endpoint):
    texts = [stimuli[key] for key in keys]
    stop, _ = shared_stop(texts[0], endpoint, [index[key] for index in enc.values() for key in keys])
    return stop, {mode: verify_family_prefix(texts, [index[key] for key in keys], stop)
                  for mode, index in enc.items()}


def development_split(rows, stimuli, count):
    windows = defaultdict(set)
    families = set()
    for row in rows:
        if row['split'] == 'train':
            family = row['group_id']
            families.add(family)
            for window in twelve_word_windows(stimuli[row['id']]):
                windows[window].add(family)
    neighbors = defaultdict(set)
    for linked in windows.values():
        for family in linked:
            neighbors[family].update(linked)
    components = []
    while families:
        component = {min(families)}
        frontier = list(component)
        while frontier:
            adjacent = neighbors[frontier.pop()] - component
            component.update(adjacent)
            frontier.extend(adjacent)
        families.difference_update(component)
        components.append(tuple(sorted(component)))
    ordered = sorted(components, key=lambda group: sha256(f'{SPLIT_SALT}:{group}'.encode()).digest())
    dev = []
    for component in ordered:
        if len(dev) + len(component) <= count:
            dev.extend(component)
    if len(dev) != count or sum(map(len, components)) <= count:
        raise ValueError('Cannot make the specified isolated development split')
    return sorted(dev)


# The cue is supplied in prose, and no question is prepended to the model input.
AB_TEXT = (
    ('a drop in pressure', 'a delay in the readings',
     'A leaking seal let the compressed air escape.',
     'A buffered clock transmitted its measurements late.',
     'A blue label distinguished the instrument from the other unit.'),
    ('a cloudy solution', 'a rise in the chamber temperature',
     'Fine suspended crystals scattered the light.',
     'A faulty thermostat kept the heater running.',
     'A paper form listed the equipment serial number.'),
    ('an increase in packet loss', 'an apparent jump in timestamps',
     'Electrical interference corrupted the network frames.',
     'The system switched from local time to UTC mid-run.',
     'The team archived the raw log in a dated folder.'),
    ('a lower final mass', 'an uneven sample color',
     'Water evaporated while the specimen was exposed to air.',
     'The dye pooled along one edge of the sample tray.',
     'The assistant logged the sample in a separate notebook.'),
    ('a sharp vibration', 'a dip in the recorded current',
     'An off-center rotor shook the motor housing.',
     'The power supply briefly limited its electrical output.',
     'The inspector attached a new inventory tag to the cabinet.'),
    ('an unexpected peak in the spectrum', 'a shorter battery lifetime',
     'A nearby transmitter emitted at the same frequency.',
     'A parasitic circuit continued to draw power overnight.',
     'The replacement enclosure was made from brushed aluminum.'),
    ('a break in the printed traces', 'a shift in the plot origin',
     'A clogged nozzle interrupted the ink flow.',
     'A reset changed the instrument reference coordinates.',
     'The folder contained older versions of the instructions.'),
    ('a weaker signal at night', 'a repeated entry in the dataset',
     'Condensation on the antenna attenuated the radio waves.',
     'An interrupted upload was retried after the first copy had saved.',
     'The team marked the files with the same project initials.'),
)


def ab_controls(stimuli):
    result = []
    for index, (a, b, explains_a, explains_b, sham) in enumerate(AB_TEXT, 1):
        first = f'The initial report could not account for {a}'
        second = f'The technicians also had no explanation for {b}'
        prefix = f'{first}. {second}.'
        anchors = {'A': [first.index(a), len(first)], 'B': [prefix.index(b, len(first)), len(prefix) - 1]}
        variants = {}
        for condition, clue in (('A', explains_a), ('B', explains_b), ('none', sham)):
            start = len(prefix) + 2
            text = prefix + '\n\n' + clue + '\n\nThe report then listed the instrument settings.'
            key = sha256(text.encode()).hexdigest()
            if key in stimuli:
                raise ValueError('Duplicated stimulus in the fresh prose fixture')
            stimuli[key] = text
            variants[condition] = {'id': key, 'cue_span': [start, start + len(clue)]}
        result.append({'group_id': f'portable-ab:{index:02}', 'split': 'test', 'anchor_spans': anchors,
                       'common_prefix_end': len(prefix), 'variants': variants})
    return result


def verified_inputs(prepared):
    manifest = read_json(prepared / 'receipt.json')
    paths = {key: prepared / key for key in ('stimuli.jsonl', 'design.json')}
    if ({name: digest(path) for name, path in paths.items()} != manifest['file_sha256']
            or manifest['schema_version'] != 1):
        raise ValueError('Existing prepared texts and annotations fail integrity verification')
    rows = list(stream_rows(paths['stimuli.jsonl']))
    if any(set(row) != {'id', 'text'} for row in rows):
        raise ValueError('Prepared text contains an annotation')
    stimuli = {row['id']: row['text'] for row in rows}
    if len(stimuli) != len(rows) or len(stimuli) != manifest['documents']:
        raise ValueError('The prepared text IDs are incomplete or repeated')
    return stimuli, read_json(paths['design.json']), manifest


def common_geometry(stimuli, design, tokenizers):
    enc = {mode: {key: EncodedText.encode(text, tokenizer) for key, text in stimuli.items()}
           for mode, tokenizer in tokenizers.items()}
    layouts = {mode: {} for mode in tokenizers}
    labels = {mode: {} for mode in tokenizers}
    adjusted = 0
    stops = {}
    family_stops = {}
    for row in design['frick']:
        key = row['id']
        original = row['sentence_spans']
        aligned = [shared_stop(stimuli[key], endpoint, [index[key] for index in enc.values()])
                   for _, endpoint in original]
        stops[key] = [stop for stop, _ in aligned]
        adjusted += sum(stop != endpoint for (_, endpoint), (stop, _) in zip(original, aligned, strict=True))
        for i, (mode, index) in enumerate(enc.items()):
            boundaries = [counts[i] for _, counts in aligned]
            if any(start >= stop for start, stop in zip(boundaries, boundaries[1:])):
                raise ValueError('Frick sentence no longer has its own visible exposure')
            layouts[mode][key] = {'kind': 'frick', 'split': row['split'], 'offsets': index[key].offsets,
                                  'sentence_tokens': [list(interval) for interval in zip(boundaries, boundaries[1:])]}

    for row in design['evidence']:
        key = row['id']
        for mode, index in enc.items():
            entry = index[key]
            risk_start = bisect_right(entry.ends, row['article_start'])
            anchors = {query['id']: entry.overlap_window(query['span']) for query in row['queries']}
            windows = {query['id']: (entry.overlap_window(query['evidence_window']) if query['evidence_window'] else None)
                       for query in row['queries']}
            if (any(stop > risk_start for _, stop in anchors.values())
                    or len({stop for _, stop in anchors.values()}) != len(anchors)
                    or any(span is not None and span[0] < risk_start for span in windows.values())):
                raise ValueError(f'{mode} violates ExpMRC question → passage → evidence order in {key}')
            layouts[mode][key] = {'kind': 'evidence', 'split': row['split'], 'offsets': entry.offsets,
                                  'risk_start': risk_start, 'anchor_tokens': anchors}
            sham = entry.overlap_window(row['earlier_sham']) if row['earlier_sham'] else None
            labels[mode][key] = {'window_tokens': windows, 'sham_tokens': sham}

    for kind in ('controls', 'ab'):
        for row in design[kind]:
            variants = {condition: value['id'] for condition, value in row['variants'].items()}
            spans = {'main': row['anchor_text_span']} if kind == 'controls' else row['anchor_spans']
            stop, prefixes = common_family_stop(stimuli, enc, variants.values(), row['common_prefix_end'])
            family_stops[row['group_id']] = {'common_prefix_end': stop}
            if kind == 'controls':
                late = [variants[label] for label in ('late', 'none')]
                stop, _ = common_family_stop(stimuli, enc, late, row['late_none_prefix_end'])
                family_stops[row['group_id']]['late_none_prefix_end'] = stop
            for mode, index in enc.items():
                for condition, key in variants.items():
                    entry = index[key]
                    anchors = {label: entry.overlap_window(span) for label, span in spans.items()}
                    if any(stop > prefixes[mode] for _, stop in anchors.values()):
                        raise ValueError(f'{mode} control anchor follows the shared prefix')
                    layouts[mode][key] = {'kind': kind, 'split': 'test', 'offsets': entry.offsets,
                                          'anchor_tokens': anchors}
                    variant = row['variants'][condition]
                    if variant['cue_span']:
                        end = variant.get('decisive_end', variant['cue_span'][1])
                        entry.overlap_window([variant['cue_span'][0], end])

    if any(rows.keys() != stimuli.keys() for rows in layouts.values()):
        raise ValueError('A native layout is missing a shared stimulus')
    audit = {'frick_shared_stops': stops, 'frick_stops_advanced_only_over_whitespace': adjusted,
             'control_shared_stops': family_stops, 'excluded_rows': 0}
    return layouts, labels, audit


def prepare(prepared, output, qwen_tokenizer, tokenizer_root):
    stimuli, design, original = verified_inputs(prepared)
    models = read_json(MODEL_REGISTRY)['models']
    paths = {'qwen': qwen_tokenizer, **{name: tokenizer_root / name / 'tokenizer.json' for name in MODELS[1:]}}
    hashes = {name: digest(path) for name, path in paths.items()}
    if hashes != {name: models[name]['sha256']['tokenizer.json'] for name in MODELS}:
        raise ValueError('A tokenizer differs from the pinned model snapshot')
    tokenizers = {name: Tokenizer.from_file(str(path)) for name, path in paths.items()}
    design['ab'] = ab_controls(stimuli)
    design['schema_version'] = 2
    design['development_families'] = {kind: development_split(design[kind], stimuli, count)
                                      for kind, count in (('frick', 4), ('evidence', 16))}
    for kind in ANNOTATED:
        for row in design[kind]:
            if row['group_id'] in design['development_families'][kind]:
                row['split'] = 'dev'
    train_ids = {row['id'] for kind in ANNOTATED for row in design[kind] if row['split'] == 'train'}
    train_windows = {window for key in train_ids for window in twelve_word_windows(stimuli[key])}
    others = stimuli.keys() - train_ids
    if any(window in train_windows for key in others for window in twelve_word_windows(stimuli[key])):
        raise ValueError('Train/dev/test isolation failed after splitting original training families')
    layouts, labels, audit = common_geometry(stimuli, design, tokenizers)
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / 'stimuli.jsonl', (dict(id=key, text=text) for key, text in sorted(stimuli.items())))
    write_json(output / 'design.json', design)
    files = ['stimuli.jsonl', 'design.json']
    for mode, layout in layouts.items():
        write_json(output / f'{mode}.json', {'schema': 1, 'model': mode, 'rows': layout})
        learning = {}
        for split in ('train', 'dev'):
            learning[split] = {kind: [dict(row, **{name: value for name, value in layout[row['id']].items()
                                                  if name not in {'kind', 'offsets'}}, **labels[mode].get(row['id'], {}))
                                      for row in design[kind] if row['split'] == split] for kind in ANNOTATED}
        write_json(output / f'{mode}-learn.json', learning)
        files.extend((f'{mode}.json', f'{mode}-learn.json'))
    receipt = {'schema': 1, 'documents': len(stimuli), 'original_receipt_sha256': digest(prepared / 'receipt.json'),
               'original_documents': original['documents'], 'tokenizer_sha256': hashes, 'dev_split_salt': SPLIT_SALT,
               'development_families': design['development_families'],
               'split_counts': {kind: dict(Counter(row['split'] for row in design[kind]))
                                for kind in ('frick', 'evidence', 'controls', 'ab')},
               'structural_alignment': audit, 'file_sha256': {name: digest(output / name) for name in files}}
    write_json(output / 'receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qwen-tokenizer', type=Path, default=TOKENIZER)
    parser.add_argument('--tokenizer-root', type=Path, default=TOKENIZER_ROOT)
    prepare(**vars(parser.parse_args()))
