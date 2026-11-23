"""Prepare the written Experiment 1 coherence judgments without retaining the raw export."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from html import unescape
import io
import json
from pathlib import Path
import re
from statistics import mean
from unicodedata import normalize
from urllib.request import urlopen
from xml.etree import ElementTree as ET
from zipfile import ZipFile


PROJECT = 'https://osf.io/m87vg/'
SOURCES = {
    'materials': ('https://osf.io/download/h53ed/', 63631, '4267b62948264fd8f39b58668c2cfc6da0605262ae8ab38ae2b9724ac7130591'),
    'analysis': ('https://osf.io/download/uf5pr/', 75672, '27ccb8f12c4cbd4d47d14488bd6fe8ad125eff33ec7e0bb3ef9d27a65de25766'),
    'responses': ('https://osf.io/download/trf8w/', 256097247, 'd972e47b4fba435748828c8cfe10affa92127f03f7de1f87c7d286b09ce1e93b'),
    'aggregate': ('https://osf.io/download/bw4ry/', 30450, '9cd0018de8bd734c53058ddcee3e5b9852bb91ddac6c1b5a6f4df74c46b5fb38'),
}
OLD_MATERIALS_SHA256 = '208f45919e2c6c2623d66c2275d11f409deb9644d720844be2b2c66f6e364580'
LICENSE = 'https://creativecommons.org/licenses/by/4.0/'
WORD_XML = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
TOKEN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
HEADING = re.compile(r'<h4>([^<>]+)</h4>\Z')


def digest(bytes_):
    return hashlib.sha256(bytes_).hexdigest()


def fetch_small(name):
    url, size, sha = SOURCES[name]
    with urlopen(url, timeout=30) as stream:
        payload = stream.read(size + 1)
    if len(payload) != size or digest(payload) != sha:
        raise ValueError(f'Unrecognized {name} release')
    return payload


def tables(payload):
    with ZipFile(io.BytesIO(payload)) as document:
        root = ET.fromstring(document.read('word/document.xml'))
    return [
        [tuple(''.join(t.text or '' for t in cell.findall('.//w:t', WORD_XML)).strip()
               for cell in row.findall('w:tc', WORD_XML))
         for row in table.findall('w:tr', WORD_XML)]
        for table in root.findall('.//w:tbl', WORD_XML)
    ]


def words(value):
    return tuple(TOKEN.findall(normalize('NFKC', value).casefold().replace('’', "'")))


def rendered_text(value):
    wrapper = HEADING.fullmatch(value)
    if not wrapper:
        raise ValueError('Expected one complete h4 display wrapper with only sentence text')
    return unescape(wrapper[1])


def located_pairs(displays):
    grouped = defaultdict(dict)
    for (family, arm), (text, noun) in displays.items():
        grouped[family][arm] = (text, noun)
    located, missing = {}, []
    for family, pair in grouped.items():
        raw = {arm: list(TOKEN.finditer(text)) for arm, (text, _) in pair.items()}
        token_lists = {arm: words(text) for arm, (text, _) in pair.items()}
        indexes = []
        if len(token_lists['Amb']) == len(token_lists['UA']):
            indexes = [i for i, (amb, ua) in enumerate(zip(token_lists['Amb'], token_lists['UA'], strict=True))
                       if amb != ua]
        verified_pair = len(indexes) == 2 and indexes[1] == len(raw['Amb']) - 1
        located[family] = {}
        for arm, (text, _) in pair.items():
            noun_span = raw[arm][indexes[0]].span() if verified_pair else None
            if noun_span is None:
                missing.append((family, arm))
            located[family][arm] = {'text': text, 'noun_span': noun_span, 'cue_span': raw[arm][-1].span()}
    return located, missing


def material_pairs(payload):
    displayed, design = tables(payload)
    if displayed[0][:3] != ('Item', 'Sentence', 'Condition') or design[0][:3] != displayed[0][:3]:
        raise ValueError('Unrecognized official materials tables')
    displayed_rows = [row for row in displayed[1:] if row[1]]
    design_rows = [row for row in design[1:] if row[1]]
    if len(displayed_rows) != 180 or len(design_rows) != 180:
        raise ValueError('Expected 90 pairs in each source table')
    rows = defaultdict(dict)
    design_families = {(item, label) for item, _, label, *_ in design_rows}
    for item, text, condition, label, *_ in displayed_rows:
        if condition != 'Mean.':
            if condition != 'Anom.':
                raise ValueError('Unexpected displayed condition')
            continue
        if (item, label) not in design_families:
            raise ValueError('Ambiguity labels disagree across author tables')
        ambiguity = {'Amb.': 'Amb', 'Unamb.': 'UA'}[label]
        if ambiguity in rows[item]:
            raise ValueError('Duplicate stimulus family condition')
        rows[item][ambiguity] = text
    if len(rows) != 63 or any(set(pair) != {'Amb', 'UA'} for pair in rows.values()):
        raise ValueError('Expected 63 coherent matched families')
    return dict(rows)


def windows(tokens, size):
    return {tokens[i:i + size] for i in range(len(tokens) + 1 - size)}


def overlap_counts(pairs, old_payload):
    if digest(old_payload) != OLD_MATERIALS_SHA256:
        raise ValueError('Historical overlap comparison must include original 2021 families')
    [old_table] = tables(old_payload)
    if len(old_table) != 49:
        raise ValueError('Historical overlap comparison needs all 48 original frames')
    old_texts, old_frames, old_lexical_pairs, old_windows = set(), set(), set(), set()
    for _, alternatives, frame in old_table[1:]:
        parts = frame.split('/')
        if parts[1] != 'KEYWORD' or len(parts) not in (5, 6):
            raise ValueError('Historical primary noun or cue regions are invalid')
        before_cue = words(' '.join(parts[2:3]))
        cue = words(parts[3])
        ending = words(' '.join(parts[4:]))
        prior = words(parts[0])
        if len(parts) == 6:
            old_frames.add(prior + ('<noun>',) + before_cue + ('<cue>',) + ending)
        for alternative in alternatives.split('/'):
            target = words(alternative)
            tokens = prior + target + before_cue + cue + ending
            old_texts.add(tokens)
            old_windows.update(windows(tokens, 5))
            old_frames.add(prior + ('<noun>',) + (before_cue + cue + ending)[:-1] + ('<cue>',))
            if len(parts) == 6:
                old_lexical_pairs.update((target, word) for word in cue)
            old_lexical_pairs.add((target, tokens[-1]))

    exact, frame, lexical, near = (set() for _ in range(4))
    for family, pair in pairs.items():
        for item in pair.values():
            text = item['text']
            c_start, c_end = item['cue_span']
            tokens = words(text)
            [final_word] = words(text[c_start:c_end])
            noun_options = [item['noun_span']] if item['noun_span'] is not None else [
                token.span() for token in TOKEN.finditer(text[:c_start])]
            if tokens in old_texts:
                exact.add(family)
            for n_start, n_end in noun_options:
                masked = (words(text[:n_start]) + ('<noun>',) + words(text[n_end:c_start])
                          + ('<cue>',) + words(text[c_end:]))
                if masked in old_frames:
                    frame.add(family)
                if (words(text[n_start:n_end]), final_word) in old_lexical_pairs:
                    lexical.add(family)
            if windows(tokens, 5) & old_windows:
                near.add(family)
    excluded = exact | frame | lexical
    return excluded, {
        'compared_2021_families': 48, 'compared_2021_variants': len(old_texts),
        'normalized_same_sentence_families': sorted(exact), 'masked_same_frame_families': sorted(frame),
        'same_noun_and_cue_families': sorted(lexical), 'shared_five_word_fragment_families': sorted(near),
        'excluded_families': sorted(excluded),
        'five_word_fragment_screen_only': True,
    }


class HashingReader(io.RawIOBase):
    def __init__(self, upstream, size):
        self.upstream = upstream
        self.size = size
        self.read_bytes = 0
        self.hash = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.upstream.read(len(buffer))
        self.read_bytes += len(data)
        if self.read_bytes > self.size:
            raise ValueError('Trial export exceeds pinned file length')
        self.hash.update(data)
        buffer[:len(data)] = data
        return len(data)


def author_exclusions(analysis):
    raw = analysis.decode('utf-8-sig')
    block = raw.split('Data.Excluded <- subset(Data,', 1)[1].split('# Remove any trials', 1)[0]
    excluded = set(re.findall(r'!Participant\.Private\.ID =="([0-9]+)"', block))
    if len(excluded) != 16 or 'Reaction.Time > 250.00' not in raw:
        raise ValueError('Unexpected participant or latency exclusions in the analysis script')
    return excluded


def project_trials(rows, materials, exclusions):
    readers, trials, original_readers = {}, defaultdict(list), set()
    counts = Counter()
    exact_text = {(family, arm): text for family, pair in materials.items() for arm, text in pair.items()}
    displays = {}
    differing_variants = set()
    for row in rows:
        counts['raw_csv_rows'] += 1
        if row['Screen.Name'] != 'Response' or row['MiniblockNumber'] == 'Practice':
            continue
        private = row['Participant.Private.ID']
        if private in exclusions:
            continue
        if not private:
            raise ValueError('Missing reader key on experimental response')
        original_readers.add(private)
        if row['Modality'] != 'Vis' or row['Plausibility'] != 'Plaus':
            continue
        counts['coherent_written_response_rows_before_filter'] += 1
        arm = row['Ambiguity']
        family = row['Item'].strip()
        if family != row['Item']:
            counts['response_rows_with_surrounding_item_whitespace'] += 1
        if (family, arm) not in exact_text:
            raise ValueError('A coherent written response has an unknown family or condition')
        if row['Sentence'] != exact_text[family, arm]:
            if words(row['Sentence']) != words(exact_text[family, arm]):
                differing_variants.add((family, arm))
                counts['response_rows_display_tokens_differ_from_material_docx'] += 1
            counts['response_rows_display_differs_from_material_docx'] += 1
        display = (row['Sentence'], row['Word'])
        if displays.setdefault((family, arm), display) != display:
            raise ValueError('One or more material versions had multiple literal displays or critical nouns')
        if row['ANSWER'] != 'sense.png':
            raise ValueError('Unexpected correct response in a coherent trial')
        raw_rt = row['Reaction.Time']
        if not raw_rt or float(raw_rt) <= 250:
            counts['latency_exclusions'] += 1
            continue
        response = row['Response']
        # A timeout retains author-coded correctness but is not an explicit reader choice.
        if response == 'timeout':
            if row['Correct'] != '0':
                raise ValueError('Unexpected timeout score')
            counts['timeouts_after_latency_filter'] += 1
        elif response not in {'sense.png', 'nonsense.png'} or (row['Correct'] == '1') != (response == 'sense.png'):
            raise ValueError('Public button encoding disagrees with recorded accuracy')
        reader = readers.setdefault(private, f'reader-{len(readers) + 1:03d}')
        source = (family, arm)
        key = (reader, *source)
        trial = {'reader_id': reader, 'family': family, 'ambiguity': arm,
                 'trial_index': int(row['Trial.Number']),
                 'sense': None if response == 'timeout' else response == 'sense.png',
                 'rt_ms': float(raw_rt), 'accuracy': int(row['Correct'])}
        trials[key].append(trial)
    if len(original_readers) != 108 or len(readers) != 108:
        raise ValueError('Expected the article’s 108 retained readers')
    if set(displays) != set(exact_text):
        raise ValueError('A coherent material lacks a written trial response')
    repeated = {key for key, group in trials.items() if len(group) != 1}
    counts['repeated_reader_family_variant_keys_after_filter'] = len(repeated)
    counts['repeated_reader_family_variant_rows_after_filter'] = sum(len(trials[key]) for key in repeated)
    counts['display_variant_tokens_differ_from_material_docx'] = len(differing_variants)
    return [dict(trial, unique_trial=key not in repeated) for key, group in trials.items() for trial in group], dict(counts), displays


def receive_trials(materials, exclusions):
    url, size, sha = SOURCES['responses']
    with urlopen(url, timeout=40) as upstream:
        source = HashingReader(upstream, size)
        with io.TextIOWrapper(io.BufferedReader(source, buffer_size=65536), encoding='cp1252', newline='') as text:
            rows = csv.DictReader(text)
            trials, counts, displays = project_trials(rows, materials, exclusions)
        if source.read_bytes != size or source.hash.hexdigest() != sha:
            raise ValueError('Trial export length or digest disagrees with the OSF manifest')
    return trials, counts, displays


def rate(rows):
    return {arm: {'trials': len(subset), 'mean': mean(subset)}
            for arm in ('Amb', 'UA')
            for subset in [[row['accuracy'] for row in rows if row['ambiguity'] == arm]]}


def summarize(trials, counts, excluded):
    grouped = defaultdict(dict)
    by_reader = defaultdict(list)
    for trial in trials:
        if trial['unique_trial']:
            grouped[trial['reader_id'], trial['family']][trial['ambiguity']] = trial
        by_reader[trial['reader_id']].append(trial)
    by_family = defaultdict(list)
    for (_, family), pair in grouped.items():
        if set(pair) == {'Amb', 'UA'} and all(trial['sense'] is not None for trial in pair.values()):
            by_family[family].append(int(pair['Amb']['sense']) - int(pair['UA']['sense']))
    if not by_family or set(by_family) != {trial['family'] for trial in trials}:
        raise ValueError('Expected paired explicit responses for every coherent written family')
    remaining = {family: values for family, values in by_family.items() if family not in excluded}
    return dict(counts, author_retained_readers=len(by_reader),
                author_accuracy_by_condition=rate(trials),
                author_accuracy_reader_macro={arm: mean(rate(rows)[arm]['mean'] for rows in by_reader.values())
                                              for arm in ('Amb', 'UA')},
                paired_explicit_differences={
                    name: {'families': len(group), 'paired_trials': sum(map(len, group.values())),
                           'amb_minus_unamb_family_macro': mean(map(mean, group.values()))}
                    for name, group in [('all_coherent', by_family), ('old_material_excluded', remaining)]},
                old_material_excluded_explicit_responses=sum(
                    row['family'] not in excluded and row['sense'] is not None and row['unique_trial'] for row in trials))


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8') as file:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), file=file)


def verified_stage(directory):
    stage = directory / 'sanitized-source.json'
    materials = directory / 'BlottRoddWarren_Stimuli.docx'
    if stage.is_file():
        payload = materials.read_bytes()
        if digest(payload) != SOURCES['materials'][2]:
            raise ValueError('Cached materials file does not match the verified release')
        sanitized = json.loads(stage.read_text(encoding='utf-8'))
        if sanitized['source_trial_sha256'] != SOURCES['responses'][2]:
            raise ValueError('Sanitized cache was not produced from the required trial release')
    else:
        print('Streaming and checking the pinned source release', flush=True)
        payload = fetch_small('materials')
        trials, counts, displays = receive_trials(material_pairs(payload), author_exclusions(fetch_small('analysis')))
        sanitized = {'source_trial_sha256': SOURCES['responses'][2], 'trials': trials, 'counts': counts,
                     'displays': [[*key, *value] for key, value in sorted(displays.items())]}
        directory.mkdir(parents=True, exist_ok=True)
        materials.write_bytes(payload)
        temporary = directory / 'sanitized-source.tmp'
        with temporary.open('x', encoding='utf-8') as stream:
            print(json.dumps(sanitized, ensure_ascii=False, allow_nan=False), file=stream)
        temporary.replace(stage)
        print('Verified and saved the sanitized trial projection', flush=True)
    displays = {(family, arm): (text, noun) for family, arm, text, noun in sanitized['displays']}
    return sanitized['trials'], sanitized['counts'], displays, digest(stage.read_bytes())


def prepare(previous, output):
    trials, counts, raw_displays, sanitized_sha = verified_stage(output.parent)
    material_file = output.parent / 'BlottRoddWarren_Stimuli.docx'
    material_table = material_pairs(material_file.read_bytes())
    displays = {key: (rendered_text(markup), noun) for key, (markup, noun) in raw_displays.items()}
    counts['verified_h4_wrappers_stripped'] = len(displays)
    counts['display_variants_with_html_entities'] = sum(
        wrapper[4:-5] != text for (wrapper, _), (text, _) in zip(raw_displays.values(), displays.values(), strict=True))
    counts['rendered_tokens_differ_from_material_docx'] = sum(
        words(text) != words(material_table[family][arm]) for (family, arm), (text, _) in displays.items())
    if counts['rendered_tokens_differ_from_material_docx']:
        raise ValueError('Rendered source tokens disagree with the published Experiment 1 material table')
    materials, missing_noun = located_pairs(displays)
    excluded, overlap = overlap_counts(materials, previous.read_bytes())
    aggregate = summarize(trials, counts, excluded)
    stimuli, layout = [], []
    for family in sorted(set(materials) - excluded, key=lambda family: digest(f'blott-2023:{family}'.encode())):
        for arm, material in sorted(materials[family].items()):
            text = material['text']
            noun_span, cue_span = material['noun_span'], material['cue_span']
            before_cue = len(text[:cue_span[0]].rstrip())
            article_id = digest(text.encode())
            stops = {before_cue, cue_span[1], len(text)}
            if noun_span is not None:
                stops.add(noun_span[1])
            stimuli.append({'id': article_id, 'text': text, 'stops': sorted(stops)})
            layout.append({'id': article_id, 'family': family, 'ambiguity': arm,
                           'noun_span': noun_span, 'cue_span': cue_span,
                           'before_cue_stop': before_cue, 'sentence_stop': len(text)})
    if len({row['id'] for row in stimuli}) != len(stimuli):
        raise ValueError('Duplicate literal stimulus text across retained families')
    ids = {(row['family'], row['ambiguity']): row['id'] for row in layout}
    outcomes = [{'id': ids[trial['family'], trial['ambiguity']], 'reader_id': trial['reader_id'],
                 'trial_index': trial['trial_index'], 'sense': trial['sense'], 'rt_ms': trial['rt_ms']}
                for trial in trials if trial['family'] not in excluded and trial['sense'] is not None and trial['unique_trial']]
    report = {
        'project': PROJECT, 'license': LICENSE, 'format': 'Experiment 1 Vis (whole sentence for 4500 ms)',
        'source': {name: {'url': url, 'size_bytes': size, 'sha256': sha}
                   for name, (url, size, sha) in SOURCES.items() if name != 'aggregate'},
        'overlap_rules': overlap, 'primary': aggregate, 'retained_stimuli': len(stimuli),
        'retained_explicit_outcomes': len(outcomes), 'prior_source_material_sha256': OLD_MATERIALS_SHA256,
        'verified_sanitized_source_sha256': sanitized_sha,
        'optional_noun_coordinates_missing': missing_noun,
        'matching_geometry': 'All stimuli use the displayed text inside exactly one h4 wrapper in CSV Sentence; any HTML entities are decoded and rendered word sequences match DOCX. Noun location uses the earlier of exactly two changes in the matched pair, the other being the terminal cue. If missing, historical overlap screens each possible single noun position without assigning a noun coordinate.',
        'workbook_role': 'First table supplies Experiment 1 planned stimuli families, second table confirms ambiguity labels.',
        'trial_stop_interpretation': 'Sense/Nonsense only after complete sentence. Main noun and final cue are stimulus coordinates.',
        'data_minimization': 'Source reader keys used only in memory to apply published exclusions and create encounter-order aliases. No source identifiers, demographic data, device data, timestamps or raw trial log written.',
    }
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / 'stimuli.jsonl', stimuli)
    write_jsonl(output / 'layout.jsonl', layout)
    write_jsonl(output / 'outcomes.jsonl', outcomes)
    for name in ('stimuli.jsonl', 'layout.jsonl', 'outcomes.jsonl'):
        report.setdefault('prepared_sha256', {})[name] = digest((output / name).read_bytes())
    (output / 'source-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-materials', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.prior_materials, args.output), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
