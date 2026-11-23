"""Freeze outcome-free stimuli for the served MedReadMe lexical localization test."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unicodedata

from research.gradio_app.checkpoint import load_checkpoint
from research.semantic_tracking.iirc_anchor_fit import WORD
from research.stage2.records import write_jsonl
from research.stage3.probe import _jsonl
from research.stage3.run import save_json
from research.stage6.medreadme_span_audit import FILES, compact, source_spans
from research.semantic_tracking.iirc_prepare import digest


ASSAY = Path(__file__).with_name('medreadme-lexical-holdout.md')
RESULTS = Path(__file__).resolve().parents[1] / 'results'
PRIMARY = frozenset({'general-complex', 'general-medical-multisense',
                     'medical-jargon-google-easy', 'medical-jargon-google-hard'})


def normalized_words(text):
    return tuple(WORD.findall(unicodedata.normalize('NFKC', text).casefold()))


def grams(sequence):
    return set(zip(sequence, sequence[1:], sequence[2:], sequence[3:], sequence[4:]))


def complex_references(path):
    saved = json.loads((RESULTS / 'stage5-complex/input-audit.json').read_bytes())
    if digest(path) != saved['output_sha256']['stimuli.jsonl']:
        raise ValueError('Expected the source sentences of the frozen lexical checkpoint')
    sequences = {normalized_words(row['text']) for row in _jsonl(path) if row['split'] in ('train', 'trial')}
    index = defaultdict(list)
    for identifier, sequence in enumerate(sorted(sequences)):
        for key in grams(sequence):
            index[key].append(identifier)
    counts = Counter(word for sequence in sequences for word in sequence)
    return sequences, index, counts


def overlap(sequence, known, index):
    if sequence in known:
        return 'identical_to_complex_fit'
    windows = grams(sequence)
    counts = Counter(reference for window in windows for reference in index.get(window, ()))
    if counts and max(counts.values()) * 4 >= len(windows):
        return 'substantial_five_word_overlap'
    return None


def outcomes(text, tokens, entities, frequencies):
    spans = list(source_spans(text, tokens, entities))
    words = []
    for word in WORD.finditer(text):
        start, end = word.span()
        labels = [(left, right, name) for left, right, name in spans if max(left, start) < min(right, end)]
        target = bool(labels)
        if any(not (left <= start and end <= right and name in PRIMARY) for left, right, name in labels):
            target = None
        normalized = normalized_words(word.group())
        count = frequencies[normalized[0]] if len(normalized) == 1 else 0
        words.append({'start': start, 'end': end, 'target': target, 'form_count': count})
    return words, [{'start': left, 'end': right, 'category': category} for left, right, category in spans]


def prepare(raw, complex_stimuli, checkpoint, output):
    for name, expected in FILES.items():
        if digest(raw / name) != expected:
            raise ValueError(f'Expected the pinned MedReadMe file: {name}')
    candidate = load_checkpoint(checkpoint)
    rated = defaultdict(list)
    with (raw / 'readability.csv').open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = row['Source'], row['Side'], row['split'], compact(row['Sentence'])
            rated[key].append(row['Sentence'])
    test = [row for row in json.loads((raw / 'jargon.json').read_bytes()) if row['split'] == 'test']
    keyed = [((*row['belongings'], 'test', compact(''.join(row['tokens']))), row) for row in test]
    counts = Counter(key for key, _ in keyed)
    test_sequences = Counter(normalized_words(text) for key, texts in rated.items() if key[2] == 'test' for text in texts)
    known, index, frequencies = complex_references(complex_stimuli)
    labels, stimuli, retired = [], [], Counter()
    for serial, (key, row) in enumerate(keyed):
        if counts[key] != 1 or len(rated.get(key, ())) != 1:
            retired['unmatched_or_ambiguous_alignment'] += 1
            continue
        text, = rated[key]
        sequence = normalized_words(text)
        if test_sequences[sequence] != 1:
            retired['duplicate_test_sentence'] += 1
            continue
        exclusion = overlap(sequence, known, index)
        if exclusion:
            retired[exclusion] += 1
            continue
        words, spans = outcomes(text, row['tokens'], row['entities'], frequencies)
        case = f'medreadme-test-{serial:04d}'
        stimuli.append({'id': case, 'text': text})
        labels.append({'id': case, 'source': key[0], 'words': words, 'spans': spans})
    coverage = Counter(word['target'] for row in labels for word in row['words'])
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix='.medreadme-lexical-') as temporary:
        stage = Path(temporary) / 'assay'
        stage.mkdir()
        write_jsonl(stage / 'stimuli.jsonl', stimuli)
        write_jsonl(stage / 'locked-outcomes.jsonl', labels)
        save_json(stage / 'assay.json', {'candidate_id': candidate.identifier,
                  'lexical_weights_sha256': digest(candidate.directory / 'lexical/heads.npz'),
                  'protocol_sha256': digest(ASSAY),
                  'preparation_code_sha256': digest(Path(__file__)), 'source_sha256': FILES,
                  'complex_stimuli_sha256': digest(complex_stimuli), 'stimuli_sha256': digest(stage / 'stimuli.jsonl'),
                  'outcomes_sha256': digest(stage / 'locked-outcomes.jsonl'), 'retained_sentences': len(stimuli),
                  'retained_by_source': Counter(row['source'] for row in labels), 'excluded_sentences': retired,
                  'words': {'positive': coverage[True], 'unmarked': coverage[False], 'ignored': coverage[None]},
                  'human_spans': sum(len(row['spans']) for row in labels)})
        stage.rename(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for value in ('raw', 'complex_stimuli', 'checkpoint', 'output'):
        parser.add_argument('--' + value.replace('_', '-'), type=Path, required=True)
    prepare(**vars(parser.parse_args()))
