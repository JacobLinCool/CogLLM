"""Freeze the professional CWI News sentence holdout for the served lexical head."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
import unicodedata
from zipfile import ZipFile

from research.core_features.background_prepare import MARKER, SOURCE, SOURCE_SHA256
from research.semantic_tracking.iirc_anchor_fit import WORD
from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3.run import save_json


PROTOCOL = Path(__file__).with_name('cwi-news-holdout.md')
MEMBER = 'testset/english/News_Test.tsv'
MEMBER_SHA256 = '3d914c936b1dfd6b63cd0555dfedf8bc8e3314dea1e5e3ea67399061fe7ba156'
COMPLEX_SHA256 = '20d1859671ffcb6b0d577b6c9cc6b1398bb7699e46c2cfea08efedab93a7068a'
CANDIDATE = '74c26ff02569b2b3bc28ed291bf15fd2f85db5b4c93a3ad7bfc34f6a5d333ea0'
LEXICAL_SHA256 = 'e9721f340b325015826bf1645d9814385d4d180689b652e772480f99a80f49d4'
CWI_REFERENCES = tuple(
    f'{directory}/english/{genre}_{split}.tsv'
    for genre in ('News', 'Wikipedia', 'WikiNews')
    for split, directory in (('Train', 'traindevset'), ('Dev', 'traindevset'), ('Test', 'testset'))
    if (genre, split) != ('News', 'Test')
)


def words(text):
    return tuple(WORD.findall(unicodedata.normalize('NFKC', text).casefold()))


def fivegrams(sequence):
    return set(zip(sequence, sequence[1:], sequence[2:], sequence[3:], sequence[4:]))


def reference_index(sequences):
    by_fivegram = defaultdict(list)
    for index, sequence in enumerate(sorted(sequences)):
        for window in fivegrams(sequence):
            by_fivegram[window].append(index)
    return sequences, by_fivegram


def overlap(sequence, reference):
    sequences, by_fivegram = reference
    if sequence in sequences:
        return 'exact'
    windows = fivegrams(sequence)
    shared = Counter(index for window in windows for index in by_fivegram.get(window, ()))
    if shared and 4 * max(shared.values()) >= len(windows):
        return 'substantial_fivegram'
    return None


def read_complex(path):
    if digest(path) != COMPLEX_SHA256:
        raise ValueError('Expected the fixed CompLex target readout source inventory')
    by_split = defaultdict(set)
    with path.open(encoding='utf-8') as source:
        for line in source:
            row = json.loads(line)
            by_split[row['split']].add(words(row['text']))
    if set(by_split) != {'train', 'trial', 'test'} or any(not part for part in by_split.values()):
        raise ValueError('Expected all three CompLex source partitions')
    frequencies = Counter(form for sentence in by_split['train'] | by_split['trial'] for form in sentence)
    return reference_index(set().union(*by_split.values())), frequencies


def read_news(path):
    if digest(path) != SOURCE_SHA256:
        raise ValueError('Expected the pinned CWI 2018 public archive')
    with ZipFile(path) as archive:
        raw = archive.read(MEMBER)
        if hashlib.sha256(raw).hexdigest() != MEMBER_SHA256:
            raise ValueError('Unexpected professional News official test member')
        references = set()
        for name in CWI_REFERENCES:
            for line in archive.read(name).decode('utf-8').splitlines():
                _, text, *_ = line.split('\t')
                references.add(words(MARKER.sub('', text)))
    by_sentence = defaultdict(dict)
    duplicate_rows = 0
    for line in raw.decode('utf-8').splitlines():
        (hit, text, left, right, target, native, nonnative,
         native_votes, nonnative_votes, label, probability) = line.split('\t')
        start, end = int(left), int(right)
        a, b = int(native_votes), int(nonnative_votes)
        proportion = float(probability)
        key = hit, text
        if (not hit or not text.strip() or (native, nonnative) != ('10', '10')
                or not 0 <= a <= 10 or not 0 <= b <= 10 or int(label) != int(a + b > 0)
                or not isfinite(proportion) or abs(proportion - (a + b) / 20) > 1e-12):
            raise ValueError('Unexpected professional News public count annotation')
        if not 0 <= start < end <= len(text) or text[start:end] != target:
            raise ValueError('Expected an exact professional News source offset')
        candidates = by_sentence[key]
        if (start, end) in candidates:
            if candidates[start, end] != (a, b):
                raise ValueError('A repeated professional News candidate has conflicting votes')
            duplicate_rows += 1
        candidates[start, end] = a, b
    if sum(map(len, by_sentence.values())) + duplicate_rows != 2095 or len({hit for hit, _ in by_sentence}) != 28:
        raise ValueError('Unexpected size of the official professional News test split')
    return by_sentence, duplicate_rows, reference_index(references)


def checkpoint_hashes(path):
    manifest_path = path / 'checkpoint.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    declared = manifest.pop('checkpoint_sha256')
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                                            allow_nan=False).encode('utf-8')).hexdigest()
    if (declared != CANDIDATE or fingerprint != CANDIDATE
            or manifest['files']['lexical/heads.npz'] != LEXICAL_SHA256
            or digest(path / 'lexical/heads.npz') != LEXICAL_SHA256):
        raise ValueError('Expected the fixed served Gradio lexical head')
    return digest(manifest_path)


def prepare(archive, complex_stimuli, checkpoint, output):
    if output.exists():
        raise FileExistsError(output)
    checkpoint_sha256 = checkpoint_hashes(checkpoint)
    protocol_sha256 = digest(PROTOCOL)
    complex_reference, frequencies = read_complex(complex_stimuli)
    source, duplicate_rows, prior_cwi = read_news(archive)
    identities = {key: words(key[1]) for key in source}
    repetitions = Counter(identities.values())
    exclusions = Counter()
    if duplicate_rows:
        exclusions['duplicate_candidate_row'] = duplicate_rows
    stimuli, outcomes = [], []
    for (hit, text), candidates in sorted(source.items()):
        sequence = identities[hit, text]
        known_complex = overlap(sequence, complex_reference)
        known_cwi = overlap(sequence, prior_cwi)
        if repetitions[sequence] > 1:
            exclusions['duplicate_heldout_sentence'] += len(candidates)
            continue
        if known_complex:
            exclusions[f'complex_{known_complex}'] += len(candidates)
            continue
        if known_cwi:
            exclusions[f'cwi_{known_cwi}'] += len(candidates)
            continue
        exact_words = {word.span() for word in WORD.finditer(text)}
        retained = []
        for (start, end), (native_votes, nonnative_votes) in sorted(candidates.items()):
            if (start, end) not in exact_words:
                exclusions['not_one_complete_served_word'] += 1
                continue
            forms = words(text[start:end])
            retained.append({'start': start, 'end': end, 'native_votes': native_votes,
                             'nonnative_votes': nonnative_votes,
                             'form_count': frequencies[forms[0]] if len(forms) == 1 else 0})
        if retained:
            identifier = 'cwi-news-' + hashlib.sha256(text.encode('utf-8')).hexdigest()
            stimuli.append({'id': identifier, 'text': text})
            outcomes.append({'id': identifier, 'hit_id': hit, 'words': retained})
    retained_candidates = sum(len(row['words']) for row in outcomes)
    hits = {row['hit_id'] for row in outcomes}
    if (not hits or len({row['id'] for row in stimuli}) != len(stimuli)
            or retained_candidates + exclusions.total() != 2095):
        raise ValueError('Expected unique and complete accounting of the official split')
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.cwi-news-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'prepared'
        stage.mkdir()
        write_jsonl(stage / 'stimuli.jsonl', stimuli)
        write_jsonl(stage / 'outcomes.jsonl', outcomes)
        audit = {
            'schema': 1, 'candidate_id': CANDIDATE, 'lexical_weights_sha256': LEXICAL_SHA256,
            'checkpoint_manifest_sha256': checkpoint_sha256, 'protocol_sha256': protocol_sha256,
            'preparation_code_sha256': digest(Path(__file__)),
            'stimuli_sha256': digest(stage / 'stimuli.jsonl'),
            'outcomes_sha256': digest(stage / 'outcomes.jsonl'),
            'source_url': SOURCE, 'source_sha256': SOURCE_SHA256,
            'official_member': MEMBER, 'official_member_sha256': MEMBER_SHA256,
            'license': 'CC BY 4.0', 'complex_stimuli_sha256': COMPLEX_SHA256,
            'complex_reference_splits': ['train', 'trial', 'test'],
            'frequency_reference_splits': ['train', 'trial'],
            'cwi_reference_members': CWI_REFERENCES,
            'overlap_rule': 'Reject normalized exact matches or at least 25% shared distinct five-word windows with one reference sentence.',
            'source_hits': len({hit for hit, _ in source}), 'source_sentences': len(source),
            'source_candidates': 2095, 'votes_per_group': {'native': 10, 'nonnative': 10},
            'retained_sentences': len(stimuli), 'retained_hits': len(hits),
            'retained_candidates': retained_candidates, 'excluded_candidates': dict(sorted(exclusions.items())),
            'presentation': 'Literal released sentence; original annotators saw a paragraph.',
            'resampling_scope': 'Original HIT; the release gives no article or reader identity key.',
            'pretraining_overlap': 'Unknown.',
        }
        save_json(stage / 'assay.json', audit)
        stage.rename(output)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive', 'complex-stimuli', 'checkpoint', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.complex_stimuli, args.checkpoint, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
