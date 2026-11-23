"""Prepare a sealed external rank assay for the two frozen PLABA ordinal readouts."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import tarfile
import unicodedata
from urllib.parse import urlparse

import numpy as np

from research.semantic_tracking.iirc_linked_pack import write_archive
from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3.causal_export import WIDTH
from research.stage3.causal_ordinal import LEVELS, PENALTY, distributions, fit_softmax
from research.stage3.probe import TARGETS, _length_standardization, _load_plaba
from research.stage3.run import MODEL, REVISION, save_json


AUTHOR_REPO = 'https://github.com/babaknaderi/TextComplexityDE'
SOURCE_HASHES = {
    'source.csv': '08881ded42eef46183b2e89845ad4b07ee128e3caa77e606a39ac059e13102f5',
    'ratings.csv': '191da27ff56f6c1f26a9db6939f400c53ef902a5d2e0e358364ce70bc284d75c',
    'LICENSE': '634f134375e113150bde4669d280b71f50e294353bfff7c365fa7912ed3d638e',
}
RATINGS = {'complexity': ('MOS_Complexity', 'Votes_Complexity'),
           'understandability': ('MOS_Understandability', 'Votes_Understandability')}
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'stage6' / 'textcomplexityde-transfer.md'
HEAD = ROOT / 'results' / 'causal-plaba-ordinal'
OUTCOMES = 'heldout-outcomes.jsonl'


def normalized_tokens(text):
    return tuple(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text).casefold()))


def unique_rows(path, key):
    rows = {}
    with path.open(encoding='cp1252', newline='') as source:
        for row in csv.DictReader(source):
            identifier = row[key]
            if identifier in rows or not identifier.isdecimal():
                raise ValueError(f'Invalid or duplicate published row ID in {path.name}')
            rows[identifier] = row
    return rows


def select_sentences(raw, prior_sentences):
    for name, expected in SOURCE_HASHES.items():
        if digest(raw / name) != expected:
            raise ValueError(f'TextComplexityDE source differs from the pinned author file: {name}')
    source = unique_rows(raw / 'source.csv', 'Sentence ID')
    ratings = unique_rows(raw / 'ratings.csv', 'ID')
    if len(source) != 1119 or len(ratings) != 1000 or ratings.keys() - source.keys():
        raise ValueError('Unexpected source or rating inventory')

    seen = {normalized_tokens(text) for _, text in prior_sentences.values()}
    identifiers, publications, stimuli, outcomes = set(), {}, [], []
    excluded = Counter()
    for identifier, rating in ratings.items():
        item = source[identifier]
        if (item['Article ID'], item['Article']) != (rating['Article_ID'], rating['Article']):
            raise ValueError(f'Conflicting article for sentence {identifier}')
        original_link = urlparse(item['Link'])
        if item['Link'] == 'Leichte Sprache':
            excluded['controls_without_wikipedia_source'] += 1
            continue
        if original_link.scheme != 'https' or original_link.netloc != 'de.wikipedia.org' or not original_link.path.startswith('/wiki/'):
            raise ValueError(f'Unexpected article provenance for sentence {identifier}')
        text = item['Sentence']
        if text != rating['Sentence']:
            excluded['source_and_rating_text_differ'] += 1
            continue
        words = normalized_tokens(text)
        if words in seen:
            excluded['exact_normalized_plaba_overlap'] += 1
            continue
        if words in identifiers or not words or text != text.strip():
            raise ValueError(f'Duplicate or unexpectedly padded text at sentence {identifier}')
        identifiers.add(words)
        group = f'wikipedia-de-{item["Article ID"]}'
        article = {'title': item['Article'], 'url': item['Link'], 'source_category': item['Category']}
        if publications.setdefault(group, article) != article:
            raise ValueError(f'Conflicting attribution for {group}')
        case = f'textde-{identifier}'
        stimuli.append({'id': case, 'text': text, 'stops': [len(text)]})
        row = {'id': case, 'group_id': group}
        for target, (score, votes) in RATINGS.items():
            value, count = float(rating[score]), int(rating[votes])
            if not 1 <= value <= 7 or count < 1:
                raise ValueError(f'Invalid aggregate human rating for {case}')
            row[target] = {'mean': value, 'votes': count}
        outcomes.append(row)
    if (len(stimuli), len(publications), excluded) != (775, 23, {'controls_without_wikipedia_source': 100,
                                                                 'source_and_rating_text_differ': 125}):
        raise ValueError('The exact-text intersection or its source coverage changed')
    return stimuli, outcomes, publications, dict(excluded)


def plaba_control(plaba):
    source_head = HEAD / 'head.npz'
    metadata = json.loads((HEAD / 'manifest.json').read_text(encoding='utf-8'))
    data_hashes = metadata['data']['current_stage2_sha256']
    if (metadata['model'] != MODEL or metadata['revision'] != REVISION or metadata['head_sha256'] != digest(source_head)
            or any(digest(plaba / name) != checksum for name, checksum in data_hashes.items())):
        raise ValueError('PLABA readout or development materials differ from the frozen checkpoint')
    _, sentences, observations, _, _ = _load_plaba(plaba)
    lengths = np.log1p([(len(text), len(text.split())) for _, text in sentences.values()])
    center, scale = _length_standardization(lengths)
    length_only = np.column_stack((np.ones(len(lengths)), (lengths - center) / scale / np.sqrt(2)))
    grades = distributions(observations, list(sentences))
    with np.load(source_head, allow_pickle=False) as archived:
        weights = archived['weights']
        if (weights.shape != (len(TARGETS), 1 + WIDTH + 2, len(LEVELS)) or not np.isfinite(weights).all()
                or tuple(archived['targets']) != TARGETS or not np.array_equal(archived['ratings'], LEVELS)
                or not np.array_equal(archived['length_mean'], center) or not np.array_equal(archived['length_scale'], scale)):
            raise ValueError('Unexpected frozen PLABA ordinal feature schema or length scaler')
    control = np.stack([fit_softmax(length_only, grades[:, i]) for i in range(len(TARGETS))])
    return sentences, {'weights': weights, 'length_only_weights': control, 'length_mean': center,
                       'length_scale': scale, 'ratings': LEVELS, 'targets': np.asarray(TARGETS)}, metadata


def prepare(raw, plaba, output, report):
    sentences, weights, metadata = plaba_control(plaba)
    stimuli, outcomes, publications, excluded = select_sentences(raw, sentences)
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / 'stimuli.jsonl', stimuli)
    write_jsonl(output / OUTCOMES, outcomes)
    np.savez_compressed(output / 'heads.npz', **weights)
    save_json(output / 'attribution.json', {
        'rating_release': AUTHOR_REPO, 'authors': ['Babak Naderi', 'Salar Mohtaj', 'Kaspar Ensikat', 'Sebastian Möller'],
        'paper': 'https://arxiv.org/abs/1904.07733', 'author_repository_license': 'MIT',
        'stimulus_source': 'German Wikipedia contributor text, kept byte-for-character after decoding the source CSV',
        'source_text_license': 'https://creativecommons.org/licenses/by-sa/3.0/',
        'article_attribution': publications, 'article_by_stimulus': {row['id']: row['group_id'] for row in outcomes},
    })
    files = {name: output / name for name in ('stimuli.jsonl', 'heads.npz', 'attribution.json')}
    files['LICENSE'] = raw / 'LICENSE'
    assay = {
        'version': 1, 'scope': 'Cross-language rank transfer to German learner judgments of isolated sentences.',
        'model': MODEL, 'revision': REVISION, 'protocol_sha256': digest(PROTOCOL),
        'preparation_code_sha256': digest(Path(__file__)), 'author_file_sha256': SOURCE_HASHES,
        'plaba_data_sha256': metadata['data']['current_stage2_sha256'], 'plaba_ordinal_head_sha256': metadata['head_sha256'],
        'plaba_candidate_array': 'weights', 'ordinal_length_control_l2': PENALTY,
        'target_order': list(TARGETS), 'observed_difficulty_order': list(RATINGS),
        'human_exposure': 'Complete isolated sentence', 'human_measurement': 'Separate aggregate mean ratings, range 1–7',
        'cases': len(stimuli), 'article_groups': len(publications), 'excluded': excluded,
        'model_input_hashes': {name: digest(path) for name, path in files.items()},
        'heldout_outcomes_sha256': digest(output / OUTCOMES),
        'prespecified_evaluation': {
            'primary_metric': 'Within-article pairwise order concordance, mean across eligible articles.',
            'polarity': 'Model scores higher=easier or better understood; human scores higher=harder or less understood.',
            'minimum_certified_sentence_coverage': 0.95, 'bootstrap_unit': 'source article', 'bootstrap_draws': 10000,
            'bootstrap_seed': 43, 'requirements': 'Each target paired improvement 95% lower bound above zero.',
        },
    }
    save_json(output / 'assay.json', assay)
    files['assay.json'] = output / 'assay.json'
    package = output / 'model-inputs.tar.gz'
    write_archive(package, files)
    with tarfile.open(package, 'r:gz') as archive:
        hashes = {}
        for member in archive:
            if not member.isfile() or member.name not in files or member.name in hashes:
                raise ValueError(f'Unexpected transfer archive member: {member.name}')
            with archive.extractfile(member) as stream:
                hashes[member.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if hashes != {**assay['model_input_hashes'], 'assay.json': digest(output / 'assay.json')}:
        raise ValueError('Transfer archive differs from its exact files')
    save_json(report, {**assay, 'archive_members_sha256': hashes, 'archive_sha256': digest(package)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('raw', 'plaba', 'output', 'report'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    prepare(**vars(parser.parse_args()))
