"""Seal a CWI Wikipedia transfer and score it after the forecasts are frozen."""

import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
from zipfile import ZipFile

import numpy as np

from research.core_features.background_prepare import MARKER, MEMBERS, SOURCE_SHA256, normalized_sentence
from research.core_features.localization_cwi_score import (
    BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, METHODS, ClusterBootstrap, group_scores, interval, load_inputs, observed,
)
from research.semantic_tracking.iirc_linked_pack import write_archive
from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3.probe import _jsonl
from research.stage3.run import MODEL, REVISION, save_json


WIKIPEDIA = {
    'traindevset/english/Wikipedia_Train.tsv': 5551,
    'traindevset/english/Wikipedia_Dev.tsv': 694,
    'testset/english/Wikipedia_Test.tsv': 870,
}
KEYS = ('id', 'group_id', 'start', 'stop', 'split')
VOTE_KEYS = ('native_total', 'nonnative_total', 'native_marked', 'nonnative_marked')
MODEL_INPUTS = ('stimuli.jsonl', 'cwi-train.jsonl', 'cwi-eval.jsonl')
OUTCOMES = 'heldout-outcomes.jsonl'
CANDIDATE = 'qwen_group_interaction'
COMPARATORS = ('qwen_pooled', 'qwen_group_offset')
MIN_COVERAGE = 0.95
GROUP_BRIER_MARGIN = 0.001
GROUP_CALIBRATION_LIMIT = 0.01
PROTOCOL = Path(__file__).resolve().parents[1] / 'core-features-background-transfer.md'


def tsv(archive, member):
    with archive.open(member) as raw, io.TextIOWrapper(raw, encoding='utf-8', newline='') as stream:
        yield from csv.reader(stream, delimiter='\t', quoting=csv.QUOTE_NONE)


def wiki_news(archive, prior, audit):
    prior_audit = json.loads(audit.read_text(encoding='utf-8'))
    hashes = prior_audit['output_sha256']
    if (prior_audit['source_archive_sha256'] != SOURCE_SHA256 or any(
            digest(prior / name) != hashes[name] for name in ('stimuli.jsonl', 'outcomes.jsonl'))):
        raise ValueError('Expected the exact WikiNews preparation used for the first sealed assay')
    original_sentences, original_hits = set(), set()
    for member in MEMBERS:
        for hit, raw_sentence, *_ in tsv(archive, member):
            marker = MARKER.match(raw_sentence)
            if marker is None:
                raise ValueError('Missing article marker in pinned WikiNews source')
            original_sentences.add(normalized_sentence(raw_sentence[marker.end():]))
            original_hits.add(hit)
    train, dev = [], []
    for row in _jsonl(prior / 'outcomes.jsonl'):
        if row['split'] == 'train':
            train.append(row)
        elif row['split'] == 'dev':
            dev.append(row)
    if ({row['group_id'] for row in train} & {row['group_id'] for row in dev}
            or len(train) != 3850 or len(dev) != 1348):
        raise ValueError('Unexpected held-in WikiNews roles or row counts')
    selected = {row['id'] for row in train + dev}
    documents = [row for row in _jsonl(prior / 'stimuli.jsonl') if row['id'] in selected]
    return documents, train, dev, original_hits, original_sentences


def wikipedia(archive, old_hits, old_sentences):
    by_hit = defaultdict(lambda: defaultdict(dict))
    hit_file = {}
    for member, n_expected in WIKIPEDIA.items():
        count = 0
        for hit, sentence, start, stop, phrase, nt, nnt, nm, nnm, label, pooled in tsv(archive, member):
            count += 1
            normalized = normalized_sentence(sentence)
            if (hit in old_hits or normalized in old_sentences or hit_file.setdefault(hit, member) != member
                    or not normalized or MARKER.match(sentence)):
                raise ValueError('Overlapping or unexpectedly marked Wikipedia input')
            start, stop, nt, nnt, nm, nnm, label = map(int, (start, stop, nt, nnt, nm, nnm, label))
            spans = by_hit[hit][sentence]
            if (not 0 <= start < stop <= len(sentence) or sentence[start:stop] != phrase or (start, stop) in spans
                    or (nt, nnt) != (10, 10) or not 0 <= nm <= nt or not 0 <= nnm <= nnt
                    or label != int(nm + nnm > 0) or abs(float(pooled) - (nm + nnm) / (nt + nnt)) > 1e-12):
                raise ValueError('Unexpected released Wikipedia candidate or vote count')
            spans[start, stop] = (nt, nnt, nm, nnm)
        if count != n_expected:
            raise ValueError(f'Unexpected row count in {member}')

    # Wikipedia has no article IDs or sentence-order markers. All sentences in a
    # HIT are linked, as are HITs sharing any normalized duplicate sentence.
    parents = {hit: hit for hit in by_hit}

    def root(hit):
        while parents[hit] != hit:
            parents[hit] = parents[parents[hit]]
            hit = parents[hit]
        return hit

    sentence_hit = {}
    for hit, sentences in by_hit.items():
        for sentence in sentences:
            owner = sentence_hit.setdefault(normalized_sentence(sentence), hit)
            left, right = sorted((root(hit), root(owner)))
            parents[right] = left

    documents, outcomes = [], []
    for hit_num, (hit, sentences) in enumerate(sorted(by_hit.items()), 1):
        for sentence_num, (sentence, spans) in enumerate(sorted(sentences.items()), 1):
            case = f'wikipedia-{hit_num:03}-{sentence_num:02}'
            documents.append({'id': case, 'text': sentence, 'stops': sorted({stop for _, stop in spans})})
            for (start, stop), votes in sorted(spans.items()):
                outcomes.append({'id': case, 'group_id': f'wikipedia-hit-family-{root(hit)}', 'start': start,
                                 'stop': stop, 'split': 'test', **dict(zip(VOTE_KEYS, votes, strict=True))})
    return documents, outcomes, len(by_hit), len(sentence_hit)


def prepare(source, prior, prior_audit, output, report):
    if digest(source) != SOURCE_SHA256:
        raise ValueError('The CWI release archive differs from the pinned author release')
    with ZipFile(source) as archive:
        documents, train, dev, old_hits, old_sentences = wiki_news(archive, prior, prior_audit)
        test_documents, test, test_hits, unique_test_sentences = wikipedia(archive, old_hits, old_sentences)
    all_documents = documents + test_documents
    combined = {row['id']: row for row in all_documents}
    referenced = defaultdict(set)
    for row in train + dev + test:
        referenced[row['id']].add(row['stop'])
    if (len(combined) != len(all_documents) or combined.keys() != referenced.keys()
            or any(set(row['stops']) != referenced[case] for case, row in combined.items())):
        raise ValueError('Prepared stops and candidates do not match')

    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / MODEL_INPUTS[0], all_documents)
    write_jsonl(output / MODEL_INPUTS[1], train)
    write_jsonl(output / MODEL_INPUTS[2], ({key: row[key] for key in KEYS} for row in dev + test))
    write_jsonl(output / OUTCOMES, dev + test)
    inputs = {name: digest(output / name) for name in MODEL_INPUTS}
    assay = {
        'version': 1, 'scope': 'CWI annotator-group lexical selection on isolated English Wikipedia sentences.',
        'model': MODEL, 'revision': REVISION, 'source_archive_sha256': SOURCE_SHA256,
        'prior_audit_sha256': digest(prior_audit), 'protocol_sha256': digest(PROTOCOL),
        'candidate_method': CANDIDATE, 'comparator_methods': list(COMPARATORS), 'forecast_methods': list(METHODS),
        'inputs_sha256': inputs, 'heldout_outcomes_sha256': digest(output / OUTCOMES),
        'counts': {name: {'candidates': len(rows), 'documents': len({row['id'] for row in rows}),
                          'material_groups': len({row['group_id'] for row in rows})}
                   for name, rows in (('wikinews_train', train), ('wikinews_dev', dev), ('wikipedia_test', test))},
        'wikipedia_source_rows': WIKIPEDIA, 'wikipedia_hits': test_hits,
        'wikipedia_unique_normalized_sentences': unique_test_sentences,
        'wikipedia_overlap_with_all_wikinews': {'exact_normalized_sentences': 0, 'hit_ids': 0},
        'acceptance': {'minimum_certified_test_coverage': MIN_COVERAGE, 'gap_mae_delta_ci_upper_below': 0,
                       'group_brier_delta_ci_upper_maximum': GROUP_BRIER_MARGIN,
                       'group_absolute_calibration_error_ci_upper_maximum': GROUP_CALIBRATION_LIMIT,
                       'bootstrap_draws': BOOTSTRAP_DRAWS, 'bootstrap_seed': BOOTSTRAP_SEED},
    }
    save_json(output / 'assay.json', assay)
    files = {name: output / name for name in (*MODEL_INPUTS, 'assay.json')}
    bundle = output / 'model-inputs.tar.gz'
    write_archive(bundle, files)
    with tarfile.open(bundle, 'r:gz') as archive:
        verified = {}
        for member in archive:
            if not member.isfile() or member.name not in files or member.name in verified:
                raise ValueError(f'Unexpected transfer archive member: {member.name}')
            with archive.extractfile(member) as stream:
                verified[member.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if verified != {**inputs, 'assay.json': digest(output / 'assay.json')}:
        raise ValueError('Transfer archive members differ from the sealed model inputs')
    save_json(report, {**assay, 'model_inputs_archive_sha256': digest(bundle), 'archive_members_sha256': verified})


def score(assay, outcomes, forecasts, seal, extraction, output):
    manifest = json.loads(assay.read_text(encoding='utf-8'))
    receipt = json.loads(seal.read_text(encoding='utf-8'))
    extract = json.loads(extraction.read_text(encoding='utf-8'))
    if (receipt['assay_sha256'] != digest(assay) or manifest['protocol_sha256'] != digest(PROTOCOL)
            or receipt['model'] != manifest['model'] or receipt['revision'] != manifest['revision']
            or receipt['extraction_sha256'] != digest(extraction)
            or extract['model'] != manifest['model'] or extract['revision'] != manifest['revision']
            or extract['dtype'] != 'float32' or extract['gpu'] != 'NVIDIA L4'
            or extract['input_sha256'] != manifest['inputs_sha256']['stimuli.jsonl']
            or digest(outcomes) != manifest['heldout_outcomes_sha256']):
        raise ValueError('Expected the sealed transfer protocol, extraction and held-out outcomes')
    rows, predictions, exclusions, prediction_sha = load_inputs(outcomes, forecasts, seal, extraction)
    test, vectors = rows['test'], predictions['test']
    bootstrap = ClusterBootstrap([row['group_id'] for row in test])
    metrics = group_scores(test, vectors, bootstrap)
    total = manifest['counts']['wikipedia_test']['candidates']
    coverage = len(test) / total
    bounds = {'certified_test_coverage': coverage >= MIN_COVERAGE}
    votes, counts = observed(test)
    by_group = {}
    for index, group in enumerate(('native', 'nonnative')):
        seen, mass = votes[:, index], counts[:, index]
        candidate = vectors[:, METHODS.index(CANDIDATE), index]
        signed, signed_draws = bootstrap.mean(candidate * mass - seen, mass)
        group_metrics = {'calibration_error': abs(signed), 'calibration_error_95_interval': interval(abs(signed_draws))}
        bounds[f'{group}_absolute_calibration'] = group_metrics['calibration_error_95_interval'][1] <= GROUP_CALIBRATION_LIMIT
        for baseline in COMPARATORS:
            base = vectors[:, METHODS.index(baseline), index]
            # The change in binomial Brier numerators cancels the constant seen term.
            delta = mass * (candidate ** 2 - base ** 2) - 2 * seen * (candidate - base)
            difference, draws = bootstrap.mean(delta, mass)
            limit = interval(draws)
            group_metrics[f'brier_delta_vs_{baseline}'] = {'difference': difference, 'paired_95_interval': limit}
            bounds[f'{group}_brier_vs_{baseline}'] = limit[1] <= GROUP_BRIER_MARGIN
        by_group[group] = group_metrics
    for baseline in COMPARATORS:
        gap = metrics['paired_differences'][f'{CANDIDATE}_minus_{baseline}']['gap_mae']
        bounds[f'gap_mae_vs_{baseline}'] = gap['article_family_bootstrap_95_interval'][1] < 0
    save_json(output, {'assay_sha256': digest(assay), 'forecasts_sha256': prediction_sha,
                       'heldout_outcomes_sha256': digest(outcomes), 'test_certified_candidates': len(test),
                       'test_certified_fraction': coverage, 'test_known_material_groups': len(bootstrap.groups),
                       'excluded_exact_stops': exclusions['test']['excluded_exact_stops'],
                       'annotator_background': metrics, 'group_calibration_and_brier': by_group,
                       'acceptance': bounds, 'all_conditions_passed': all(bounds.values())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, fields in (('prepare', ('source', 'prior', 'prior-audit', 'output', 'report')),
                         ('score', ('assay', 'outcomes', 'forecasts', 'seal', 'extraction', 'output'))):
        command = commands.add_parser(name)
        for field in fields:
            command.add_argument(f'--{field}', type=Path, required=True)
    arguments = vars(parser.parse_args())
    {'prepare': prepare, 'score': score}[arguments.pop('command')](**arguments)
