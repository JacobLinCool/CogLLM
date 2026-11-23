"""Prepare the crossed Kerwer reader-background assay without fitting a readout."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

from research.stage2.records import write_jsonl

from . import kerwer_score as kerwer


RESEARCH = Path(__file__).resolve().parents[1]
RESULTS = RESEARCH / 'results'
PROTOCOL = Path(__file__).with_name('kerwer-reader-background-protocol.md')
PROFILE_PATH = 'scoring/prereading-profiles.jsonl'
PROFILE_SHA256 = '0a7d5a2fb911272fa0d9d74cefdeae684b2a91218c02cb0f6141c50ac240bebd'
ORIGINAL_TRANSFER_SHA256 = '2fe2448c31ef0273bda16582b8b50eb727a1d4a47c04c59424160e800d035bd7'
READER_FOLDS, DOI_FOLDS = 5, 4
READER_NAMESPACE, DOI_NAMESPACE = 'kerwer-background-audit-reader', 'kerwer-background-audit-doi'
GRADES = list(range(1, 9))


def folds(keys, width, namespace):
    ordered = sorted(set(keys), key=lambda key: (sha256(f'{namespace}:{key}'.encode('utf-8')).digest(), key))
    return {key: rank % width for rank, key in enumerate(ordered)}


def validate_source(holdout, v8):
    release_path = v8 / 'run-verification.json'
    release = kerwer.read_json(release_path)
    forecasts = v8 / 'kerwer/predictions.jsonl'
    seal = v8 / 'kerwer/seal.json'
    if (release['phase'] != 'finished' or release['schema_version'] != 8 or not release['release_verified']
            or not release['endpoint_absent_after_stop'] or not release['release_usage_recorded']
            or release['checkpoint_sha256'] != kerwer.CANDIDATE_ID):
        raise ValueError('Requires the completed, verified and released V8 candidate')
    stage = release['stages']['forecast']
    if (stage['state'] != 'complete' or stage['verified'] is not True
            or any(kerwer.digest(path) != stage['members'][f'kerwer/{path.name}'] for path in (forecasts, seal))):
        raise ValueError('The reconstructed Kerwer forecasts differ from the V8 receipt')
    transfer_path = RESULTS / 'kerwer-transfer-protocol.json'
    transfer_sha = kerwer.digest(transfer_path)
    transfer = kerwer.read_json(transfer_path)
    alignment = transfer.pop('alignment')
    if (transfer_sha != ORIGINAL_TRANSFER_SHA256 or transfer != kerwer.rules()
            or kerwer.fingerprint(alignment) != kerwer.ALIGNMENT):
        raise ValueError('The original Kerwer protocol or exact-coordinate selection changed')
    manifest_path = holdout / 'manifest.json'
    source_manifest = kerwer.read_json(manifest_path)
    expected = {**kerwer.INPUTS, PROFILE_PATH: PROFILE_SHA256}
    if (source_manifest['counts'] != {'families': 12, 'variants': 36, 'readers': 166, 'outcomes': 1947}
            or any(source_manifest['sha256'][name] != digest or kerwer.digest(holdout / name) != digest
                   for name, digest in expected.items())):
        raise ValueError('Kerwer design, readers, stimuli or response ledger differ from the frozen sources')
    documents, _ = kerwer.read_study(holdout)
    values = kerwer.read_forecasts(forecasts, seal, alignment['served'])
    provenance = {'holdout_manifest_sha256': kerwer.digest(manifest_path), 'holdout_files_sha256': expected,
                  'original_transfer_protocol_sha256': transfer_sha, 'v8_run_id': release['run_id'],
                  'v8_release_receipt_sha256': kerwer.digest(release_path), 'forecasts_sha256': kerwer.digest(forecasts),
                  'forecast_seal_sha256': kerwer.digest(seal), 'candidate_id': kerwer.CANDIDATE_ID}
    return documents, values, alignment['eligible'], provenance


def article_features(design, documents, values, eligible, assignment):
    features = []
    for item in sorted(design, key=lambda row: row['id']):
        key = item['id']
        row = documents[key]
        spans = eligible['rewrite'].get(key) if item['arm'] != 'plain_with_headings' else eligible['headings'].get(key)
        if not spans or any(not any(left <= start < end <= right for left, right in row['body_spans'])
                            for start, end in spans):
            raise ValueError(f'Incomplete original certified body-only selection at {key}')
        body = [row['text'][left:right] for left, right in row['body_spans']]
        scores = [values[key][tuple(span)] for span in spans]
        scalar = math.fsum(scores) / len(scores)
        if not 1 <= scalar <= 5:
            raise ValueError(f'Corrupt ordinal article readout at {key}')
        features.append({'id': key, 'doi': item['doi'], 'arm': item['arm'], 'doi_fold': assignment[item['doi']],
                         'body_characters': sum(map(len, body)), 'body_words': sum(len(part.split()) for part in body),
                         'served_understanding': scalar, 'sentence_spans': spans})
    if len(features) != 36 or sum(len(row['sentence_spans']) for row in features) != 458:
        raise ValueError('The body-only article score does not cover the frozen 36 variants and 458 sentences')
    return features


def trial_keys(holdout, articles, readers):
    trials, seen = [], set()
    for row in kerwer.rows(holdout / kerwer.OUTCOMES):
        # Derive assignment from the already captured keys. Rating values are neither copied nor consulted.
        case, reader = row['id'], row['reader']
        doi, doi_fold = articles[case]['doi'], articles[case]['doi_fold']
        if (reader, doi) in seen:
            raise ValueError('Each reader can have one randomized variant per DOI family')
        seen.add((reader, doi))
        trials.append({'reader': reader, 'id': case, 'reader_fold': readers[reader], 'doi_fold': doi_fold})
    if len(trials) != 1947 or {row['reader'] for row in trials} != readers.keys():
        raise ValueError('The trial ledger differs from the fixed observed participant and DOI design')
    return sorted(trials, key=lambda row: (row['reader'], row['id']))


def validate_folds(trials, article_by_id):
    cells = []
    held_out = Counter()
    for r in range(READER_FOLDS):
        for d in range(DOI_FOLDS):
            train = [row for row in trials if row['reader_fold'] != r and row['doi_fold'] != d]
            test = [row for row in trials if row['reader_fold'] == r and row['doi_fold'] == d]
            train_readers, test_readers = ({row['reader'] for row in split} for split in (train, test))
            train_dois, test_dois = ({article_by_id[row['id']]['doi'] for row in split} for split in (train, test))
            fit_arms = {article_by_id[row['id']]['arm'] for row in train}
            if (not train or not test or train_readers & test_readers or train_dois & test_dois
                    or len(train_dois) != 9 or len(test_dois) != 3 or fit_arms != set(kerwer.ARMS.values())):
                raise ValueError(f'Crossed cell {(r, d)} lost first-seen reader, DOI or arm coverage')
            held_out.update((row['reader'], row['id']) for row in test)
            cells.append({'reader_fold': r, 'doi_fold': d, 'fit_trials': len(train), 'test_trials': len(test),
                          'fit_readers': len(train_readers), 'test_readers': len(test_readers),
                          'fit_doi_families': len(train_dois), 'test_doi_families': len(test_dois)})
    if len(held_out) != len(trials) or set(held_out.values()) != {1}:
        raise ValueError('Every observed trial must appear in exactly one test cell')
    return cells


def specification():
    return {'grades': GRADES, 'primary': 'normalized_ranked_probability_score',
            'models': {'L': ['log_body_characters', 'log_body_words', 'arm'], 'LP': ['L', 'epa_z'],
                       'T': ['L', 'served_understanding_z'], 'TP': ['T', 'epa_z'],
                       'TPI': ['TP', 'epa_z * served_understanding_z']},
            'solver': {'family': 'proportional_odds_logit', 'optimizer': 'L-BFGS-B', 'maxiter': 400,
                       'ftol': 1e-12, 'gtol': 1e-7, 'mean_trial_nll': True, 'squared_slope_penalty': 0.01,
                       'penalize_thresholds': False},
            'blank_background': {'models': ['TP', 'TPI'], 'marginal_over': 'unique fit readers with equal weights',
                                 'reference': 'T', 'rps_increase_ci95_upper_below': 0.01,
                                 'calibration_increase_ci95_upper_below': 0.01},
            'calibration': {'measure': 'cumulative_probability_ece', 'cuts': list(range(1, 8)),
                            'bin_edges': [0, 0.2, 0.4, 0.6, 0.8, 1], 'increase_ci95_upper_below': 0.01},
            'gains': {'reader': ['T', 'TP'], 'joint_vs_length_arm_and_reader': ['LP', 'TP'],
                      'interaction': ['TP', 'TPI'], 'rps_reduction_ci95_lower_above': 0},
            'interval': {'reader_doi_paired_bootstrap_draws': 10000, 'seed': 20261121,
                         'quantiles': [0.025, 0.975], 'oof_refit': False}}


def prepare(holdout, v8, output):
    if output.exists():
        raise FileExistsError(output)
    documents, values, eligible, provenance = validate_source(holdout, v8)
    design = list(kerwer.rows(holdout / kerwer.DESIGN))
    profiles = list(kerwer.rows(holdout / PROFILE_PATH))
    if (len(profiles) != 166 or len({row['reader'] for row in profiles}) != len(profiles)
            or any(type(row['english_e_pa']) is not int or not 27 <= row['english_e_pa'] <= 60 for row in profiles)):
        raise ValueError('Expected one independently measured E-PA score for each of 166 readers')
    reader_folds = folds((row['reader'] for row in profiles), READER_FOLDS, READER_NAMESPACE)
    doi_folds = folds((row['doi'] for row in design), DOI_FOLDS, DOI_NAMESPACE)
    articles = article_features(design, documents, values, eligible, doi_folds)
    article_by_id = {row['id']: row for row in articles}
    readers = [{'reader': row['reader'], 'english_e_pa': row['english_e_pa'], 'reader_fold': reader_folds[row['reader']]}
               for row in sorted(profiles, key=lambda row: row['reader'])]
    trials = trial_keys(holdout, article_by_id, reader_folds)
    cells = validate_folds(trials, article_by_id)

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.kerwer-background-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'prepared'
        staged.mkdir(mode=0o700)
        local = staged / 'local'
        local.mkdir(mode=0o700)
        files = {'local/articles.jsonl': articles, 'local/readers.jsonl': readers, 'local/trial-keys.jsonl': trials}
        for name, rows in files.items():
            write_jsonl(staged / name, rows)
            (staged / name).chmod(0o600)
        receipt = {'status': 'prepared_before_kerwer_reader_background_fitting_or_scoring',
                   'source': provenance, 'prior_use': 'same readers and DOI families used for published and project rewrite effects',
                   'protocol_sha256': kerwer.digest(PROTOCOL), 'preparation_code_sha256': kerwer.digest(Path(__file__)),
                   'counts': {'readers': len(readers), 'doi_families': len(doi_folds), 'variants': len(articles),
                              'served_body_sentences': sum(len(row['sentence_spans']) for row in articles), 'trials': len(trials)},
                   'fold_design': {'reader_folds': READER_FOLDS, 'doi_folds': DOI_FOLDS,
                                   'reader_namespace': READER_NAMESPACE, 'doi_namespace': DOI_NAMESPACE,
                                   'fit': 'both folds differ', 'test': 'both folds match', 'cells': cells},
                   'fit_and_score': specification(), 'outcomes': 'Original frozen outcomes remain local, referenced by checksum only',
                   'prepared_sha256': {name: kerwer.digest(staged / name) for name in files}}
        kerwer.write_json(staged / 'preparation-receipt.json', receipt)
        staged.rename(output)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--holdout', type=Path, required=True)
    parser.add_argument('--v8', type=Path, default=RESULTS / 'gradio-candidate-l4')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    result = prepare(args.holdout, args.v8, args.output)
    if args.receipt is not None:
        kerwer.write_json(args.receipt, result)
