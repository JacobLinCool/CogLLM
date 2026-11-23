"""Seal TECO article inputs and local crossed reader/passage scoring records."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.core_features.extract import plan
from research.semantic_tracking.iirc_linked_pack import write_archive
from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3.run import MODEL, REVISION, TRANSFORMERS_VERSION, save_json
from research.stage5 import nahatame_audit


PROTOCOL = Path(__file__).with_name('teco-proficiency-protocol.md')
SOURCE_AUDIT = PROTOCOL.parents[1] / 'results' / 'nahatame-teco-audit.json'
TOKENIZER_SHA256 = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
FOLD_COUNT = 5
FOLD_NAMESPACE = 'cogllm-teco-proficiency-v1'


def balanced_folds(ids, stratum):
    ranked = sorted(ids, key=lambda key: (sha256(f'{FOLD_NAMESPACE}/{stratum}/{key}'.encode()).digest(), key))
    return {key: rank % FOLD_COUNT for rank, key in enumerate(ranked)}


def make_records(raw, source, bodies):
    readers = {int(row['subjectid']): int(row['pro'])
               for row in nahatame_audit.csv_rows(raw / 'teco_participant_info.csv')}
    grades = {row['text_id']: row['grade'] for row in source['passage_metadata']}
    strata = defaultdict(list)
    for text_id, grade in grades.items():
        strata[grade].append(text_id)
    if (len(readers) != 41 or not all(0 <= score <= 19 for score in readers.values())
            or grades.keys() != bodies.keys() or Counter(grades.values()) != {'p2': 10, '2': 10, 'p1': 10}):
        raise ValueError('Unexpected independently measured readers, passages or EIKEN strata')
    reader_folds = balanced_folds(readers, 'reader')
    passage_folds = {text_id: fold for grade, ids in strata.items()
                     for text_id, fold in balanced_folds(ids, f'passage/{grade}').items()}

    ratings = {(int(row['subject']), int(row['text'])): int(row['Subjective'])
               for row in nahatame_audit.csv_rows(raw / 'df_individual.csv') if row['Subjective'] != 'NA'}
    if len(ratings) != 1208 or {x for x, _ in ratings} != readers.keys() or {y for _, y in ratings} != bodies.keys():
        raise ValueError('Observed difficulty trials differ from the audited inventory')
    trial_design = [{'reader_id': person, 'id': str(passage)} for person, passage in sorted(ratings)]
    stimuli = [{'id': str(text_id), 'text': body, 'stops': [len(body)]} for text_id, body in sorted(bodies.items())]
    local_records = {
        'trial-design.jsonl': trial_design,
        'outcomes.jsonl': [{'reader_id': person, 'id': str(passage), 'rating': rating}
                           for (person, passage), rating in sorted(ratings.items())],
        'reader-background.jsonl': [{'reader_id': person, 'reading_proficiency': score}
                                    for person, score in sorted(readers.items())],
        'reader-folds.jsonl': [{'reader_id': person, 'fold': fold}
                               for person, fold in sorted(reader_folds.items())],
        'passage-folds.jsonl': [{'id': str(passage), 'fold': fold, 'eiken_grade': grades[passage]}
                                for passage, fold in sorted(passage_folds.items())],
    }
    return stimuli, local_records, crossed_cells(ratings, reader_folds, passage_folds)


def crossed_cells(ratings, reader_folds, passage_folds):
    cells = []
    evaluation_totals = Counter()
    pairs = sorted(ratings)
    for reader_fold in range(FOLD_COUNT):
        for passage_fold in range(FOLD_COUNT):
            fit, evaluation = [], []
            for person, passage in pairs:
                match_reader = reader_folds[person] == reader_fold
                match_passage = passage_folds[passage] == passage_fold
                if match_reader and match_passage:
                    evaluation.append((person, str(passage)))
                elif not match_reader and not match_passage:
                    fit.append((person, str(passage)))
            if not fit or not evaluation:
                raise ValueError('Each crossed cell needs both fit and evaluation trials')
            fit_readers, fit_passages = map(set, zip(*fit))
            eval_readers, eval_passages = map(set, zip(*evaluation))
            if fit_readers & eval_readers or fit_passages & eval_passages:
                raise ValueError('Crossed cell shares a held-out reader or passage with the fit')
            evaluation_totals.update(evaluation)
            cells.append({'reader_fold': reader_fold, 'passage_fold': passage_fold,
                          'fit': fit, 'evaluation': evaluation})
    if evaluation_totals != Counter((person, str(passage)) for person, passage in ratings):
        raise ValueError('Every observed rating must appear in exactly one evaluation cell')
    return cells


def certify_stops(stimuli, tokenizer_path):
    from tokenizers import Tokenizer, __version__

    if digest(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError('Expected the pinned Qwen3.5-2B tokenizer bytes')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(text, *, add_special_tokens, return_offsets_mapping=False):
        encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        result = {'input_ids': encoded.ids}
        if return_offsets_mapping:
            result['offset_mapping'] = encoded.offsets
        return result

    documents, rejected = plan(encode, stimuli)
    if rejected or len(documents) != len(stimuli):
        raise ValueError('Each passage needs one exact causal body-end stop')
    stops = []
    for identifier, tokens, ((stop, position),) in sorted(documents, key=lambda row: int(row[0])):
        if position != len(tokens) - 1:
            raise ValueError(f'The body-end stop is not the final token of passage {identifier}')
        stops.append({'id': identifier, 'stop': stop, 'token': position, 'tokens': len(tokens)})
    return stops, __version__


def summarize_cells(cells, trial_count):
    return [{'reader_fold': cell['reader_fold'], 'passage_fold': cell['passage_fold'],
             'fit': len(cell['fit']), 'evaluation': len(cell['evaluation']),
             'excluded_one_dimension': trial_count - len(cell['fit']) - len(cell['evaluation'])}
            for cell in cells]


def prepare(raw, tokenizer, output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    source, bodies = nahatame_audit.audit(raw)
    audited = json.loads(SOURCE_AUDIT.read_text(encoding='utf-8'))
    if json.loads(json.dumps(source)) != audited:
        raise ValueError('Source inventory differs from the committed release audit')
    stimuli, records, cells = make_records(raw, source, bodies)
    stops, tokenizer_version = certify_stops(stimuli, tokenizer)
    records['crossed-cells.jsonl'] = cells

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.teco-proficiency-', dir=output.parent) as temporary:
        working = Path(temporary) / 'prepared'
        model_input = working / 'model_input'
        model_input.mkdir(parents=True)
        local = working / 'local'
        local.mkdir(mode=0o700)
        for name, rows in records.items():
            path = local / name
            write_jsonl(path, rows)
            path.chmod(0o600)
        write_jsonl(model_input / 'stimuli.jsonl', stimuli)
        write_jsonl(model_input / 'certified-stops.jsonl', stops)
        save_json(model_input / 'extraction-plan.json', {
            'model': MODEL, 'model_revision': REVISION, 'transformers': TRANSFORMERS_VERSION,
            'tokenizer_preflight_version': tokenizer_version, 'tokenizer_sha256': TOKENIZER_SHA256,
            'dtype': 'float32', 'planned_accelerator': 'NVIDIA L4', 'record_fields': ['id', 'text', 'stops'],
            'extractor': 'research.core_features.extract', 'runtime_causality_checks': 'certified_token_controls',
            'tokenized_field': 'text', 'stops': 'Unicode code-point offset at the end of each source body',
            'documents': len(stimuli), 'expected_certified_stops': len(stops),
            'total_input_tokens': sum(row['tokens'] for row in stops), 'max_input_tokens': max(row['tokens'] for row in stops),
            'sha256': {path.name: digest(path) for path in sorted(model_input.glob('*.jsonl'))},
        })
        write_archive(working / 'model-inputs.tar.gz', {path.name: path for path in sorted(model_input.iterdir())})

        trial_count = len(records['trial-design.jsonl'])
        passage_folds = records['passage-folds.jsonl']
        receipt = {
            'status': 'prepared_before_any_model_extraction_or_fitting',
            'source': {'ratings': source['rating_source'], 'passages': source['teco_source']},
            'source_use': {
                'teco_project_license': 'https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/',
                'passage_rightsholder_guideline': 'https://www.eiken.or.jp/eiken/en/pdf-guideline01.pdf#page=5',
                'model_extraction_authorization': 'unverified',
            },
            'source_sha256': source['files_sha256'], 'source_audit_sha256': digest(SOURCE_AUDIT),
            'source_audit_code_sha256': digest(Path(nahatame_audit.__file__)),
            'protocol_sha256': digest(PROTOCOL), 'preparation_code_sha256': digest(Path(__file__)),
            'counts': {'readers': len(records['reader-folds.jsonl']), 'passages': len(stimuli),
                       'observed_ratings': trial_count, 'missing_ratings': source['trial_rows'] - trial_count,
                       'exact_body_end_stops': len(stops)},
            'target': {'name': 'post_reading_passage_difficulty', 'levels': list(range(1, 10)),
                       'anchors': {'1': 'very easy', '9': 'very difficult'}},
            'background': {'name': 'pre_reading_english_reading_proficiency', 'range': [0, 19],
                           'observed_range': source['proficiency_measured_score_range'],
                           'use': 'optional downstream readout only'},
            'folds': {'count': FOLD_COUNT, 'index_base': 0, 'hash': 'SHA-256', 'namespace': FOLD_NAMESPACE,
                      'rank_input_utf8': '{namespace}/reader/{reader_id} or {namespace}/passage/{eiken_grade}/{text_id}',
                      'allocation': 'Sort by digest and numeric source ID within each stratum, then assign rank % 5.',
                      'reader_counts': [sum(row['fold'] == fold for row in records['reader-folds.jsonl'])
                                        for fold in range(FOLD_COUNT)],
                      'passage_counts_by_grade': [{grade: sum(row['fold'] == fold and row['eiken_grade'] == grade
                                                                for row in passage_folds)
                                                   for grade in ('p2', '2', 'p1')} for fold in range(FOLD_COUNT)],
                      'cells': summarize_cells(cells, trial_count), 'trial_key': ['reader_id', 'id']},
            'body_sha256_by_passage': {str(key): sha256(body.encode()).hexdigest() for key, body in sorted(bodies.items())},
            'sha256': {str(path.relative_to(working)): digest(path) for path in sorted(working.rglob('*')) if path.is_file()},
        }
        save_json(working / 'preparation-receipt.json', receipt)
        working.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path, help='Directory of the audited TECO and Nahatame source files')
    parser.add_argument('tokenizer', type=Path, help='Pinned Qwen3.5-2B tokenizer.json')
    parser.add_argument('output', type=Path, help='New working directory for the preparation')
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw, args.tokenizer, args.output), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
