"""Prepare grade-free L4 input and local crossed-fold records for StudentAnnotations."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from research.core_features.extract import plan
from research.gradio_app.plaba import TOKENIZER_SHA256
from research.semantic_tracking.iirc_linked_pack import write_archive
from research.semantic_tracking.iirc_prepare import digest
from research.stage2.records import write_jsonl
from research.stage3.run import MODEL, REVISION, TRANSFORMERS_VERSION, save_json

from . import student_annotations_audit as audit


PROTOCOL = Path(__file__).with_name('student-annotations-background.md')
SOURCE_AUDIT = PROTOCOL.parents[1] / 'results' / 'student-annotations-audit.json'
PROFILE_ENCODING = {profile: index for index, profile in enumerate(audit.LANGUAGES)}


def identifier(namespace, value):
    return hashlib.sha256(f'cogllm-studentannotations-v1/{namespace}/{value}'.encode()).hexdigest()[:24]


def make_records(guide):
    stimuli, sentences, readers, design, outcomes = {}, {}, {}, [], []
    for row in guide:
        case = identifier('sentence', row.text)
        normalized = ' '.join(row.text_key)
        reader = identifier('reader', row.reader)
        trial = identifier('trial', f'{reader}/{case}')
        if stimuli.setdefault(case, row.text) != row.text:
            raise ValueError('Stimulus key collision')
        if case not in sentences:
            sentences[case] = {'id': case, 'group': identifier('normalized_sentence', normalized),
                               'fold': audit.partition('sentence', normalized)}
        if reader not in readers:
            readers[reader] = {'reader': reader, 'fold': audit.partition('reader', row.reader),
                               'language': PROFILE_ENCODING[row.language]}
        design.append({'trial': trial, 'id': case, 'reader': reader})
        outcomes.append({'trial': trial, 'grade': row.grade})
    return ([{'id': case, 'text': text, 'stops': [len(text.rstrip())]} for case, text in sorted(stimuli.items())],
            {'sentences.jsonl': list(sentences.values()), 'readers.jsonl': list(readers.values()),
             'trial-design.jsonl': design, 'outcomes.jsonl': outcomes})


def certify(stimuli, path):
    from tokenizers import Tokenizer, __version__

    if digest(path) != TOKENIZER_SHA256:
        raise ValueError('Exact-stop preflight requires the pinned Qwen tokenizer bytes')
    tokenizer = Tokenizer.from_file(str(path))

    def encode(text, *, add_special_tokens, return_offsets_mapping=False):
        encoded = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        result = {'input_ids': encoded.ids}
        if return_offsets_mapping:
            result['offset_mapping'] = encoded.offsets
        return result

    documents, rejected = plan(encode, stimuli)
    positions = {case: (len(tokens), stops[0]) for case, tokens, stops in documents}
    if rejected or len(positions) != len(stimuli):
        raise ValueError('Not every published sentence has a certified exact stop')
    inventory = [{'id': row['id'], 'stop': positions[row['id']][1][0], 'token': positions[row['id']][1][1],
                  'tokens': positions[row['id']][0]} for row in stimuli]
    return inventory, __version__


def validate_local(records, summary):
    sentences = {row['id']: row for row in records['sentences.jsonl']}
    readers = {row['reader']: row for row in records['readers.jsonl']}
    design = records['trial-design.jsonl']
    outcomes = records['outcomes.jsonl']
    expected = (summary['study_guide_sentences'], summary['readers'], summary['study_guide_normalized_sentences'],
                summary['study_guide_observations'])
    actual = len(sentences), len(readers), len({row['group'] for row in sentences.values()}), len(design)
    if actual != expected or len({row['trial'] for row in design}) != len(design):
        raise ValueError('Rekeying changed the guide trial, sentence or reader inventory')
    if any(trial['trial'] != outcome['trial'] for trial, outcome in zip(design, outcomes, strict=True)):
        raise ValueError('Local design and outcome keys differ')
    grid = Counter((readers[row['reader']]['fold'], sentences[row['id']]['fold']) for row in design)
    counts = [[grid[i, j] for j in range(audit.FOLDS)] for i in range(audit.FOLDS)]
    if counts != summary['crossed_reader_sentence_fold_counts']:
        raise ValueError('Rekeying changed the sealed reader and sentence folds')
    return {'test_counts': counts,
            'fit_counts': [[sum(count for (i, j), count in grid.items() if i != reader and j != sentence)
                            for sentence in range(audit.FOLDS)] for reader in range(audit.FOLDS)]}


def prepare(student, textde, tokenizer, output):
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    selected = audit.select_guide(student, textde)
    summary = audit.summarize(*selected)
    if summary != json.loads(SOURCE_AUDIT.read_text(encoding='utf-8')):
        raise ValueError('Guide selection differs from the audited inventory')
    stimuli, records = make_records(selected[0])
    folds = validate_local(records, summary)
    stops, tokenizer_version = certify(stimuli, tokenizer)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.student-annotations-', dir=output.parent) as temporary:
        working = Path(temporary) / 'assay'
        model_input = working / 'model_input'
        model_input.mkdir(parents=True)
        local = working / 'local'
        local.mkdir(mode=0o700)
        for filename, rows in records.items():
            path = local / filename
            write_jsonl(path, rows)
            path.chmod(0o600)
        write_jsonl(model_input / 'stimuli.jsonl', stimuli)
        write_jsonl(model_input / 'certified-stops.jsonl', stops)
        plan_path = model_input / 'extraction-plan.json'
        save_json(plan_path, {
            'model': MODEL, 'model_revision': REVISION, 'transformers': TRANSFORMERS_VERSION,
            'tokenizer_preflight_version': tokenizer_version, 'tokenizer_sha256': TOKENIZER_SHA256,
            'dtype': 'float32', 'planned_accelerator': 'NVIDIA L4', 'record_fields': ['id', 'text', 'stops'],
            'extractor': 'research.core_features.extract', 'runtime_causality_checks': 'certified_token_controls',
            'stops': 'Unicode code-point offset just after the final visible character of each source sentence',
            'documents': len(stimuli), 'expected_certified_stops': len(stops),
            'total_input_tokens': sum(row['tokens'] for row in stops), 'max_input_tokens': max(row['tokens'] for row in stops),
            'sha256': {path.name: digest(path) for path in sorted(model_input.glob('*.jsonl'))},
        })
        packet = working / 'model-inputs.tar.gz'
        write_archive(packet, {path.name: path for path in sorted(model_input.iterdir())})
        receipt = {
            'status': 'prepared_before_any_model_extraction_or_fitting', 'scope': 'German standalone guide sentence difficulty',
            'source': summary['author_release'], 'comparison_source_sha256': summary['comparison_source_sha256'],
            'source_audit_sha256': digest(SOURCE_AUDIT), 'protocol_sha256': digest(PROTOCOL),
            'source_selection_code_sha256': digest(Path(audit.__file__)), 'prep_code_sha256': digest(Path(__file__)),
            'counts': {'sentences': len(stimuli), 'normalized_sentence_groups': summary['study_guide_normalized_sentences'],
                       'readers': summary['readers'], 'trials': summary['study_guide_observations'],
                       'exact_causal_stops': len(stops)},
            'crossed_four_fold': folds, 'language_code': PROFILE_ENCODING,
            'grade_meaning': {'1': 'very easy to understand', '5': 'very difficult to understand'},
            'sha256': {str(path.relative_to(working)): digest(path) for path in sorted(working.rglob('*')) if path.is_file()},
        }
        receipt_path = working / 'preparation-receipt.json'
        save_json(receipt_path, receipt)
        for path in (model_input / 'stimuli.jsonl', model_input / 'certified-stops.jsonl', plan_path, packet, receipt_path):
            path.chmod(0o444)
        working.rename(output)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('student', 'textde', 'tokenizer', 'output'):
        parser.add_argument(f'--{argument}', type=Path, required=True)
    print(json.dumps(prepare(**vars(parser.parse_args()))['counts']))
