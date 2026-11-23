"""Seal and score a positive-only transfer diagnostic for the frozen IIRC anchor heads."""

import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
from shutil import copyfile

import numpy as np

from research.core_features.readback_score import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, paired_material_interval
from research.semantic_tracking.iirc_anchor_fit import (
    FORM_FEATURES, MAX_WORDS, METHODS, TOP_SPANS, WORD, candidate_logits, candidate_spans, passage_design, top_spans,
)
from research.semantic_tracking.iirc_anchor_score import MIN_IOU, iou, proposals_for
from research.semantic_tracking.iirc_linked_pack import write_archive
from research.semantic_tracking.iirc_prepare import digest
from research.semantic_tracking.joint_score import rows_by_key
from research.stage2.records import write_jsonl
from research.stage3.probe import _jsonl
from research.stage3.run import MODEL, REVISION, save_json


SOURCE_HEADS_SHA256 = 'c7f45af167c320e8f7065a3fb9a45d59aa4c2ff912f6f415f93500c028a90f18'
PUBLIC = ('stimuli.jsonl', 'sentences.jsonl', 'heads.npz')
METRICS = ('top1_overlap', 'top5_overlap', 'top1_exact', 'top5_exact')
PRIMARY = 'top5_overlap'


def prepare(prepared, labels, snapshot, heads, output):
    from research.semantic_tracking.iirc_linked_prepare import load_tokenizer

    gold = json.loads(labels.read_text(encoding='utf-8'))
    for name in ('stimuli.jsonl', 'trials.jsonl'):
        if digest(prepared / name) != gold['source']['prepared_sha256'][name]:
            raise ValueError('Prepared stimuli differ from the separately audited PLABA endpoint')
    if digest(heads) != SOURCE_HEADS_SHA256:
        raise ValueError('Expected the previously frozen IIRC form and anchor heads')
    tokenizer = load_tokenizer(snapshot)
    round_one_documents = {row['stimulus_id'] for row in _jsonl(prepared / 'trials.jsonl')
                           if row['source_ref']['file'] == 'annotations.csv' and row['metadata']['round'] == 'round1'}
    materials, sentences = [], []
    tokens = 0
    for document in _jsonl(prepared / 'stimuli.jsonl'):
        identifier, text = document['id'], document['text']
        if identifier not in round_one_documents:
            continue
        full = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = full['offset_mapping']
        positions = {end: index + 1 for index, (start, end) in enumerate(offsets) if start < end}
        certified = []
        for segment in document['segments']:
            end = segment['end']
            prefix = tokenizer(text[:end], add_special_tokens=False, return_offsets_mapping=True)
            count = positions.get(end)
            exact = (count is not None and prefix['input_ids'] == full['input_ids'][:count]
                     and prefix['offset_mapping'] == offsets[:count])
            sentences.append({'document_id': identifier, 'sentence_id': segment['id'], 'start': segment['start'],
                              'end': end, 'token_count': count if exact else None})
            if exact:
                certified.append(end)
        if not certified:
            raise ValueError(f'No independently certified sentence stops in {identifier}')
        materials.append({'id': identifier, 'text': text, 'stops': certified})
        tokens += len(full['input_ids'])
    if {row['id'] for row in materials} != round_one_documents:
        raise ValueError('The author Round 1 material inventory is incomplete')

    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / 'stimuli.jsonl', materials)
    write_jsonl(output / 'sentences.jsonl', sentences)
    copyfile(heads, output / 'heads.npz')
    save_json(output / 'assay.json', {
        'assay': 'plaba-span-transfer-v1', 'model': MODEL, 'revision': REVISION,
        'source_revision': gold['source']['revision'], 'local_gold_sha256': digest(labels),
        'inputs_sha256': {name: digest(output / name) for name in PUBLIC},
        'documents': len(materials), 'sentences': len(sentences), 'tokens': tokens,
        'certified_sentences': sum(row['token_count'] is not None for row in sentences),
        'methods': list(METHODS), 'metrics': list(METRICS), 'primary': PRIMARY,
        'iou_threshold': MIN_IOU, 'top_k': TOP_SPANS, 'candidate_max_words': MAX_WORDS,
        'bootstrap': {'unit': 'Round 1 reader-connected article component', 'draws': BOOTSTRAP_DRAWS,
                      'seed': BOOTSTRAP_SEED},
    })
    write_archive(output / 'model-inputs.tar.gz', {name: output / name for name in (*PUBLIC, 'assay.json')})


def read_inputs(inputs):
    path = inputs / 'assay.json'
    assay = json.loads(path.read_text(encoding='utf-8'))
    expected = {'assay': 'plaba-span-transfer-v1', 'model': MODEL, 'revision': REVISION, 'methods': list(METHODS),
                'metrics': list(METRICS), 'primary': PRIMARY, 'iou_threshold': MIN_IOU, 'top_k': TOP_SPANS,
                'candidate_max_words': MAX_WORDS}
    if (any(assay[key] != value for key, value in expected.items()) or set(assay['inputs_sha256']) != set(PUBLIC)
            or assay['inputs_sha256']['heads.npz'] != SOURCE_HEADS_SHA256
            or any(digest(inputs / name) != assay['inputs_sha256'][name] for name in PUBLIC)):
        raise ValueError('PLABA transfer inputs differ from the sealed assay')
    materials = rows_by_key(inputs / 'stimuli.jsonl', lambda row: row['id'])
    sentences = rows_by_key(inputs / 'sentences.jsonl', lambda row: row['sentence_id'])
    observed_stops = defaultdict(list)
    for row in sentences.values():
        if (set(row) != {'document_id', 'sentence_id', 'start', 'end', 'token_count'}
                or not 0 <= row['start'] < row['end'] <= len(materials[row['document_id']]['text'])):
            raise ValueError('Unexpected field or coordinate in author sentence inventory')
        if row['token_count'] is not None:
            observed_stops[row['document_id']].append(row['end'])
    if (len(materials) != assay['documents'] or len(sentences) != assay['sentences']
            or sum(map(len, observed_stops.values())) != assay['certified_sentences']
            or any(row['stops'] != observed_stops[key] for key, row in materials.items())):
        raise ValueError('Sentence inventory disagrees with the extractor stops')
    return assay, digest(path), materials, sentences


def sentence_predictions(text, row, offsets, states, weights):
    count, end, start = (row[key] for key in ('token_count', 'end', 'start'))
    if count is None:
        return {method: [] for method in METHODS}
    if not 0 < count <= len(offsets) or int(offsets[:count, 1].max()) != end:
        raise ValueError('Frozen source offsets do not certify the declared author sentence end')
    design = passage_design(text[:end], offsets[:count])
    bounds = candidate_spans(design)
    visible = (bounds[:, 0] >= start) & (bounds[:, 1] <= end)
    design = replace(design, endpoints=design.endpoints[visible], forms=design.forms[visible])
    return {
        'form': top_spans(design, candidate_logits(design, states[:count], weights['form'])),
        'qwen': top_spans(design, candidate_logits(design, states[:count], weights['qwen_form'],
                                                  weights['qwen_start'], weights['qwen_end'])),
    }


def forecast(inputs, output):
    from research.core_features.extract import run as extract

    assay, assay_sha, materials, sentences = read_inputs(inputs)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(inputs / 'heads.npz', allow_pickle=False) as checkpoint:
        weights = {key: checkpoint[key] for key in checkpoint.files}
    shapes = {'form': (len(FORM_FEATURES),), 'qwen_form': (len(FORM_FEATURES),),
              'qwen_start': (128,), 'qwen_end': (128,), 'projection': (2048, 128)}
    if (set(weights) != set(shapes) or any(weights[key].shape != shape or weights[key].dtype != np.float32
                                        or not np.isfinite(weights[key]).all() for key, shape in shapes.items())):
        raise ValueError('Unexpected frozen IIRC heads or projection layout')
    features = output / 'extraction'
    extract(inputs / 'stimuli.jsonl', features, max_batch_tokens=6144, max_batch_examples=24,
            projected_tokens=(set(materials), weights['projection']), projected_only=True)
    dense_path = features / 'token-features.npz'
    receipt_path = features / 'extraction.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (receipt['input_sha256'] != assay['inputs_sha256']['stimuli.jsonl'] or receipt['model'] != MODEL
            or receipt['revision'] != REVISION or receipt['tokens'] != assay['tokens']
            or receipt['projected_tokens']['sha256'] != digest(dense_path)):
        raise ValueError('L4 extraction differs from the sealed inputs or pinned Qwen version')
    with np.load(dense_path, allow_pickle=False) as dense:
        if set(dense.files) != {'ids', 'indptr', 'offsets', 'states', 'projection'} or not np.array_equal(
                dense['projection'], weights['projection']):
            raise ValueError('Unexpected dense projected token states')
        identifiers, boundaries, offsets, states = (dense[key] for key in ('ids', 'indptr', 'offsets', 'states'))
    if (len(identifiers) != len(materials) or set(identifiers) != materials.keys()
            or boundaries.shape != (len(identifiers) + 1,) or boundaries[0] != 0
            or np.any(np.diff(boundaries) <= 0) or boundaries[-1] != len(states)
            or offsets.shape != (len(states), 2) or states.shape != (assay['tokens'], 128)):
        raise ValueError('Projected states do not cover every token of every document')
    slices = {key: slice(boundaries[index], boundaries[index + 1]) for index, key in enumerate(identifiers)}
    rows = []
    for sentence_id, row in sentences.items():
        key = row['document_id']
        slc = slices[key]
        methods = sentence_predictions(materials[key]['text'], row, offsets[slc], states[slc], weights)
        rows.append({'sentence_id': sentence_id, 'methods': methods})
    predictions = output / 'predictions.jsonl'
    write_jsonl(predictions, rows)
    save_json(output / 'forecast-manifest.json', {
        'assay_sha256': assay_sha, 'model': MODEL, 'revision': REVISION, 'heads_sha256': SOURCE_HEADS_SHA256,
        'predictions_sha256': digest(predictions), 'extraction_sha256': digest(receipt_path),
        'causality_sha256': digest(features / 'causality.json'), 'sentences': len(rows),
    })


def score(inputs, labels, predictions, seal, output):
    assay, assay_sha, materials, sentences = read_inputs(inputs)
    receipt = json.loads(seal.read_text(encoding='utf-8'))
    if (receipt['assay_sha256'] != assay_sha or receipt['model'] != MODEL or receipt['revision'] != REVISION
            or receipt['heads_sha256'] != SOURCE_HEADS_SHA256 or receipt['sentences'] != len(sentences)
            or receipt['predictions_sha256'] != digest(predictions) or assay['local_gold_sha256'] != digest(labels)):
        raise ValueError('Forecasts or local labels fail their sealed hash')
    forecasts = rows_by_key(predictions, lambda row: row['sentence_id'])
    if forecasts.keys() != sentences.keys():
        raise ValueError('Missing or extra supplied sentences')
    rendered = {}
    for key, sentence in sentences.items():
        methods = forecasts[key]['methods']
        start, end = (sentence[field] for field in ('start', 'end'))
        text = materials[sentence['document_id']]['text']
        if set(forecasts[key]) != {'sentence_id', 'methods'} or set(methods) != set(METHODS):
            raise ValueError(f'Unexpected prediction fields at {key}')
        if sentence['token_count'] is None:
            if any(methods.values()):
                raise ValueError('An uncertified sentence must have no proposed spans')
            rendered[key] = methods
            continue
        rendered[key] = proposals_for({'paragraph_id': key, 'methods': methods}, text)
        if any(not start <= a < b <= end for proposals in rendered[key].values() for a, b in proposals):
            raise ValueError('A prediction extends outside its supplied sentence')

    gold = json.loads(labels.read_text(encoding='utf-8'))
    targets = {}
    for row in gold['positives']:
        sentence = sentences[row['sentence_id']]
        text = materials[row['document_id']]['text']
        for span in row['spans']:
            a, b = span['start'], span['end']
            if (row['document_id'] != sentence['document_id'] or not sentence['start'] <= a < b <= sentence['end']
                    or text[a:b] != span['expression']):
                raise ValueError('Gold span does not align to its published source sentence')
            key = row['sentence_id'], a, b
            component = row['round1_reader_connected_group']
            if targets.setdefault(key, component) != component:
                raise ValueError('A repeated span has inconsistent reader grouping')
    if len(targets) != gold['unique_exact_document_spans']:
        raise ValueError('Retrospective union differs from the independently audited positive inventory')

    values = {metric: {method: [] for method in METHODS} for metric in METRICS}
    within_support = {name: 0 for name in ('overlap', 'exact')}
    groups = []
    for (sentence_id, a, b), component in sorted(targets.items()):
        row = sentences[sentence_id]
        text = materials[row['document_id']]['text']
        words = list(WORD.finditer(text, row['start'], row['end']))
        candidates = [(left.start(), right.end()) for index, left in enumerate(words) for right in words[index:index + MAX_WORDS]]
        within_support['exact'] += (a, b) in candidates
        within_support['overlap'] += any(iou((a, b), candidate) >= MIN_IOU for candidate in candidates)
        groups.append({'group_id': component})
        for method, proposals in rendered[sentence_id].items():
            for k in (1, TOP_SPANS):
                values[f'top{k}_overlap'][method].append(any(iou((a, b), candidate) >= MIN_IOU for candidate in proposals[:k]))
                values[f'top{k}_exact'][method].append((a, b) in proposals[:k])
    scores = {}
    for name, methods in values.items():
        scores[name] = {'hits': {method: sum(hits) for method, hits in methods.items()},
                        'means': {method: float(np.mean(hits)) for method, hits in methods.items()},
                        'qwen_minus_form': paired_material_interval(
                            np.asarray(methods['qwen'], dtype=float) - np.asarray(methods['form'], dtype=float), groups)}
    save_json(output, {
        'assay_sha256': assay_sha, 'forecasts_sha256': digest(predictions), 'label_audit_sha256': digest(labels),
        'scope': 'Positive-only phrase recovery in a supplied PLABA author sentence with frozen IIRC weights.',
        'targets': len(targets), 'sentences_containing_targets': len({key[0] for key in targets}),
        'targets_with_certified_sentence_end': sum(sentences[key[0]]['token_count'] is not None for key in targets),
        'candidate_support_count': within_support, 'primary': PRIMARY, 'scores': scores, 'bootstrap': assay['bootstrap'],
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    steps = parser.add_subparsers(dest='step', required=True)
    for function, names in ((prepare, ('prepared', 'labels', 'snapshot', 'heads', 'output')),
                            (forecast, ('inputs', 'output')),
                            (score, ('inputs', 'labels', 'predictions', 'seal', 'output'))):
        command = steps.add_parser(function.__name__)
        command.set_defaults(callback=function)
        for name in names:
            command.add_argument(f'--{name}', type=Path, required=True)
    args = vars(parser.parse_args())
    args.pop('step')
    args.pop('callback')(**args)
