"""Compare archived PLABA sentence ranks on the existing development families."""

import argparse
from collections import defaultdict
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import re

import numpy as np

from research.backbone_transfer.extract import digest, read_json, stream_rows, write_json
from research.union_training.verify_observations import require, verified_member


TASK = 'plaba_processing_ease'
RATINGS = np.arange(1, 6, dtype=np.float64)
METHODS = ('sentence_length', 'epoch_12', 'epoch_31')


def forecast(path, receipt_sha256, task_rows, epoch):
    seal = read_json(path / 'seal.json')
    require(seal['packet_receipt_sha256'] == receipt_sha256, f'Epoch {epoch} uses another packet')
    fitting = verified_member(path, 'fitting.json', seal['files_sha256']['fitting.json'])
    require(read_json(fitting)['selected_epoch'] == epoch, f'Expected epoch {epoch} weights')
    predictions = verified_member(path, 'predictions.npz', seal['files_sha256']['predictions.npz'])
    expected = np.asarray([i for i, row in enumerate(task_rows) if row['role'] == 1])
    with np.load(predictions, allow_pickle=False) as data:
        key = f'{TASK}.development'
        require(np.array_equal(data[f'{key}.rows'], expected), 'PLABA development row order changed')
        probabilities = data[f'{key}.probabilities'].astype(np.float64)
    require(probabilities.shape == (len(expected), len(RATINGS))
            and np.isfinite(probabilities).all() and np.all(probabilities >= 0)
            and np.allclose(probabilities.sum(axis=1), 1.0, atol=2e-6, rtol=0),
            'Invalid archived ordinal forecasts')
    return dict(zip(expected.tolist(), (probabilities @ RATINGS).tolist(), strict=True))


def sentence_geometry(materials, ratings):
    groups = {row['id']: row['group_id'] for row in stream_rows(ratings)}
    positions = {}
    for row in stream_rows(materials):
        text = row['text']
        key = sha256(text.encode()).hexdigest()
        for number, segment in enumerate(row['segments'], 1):
            start, end = segment['start'], segment['end']
            require(0 <= start < end <= len(text) and (key, end) not in positions,
                    'PLABA source sentence is ambiguous')
            sentence = text[start:end]
            positions[key, end] = {
                'variant': row['id'], 'group': groups[row['id']], 'sentence': number,
                'start': start, 'end': end, 'document_end': len(text), 'text': sentence,
                'words': len(re.findall(r"\b[^\W_]+(?:[-’'][^\W_]+)*\b", sentence)),
            }
    return positions


def evaluate_document(sentences):
    pairs = [(l, r) for l, r in combinations(sentences, 2) if l['human'] != r['human']]
    hardest = min(row['human'] for row in sentences)
    metrics = {}
    for name in METHODS:
        score = sum((1.0 if (l['human'] - r['human']) * (l[name] - r[name]) > 0
                     else 0.5 if l[name] == r[name] else 0.0) for l, r in pairs)
        model_min = min(row[name] for row in sentences)
        model_hardest = [row for row in sentences if row[name] == model_min]
        metrics[name] = {
            'pair_order_accuracy': score / len(pairs) if pairs else None,
            'hardest_sentence_hit': sum(row['human'] == hardest for row in model_hardest) / len(model_hardest),
            'mae_rating_points': float(np.mean([abs(row[name] - row['human']) for row in sentences])),
        }
    return {'sentences': len(sentences), 'nontied_pairs': len(pairs), 'metrics': metrics}


def evaluate_rewrites(path, points):
    by_group = defaultdict(list)
    for row in stream_rows(path):
        groups = []
        for side in ('original', 'plain'):
            groups.append([points.get((doc, stop)) for doc, stop in row[side]])
        if any(point is None for side in groups for point in side):
            continue
        original, plain = groups
        require(all(point['group'] == row['group_id'] for point in original + plain),
                'Rewrite linkage crosses abstract families')
        by_group[row['group_id']].append({
            field: float(np.mean([point[field] for point in plain]) - np.mean([point[field] for point in original]))
            for field in ('human', *METHODS)
        })
    require(by_group, 'No eligible PLABA development rewrites')
    return {group: {'alignments': len(rows), **{
        field: float(np.mean([row[field] for row in rows])) for field in ('human', *METHODS)
    }} for group, rows in sorted(by_group.items())}


def run(packet, baseline, candidate, cache, output):
    receipt_path = packet / 'receipt.json'
    receipt = read_json(receipt_path)
    index = verified_member(packet, 'index.json', receipt['packet_sha256']['index.json'])
    task, = (task for task in read_json(index)['tasks'] if task['name'] == TASK)
    require(task['coordinates'] == 'target_and_document_end' and task['labels'] == RATINGS.tolist(),
            'PLABA measurement or rating order changed')
    task_rows = list(stream_rows(verified_member(packet, task['file'], receipt['packet_sha256'][task['file']])))
    plaba = receipt['source_files_sha256']['plaba']
    stimuli, ratings, pairs = (verified_member(cache, name, plaba[name]) for name in (
        'stage2/plaba/stimuli.jsonl', 'expository-lora/prepared/plaba/ratings.jsonl',
        'expository-lora/prepared/plaba/pairs.jsonl',
    ))
    segments = sentence_geometry(stimuli, ratings)
    previous = forecast(baseline, digest(receipt_path), task_rows, 12)
    current = forecast(candidate, digest(receipt_path), task_rows, 31)
    train, dev = [], []
    for index, row in enumerate(task_rows):
        ((doc, stop), (end_doc, end)) = row['coordinates']
        segment = segments[doc, stop]
        require(end_doc == doc and end == segment['document_end'] and row['family'] == f"plaba:{segment['group']}",
                'Archived sentence or full-document position does not match the source')
        counts = np.asarray(row['outcome'])
        require(counts.shape == RATINGS.shape and np.all(counts >= 0) and counts.sum() > 0,
                'Invalid PLABA rating counts')
        result = {**segment, 'human': float(counts @ RATINGS / counts.sum()), 'ratings': int(counts.sum())}
        if row['role'] == 0:
            train.append(result)
        else:
            dev.append({**result, 'epoch_12': previous[index], 'epoch_31': current[index]})
    require(not {r['group'] for r in train} & {r['group'] for r in dev}, 'Development reuses a training family')
    design = np.asarray([[1.0, np.log1p(row['words'])] for row in train])
    intercept, slope = np.linalg.lstsq(design, [row['human'] for row in train], rcond=None)[0]
    by_doc = defaultdict(list)
    for row in dev:
        row['sentence_length'] = float(np.clip(intercept + slope * np.log1p(row['words']), 1, 5))
        by_doc[row['variant']].append(row)
    by_doc = {doc: sorted(rows, key=lambda r: r['sentence']) for doc, rows in sorted(by_doc.items())}
    docs = {doc: evaluate_document(rows) for doc, rows in by_doc.items()}
    rewrites = evaluate_rewrites(pairs, {(row['variant'], row['end']): row for row in dev})
    summary = {}
    for method in METHODS:
        summary[method] = {}
        for metric in ('pair_order_accuracy', 'hardest_sentence_hit', 'mae_rating_points'):
            values = [d['metrics'][method][metric] for d in docs.values() if d['metrics'][method][metric] is not None]
            summary[method][metric] = float(np.mean(values))
    write_json(output, {
        'role': 'model-selection development diagnostic', 'task': TASK,
        'receipt_sha256': digest(receipt_path), 'candidate_seal_sha256': digest(candidate / 'seal.json'),
        'training_sentences': len(train), 'development_sentences': len(dev),
        'development_ratings': sum(row['ratings'] for row in dev), 'development_documents': len(by_doc),
        'development_families': len({row['group'] for row in dev}),
        'sentence_length_fit': {'intercept': float(intercept), 'log_word_count_slope': float(slope)},
        'nontied_pairs': sum(doc['nontied_pairs'] for doc in docs.values()),
        'metrics_equal_weight_document': summary, 'document_metrics': docs,
        'plain_minus_original_by_abstract_family': rewrites,
        'plain_minus_original_equal_weight_family': {field: float(np.mean([r[field] for r in rewrites.values()]))
                                                    for field in ('human', *METHODS)},
        'source_sentence_forecasts': by_doc,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packet', 'baseline', 'candidate', 'cache', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    run(**vars(parser.parse_args()))
