"""Forecast ExpMRC evidence arrival with the frozen QASPER heads."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory


HEADS_SHA256 = '5bf54b491e1d1be375a12fa1e2e1dbc8c5c5127ed242d203465f1c1fee679ec7'
METHODS = ('position', 'question_only', 'pooled_rank', 'token_rank')


def digest(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def rows(path, expected_sha, fields):
    if digest(path) != expected_sha:
        raise ValueError(f'Input differs from the frozen ExpMRC material: {path.name}')
    values = [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line]
    if (not values or any(set(row) != fields or not isinstance(row['id'], str) or not row['id']
                          or not isinstance(row['text'], str) or not row['text'].strip() for row in values)
            or len({row['id'] for row in values}) != len(values)):
        raise ValueError(f'Unexpected ExpMRC inference fields or identifiers: {path.name}')
    return values


def material(articles_path, queries_path, articles_sha, queries_sha):
    articles = rows(articles_path, articles_sha, {'id', 'text'})
    queries = rows(queries_path, queries_sha, {'id', 'article_id', 'text'})
    ids = {row['id'] for row in articles}
    if ids & {row['id'] for row in queries} or {row['article_id'] for row in queries} != ids:
        raise ValueError('The ExpMRC queries must refer to exactly the sealed articles')
    return articles, queries


def prepare_stimuli(articles, queries, tokenizer, segmenter):
    from research.gradio_app.plaba import sentence_spans

    stops = {}
    candidates, tokens = 0, 0
    for row in (*articles, *queries):
        text = row['text']
        complete = tokenizer.encode(text, add_special_tokens=False)
        tokens += len(complete.ids)
        if not 0 < len(complete.ids) <= 8192:
            raise ValueError('Invalid token length for a frozen ExpMRC input')
        candidate = [end for _, end in sentence_spans(text, segmenter) if end < len(text)] if 'article_id' not in row else []
        candidate.append(len(text))
        candidates += len(candidate)
        last_token = {right: index for index, (left, right) in enumerate(complete.offsets) if left < right}
        certified = []
        for end in candidate:
            index = last_token.get(end)
            if index is not None and complete.ids[:index + 1] == tokenizer.encode(text[:end], add_special_tokens=False).ids:
                certified.append(end)
        if not certified or certified[-1] != len(text):
            raise ValueError(f'Complete source is uncertified for {row["id"]}')
        stops[row['id']] = certified
    stimuli = [{key: row[key] for key in ('id', 'text')} | {'stops': stops[row['id']]} for row in (*articles, *queries)]
    count = sum(len(value) for value in stops.values())
    return stimuli, {'documents': len(articles), 'queries': len(queries), 'candidate_stops': candidates,
                     'certified_stops': count, 'coalesced_stops': candidates - count, 'input_tokens': tokens}


def export(args, output):
    import numpy as np
    import torch
    from tokenizers import Tokenizer

    from research.core_features.extract import run
    from research.gradio_app.causal_plaba import QWEN_FILES
    from research.gradio_app.checkpoint import load_checkpoint
    from research.gradio_app.plaba import BASE_REVISION
    from research.semantic_tracking.qasper_factorial_fit import (
        load_token_states, paragraph_logits, question_features, question_only_features, token_design,
    )
    from research.semantic_tracking.qasper_fit import assemble, load_states, pack, projection
    from research.stage2.records import write_jsonl

    articles, queries = material(args.articles, args.queries, args.articles_sha256, args.queries_sha256)
    if (args.model.name != BASE_REVISION or any(digest(args.model / name) != sha for name, sha in QWEN_FILES.items())
            or digest(args.heads) != HEADS_SHA256):
        raise ValueError('Expected exactly the fixed Qwen backbone and QASPER readout weights')
    checkpoint = load_checkpoint(args.checkpoint)
    tokenizer = Tokenizer.from_file(str(args.model / 'tokenizer.json'))
    stimuli, inventory = prepare_stimuli(articles, queries, tokenizer, checkpoint.segmenter())
    documents = {row['id']: row for row in stimuli}
    article_ids = {row['id'] for row in articles}
    matrix = projection()
    stimuli_path = output / 'stimuli.jsonl'
    write_jsonl(stimuli_path, stimuli)
    extraction = output / 'extraction'
    run(stimuli_path, extraction, segment_means=True, max_batch_tokens=6144,
        projected_tokens=(article_ids, matrix))
    dense = extraction / 'token-features.npz'
    features = extraction / 'features.npz'
    states, means = load_states(features, documents)
    paper_tokens = load_token_states(dense, features, documents, matrix, article_ids)
    eval_rows = [{'paper_id': row['article_id'], 'question_id': row['id']} for row in queries]
    question_states = question_features(documents, means, matrix, eval_rows)
    geometry, body = assemble(eval_rows, documents, states, means, matrix)
    device = torch.device('cuda')
    with np.load(args.heads, allow_pickle=False) as heads, torch.inference_mode():
        if not np.array_equal(heads['projection'], matrix):
            raise ValueError('Frozen QASPER projection changed')
        center, scale = heads['position_mean'], heads['position_scale']
        body_design = torch.as_tensor(pack(geometry, body, center, scale), device=device)
        question_design = torch.as_tensor(question_only_features(geometry, eval_rows, question_states, center, scale),
                                         device=device)
        tokens = token_design(eval_rows, paper_tokens, question_states, body_design.shape[1], device, training=False)
        designs = (body_design[..., :4], question_design, body_design, body_design)
        forecasts = {}
        for name, design in zip(METHODS, designs, strict=True):
            weights = torch.as_tensor(heads[name], device=device)
            hazards = torch.sigmoid(paragraph_logits(weights, design, tokens if name == 'token_rank' else None).double())
            forecasts[name] = hazards.cpu().numpy()
    if any(not np.all(np.isfinite(values) & (0 < values) & (values < 1)) for values in forecasts.values()):
        raise ValueError('ExpMRC interval hazards must be finite probabilities')
    predictions = output / 'predictions.jsonl'
    write_jsonl(predictions, (
        {'id': row['id'], 'article_id': row['article_id'], 'stops': documents[row['article_id']]['stops'],
         'methods': {name: values[index, :len(documents[row['article_id']]['stops'])].tolist()
                     for name, values in forecasts.items()}}
        for index, row in enumerate(queries)))
    for name in ('extraction.json', 'causality.json', 'telemetry.jsonl'):
        shutil.copyfile(extraction / name, output / name)
    seal = {**inventory, 'predictions_sha256': digest(predictions), 'articles_sha256': args.articles_sha256,
            'queries_sha256': args.queries_sha256, 'assay_sha256': args.assay_sha256,
            'protocol_sha256': args.protocol_sha256, 'heads_sha256': HEADS_SHA256, 'candidate_id': checkpoint.identifier,
            'model_revision': BASE_REVISION, 'source_code_sha256': digest(Path(__file__)),
            'extraction_sha256': digest(output / 'extraction.json')}
    (output / 'seal.json').write_text(json.dumps(seal, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('articles', 'queries', 'model', 'checkpoint', 'heads', 'output'):
        parser.add_argument(f'--{flag}', type=Path, required=True)
    for flag in ('articles', 'queries', 'assay', 'protocol'):
        parser.add_argument(f'--{flag}-sha256', required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent, prefix='.expmrc-forecast-') as temporary:
        output = Path(temporary) / 'return'
        output.mkdir()
        export(args, output)
        output.rename(args.output)


if __name__ == '__main__':
    main()
