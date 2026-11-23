"""Verify the packed preflight and actual extraction token counts independently."""

import argparse
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from research.gradio_app import cuda_verification_batch_worker as worker
from research.semantic_tracking.expmrc_forecast import material, prepare_stimuli
from research.stage2.records import write_jsonl


STAGE = 'nq-evidence'


def audit_inputs(assay, snapshot, checkpoint_path, plan, *, runtime_budget):
    os.environ['USE_TORCH'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    from tokenizers import Tokenizer, __version__ as rust_version
    from transformers import AutoTokenizer, __version__ as transformers_version

    from research.gradio_app.checkpoint import BACKBONE, load_checkpoint

    if worker.digest(assay / 'assay.json') != plan['assay_sha256']:
        raise ValueError('Tokenizer verification must use the exact packed assay')
    if (snapshot.name != BACKBONE['revision'] or transformers_version != BACKBONE['transformers']
            or any(worker.digest(snapshot / name) != BACKBONE['snapshot_sha256'][name]
                   for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json'))):
        raise ValueError('Expected the pinned tokenizer files and Transformers version')
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint.identifier != plan['candidate_id']:
        raise ValueError('Expected bundled sentence boundaries for the exact checkpoint')
    articles, queries = material(assay / 'articles.jsonl', assay / 'queries.jsonl',
                                 plan['articles_sha256'], plan['queries_sha256'])
    rust = Tokenizer.from_file(str(snapshot / 'tokenizer.json'))
    runtime = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    stimuli, inventory = prepare_stimuli(articles, queries, rust, checkpoint.segmenter())
    if any(value != plan[key] for key, value in inventory.items()):
        raise ValueError('The original Rust preflight cannot be reproduced')
    with TemporaryDirectory() as directory:
        rebuilt = Path(directory) / 'stimuli.jsonl'
        write_jsonl(rebuilt, stimuli)
        stimuli_sha256 = worker.digest(rebuilt)

    counts = []
    for row in stimuli:
        text = row['text']
        rust_ids = rust.encode(text, add_special_tokens=False).ids
        encoded = runtime(text, add_special_tokens=False, return_offsets_mapping=True).encodings[0]
        positions = {end: index for index, (start, end) in enumerate(encoded.offsets) if start < end}
        for stop in row['stops']:
            independently_encoded = runtime.encode(text[:stop], add_special_tokens=False)
            if stop not in positions or independently_encoded != encoded.ids[:positions[stop] + 1]:
                raise ValueError(f'The runtime tokenizer cannot certify the sealed stop at {row["id"]}:{stop}')
        counts.append({'id': row['id'], 'rust': len(rust_ids), 'extraction': len(encoded.ids),
                       'token_ids_differ': rust_ids != encoded.ids, 'stops': len(row['stops'])})
    paragraphs = {row['id']: row['stops'] for row in stimuli}
    expected = {row['id']: (row['article_id'], paragraphs[row['article_id']]) for row in queries}
    actual_tokens = sum(row['extraction'] for row in counts)
    if ('torch' in sys.modules or not runtime.is_fast or not 0 < actual_tokens <= runtime_budget
            or max(row['extraction'] for row in counts) > 6144 or sum(row['rust'] for row in counts) != inventory['input_tokens']):
        raise ValueError('Tokenizer audit must use the pinned fast tokenizer without loading Torch or exceeding the budget')
    return {
        'reconstructed_stimuli_sha256': stimuli_sha256, 'rust_preflight_tokens': inventory['input_tokens'],
        'runtime_extraction_tokens': actual_tokens, 'transformers': transformers_version, 'tokenizers': rust_version,
        'transformers_tokenizer': type(runtime).__name__,
        'pretokenizer_configs_equal': str(rust.pre_tokenizer) == str(runtime.backend_tokenizer.pre_tokenizer),
        'certified_stops': inventory['certified_stops'], 'inputs': counts,
    }, expected


def verify(job, assay, snapshot, checkpoint_path, output):
    from research.gradio_app import cuda_verification_batch as host

    metadata_path = host.manifest_path(job)
    metadata = worker.read_json(metadata_path)
    host.verify(job, metadata)
    stage = metadata['stages'][STAGE]
    if (metadata['schema_version'] != 18 or not metadata['release_verified']
            or not metadata['stages']['setup']['verified'] or stage['state'] != 'complete'):
        raise ValueError('Expected a completed extraction from a released Natural Questions L4 instance')
    stage_dir = job / 'outputs' / STAGE
    stage_archive = job / f'{STAGE}.tar.gz'
    if (stage['members'].keys() != worker.OUTPUTS[STAGE]
            or stage['archive_bytes'] != stage_archive.stat().st_size
            or stage['archive_sha256'] != worker.digest(stage_archive)
            or any(worker.digest(stage_dir / name) != sha for name, sha in stage['members'].items())):
        raise ValueError('Retrieved evidence predictions differ from their sealed L4 output')
    worker.unpack(stage_archive, stage['members'])
    bundle = worker.read_json(job / 'bundle.json')
    plan = bundle['nq_evidence']
    inventory, expected = audit_inputs(assay, snapshot, checkpoint_path, plan, runtime_budget=120_000)
    extraction = worker.read_json(stage_dir / 'extraction.json')
    if inventory['reconstructed_stimuli_sha256'] != extraction['input_sha256']:
        raise ValueError('The stage did not extract states from the independently reconstructed stimuli')
    with (stage_dir / 'predictions.jsonl').open(encoding='utf-8') as stream:
        predictions = list(map(json.loads, stream))
    if ({row['id']: (row['article_id'], row['stops']) for row in predictions} != expected
            or len(predictions) != len(expected)):
        raise ValueError('The returned forecasts differ from their exact article and stop inventory')
    host.verify_evidence_forecast(stage_dir, bundle, plan, inventory['runtime_extraction_tokens'])
    result = {
        **inventory,
        'run_id': metadata['run_id'], 'original_transport_phase': metadata['phase'],
        'original_transport_error': stage.get('error'), 'stage_verified_with_independent_runtime_token_inventory': True,
        'metadata_sha256': worker.digest(metadata_path), 'protocol_sha256': plan['protocol_sha256'],
        'assay_sha256': plan['assay_sha256'], 'predictions_sha256': stage['members']['predictions.jsonl'],
        'stage_archive_sha256': stage['archive_sha256'],
        'release_verified': metadata['release_verified'], 'code_sha256': worker.digest(Path(__file__)),
    }
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False, sort_keys=True)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('job', 'assay', 'snapshot', 'checkpoint-path', 'output'):
        parser.add_argument(f'--{option}', required=True, type=Path)
    verify(**vars(parser.parse_args()))
