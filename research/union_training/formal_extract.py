"""Extract independent grade-free assessment packages with a single frozen L4 load."""

import argparse
import os
from pathlib import Path

from research.backbone_transfer import extract as projection
from research.backbone_transfer.extract import checked_digest, digest, extract_loaded, load_frozen_backbone, read_json, write_json


def extract(candidate, model, stimuli, output, protocol_sha256):
    if candidate != 'qwen' or stimuli.name != 'stimuli.jsonl':
        raise ValueError('Expected the registered frozen Qwen and formal bundle sentinel')
    os.environ['HF_HUB_OFFLINE'] = '1'
    source = stimuli.parent / 'formal'
    registered = read_json(source / 'inventory.json')
    if not registered or any(Path(name).name != name for name in registered):
        raise ValueError('Invalid grade-free assessment inventory')
    for name, entry in registered.items():
        if digest(source / name / 'stimuli.jsonl') != entry['stimuli_sha256']:
            raise ValueError(f'Assessment inputs changed for {name}')
    frozen = load_frozen_backbone(candidate, model)
    output.mkdir(exist_ok=False)
    results = {}
    for name in registered:
        child = output / name
        default_limit = projection.MAX_INPUT_TOKENS
        observation_limit = 4096 if name == 'anglin' else default_limit
        if observation_limit > default_limit:
            config = read_json(model / 'config.json')
            text_config = config.get('text_config', config)
            if text_config['max_position_embeddings'] < observation_limit:
                raise ValueError('Anglin observation exceeds the pinned context capacity')
        try:
            projection.MAX_INPUT_TOKENS = observation_limit
            extract_loaded(frozen, source / name / 'stimuli.jsonl', child, registered[name]['protocol_sha256'],
                           max_batch_documents=48, max_batch_tokens=6144)
        finally:
            projection.MAX_INPUT_TOKENS = default_limit
        seal = read_json(child / 'seal.json')
        if (seal['stimuli_sha256'] != registered[name]['stimuli_sha256']
                or seal['protocol_sha256'] != registered[name]['protocol_sha256']
                or seal['content_forward_per_document'] != 1):
            raise ValueError(f'Invalid extracted assessment for {name}')
        results[name] = {'seal_sha256': digest(child / 'seal.json'), 'documents': seal['documents'],
                         'positions': seal['certified_stops'], 'input_token_limit': observation_limit}
    write_json(output / 'seal.json', {'protocol_sha256': protocol_sha256, 'candidate': candidate,
                                    'grade_free_extraction': True, 'assessments': results})
    write_json(output / 'causality.json', {name: digest(output / name / 'causality.json') for name in results})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    for name in ('model', 'stimuli', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--protocol-sha256', type=checked_digest, required=True)
    extract(**vars(parser.parse_args()))
