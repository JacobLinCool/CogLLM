"""Forecast an expert edit and its source with identical incremental checkpoint history."""
import hashlib
import json
from pathlib import Path

import torch

from research.reading_validation.transition_external_forecast import directed


@torch.inference_mode()
def run(runtime, reader, checkpoint, input_path, output_dir):
    packet = Path(input_path).read_bytes()
    rows = [json.loads(s) for s in packet.splitlines()]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions = {}
    calls = [0]
    def increment(_encoder, _args):
        calls[0] += 1
    with runtime.encoder.register_forward_pre_hook(increment):
        for n,row in enumerate(rows):
            probabilities = directed(runtime,row)
            if row['stops'][-1]['id'] != row['id'] or calls[0] != n+1:
                raise ValueError('Every variant must receive one forward ending in its target sentence')
            predictions[row['id']] = probabilities[row['id']]
    if len(rows)!=len(predictions) or not rows:
        raise ValueError('Duplicated expert-edit keys')
    forecast = out/'matched-logits.json'
    forecast.write_text(json.dumps(predictions, sort_keys=True, allow_nan=False))
    (out/'receipt.json').write_text(json.dumps({'candidate':runtime.candidate, 'revision':runtime.revision,
        'input_sha256':hashlib.sha256(packet).hexdigest(), 'documents':len(rows), 'directed_forward_calls':calls[0],
        'forecast_sha256':hashlib.sha256(forecast.read_bytes()).hexdigest()},indent=2))
