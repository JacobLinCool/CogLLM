"""Gradio factory for fitted portable CogLLM checkpoints."""

from math import isfinite
from pathlib import Path

import torch

from research.backbone_transfer.extract import digest
from research.token_readout.backbone import Backbone
from research.token_readout.readout import PortableReadout
from research.token_readout.runner import TokenRunner
from research.token_readout.transfer_fit import CHECKPOINT_SCHEMA, DEFAULT_SCALE


def load(*, model: Path, checkpoint: Path, english_head=None, transition_head=None, narrative_head=None,
         terminology_head=None, serving_device=None):
    identity = digest(checkpoint)
    packet = torch.load(checkpoint, map_location='cpu', weights_only=True)
    fields = {'schema', 'candidate', 'revision', 'variant', 'relations', 'reader', 'priors', 'scale'}
    if not isinstance(packet, dict) or packet.keys() != fields or packet['schema'] != CHECKPOINT_SCHEMA:
        raise ValueError('Expected a portable readout checkpoint with an identified backbone and variant')
    if (packet['candidate'] not in ('qwen', 'lfm3', 'gemma4') or packet['variant'] not in ('point', 'span')
            or type(packet['relations']) is not bool):
        raise ValueError('Checkpoint identifies an unsupported text backbone or readout')
    scale = packet['scale']
    priors = packet['priors']
    if (not isinstance(scale, dict) or scale.keys() != DEFAULT_SCALE.keys()
            or any(type(value) not in (int, float) or not isfinite(value) for value in scale.values())
            or min(scale['opening_gain'], scale['resolution_gain']) <= 0
            or not isinstance(priors, dict) or any(type(priors.get(key)) not in (int, float) or not isfinite(priors[key])
                                                  for key in ('opening', 'resolution'))):
        raise ValueError('Checkpoint calibration must provide finite priors and positive gains')
    if (english_head is None) != (transition_head is None):
        raise ValueError('Writing review requires both the wording and coherence readouts.')
    if english_head is not None:
        from research.reading_validation.transition_inference import CHECKPOINT
        if identity != CHECKPOINT:
            raise ValueError('The writing readouts require the validated Qwen token checkpoint.')
    runtime = Backbone.load(packet['candidate'], model, native_attention=packet['relations'], serving_device=serving_device)
    if runtime.revision != packet['revision']:
        raise ValueError('Readout weights belong to another model revision')
    if english_head is not None:
        from research.reading_validation.local_inference import REVISION
        if packet['candidate'] != 'qwen' or runtime.revision != REVISION:
            raise ValueError('English writing review requires the Qwen revision used to train the head.')
    reader = PortableReadout(runtime.width, packet['variant'], packet['relations']).to(runtime.device)
    reader.load_state_dict(packet['reader'], strict=True)
    reader.eval().requires_grad_(False)
    return TokenRunner(runtime, reader, None, priors, identity, scale=scale, anchor_width=reader.anchor_tokens,
                       english_head=english_head, transition_head=transition_head, narrative_head=narrative_head,
                       terminology_head=terminology_head)
