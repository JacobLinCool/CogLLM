"""Frozen causal token features with native full-attention relationships."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import inspect
from math import isfinite, log
from pathlib import Path
from threading import RLock

import torch

from research.backbone_transfer.extract import RELATIVE_RMS_LIMIT, load_language_model


MAX_TOKENS = 2048
RELATION_ABSOLUTE_LIMIT = 1e-3
_ATTENTION_LOCK = RLock()


@dataclass(frozen=True)
class Features:
    token_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    hidden: torch.Tensor
    mid: torch.Tensor
    relations: torch.Tensor | None


class _GlobalRelationships:
    """Aggregate the SDPA weights from its actual Q, K, mask, and scale."""

    def __init__(self, original, layers, size, device):
        self.original = original
        self.layers = frozenset(layers)
        self.seen = set()
        self.checked_masks = []
        self.positions = torch.arange(size, device=device)
        self.total = torch.zeros((size, size), device=device, dtype=torch.float32)
        self.peak = torch.zeros_like(self.total)
        self.head_count = 0

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        if module in self.layers:
            if module in self.seen:
                raise RuntimeError(f'Duplicate global attention capture at layer {module.layer_idx}')
            self.capture(module, query, key, value, attention_mask, kwargs)
            self.seen.add(module)
        return self.original(module, query, key, value, attention_mask, **kwargs)

    def capture(self, module, query, key, value, attention_mask, kwargs):
        size = len(self.positions)
        scale = kwargs.get('scaling')
        is_causal = kwargs.get('is_causal')
        if is_causal is None:
            is_causal = module.is_causal
        if (query.ndim != 4 or key.ndim != 4 or query.shape[0] != 1 or key.shape[0] != 1
                or query.shape[-2:] != key.shape[-2:] or query.shape[-2] != size
                or not query.shape[1] or not key.shape[1] or query.shape[1] % key.shape[1]
                or getattr(module, 'num_key_value_groups', 1) != query.shape[1] // key.shape[1]
                or query.dtype != torch.float32 or key.dtype != torch.float32 or value.dtype != torch.float32
                or type(scale) not in (int, float) or not isfinite(scale) or scale <= 0
                or kwargs.get('dropout', 0) != 0 or not is_causal
                or kwargs.get('position_bias') is not None or kwargs.get('sliding_window') is not None):
            raise RuntimeError('Global relation capture requires unpadded causal FP32 native attention')
        self.check_mask(attention_mask)

        heads = query.shape[1]
        key_heads = key.shape[1]
        grouped_queries = query[0].unflatten(0, (key_heads, heads // key_heads))
        transposed_keys = key[0].transpose(-2, -1).unsqueeze(1)
        for start in range(0, size, 32):
            stop = min(start + 32, size)
            positions = self.positions[start:stop, None]
            scores = (grouped_queries[:, :, start:stop] @ transposed_keys[:, :, :, :stop]) * scale
            visible = self.positions[None, :stop] <= positions
            weights = scores.masked_fill(~visible, -torch.inf).softmax(dim=-1).flatten(0, 1)
            self.total[start:stop, :stop].add_(weights.sum(dim=0))
            current_peak = self.peak[start:stop, :stop]
            torch.maximum(current_peak, weights.amax(dim=0), out=current_peak)
        self.head_count += heads

    def check_mask(self, mask):
        if mask is None or any(mask is checked for checked in self.checked_masks):
            return
        size = len(self.positions)
        if mask.shape != (1, 1, size, size):
            raise RuntimeError('Expected a single square mask for global attention')
        visible = self.positions[:, None] >= self.positions[None, :]
        values = mask[0, 0]
        if values.dtype == torch.bool:
            correct = torch.equal(values, visible)
        elif values.is_floating_point():
            correct = bool(torch.where(visible, values == 0, values <= -1e8).all())
        else:
            correct = False
        if not correct:
            raise RuntimeError('Native global attention does not match causal prefix visibility')
        self.checked_masks.append(mask)

    def result(self):
        if self.seen != self.layers or not self.head_count:
            raise RuntimeError(f'Captured {len(self.seen)} of {len(self.layers)} native global attention layers')
        uniform_reference = self.positions[:, None] + 1
        mean = self.total / self.head_count
        relations = (torch.stack((mean, self.peak), dim=-1) * uniform_reference.unsqueeze(-1)).log1p() - log(2)
        return relations.masked_fill((self.positions[:, None] < self.positions[None, :]).unsqueeze(-1), 0)


@contextmanager
def _capture_global(module, layers, size, device):
    # Override the owning model module, leaving Transformers' global SDPA registry untouched.
    with _ATTENTION_LOCK:
        original = module.ALL_ATTENTION_FUNCTIONS
        registry = type(original)()
        capture = _GlobalRelationships(original['sdpa'], layers, size, device)
        registry['sdpa'] = capture
        module.ALL_ATTENTION_FUNCTIONS = registry
        try:
            yield capture
        finally:
            module.ALL_ATTENTION_FUNCTIONS = original


class Backbone:
    """Single-document causal forward for the pinned text backbones."""

    def __init__(self, candidate, encoder, tokenizer, spec, *, native_attention=True):
        self.candidate = candidate
        self.revision = spec['revision']
        self.width = spec['hidden_width']
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.device = next(encoder.parameters()).device
        if encoder.training or not len(encoder.layers):
            raise ValueError('Expected a frozen backbone in evaluation mode with identifiable layers')
        self.mid_index = (len(encoder.layers) - 1) // 2
        self.global_layers = ()
        self.attention_module = None
        if not native_attention:
            return
        if encoder.config._attn_implementation != 'sdpa':
            raise ValueError('Native relations require the validated SDPA attention backend')
        self.global_layers = tuple(
            layer.self_attn for layer, layer_type in zip(encoder.layers, encoder.config.layer_types, strict=True)
            if layer_type == 'full_attention'
        )
        modules = {inspect.getmodule(type(layer)) for layer in self.global_layers}
        if len(modules) != 1:
            raise ValueError('Expected a single owning attention module and visible native global attention layers')
        self.attention_module = modules.pop()
        if not hasattr(self.attention_module, 'ALL_ATTENTION_FUNCTIONS'):
            raise ValueError('The pinned architecture does not expose its native attention interface')

    @classmethod
    def load(cls, candidate, model, *, native_attention=True, serving_device=None):
        from tokenizers import Tokenizer
        import transformers

        model = Path(model)
        encoder, spec = load_language_model(model, candidate, torch, transformers, serving_device=serving_device)
        tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
        return cls(candidate, encoder, tokenizer, spec, native_attention=native_attention)

    def encode(self, text):
        if type(text) is not str or not text.strip():
            raise ValueError('Provide nonempty original text')
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        if any(not 0 <= left <= right <= len(text) for left, right in encoded.offsets):
            raise ValueError('Tokenizer offsets fall outside the original Unicode text')
        return self.encode_tokens(tuple(encoded.ids), tuple(encoded.offsets))

    def encode_tokens(self, token_ids, offsets):
        size = len(token_ids)
        if not 1 <= size <= MAX_TOKENS or len(offsets) != size:
            raise ValueError(f'Expected one unpadded document with 1–{MAX_TOKENS} tokens and exact offsets')
        inputs = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        middle = []

        def collect(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or middle:
                raise RuntimeError('Expected exactly one intermediate token tensor')
            middle.append(output[0])

        capture = (_capture_global(self.attention_module, self.global_layers, size, self.device)
                   if self.attention_module is not None else nullcontext())
        with capture as native, self.encoder.layers[self.mid_index].register_forward_hook(collect):
            with torch.inference_mode():
                hidden = self.encoder(input_ids=inputs, use_cache=False, output_attentions=False,
                                      output_hidden_states=False).last_hidden_state[0]
                relations = native.result() if native is not None else None
            if len(middle) != 1:
                raise RuntimeError('Missing intermediate token tensor')
            hidden, mid = hidden.detach().clone(), middle[0].detach().clone()
            if relations is not None:
                relations = relations.detach().clone()
        if (hidden.shape != (size, self.width) or mid.shape != hidden.shape
                or (relations is not None and relations.shape != (size, size, 2))
                or any(value.dtype != torch.float32 or not bool(value.isfinite().all())
                       for value in (hidden, mid, relations) if value is not None)):
            raise RuntimeError('Malformed or nonfinite frozen backbone features')
        return Features(token_ids, offsets, hidden, mid, relations)

    def validate(self, text):
        """Verify suffix invariance at every frozen feature including native relationships."""
        original = self.encode(text)
        size = len(original.token_ids)
        if size < 4:
            raise ValueError('Validation requires at least four tokens')
        cut = size // 2
        short = self.encode_tokens(original.token_ids[:cut], original.offsets[:cut])
        # Swap in a distinct, verified tokenizer ID while holding indices fixed.
        alternatives = set(original.token_ids)
        if len(alternatives) < 2:
            raise ValueError('Validation requires distinct token IDs')
        first, second = sorted(alternatives)[:2]
        changed_ids = tuple(second if token == first else first for token in original.token_ids[cut:])
        changed = self.encode_tokens(original.token_ids[:cut] + changed_ids, original.offsets)

        def prefix(value):
            features = {'hidden': value.hidden[:cut], 'mid': value.mid[:cut]}
            if value.relations is not None:
                features['relations'] = value.relations[:cut, :cut]
            return features

        expected = prefix(original)
        variants = {'truncation': prefix(short), 'suffix_replacement': prefix(changed)}
        absolute = {
            label: {name: float((tensor - expected[name]).abs().max()) for name, tensor in actual.items()}
            for label, actual in variants.items()
        }
        relative = {}
        for label, actual in variants.items():
            relative[label] = {}
            for name in ('hidden', 'mid'):
                difference = (actual[name] - expected[name]).square().mean(dim=-1).sqrt()
                scale = expected[name].square().mean(dim=-1).sqrt().clamp_min(1e-8)
                relative[label][name] = float((difference / scale).max())
        if (any(error > RELATIVE_RMS_LIMIT for check in relative.values() for error in check.values())
                or any(check.get('relations', 0.0) > RELATION_ABSOLUTE_LIMIT for check in absolute.values())):
            raise RuntimeError(f'{self.candidate} causal-prefix feature check failed: {relative}, {absolute}')
        return {'tokens': size, 'common_prefix': cut, 'global_layers': len(self.global_layers),
                'mid_layer': self.mid_index, 'max_absolute_difference': absolute, 'max_relative_rms': relative,
                'hidden_relative_tolerance': RELATIVE_RMS_LIMIT, 'relation_absolute_tolerance': RELATION_ABSOLUTE_LIMIT}
