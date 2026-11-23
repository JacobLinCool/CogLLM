"""Capture exact causal Qwen token states and replay its final attention block."""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import torch
from torch import nn

from research.backbone_transfer.extract import load_language_model


ATTENTION_LAYERS = (3, 7, 11, 15, 19, 23)
MID_LAYER = 11
QUERY_HEADS = 8
KEY_HEADS = 2
HEAD_WIDTH = 256
HIDDEN_WIDTH = 2048
VOCAB_SIZE = 248_320
MAX_TOKENS = 2048
_REGISTRY_LOCK = RLock()


@dataclass(frozen=True)
class TokenFeatures:
    token_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    hidden: torch.Tensor
    mid: torch.Tensor
    queries: torch.Tensor
    keys: torch.Tensor
    layer_input: torch.Tensor


def causal_attention_blocks(queries, keys, block_rows=32):
    """Yield (first query index, weights[layer, head, row, key]).

    Grouped query heads share their matching K head. The softmax includes the
    complete visible prefix for each row, with future positions exactly zero.
    Only a block of query rows is materialized at a time.
    """
    layers, heads, length, width = queries.shape
    key_layers, key_heads, key_length, key_width = keys.shape
    if (min(layers, heads, length, width, key_heads, block_rows) < 1 or heads % key_heads
            or (key_layers, key_length, key_width) != (layers, length, width)
            or queries.device != keys.device or queries.dtype != keys.dtype
            or not queries.is_floating_point()):
        raise ValueError("Invalid grouped causal Q/K geometry")
    group_size = heads // key_heads
    positions = torch.arange(length, device=queries.device)
    grouped_keys = keys.float().transpose(-2, -1).unsqueeze(2)
    for start in range(0, length, block_rows):
        stop = min(length, start + block_rows)
        query = queries[:, :, start:stop].float().reshape(layers, key_heads, group_size, stop - start, width)
        scores = (query @ grouped_keys).reshape(layers, heads, stop - start, length) * width**-0.5
        future = positions[None, :] > positions[start:stop, None]
        yield start, scores.masked_fill(future, -torch.inf).softmax(dim=-1)


class LowRankUpdate(nn.Module):
    def __init__(self, inputs, outputs, rank):
        super().__init__()
        self.down = nn.Linear(inputs, rank, bias=False)
        self.up = nn.Linear(rank, outputs, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        # LoRA alpha equals rank so the branch scale is one.
        return self.up(self.down(value))


class AttentionAdapter(nn.Module):
    def __init__(self, rank=4):
        super().__init__()
        if type(rank) is not int or not 0 < rank <= KEY_HEADS * HEAD_WIDTH:
            raise ValueError("Adapter rank must be a positive integer within the K projection width")
        self.query = LowRankUpdate(HIDDEN_WIDTH, QUERY_HEADS * HEAD_WIDTH, rank)
        self.key = LowRankUpdate(HIDDEN_WIDTH, KEY_HEADS * HEAD_WIDTH, rank)

    def query_hook(self, _module, inputs, output):
        # Qwen interleaves Q and gate within each head, not across two global halves.
        per_head = output.unflatten(-1, (QUERY_HEADS, HEAD_WIDTH * 2))
        query, gate = per_head.chunk(2, dim=-1)
        update = self.query(inputs[0]).unflatten(-1, (QUERY_HEADS, HEAD_WIDTH))
        return torch.cat((query + update, gate), dim=-1).flatten(-2)

    def key_hook(self, _module, inputs, output):
        return output + self.key(inputs[0])


class _AttentionCapture:
    def __init__(self, original, expected, length):
        self.original = original
        self.expected = expected
        self.length = length
        self.entries = {}
        self.checked_mask = None

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        index = module.layer_idx
        if index not in self.expected or index in self.entries:
            raise RuntimeError(f"Unexpected Qwen attention capture at layer {index}")
        if (query.shape != (1, QUERY_HEADS, self.length, HEAD_WIDTH)
                or key.shape != (1, KEY_HEADS, self.length, HEAD_WIDTH)
                or value.shape != key.shape or not module.is_causal or query.dtype != torch.float32
                or key.dtype != torch.float32 or value.dtype != torch.float32
                or kwargs.get("dropout", 0) != 0 or kwargs.get("is_causal") is False
                or kwargs.get("position_bias") is not None or kwargs.get("scaling") != HEAD_WIDTH**-0.5):
            raise ValueError("Capture requires the pinned unpadded FP32 causal Qwen attention path")
        if attention_mask is not None and attention_mask is not self.checked_mask:
            indices = torch.arange(self.length, device=query.device)
            causal = indices[:, None] >= indices[None, :]
            if attention_mask.shape != (1, 1, self.length, self.length):
                raise ValueError("Unexpected full-attention mask shape for a single unpadded input")
            mask = attention_mask[0, 0]
            if mask.dtype == torch.bool:
                valid = torch.equal(mask, causal)
            elif mask.is_floating_point():
                valid = bool((mask[causal] == 0).all() and (mask[~causal] <= -1e8).all())
            else:
                valid = False
            if not valid:
                raise ValueError("The original attention mask differs from the complete causal prefix")
            self.checked_mask = attention_mask
        result = self.original(module, query, key, value, attention_mask, **kwargs)
        self.entries[index] = (query[0], key[0])
        return result


@contextmanager
def _capture(expected, length):
    from transformers.models.qwen3_5 import modeling_qwen3_5

    # Transformers resolves the SDPA interface in this module. Serialize only
    # its scoped substitution and restore it even when the model raises.
    with _REGISTRY_LOCK:
        original = modeling_qwen3_5.ALL_ATTENTION_FUNCTIONS
        capture = _AttentionCapture(original["sdpa"], expected, length)
        registry = type(original)()
        registry["sdpa"] = capture
        modeling_qwen3_5.ALL_ATTENTION_FUNCTIONS = registry
        try:
            yield capture.entries
            if tuple(capture.entries) != expected:
                raise RuntimeError(f"Incomplete Qwen attention capture: {tuple(capture.entries)}")
        finally:
            modeling_qwen3_5.ALL_ATTENTION_FUNCTIONS = original


def _return_tokens(tensors):
    return tuple(tensor.detach().clone() for tensor in tensors)


class QwenRuntime:
    def __init__(self, encoder, tokenizer, revision):
        layers = encoder.layers
        attention = tuple(i for i, layer in enumerate(layers) if layer.block_type == "full_attention")
        if (len(layers) != 24 or attention != ATTENTION_LAYERS
                or encoder.config._attn_implementation != "sdpa" or encoder.training):
            raise ValueError("Unexpected Qwen3.5-2B decoder architecture or attention backend")
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.revision = revision
        self.last = layers[-1]
        self.device = next(encoder.parameters()).device

    @classmethod
    def load(cls, snapshot):
        import transformers
        from tokenizers import Tokenizer

        snapshot = Path(snapshot)
        encoder, spec = load_language_model(snapshot, "qwen", torch, transformers)
        tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
        return cls(encoder, tokenizer, spec["revision"])

    def new_adapter(self, rank=4):
        return AttentionAdapter(rank).to(device=self.device, dtype=torch.float32)

    def _install_adapter(self, stack, adapter):
        if adapter is not None:
            attention = self.last.self_attn
            stack.enter_context(attention.q_proj.register_forward_hook(adapter.query_hook))
            stack.enter_context(attention.k_proj.register_forward_hook(adapter.key_hook))

    def encode(self, text, adapter=None):
        if type(text) is not str or not text.strip():
            raise ValueError("Provide nonempty original text")
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if any(not 0 <= begin <= end <= len(text) for begin, end in tokens.offsets):
            raise ValueError("Tokenizer offsets fall outside the original Unicode text")
        return self.encode_tokens(tokens.ids, tokens.offsets, adapter)

    def encode_tokens(self, token_ids, offsets, adapter=None):
        token_ids = tuple(token_ids)
        offsets = tuple(tuple(span) for span in offsets)
        length = len(token_ids)
        if (not 1 <= length <= MAX_TOKENS or len(offsets) != length
                or any(type(token) is not int or not 0 <= token < VOCAB_SIZE for token in token_ids)
                or any(len(span) != 2 or not 0 <= span[0] <= span[1] for span in offsets)):
            raise ValueError(f"Expected one unpadded 1–{MAX_TOKENS} token input with exact offsets")
        input_ids = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        stages = {}

        def save_mid(_module, _inputs, output):
            stages["mid"] = output[0]

        def save_last_input(_module, inputs):
            stages["last_input"] = inputs[0][0]

        with _capture(ATTENTION_LAYERS, length) as recorded, ExitStack() as handles:
            handles.enter_context(self.encoder.layers[MID_LAYER].register_forward_hook(save_mid))
            handles.enter_context(self.last.register_forward_pre_hook(save_last_input))
            self._install_adapter(handles, adapter)
            with torch.inference_mode():
                final = self.encoder(input_ids=input_ids, use_cache=False, output_attentions=False,
                                     output_hidden_states=False).last_hidden_state[0]
            # Materialize ordinary tensors outside inference_mode so the cached
            # block input and frozen features can be used in later autograd.
            hidden, mid, layer_input = _return_tokens((final, stages["mid"], stages["last_input"]))
            queries = torch.stack([recorded[index][0] for index in ATTENTION_LAYERS])
            keys = torch.stack([recorded[index][1] for index in ATTENTION_LAYERS])
        if (any(tensor.shape != (length, HIDDEN_WIDTH) or tensor.dtype != torch.float32
                for tensor in (hidden, mid, layer_input))
                or not all(bool(torch.isfinite(tensor).all()) for tensor in (hidden, mid, layer_input, queries, keys))):
            raise RuntimeError("Nonfinite or malformed all-position Qwen features")
        return TokenFeatures(token_ids, offsets, hidden, mid, queries, keys, layer_input)

    def replay_last(self, layer_input, adapter):
        """Run the original last block and norm with differentiable Q/K updates."""
        if (layer_input.ndim != 2 or layer_input.shape[1] != HIDDEN_WIDTH
                or not 1 <= layer_input.shape[0] <= MAX_TOKENS or layer_input.device != self.device
                or layer_input.dtype != torch.float32 or layer_input.is_inference()):
            raise ValueError("Expected ordinary FP32 cached Qwen layer-24 inputs on the runtime device")
        length = layer_input.shape[0]
        text_positions = torch.arange(length, device=self.device).unsqueeze(0)
        rotary_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
        layer_input = layer_input.unsqueeze(0)
        rotary = self.encoder.rotary_emb(layer_input, rotary_positions)
        with _capture((ATTENTION_LAYERS[-1],), length) as recorded, ExitStack() as handles:
            self._install_adapter(handles, adapter)
            result = self.last(layer_input, position_embeddings=rotary, position_ids=text_positions,
                               attention_mask=None, past_key_values=None, use_cache=False)
            hidden = self.encoder.norm(result)[0]
            query, key = recorded[ATTENTION_LAYERS[-1]]
        return hidden, query, key

    def validate(self, text):
        """Probe no-op replay, exact token coverage and suffix-causal features."""
        original = self.encode(text)
        if len(original.token_ids) < 4:
            raise ValueError("Validation requires at least four input tokens")
        with torch.no_grad():
            replay = self.replay_last(original.layer_input, self.new_adapter())
        prefix = len(original.token_ids) // 2
        shortened = self.encode_tokens(original.token_ids[:prefix], original.offsets[:prefix])
        changed_tokens = original.token_ids[:prefix] + tuple(
            (token + 1) % VOCAB_SIZE for token in original.token_ids[prefix:]
        )
        changed = self.encode_tokens(changed_tokens, original.offsets)

        def relative_rms(left, right):
            difference = (left - right).square().mean().sqrt()
            return float(difference / left.square().mean().sqrt().clamp_min(1e-8))

        def compare(expected, actual):
            return dict(zip(("hidden", "mid", "q", "k", "layer_input"),
                            (relative_rms(left, right) for left, right in zip(expected, actual, strict=True)),
                            strict=True))

        def prefix_features(value):
            return (value.hidden[:prefix], value.mid[:prefix], value.queries[:, :, :prefix],
                    value.keys[:, :, :prefix], value.layer_input[:prefix])

        replay_errors = {
            name: relative_rms(left, right)
            for name, left, right in zip(("hidden", "q", "k"),
                                         (original.hidden, original.queries[-1], original.keys[-1]), replay,
                                         strict=True)
        }
        comparisons = {"no_op_replay": replay_errors,
                       "truncation": compare(prefix_features(original), prefix_features(shortened)),
                       "suffix_replacement": compare(prefix_features(original), prefix_features(changed))}
        if any(not 0 <= value <= 1e-3 for data in comparisons.values() for value in data.values()):
            raise RuntimeError(f"Qwen replay or causal-prefix control failed: {comparisons}")
        return {"tokens": len(original.token_ids), "common_prefix": prefix, "relative_rms": comparisons}
