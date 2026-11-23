"""Run the application readouts with the certified L4 Qwen text backbone."""

from pathlib import Path
from threading import Lock

import numpy as np

from research.stage3.run import TRANSFORMERS_VERSION, encode as encode_frozen

from .causal_plaba import CausalEncoding, HIDDEN_WIDTH, MAX_TOKENS, QWEN_FILES, digest
from .plaba import BASE_REVISION, MAX_CHARACTERS
from .tokenizer import ExtractionTokenizer


class CudaCausalQwen:
    """Expose one causal FP32 text-backbone forward for a plain-text input."""

    def __init__(self, model_dir: Path):
        if model_dir.name != BASE_REVISION or not model_dir.is_dir():
            raise ValueError(f'--model must point to the local Qwen snapshot {BASE_REVISION}')

        import torch
        import transformers

        if transformers.__version__ != TRANSFORMERS_VERSION or torch.cuda.device_count() != 1:
            raise RuntimeError('Expected pinned Transformers and one CUDA device')
        gpu = torch.cuda.get_device_properties(0)
        if gpu.name != 'NVIDIA L4' or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError(f'Expected L4 with native BF16, got {gpu.name}')
        for name, expected in QWEN_FILES.items():
            if digest(model_dir / name) != expected:
                raise ValueError(f'本機 Qwen 快照中的 {name} 與驗證版本不同。')
        self.tokenizer = ExtractionTokenizer(model_dir)

        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.allow_tf32 = False
        model, loading = transformers.Qwen3_5ForCausalLM.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa',
            use_safetensors=True, output_loading_info=True,
        )
        if any(loading.values()) or next(model.model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError(f'Pinned BF16 checkpoint did not load exactly: {loading}')
        self.encoder = model.model.eval().requires_grad_(False).to(device='cuda', dtype=torch.float32)
        self._torch = torch
        self._forward_lock = Lock()

    def encode(self, text: str) -> CausalEncoding:
        if not text.strip():
            raise ValueError('請貼上文章正文。')
        if len(text) > MAX_CHARACTERS:
            raise ValueError(f'每篇最多 {MAX_CHARACTERS:,} 個字元。請選擇一個完整段落再分析。')
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        tokens, offsets = tuple(encoded.ids), tuple(encoded.offsets)
        if not tokens or len(tokens) != len(offsets) or any(not 0 <= start <= end <= len(text) for start, end in offsets):
            raise ValueError('這篇正文沒有可對齊的模型 token。')
        if len(tokens) > MAX_TOKENS:
            raise ValueError(f'文章共超過 {MAX_TOKENS:,} 個模型 token。請縮短文章。')

        with self._forward_lock:
            hidden = encode_frozen(self.encoder, [list(tokens)], self.tokenizer.pad_token_id, self._torch)
            if hidden.dtype != self._torch.float32 or hidden.shape != (1, len(tokens), HIDDEN_WIDTH):
                raise RuntimeError('Expected FP32 Qwen hidden states aligned to every source token')
            states = hidden[0].cpu().numpy()
            del hidden
        if not np.isfinite(states).all():
            raise RuntimeError('Nonfinite Qwen hidden states')
        return CausalEncoding(text, tokens, offsets, states)
