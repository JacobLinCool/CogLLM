"""Expose the pinned extraction tokenizer to the application readouts."""

from pathlib import Path

from research.stage3.run import TRANSFORMERS_VERSION

from .causal_plaba import digest


TOKENIZER_CONFIG_SHA256 = '49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c'


class ExtractionTokenizer:
    """Return plain-text tokenization with source offsets and original token IDs."""

    def __init__(self, model_dir: Path):
        from transformers import AutoTokenizer, __version__

        if (__version__ != TRANSFORMERS_VERSION
                or digest(model_dir / 'tokenizer_config.json') != TOKENIZER_CONFIG_SHA256):
            raise ValueError('本機 tokenizer 與 IIRC 權重使用的版本不同。')
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        if not self._tokenizer.is_fast or self._tokenizer.pad_token_id is None:
            raise ValueError('IIRC 讀出需要已設定 padding token 的 Transformers 快速 tokenizer。')
        self.pad_token_id = self._tokenizer.pad_token_id

    def encode(self, text: str, *, add_special_tokens: bool):
        return self._tokenizer(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True).encodings[0]
