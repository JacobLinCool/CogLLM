"""Apply the fixed PLABA sentence-end readout to one causal Qwen forward."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from nltk.tokenize import PunktSentenceTokenizer, PunktTokenizer

from research.stage3.probe import TARGETS
from research.stage3.run import MODEL, sentence_position
from .plaba import ArticleRatings, BASE_REVISION, MAX_CHARACTERS, SentenceRating, TOKENIZER_SHA256, sentence_spans


CHECKPOINT = Path(__file__).resolve().parents[1] / 'results' / 'causal-plaba-checkpoint'
QWEN_FILES = {
    'tokenizer.json': TOKENIZER_SHA256,
    'config.json': 'ed1c1723241f23f7f4e23430759cbd7dcfb4103cbdfe052bfe7626b57c2615b4',
    'model.safetensors.index.json': 'aca8afed9da75b0f050b408d270766fd77627f1af401e240f61c3b47d0db02f9',
    'model.safetensors-00001-of-00001.safetensors':
        'aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1',
}
MAX_TOKENS = 8_192
HIDDEN_WIDTH = 2_048


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


@dataclass(frozen=True)
class CausalEncoding:
    text: str
    tokens: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    states: np.ndarray


class CausalQwen:
    """Expose plain-text token alignment and the FP32 states of a single causal forward."""

    def __init__(self, model_dir: Path):
        if model_dir.name != BASE_REVISION or not model_dir.is_dir():
            raise ValueError(f'--model must point to the local Qwen snapshot {BASE_REVISION}')
        for name, expected in QWEN_FILES.items():
            if digest(model_dir / name) != expected:
                raise ValueError(f'本機 Qwen 快照中的 {name} 與驗證版本不同。')

        import mlx.core as mx
        from mlx_lm import load
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
        original, _ = load(str(model_dir))
        self.encoder = original.language_model.model.eval()
        self.encoder.set_dtype(mx.float32)
        mx.eval(self.encoder.parameters())
        self.mx = mx

    def encode(self, text: str) -> CausalEncoding:
        if not text.strip():
            raise ValueError('請貼上文章正文。')
        if len(text) > MAX_CHARACTERS:
            raise ValueError(f'每篇最多 {MAX_CHARACTERS:,} 個字元。請選擇一個完整段落再分析。')
        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = tuple(encoded.ids)
        offsets = tuple(encoded.offsets)
        if not tokens or len(tokens) != len(offsets) or any(not 0 <= start <= end <= len(text) for start, end in offsets):
            raise ValueError('這篇正文沒有可對齊的模型 token。')
        if len(tokens) > MAX_TOKENS:
            raise ValueError(f'文章共超過 {MAX_TOKENS:,} 個模型 token。請縮短文章。')
        output = self.encoder(self.mx.array([tokens], dtype=self.mx.int32))
        if output.dtype != self.mx.float32 or output.shape != (1, len(tokens), HIDDEN_WIDTH):
            raise RuntimeError('Expected FP32 Qwen hidden states aligned to every source token')
        states = np.array(output[0], dtype=np.float32, copy=True)
        if not np.isfinite(states).all():
            raise RuntimeError('Nonfinite Qwen hidden states')
        return CausalEncoding(text, tokens, offsets, states)


@dataclass(frozen=True)
class CausalArticleRatings(ArticleRatings):
    sentence_numbers: tuple[int, ...]
    unavailable_sentences: tuple[tuple[int, int], ...]
    sentence_tokens: tuple[int, ...]


class CausalPlabaReader:
    def __init__(self, model_dir: Path, *, backbone: CausalQwen | None = None, checkpoint: Path = CHECKPOINT,
                 segmenter: PunktSentenceTokenizer | None = None):
        weights_path = checkpoint / 'head.npz'
        metadata = json.loads((checkpoint / 'manifest.json').read_text(encoding='utf-8'))
        if (metadata['model'] != MODEL or metadata['revision'] != BASE_REVISION
                or metadata['source_feature_compute_dtype'] != 'float32'
                or metadata['source_feature_block'] != 'sentence_end_final_hidden'
                or metadata['deployment_readout'] != 'qwen_plus_length'
                or metadata['head_sha256'] != digest(weights_path)):
            raise ValueError('The PLABA weights do not match the certified causal readout manifest')
        with np.load(weights_path, allow_pickle=False) as head:
            self.weights, self.intercept = head['combined_weights'], head['combined_intercept']
            self.length_mean, self.length_scale = head['length_mean'], head['length_scale']
            if (tuple(head['targets']) != TARGETS or self.weights.shape != (HIDDEN_WIDTH + 2, 2)
                    or self.intercept.shape != (2,) or self.length_mean.shape != (2,) or self.length_scale.shape != (2,)
                    or not all(np.isfinite(v).all() for v in (self.weights, self.intercept, self.length_mean, self.length_scale))
                    or np.any(self.length_scale <= 0)):
                raise ValueError('Unexpected causal PLABA readout parameters')
        self.segmenter = segmenter if segmenter is not None else PunktTokenizer('english')
        self.backbone = backbone if backbone is not None else CausalQwen(model_dir)

    def position(self, encoded: CausalEncoding, start: int, end: int) -> int | None:
        try:
            position = sentence_position(encoded.text, {'start': start, 'end': end, 'id': f'{start}:{end}'}, encoded.offsets)
        except ValueError:
            return None
        boundary = start + len(encoded.text[start:end].rstrip())
        prefix = self.backbone.tokenizer.encode(encoded.text[:boundary], add_special_tokens=False).ids
        return position if encoded.tokens[:position + 1] == tuple(prefix) else None

    def score(self, hidden: np.ndarray, sentences: list[str]) -> np.ndarray:
        hidden = np.asarray(hidden, dtype=np.float64)
        if hidden.shape != (len(sentences), HIDDEN_WIDTH):
            raise ValueError('Expected one sentence-end state per sentence')
        norms = np.linalg.norm(hidden, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms) & (norms > 0)):
            raise RuntimeError('Invalid hidden-state norms for causal PLABA readout')
        lengths = np.log1p([(len(sentence), len(sentence.split())) for sentence in sentences])
        lengths = (lengths - self.length_mean) / self.length_scale / np.sqrt(2)
        design = np.column_stack((hidden / norms, lengths))
        return np.clip(design @ self.weights + self.intercept, 1, 5)

    def rate(self, encoded: CausalEncoding, spans: list[tuple[int, int]], *,
             allow_empty: bool = False) -> CausalArticleRatings:
        available, unavailable = [], []
        for number, (start, end) in enumerate(spans, 1):
            position = self.position(encoded, start, end)
            if position is None:
                unavailable.append((start, end))
            else:
                available.append((number, start, end, position))
        if not available:
            if allow_empty:
                return CausalArticleRatings(encoded.text, (), (), tuple(unavailable), ())
            raise ValueError('找不到可精確對齊的句末。這篇文章目前無法提供逐句分數。')
        states = encoded.states[[position for _, _, _, position in available]]
        texts = [encoded.text[start:end] for _, start, end, _ in available]
        scores = self.score(states, texts)
        return CausalArticleRatings(
            encoded.text,
            tuple(SentenceRating(start, end, float(ease), float(understanding))
                  for (_, start, end, _), (ease, understanding) in zip(available, scores, strict=True)),
            tuple(number for number, _, _, _ in available),
            tuple(unavailable),
            tuple(position for _, _, _, position in available),
        )

    def analyze(self, text: str) -> CausalArticleRatings:
        spans = sentence_spans(text, self.segmenter)
        return self.rate(self.backbone.encode(text), spans)
