"""Run the verified PLABA-EVAL sentence head on pasted English text."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from nltk.tokenize import PunktTokenizer


BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
HEAD_PATH = Path(__file__).resolve().parents[1] / "results" / "plaba-eval-head.npz"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
MAX_CHARACTERS = 16_000
MAX_PROMPT_TOKENS = 8_192


@dataclass(frozen=True)
class SentenceRating:
    start: int
    end: int
    ease: float
    understanding: float


@dataclass(frozen=True)
class ArticleRatings:
    text: str
    sentences: tuple[SentenceRating, ...]


def validate_article_text(text: str) -> None:
    if not text.strip():
        raise ValueError("請貼上文章正文。")
    if len(text) > MAX_CHARACTERS:
        raise ValueError(f"每篇最多 {MAX_CHARACTERS:,} 個字元。請選擇一個完整段落再分析。")


def sentence_spans(text: str, segmenter: PunktTokenizer) -> list[tuple[int, int]]:
    validate_article_text(text)
    spans = []
    previous_end = 0
    for start, end in segmenter.span_tokenize(text):
        sentence = text[start:end]
        start += len(sentence) - len(sentence.lstrip())
        end -= len(sentence) - len(sentence.rstrip())
        if start >= end:
            continue
        if start < previous_end or text[previous_end:start].strip():
            raise RuntimeError("Sentence spans do not cover the source text in order")
        spans.append((start, end))
        previous_end = end
    if not spans or text[previous_end:].strip():
        raise RuntimeError("Sentence spans do not cover the source text in order")
    return spans


class PlabaReader:
    def __init__(self, model_dir: Path):
        if model_dir.name != BASE_REVISION or not (model_dir / "config.json").is_file():
            raise ValueError(f"--model must point to the local Qwen snapshot {BASE_REVISION}")
        if sha256((model_dir / "tokenizer.json").read_bytes()).hexdigest() != TOKENIZER_SHA256:
            raise ValueError("The local Qwen tokenizer does not match the verified snapshot")

        # Keep MLX loading behind the model boundary so text and UI checks can run independently.
        import mlx.core as mx
        from mlx_lm import load

        from research.train_plaba_probe import PROMPT, TARGETS, predict

        with np.load(HEAD_PATH, allow_pickle=False) as artifact:
            if tuple(artifact["targets"]) != TARGETS:
                raise ValueError("Unexpected PLABA readout targets")
            weights = artifact["weights"]
            intercept = artifact["intercept"]
            if weights.shape != (2_048, 2) or intercept.shape != (2,):
                raise ValueError("Unexpected PLABA readout shape")
            self.head = weights, intercept

        self.segmenter = PunktTokenizer("english")
        model, self.tokenizer = load(str(model_dir))
        self.encoder = model.language_model.model
        self.mx = mx
        self.prompt = PROMPT
        self.predict = predict

    def analyze(self, text: str) -> ArticleRatings:
        spans = sentence_spans(text, self.segmenter)
        sentences = [text[start:end] for start, end in spans]
        numbered = list(enumerate(sentences, start=1))
        article = "\n".join(f"[{index}] {sentence}" for index, sentence in numbered)
        tokens = self.tokenizer.encode(f"{self.prompt}\n\n{article}\n", add_special_tokens=False)
        positions = []
        for index, sentence in numbered:
            focus = f"\nFocal sentence [{index}]: {sentence}\nAssessment:"
            tokens.extend(self.tokenizer.encode(focus, add_special_tokens=False))
            positions.append(len(tokens) - 1)
        if len(tokens) > MAX_PROMPT_TOKENS:
            raise ValueError(f"文章及其逐句輸入共超過 {MAX_PROMPT_TOKENS:,} 個模型 token。請縮短文章。")

        mx = self.mx
        hidden = self.encoder(mx.array([tokens]))[0, mx.array(positions), :]
        features = np.array(hidden.astype(mx.float32), dtype=np.float64, copy=True)
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms) & (norms > 0)):
            raise RuntimeError("Invalid hidden-state norms for PLABA readout")
        scores = self.predict(features / norms, self.head)
        return ArticleRatings(
            text,
            tuple(
                SentenceRating(start, end, float(ease), float(understanding))
                for (start, end), (ease, understanding) in zip(spans, scores, strict=True)
            ),
        )
