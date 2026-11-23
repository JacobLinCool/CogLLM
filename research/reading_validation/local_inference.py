"""Local reading difficulty on a single already-computed Qwen feature sequence."""

from collections import Counter
from dataclasses import dataclass
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

from research.reading_validation.local_features import Lexicon, disjoint_ranking, sentence_features
from research.reading_validation.local_forecast import WORD, digest
from research.stage5.complex_tokenization import token_span


DEFAULT = Path(__file__).resolve().parents[1] / "results/reading-validation-local/serving"
BOUNDARY = re.compile(r'(?<=[.!?])(?=\s+[A-Z0-9“"(])|(?<=\n)(?=\s*\S)')
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SCHEMA = "cogllm-local-reading-en-v1"
LEXICON_DIGEST = "9184bd90fefd7f01b5cc9108019bfe5dae7685575d375c25db444121de0a741c"


def english_sentences(text):
    stops = [0] + [m.start() for m in BOUNDARY.finditer(text)] + [len(text)]
    spans = []
    for a, b in zip(stops, stops[1:]):
        segment = text[a:b]
        left = a + len(segment) - len(segment.lstrip())
        right = a + len(segment.rstrip())
        if left < right and WORD.search(text, left, right):
            spans.append((left, right))
    return spans


def _matrix_blocks(parts, centers, scales, values):
    return np.column_stack([(values[k] - centers[k]) / scales[k] for k in parts])


@dataclass(frozen=True)
class LocalInference:
    lexicon: Lexicon
    rank: dict
    rating: dict
    lexical_weight: np.ndarray
    lexical_bias: float
    projection: np.ndarray
    bundle_sha256: str

    @classmethod
    def load(cls, directory=DEFAULT):
        directory = Path(directory)
        packed = directory / "lexicon.json.gz"
        supplied = json.loads((directory / "manifest.json").read_bytes())["files_sha256"]["lexicon.json.gz"]
        if digest(packed) != supplied:
            raise ValueError("Local reading lexicon differs from its manifest")
        return cls.from_lexicon(directory, gzip.decompress(packed.read_bytes()))

    @classmethod
    def from_lexicon(cls, directory, lexical_bytes):
        """Load the same sealed reading model from the uncompressed lexical parameters."""
        directory = Path(directory)
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        if (
            manifest["schema"] != SCHEMA
            or manifest["revision"] != REVISION
            or set(manifest["files_sha256"]) != {"lexicon.json.gz", "weights.npz"}
        ):
            raise ValueError("Unrecognized English local-reading checkpoint")
        if digest(directory / "weights.npz") != manifest["files_sha256"]["weights.npz"]:
            raise ValueError("Local reading weights differ from their manifest")
        if sha256(lexical_bytes).hexdigest() != LEXICON_DIGEST:
            raise ValueError("Incorrect lexical counts for this reading model")
        lex = json.loads(lexical_bytes)
        dictionaries = {k: Counter(v) for k, v in lex.items() if k != "phrases"}
        dictionaries["terms"] = Counter({tuple(words): n for words, n, count in lex["phrases"]})
        dictionaries["term_counts"] = Counter({tuple(words): count for words, n, count in lex["phrases"]})
        with np.load(directory / "weights.npz", allow_pickle=False) as arrays:
            rank = dict(parts=manifest["rank_features"], coef=arrays["rank_coef"])
            rating = dict(
                parts=manifest["rating_features"],
                coef=arrays["rating_coef"],
                offset=arrays["rating_offset"],
            )
            for model, prefix in ((rank, "rank"), (rating, "rating")):
                model["center"] = {name: arrays[f"{prefix}_mean_{name}"] for name in model["parts"]}
                model["scale"] = {name: arrays[f"{prefix}_scale_{name}"] for name in model["parts"]}
            w, b, proj = arrays["lexical_weight"], float(arrays["lexical_bias"]), arrays["projection"]
        if (
            w.shape != (2048,)
            or proj.shape != (2048, 32)
            or len(rank["coef"]) != 56
            or rating["coef"].shape != (2061, 2)
            or rating["offset"].shape != (2, 5)
        ):
            raise ValueError("Unexpected source projection or local reading head dimension")
        return cls(Lexicon(**dictionaries), rank, rating, w, b, proj, digest(manifest_path))

    def _infer_arrays(self, text, arrays, normalized_at, spans):
        output = []
        for i, (start, end) in enumerate(spans):
            if not (0 <= start < end <= len(text)) or (i and start < spans[i - 1][1]):
                raise ValueError("Sentence intervals must be ordered spans in the original text")
            if not WORD.search(text, start, end):
                continue
            word_table = arrays["words"]
            words = (word_table[:, 0] >= start) & (word_table[:, 1] <= end)
            valid = int(np.count_nonzero(word_table[words, 2] >= 0))
            can_rate = valid > 0 and end in normalized_at
            node = sentence_features(
                str(end),
                "article",
                text,
                start,
                end,
                self.lexicon,
                arrays,
                i,
                0.0,
                sentence_endpoint=can_rate,
            )
            rank = (
                _matrix_blocks(self.rank["parts"], self.rank["center"], self.rank["scale"], node.features)
                @ self.rank["coef"]
            )
            proposals = []
            for j in disjoint_ranking(node.coords, rank, 5):
                a, b = node.coords[j]
                proposals.append(
                    dict(start=int(a), end=int(b), rank=len(proposals) + 1, score=float(rank[j]))
                )
            rating = None
            if can_rate:
                values = dict(node.sentence)
                values["surface"] = values["surface"][[0, 1, 2, 4, 5, 6]]
                values["sentence_state"] = normalized_at[end]
                x = np.concatenate(
                    [
                        (values[k] - self.rating["center"][k]) / self.rating["scale"][k]
                        for k in self.rating["parts"]
                    ]
                )
                z = (x @ self.rating["coef"])[:, None] * (np.arange(1, 6) - 3) + self.rating["offset"]
                p = np.exp(z - z.max(axis=1, keepdims=True))
                p /= p.sum(axis=1, keepdims=True)
                rating = {k: float(p[j] @ np.arange(1, 6)) for j, k in enumerate(("ease", "understanding"))}
            output.append(
                dict(
                    start=int(start),
                    end=int(end),
                    phrases=proposals,
                    rating=rating,
                    words=int(words.sum()),
                    certified_words=valid,
                )
            )
        return dict(schema=SCHEMA, language="en", bundle_sha256=self.bundle_sha256, sentences=output)

    def score(self, text, features, tokenizer, reader, *, sentence_spans=None):
        """Return source-offset phrases and 1–5 sentence ratings from provided native features.

        The caller chooses the English endpoint. Custom sentence bounds are Unicode half-open
        intervals. Ranking logits are conditional on the supplied sentence. A missing
        certified sentence stop has a null rating and preserves the available phrase ranks.
        """
        if reader.input_width != 2048 or features.hidden.shape != (len(features.offsets), 2048):
            raise ValueError("Local readout requires one native Qwen3.5-2B feature sequence")
        if not text.strip():
            raise ValueError("Expected the original article text")
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if tuple(encoded.ids) != tuple(features.token_ids) or tuple(map(tuple, encoded.offsets)) != tuple(
            features.offsets
        ):
            raise ValueError("The provided features refer to a different article or tokenizer")
        spans = list(sentence_spans) if sentence_spans is not None else english_sentences(text)
        words = []
        for m in WORD.finditer(text):
            _, hit = token_span(text, features.offsets, *m.span())
            index = -1
            if hit is not None:
                prefix = tokenizer.encode(text[: m.end()], add_special_tokens=False).ids
                if tuple(prefix) == tuple(features.token_ids[: hit[1] + 1]):
                    index = hit[1]
            words.append((*m.span(), index))
        offsets = np.asarray(features.offsets, dtype=int)
        # Project token features in blocks to limit temporary memory on the host.
        n = len(offsets)
        lexical = np.empty(n, dtype=np.float32)
        projected = np.empty((n, 32), dtype=np.float32)
        normalized_at = {}
        certified = {}
        for _, end in spans:
            candidates = np.flatnonzero(offsets[:, 1] == end)
            if len(candidates):
                j = int(candidates[-1])
                if tuple(tokenizer.encode(text[:end], add_special_tokens=False).ids) == tuple(
                    features.token_ids[: j + 1]
                ):
                    certified[j] = end
        for start in range(0, n, 128):
            stop = min(start + 128, n)
            matrix = features.hidden[start:stop].detach().cpu().numpy().astype(np.float64)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or (norms <= 0).any():
                raise ValueError("Invalid native final-layer state")
            matrix /= norms
            lexical[start:stop] = np.clip(matrix @ self.lexical_weight + self.lexical_bias, 0, 1)
            projected[start:stop] = matrix.astype(np.float32) @ self.projection
            for j, end in certified.items():
                if start <= j < stop:
                    normalized_at[end] = matrix[j - start].copy()
        arrays = dict(
            words=np.asarray(words, dtype=int).reshape(-1, 3),
            offsets=offsets,
            lexical=lexical,
            projection=projected,
        )
        return self._infer_arrays(text, arrays, normalized_at, spans)
