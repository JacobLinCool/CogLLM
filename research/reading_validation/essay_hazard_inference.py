"""Apply the fixed eight-topic essay readout to certified, already-encoded sentence ends."""
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from research.reading_validation.transition_inference import CHECKPOINT, REVISION

WEIGHTS = Path(__file__).resolve().parents[1] / "results/reading-validation-transition-external/essay-cross-topic/serving.json"
WEIGHT_SHA = "77a0634d12f400d1987d425a56f89c27e92d48025210f38c8e9cf7d0034cff13"


@dataclass(frozen=True)
class EssayHazardInference:
    baseline: np.ndarray
    residual: np.ndarray
    reason_prior: np.ndarray
    reliability: np.ndarray

    @classmethod
    def load(cls, path=WEIGHTS):
        raw = Path(path).read_bytes()
        if sha256(raw).hexdigest() != WEIGHT_SHA:
            raise ValueError("Essay observer weights differ from the source validation")
        data = json.loads(raw)
        if data["revision"] != REVISION or data["checkpoint_sha256"] != CHECKPOINT:
            raise ValueError("Essay observer feature identity changed")
        arrays = [np.stack([m[k] for m in data["models"]]) for k in ("baseline", "residual", "reason_prior", "reliability")]
        if [a.shape for a in arrays] != [(8, 299), (8, 299, 7), (8, 7), (8,)]:
            raise ValueError("Essay observer feature dimensions changed")
        return cls(*arrays)

    def probabilities(self, matrix, *, per_model=False):
        """Return hazard and lexical risks for each observation, optionally in each source fold."""
        x = np.asarray(matrix, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 298 or not np.isfinite(x).all():
            raise ValueError("Expected a finite matrix with the original 298 boundary features")
        x = np.column_stack((np.ones(len(x)), x))
        logit = x @ self.baseline.T
        lexical = np.exp(-np.logaddexp(0., -logit))
        log_base = np.where(logit < -35., logit, np.log(np.logaddexp(0., np.maximum(logit, -36.))))
        residual = np.einsum("nf,mfr->nmr", x, self.residual, optimize=True)
        reason_component = np.logaddexp.reduce(np.log(self.reason_prior) + residual, axis=-1)
        with np.errstate(divide="ignore"):
            log_scale = np.logaddexp(np.log1p(-self.reliability), np.log(self.reliability) + reason_component)
        risk = -np.expm1(-np.exp(np.minimum(log_base + log_scale, np.log(745.))))
        return (risk, lexical) if per_model else (risk.mean(axis=1), lexical.mean(axis=1))

    def score(self, text, features, runtime, reader, checkpoint, sentence_spans, token_surprisal):
        """Read a whole original article once upstream, then score eligible noninitial sentence stops.

        Spans are ordered Unicode half-open string intervals including the first sentence. Surprisal
        is one CPU NumPy token NLL per original token (initial zero). No encoder call is performed.
        """
        import torch
        from research.reading_validation.transition_external_forecast import pooled_boundary_features, trace_rows

        if runtime.candidate != "qwen" or runtime.revision != REVISION or features.relations is None:
            raise ValueError("Essay observer requires the selected Qwen 2B native relationship features")
        tok = runtime.tokenizer
        encoding = tok.encode(text, add_special_tokens=False)
        ids = tuple(features.token_ids)
        if tuple(encoding.ids) != ids or tuple(map(tuple, encoding.offsets)) != tuple(features.offsets):
            raise ValueError("Essay observer received states from a different article")
        previous = 0
        stops = []
        for pos, (start, end) in enumerate(sentence_spans, 1):
            if not (previous <= start < end <= len(text)) or text[start:end] != text[start:end].strip():
                raise ValueError("Expected ordered original Unicode sentence spans without surrounding whitespace")
            previous = end
            if pos == 1 or (start and not text[start-1].isspace()):
                continue
            prefix = tuple(tok.encode(text[:end], add_special_tokens=False).ids)
            if not prefix or prefix != ids[:len(prefix)] or features.offsets[len(prefix)-1][1] != end:
                continue
            stops.append(dict(id=pos, index=pos, start=start, end=end, archive_final=False))
        if not stops:
            return []
        with torch.inference_mode():
            final, middle = reader.project(features)
            scores = reader.score_projected(final, middle, features.relations)
            scale = checkpoint["scale"]
            scores = replace(scores, opening=scores.opening * scale["opening_gain"],
                             resolution=scores.resolution * scale["resolution_gain"] + scale["resolution_bias"])
            traces = trace_rows(text, features.offsets, scores, checkpoint["priors"], reader.anchor_tokens)
            rows, _ = pooled_boundary_features(runtime, features, middle, final, traces,
                                                {"text": text, "stops": stops}, token_surprisal)
        matrix = np.stack([np.r_[np.log1p(row["position"]), row["features"], row["representation"]] for row in rows])
        hazards, lexical = self.probabilities(matrix)
        return [{"position": s["index"], "start": s["start"], "end": s["end"],
                 "essay_hazard": float(h), "essay_lexical": float(l)}
                for s, h, l in zip(stops, hazards, lexical, strict=True)]
