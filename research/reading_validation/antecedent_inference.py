"""Optional, source-calibrated links from English personal pronouns to earlier words."""

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from research.reading_validation.narrative_inference import certified_words
from research.reading_validation.transition_pointer_features import CONTROL, FORMS, design, sentence_ids


BUNDLE = Path(__file__).resolve().parents[1] / "results/reading-validation-transition-pointer/serving"
BUNDLE_SHA = "5dd01ad44e15a62b79ae7c4a70ddb95d0e32407cfa596fbc922d96e28bea5adf"
POS_SHA = "6025f530624335c67d6547d44757b357b4e79bae030a0383e9887a92c1718f0b"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"


def load_tagger(path):
    """Load the same public English POS weights that supplied the source control features."""
    from nltk.tag import PerceptronTagger

    if sha256(path.read_bytes()).hexdigest() != POS_SHA:
        raise ValueError("The original English POS resource has changed")
    folder = "averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng."
    with ZipFile(path) as archive:
        weights, word_tags, classes = (json.loads(archive.read(folder + field + ".json"))
                                       for field in ("weights", "tagdict", "classes"))
    tagger = PerceptronTagger(load=False)
    tagger.model.weights = weights
    tagger.tagdict = word_tags
    tagger.classes = tagger.model.classes = set(classes)
    return tagger


def material(text, runtime):
    encoding = runtime.tokenizer.encode(text, add_special_tokens=False)
    ids, offsets = tuple(encoding.ids), tuple(map(tuple, encoding.offsets))
    if not 1 <= len(ids) <= 2048:
        raise ValueError("The complete passage must be within the validated 2,048-token limit")
    words = certified_words(text, runtime.tokenizer, ids, offsets, [(0, len(text))])
    forms = [text[a:b] for a, b, *_ in words]
    queries = [i for i, w in enumerate(forms) if i and w.casefold() in FORMS]
    return ids, offsets, words, forms, queries


def native_signals(features, cap, query, starts, ends):
    """Use the same subgrouped native-attention masses and read states as the source model."""
    import torch
    from torch.nn import functional as F

    if not cap.seen or any(cap.seen[i].shape[1:] != (len(query), len(starts)) for i in cap.seen):
        raise ValueError("Original attention rows have not been captured")
    heads = torch.cat([cap.seen[i] for i in sorted(cap.seen)], 0)
    if heads.shape[0] != 48:
        raise ValueError("Expected all 48 native full-attention heads of the pinned Qwen revision")
    stages = [F.normalize(v[ends]) for v in (features.mid, features.hidden)]
    cosine = [(x[query] @ x.T).clamp(-1, 1) for x in stages]
    global_view = torch.stack([*cosine, torch.log1p(heads.mean(0)), torch.log1p(heads.amax(0))], -1)
    return torch.cat((global_view, torch.log1p(heads).permute(1, 2, 0)), -1).cpu().numpy()


@dataclass(frozen=True)
class AntecedentInference:
    """A word-location model calibrated on the original choices of English reading-game players."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    threshold: float
    tagger: object

    @classmethod
    def load(cls, directory=BUNDLE):
        raw = (Path(directory) / "word-pointer.json").read_bytes()
        if sha256(raw).hexdigest() != BUNDLE_SHA:
            raise ValueError("Unexpected English antecedent readout")
        packet = json.loads(raw)
        if (packet["schema"] != "cogllm-english-pronoun-link-v1" or packet["revision"] != REVISION
                or packet["native_attention_heads"] != 48 or packet["pos_sha256"] != POS_SHA
                or packet["threshold"] != .8):
            raise ValueError("Antecedent architecture or source operating point differs")
        vectors = [np.asarray(packet["weights"][k], dtype=float) for k in ("mean", "sd", "beta")]
        if any(v.shape != (len(CONTROL) + 52,) or not np.isfinite(v).all() for v in vectors) or (vectors[1] <= 0).any():
            raise ValueError("Invalid native antecedent weights")
        return cls(*vectors, packet["threshold"], load_tagger(Path(directory) / "averaged_perceptron_tagger_eng.zip"))

    @contextmanager
    def capture(self, text, runtime):
        """Wrap the caller's existing single forward; the context never starts an encoder itself."""
        import torch
        from research.reading_validation.transition_pointer_l4 import watch

        if runtime.candidate != "qwen" or runtime.revision != REVISION or len(runtime.global_layers) != 6:
            raise ValueError("English antecedent links require the validated native Qwen3.5-2B runtime")
        ids, offsets, words, forms, queries = material(text, runtime)
        bounds = [[a, b] for a, b, *_ in words]
        row = {"text": text, "words": bounds, "sentences": sentence_ids(text, bounds)}
        controls = []
        for q in queries:
            # The tagger sees only the current pronoun and preceding words.
            tags = [p for _, p in self.tagger.tag(forms[:q + 1])]
            controls.append(design({**row, "pos": tags, "pos_last": [tags[q - 1]]}, q, 0))
        saved = PointerPass(self, ids, offsets, words, queries, controls, None, None, None)
        if queries:
            locations = torch.tensor([w[2:4] for w in words], device=runtime.device, dtype=torch.long)
            starts, ends = locations[:, 0], locations[:, 1]
            query = torch.tensor(queries, dtype=torch.long, device=runtime.device)
            with watch(runtime, len(ids), ends[query], starts, ends, False) as cap:
                saved.cap, saved.starts, saved.ends = cap, starts, ends
                yield saved
        else:
            yield saved


@dataclass
class PointerPass:
    model: AntecedentInference
    token_ids: tuple
    offsets: tuple
    words: list
    queries: list
    controls: list
    cap: object
    starts: object
    ends: object

    def logits(self, features):
        """Emit a vector over every prior word at every eligible pronoun, without another model call."""
        import torch

        if tuple(features.token_ids) != self.token_ids or tuple(features.offsets) != self.offsets:
            raise ValueError("Antecedent head received another article's features")
        if not self.queries:
            return {}
        with torch.inference_mode():
            query = torch.tensor(self.queries, dtype=torch.long, device=self.ends.device)
            blocks = native_signals(features, self.cap, query, self.starts, self.ends)
        result = {}
        for i, q in enumerate(self.queries):
            X = np.column_stack([self.controls[i], blocks[i, :q]]).astype(np.float64)
            result[q] = (X - self.model.mean) / self.model.scale @ self.model.weights
        return result

    def links(self, features):
        """Return only source-calibrated links, with half-open Unicode character offsets."""
        links = []
        for q, values in self.logits(features).items():
            first = int(values.argmax())
            confidence = float(1 / np.exp(values - values[first]).sum())
            if confidence >= self.model.threshold:
                a, b = self.words[q][:2]
                x, y = self.words[first][:2]
                if not 0 <= x < y <= a < b:
                    raise ValueError("Antecedent is not an earlier visible source word")
                links.append({"pronoun_start": a, "pronoun_end": b, "antecedent_start": x, "antecedent_end": y,
                              "model_confidence": confidence})
        return links
