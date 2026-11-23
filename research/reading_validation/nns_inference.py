"""Estimate directly reported English word difficulty from an existing Qwen forward."""

import gzip
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from wordfreq import zipf_frequency

from research.reading_validation.local_forecast import WORD, digest
from research.reading_validation.local_inference import REVISION, LocalInference
from research.reading_validation.nns_features import single_word_rank, stem
from research.reading_validation.nns_fit import all_features
from research.stage5.complex_tokenization import token_span

DEFAULT = Path(__file__).resolve().parents[1] / "results/reading-validation-nns"
SCHEMA = "nns-self-report-en-v1"


@dataclass(eq=False)
class WordReport:
    model: LocalInference
    weights: dict
    profile: dict
    levels: list
    word_rates: dict
    prevalence: float
    extra: np.ndarray
    checksum: str
    _form: object = field(init=False, repr=False)

    def __post_init__(self):
        self._form = lru_cache(maxsize=12000)(self.word_form)

    @classmethod
    def load(cls, directory=DEFAULT, *, native=None):
        directory = Path(directory)
        manifest = directory / "serving.json"
        card = json.loads(manifest.read_bytes())
        if (
            card["schema"] != SCHEMA
            or card["revision"] != REVISION
            or any(version(k) != v for k, v in card["dependencies"].items())
        ):
            raise ValueError(
                "Unrecognized word-study checkpoint or English lexical resource version"
            )
        for k, v in (
            ("weights.npz", "weights_sha256"),
            ("vocabulary.json.gz", "vocabulary_sha256"),
        ):
            if digest(directory / k) != card[v]:
                raise ValueError("The word-study checkpoint or reference changed")
        with gzip.open(directory / "vocabulary.json.gz", "rt") as f:
            profiles = json.load(f)
        profile = {
            axis: profiles["cefrj"][axis] | profiles["octanove"][axis]
            for axis in ("surface", "stem")
        }
        native = LocalInference.load() if native is None else native
        if native.bundle_sha256 != card["local_source_sha256"]:
            raise ValueError("Incorrect PLABA source readout")
        with np.load(directory / "weights.npz", allow_pickle=False) as z:
            weights = {k: z[k] for k in z.files}
        if len(set(weights["stem"])) != len(weights["stem"]):
            raise ValueError("Duplicate source stem prior")
        rates = {
            str(k): (float(y), float(n))
            for k, y, n in zip(
                weights["stem"],
                weights["stem_flags"],
                weights["stem_answers"],
                strict=True,
            )
        }
        freq = float(weights["stem_flags"].sum() / weights["stem_answers"].sum())
        extra = np.linalg.qr(np.random.default_rng(428371).standard_normal((2048, 64)))[
            0
        ].astype(np.float32)
        return cls(
            native,
            weights,
            profile,
            list(card["proficiency"]),
            rates,
            freq,
            extra,
            digest(manifest),
        )

    def word_form(self, word):
        w = word.casefold()
        root = stem(w)
        f = zipf_frequency(w, "en")
        grade = self.profile["surface"].get(w, self.profile["stem"].get(root))
        n = len(w)
        return [
            n,
            np.log1p(n),
            f,
            max(0, 3 - f),
            max(0, f - 5),
            len(re.findall(r"[aeiouy]+", w)),
            grade if grade is not None else 3.5,
            float(grade is None),
            float(w not in self.profile["surface"]),
            float("-" in w),
            float(any(s.isdigit() for s in w)),
            float(word[:1].isupper()),
            float(word.isupper()),
        ]

    def _word_design(self, text, positions, certified, lexical, primary, extra):
        selected = np.flatnonzero(certified)
        if not len(selected):
            return positions[selected], {}, np.empty((0, 2))
        ranked = single_word_rank(
            self.model, text, "article", positions, certified, lexical, primary
        )
        form = []
        source = []
        rates = []
        for j in selected:
            a, b = positions[j]
            w = text[a:b]
            y, n = self.word_rates.get(stem(w.casefold()), (0.0, 0.0))
            form.append([*self._form(w), np.log1p(j)])
            source.append((lexical[j], ranked[a, b]))
            rates.append(
                (
                    logit(
                        np.clip((y + 20 * self.prevalence) / (n + 20), 1e-7, 1 - 1e-7)
                    ),
                    np.log1p(n),
                )
            )
        base = {
            "form": np.asarray(form),
            "source": np.asarray(source),
            "context": np.column_stack(
                [primary[selected], extra[selected].astype(np.float16)]
            ),
        }
        return positions[selected], base, np.asarray(rates)

    def _probabilities(self, design, proficiency):
        positions, base, rates = design
        if proficiency is not None and proficiency not in self.levels:
            raise ValueError("Expected English proficiency A1, A2, B1, B2, C1 or C2")
        if not len(positions):
            return np.empty(0)
        aware = proficiency is not None
        prefix = "cefr" if aware else "text"
        level = self.levels.index(proficiency) if aware else 0
        pairs = np.column_stack(
            [np.arange(len(positions)), np.full(len(positions), level)]
        )
        prob = []
        for arm in ("lexical", "contextual"):
            name = prefix + "." + arm
            x = all_features(base, pairs, rates, arm, aware)
            t = self.weights[name + ".theta"]
            prob.append(
                expit(
                    (x - self.weights[name + ".mean"])
                    / self.weights[name + ".scale"]
                    @ t[:-1]
                    + t[-1]
                )
            )
        h = float(self.weights[prefix + ".mixture"])
        return (1 - h) * prob[0] + h * prob[1]

    def _packed(self, text, positions, certified, lexical, primary, extra, proficiency):
        design = self._word_design(text, positions, certified, lexical, primary, extra)
        return self._rows(design[0], self._probabilities(design, proficiency))

    @staticmethod
    def _rows(positions, probabilities):
        return [
            {"start": int(a), "end": int(b), "reported_unknown_probability": float(p)}
            for (a, b), p in zip(positions, probabilities, strict=True)
        ]

    def _native_words(self, text, features, tokenizer, reader):
        if reader.input_width != 2048 or features.hidden.shape != (
            len(features.offsets),
            2048,
        ):
            raise ValueError(
                "Expected native Qwen3.5-2B features for one English sentence"
            )
        enc = tokenizer.encode(text, add_special_tokens=False)
        if (
            tuple(enc.ids) != tuple(features.token_ids)
            or tuple(map(tuple, enc.offsets)) != tuple(features.offsets)
            or not 0 < len(enc.ids) <= 1024
        ):
            raise ValueError(
                "The provided sentence and native Qwen features must agree"
            )
        positions = np.asarray([m.span() for m in WORD.finditer(text)], int).reshape(
            -1, 2
        )
        indices = np.full(len(positions), -1, int)
        for i, (a, b) in enumerate(positions):
            _, t = token_span(text, enc.offsets, a, b)
            if (
                t
                and tokenizer.encode(text[:b], add_special_tokens=False).ids
                == enc.ids[: t[1] + 1]
            ):
                indices[i] = t[1]
        ok = indices >= 0
        positions_ok = np.flatnonzero(ok)
        p = np.zeros((len(ok), 32), np.float32)
        x = np.zeros((len(ok), 64), np.float32)
        v = np.zeros(len(ok), np.float32)
        for start in range(0, len(positions_ok), 96):
            idx = positions_ok[start : start + 96]
            vals = (
                features.hidden[indices[idx].tolist()]
                .detach()
                .cpu()
                .numpy()
                .astype(float)
            )
            norms = np.linalg.norm(vals, axis=1, keepdims=True)
            if np.any(norms <= 0) or not np.isfinite(norms).all():
                raise ValueError("Invalid Qwen word state")
            vals /= norms
            v[idx] = np.clip(
                vals @ self.model.lexical_weight + self.model.lexical_bias, 0, 1
            )
            vals = vals.astype(np.float32)
            p[idx] = vals @ self.model.projection
            x[idx] = vals @ self.extra
        return len(positions), self._word_design(text, positions, ok, v, p, x)

    def score(self, text, features, tokenizer, reader, *, proficiency=None):
        """Estimate the report probability at certified word ends of one English sentence."""
        count, design = self._native_words(text, features, tokenizer, reader)
        return {
            "schema": SCHEMA,
            "checkpoint_sha256": self.checksum,
            "proficiency": proficiency,
            "word_count": count,
            "certified_words": len(design[0]),
            "words": self._rows(design[0], self._probabilities(design, proficiency)),
        }

    def score_levels(self, text, features, tokenizer, reader):
        """One shared word extraction, seven small population readouts and original offsets."""
        count, design = self._native_words(text, features, tokenizer, reader)
        return {
            "schema": SCHEMA,
            "text": text,
            "checkpoint_sha256": self.checksum,
            "word_count": count,
            "positions": design[0].astype(int).tolist(),
            "probabilities": {
                key: self._probabilities(design, level).tolist()
                for key, level in [
                    ("unspecified", None),
                    *((s, s) for s in self.levels),
                ]
            },
        }
