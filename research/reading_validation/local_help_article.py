"""Build the frozen Guo help features for named spans after one existing Qwen forward."""

import json
from collections import defaultdict
from math import log
from pathlib import Path

import numpy as np

from research.reading_validation.definition_data import word_geometry
from research.reading_validation.definition_features import Page
from research.reading_validation.definition_fit import all_spans
from research.reading_validation.definition_predict import feature_series
from research.reading_validation.local_features import sentence_features
from research.reading_validation.local_forecast import WORD, digest
from research.reading_validation.local_inference import LocalInference
from research.reading_validation.local_jargon import restore_predictor
from research.reading_validation.local_jargon_prepare import sentence_bounds
from research.stage5.complex_tokenization import token_span

RESULTS = Path(__file__).resolve().parents[1] / "results"


class ArticleHelpFeatures:
    def __init__(self):
        self.local = LocalInference.load()
        base = RESULTS / "reading-validation-local"
        rank = base / "localization/deployment-weights.json"
        lock = json.loads((base / "jargon/prediction-lock.json").read_bytes())
        if digest(rank) != lock["probe_weights_sha256"]:
            raise ValueError("Local phrase or lexical-form coefficients changed")
        self.phrase = json.loads(rank.read_bytes())
        deft = RESULTS / "reading-validation-definitions"
        if (
            digest(deft / "weights.json")
            != json.loads((deft / "forecast-seal.json").read_bytes())["weights_sha256"]
        ):
            raise ValueError("DEFT evidence model changed")
        self.deft = json.loads((deft / "weights.json").read_bytes())

    def from_components(self, text, queries, local_arrays, page):
        """Map each stable target ID to (its literal term, inclusive start, exclusive end)."""
        if not page.n:
            return {}, {target: "no eligible word spans" for target in queries}
        l, r, logit = all_spans(page, self.deft["term"]["semantic"], "semantic")
        deft_positions = {
            (int(a), int(b)): i for i, (a, b) in enumerate(zip(l, r, strict=True))
        }
        by_sentence = defaultdict(list)
        excluded = {}
        sentences = sentence_bounds(text)
        for target, (name, a, b) in queries.items():
            if (
                not isinstance(name, str)
                or not 0 <= a < b <= len(text)
                or name.casefold() != text[a:b].casefold()
            ):
                raise ValueError(
                    "Candidate term must match its displayed source character interval"
                )
            span = page.word_span((a, b))
            if span not in deft_positions:
                excluded[target] = (
                    "not an exact eligible single-clause DEFT span of one to eight words"
                )
                continue
            containing = [
                i for i, (lo, hi) in enumerate(sentences) if lo <= a < b <= hi
            ]
            if len(containing) != 1:
                excluded[target] = "crosses an editorial sentence boundary"
                continue
            by_sentence[containing[0]].append((target, name, a, b, span))
        values = {}
        for i, targets in sorted(by_sentence.items()):
            a, b = sentences[i]
            node = sentence_features(
                str(i),
                "article",
                text,
                a,
                b,
                self.local.lexicon,
                local_arrays,
                i,
                0.0,
                sentence_endpoint=False,
            )
            positions = {tuple(z): j for j, z in enumerate(node.coords)}
            surface = restore_predictor(self.phrase["form"], node)
            phrase = restore_predictor(self.phrase["local"], node)
            for target, name, lo, hi, span in targets:
                j = positions.get((lo, hi))
                if j is None:
                    excluded[target] = "outside the local wording grid"
                    continue
                peaks = {}
                for model in ("surface", "semantic"):
                    _, prob = feature_series(
                        page, span, self.deft["link"][model], model
                    )
                    peaks[model] = float(max(prob[span[1] + 1 :], default=0))
                values[target] = np.array(
                    [
                        node.raw["external_lexical"][j],
                        surface[j],
                        np.log1p(len(WORD.findall(name))),
                        np.log1p(len(name)),
                        lo / len(text),
                        peaks["surface"],
                        phrase[j],
                        logit[deft_positions[span]],
                        peaks["semantic"],
                    ],
                    dtype=float,
                )
        return values, excluded

    def from_native(self, text, queries, features, tokenizer):
        """Consume a native Features object with attention; perform no backbone calls or decoding."""
        import torch
        from torch.nn.functional import layer_norm, normalize

        from research.reading_validation.identity_causal_features import project_basis

        ids = tokenizer.encode(text, add_special_tokens=False)
        if (
            tuple(ids.ids) != features.token_ids
            or tuple(map(tuple, ids.offsets)) != features.offsets
        ):
            raise ValueError("Text and causal features refer to different input")
        if (
            features.relations is None
            or features.hidden.shape != (len(ids.ids), 2048)
            or len(ids.ids) > 2048
        ):
            raise ValueError(
                "Expected at most 2048 Qwen3.5-2B tokens with full native attention"
            )
        words = []
        for m in WORD.finditer(text):
            _, hit = token_span(text, features.offsets, *m.span())
            valid = (
                hit is not None
                and tokenizer.encode(text[: m.end()], add_special_tokens=False).ids
                == ids.ids[: hit[1] + 1]
            )
            words.append((*m.span(), hit[1] if valid else -1))
        device = features.hidden.device
        normalized = normalize(features.hidden.double(), dim=1)
        w = torch.as_tensor(self.local.lexical_weight, device=device)
        projection = torch.as_tensor(self.local.projection, device=device)
        arrays = {
            "words": np.asarray(words, dtype=int),
            "offsets": np.asarray(features.offsets),
            "projection": (normalized.float() @ projection).cpu().numpy(),
            "lexical": (normalized @ w + self.local.lexical_bias)
            .clamp(0, 1)
            .float()
            .cpu()
            .numpy(),
        }
        geometry, _ = word_geometry(text, tokenizer)
        if not geometry:
            return {}, {target: "no eligible word spans" for target in queries}
        basis = project_basis(2048, device)
        states = torch.cat(
            [
                layer_norm(hidden, (2048,)) @ p[:, :40].contiguous()
                for hidden, p in zip(
                    (features.mid, features.hidden), basis, strict=True
                )
            ],
            dim=1,
        )
        ends = torch.tensor([t[3] for t in geometry], device=device)
        visible = torch.arange(len(ids.ids), device=device)[None, :] <= ends[:, None]
        relations = torch.where(
            visible[:, :, None], features.relations[ends] + log(2), 0
        )
        packed = torch.round(32 * relations)
        if packed.min() < 0 or packed.max() > 255:
            raise ValueError("Unexpected native attention quantizer range")
        page = Page(
            text, geometry, states.half().cpu().numpy(), packed.byte().cpu().numpy()
        )
        return self.from_components(text, queries, arrays, page)
