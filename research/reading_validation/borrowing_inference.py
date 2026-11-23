"""Serve fitted observation-specific reading distributions from one existing Qwen text encoding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from research.backbone_transfer.extract import digest, read_json
from research.reading_validation.borrowing_fit import Private
from research.union_training.fit_observations import COUNT_MODELS, Readouts

RESULTS = Path(__file__).resolve().parents[1] / "results"
DEFAULT = RESULTS / "reading-validation-family-bootstrap" / "global"
SCHEMA = DEFAULT / "serving-schema.json"
DEFAULT_PROJECTION = (
    RESULTS / "portable-token-readout-l4" / "linked" / "span-calibrated.pt"
)


@dataclass(frozen=True)
class CategoryForecast:
    """One probability per registered category. Rows correspond to submitted exposures."""

    endpoint: str
    labels: tuple
    probabilities: np.ndarray


@dataclass(frozen=True)
class GaussianForecast:
    """Mean vectors and covariance matrices in the source response units, including known error."""

    endpoint: str
    mean: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class Endpoint:
    """Measurement shape required by the original training implementation."""

    name: str
    source: str
    likelihood: str
    states: tuple[str, ...]
    coordinates: Tensor
    output_width: int
    center: float
    spread: float
    labels: tuple
    support: str


class AdaptiveObservationReader:
    """Source-trained measurement heads. The caller supplies its existing causal text features."""

    def __init__(
        self,
        seed: int = 1729,
        *,
        directory: Path = DEFAULT,
        projection: Path = DEFAULT_PROJECTION,
    ):
        self.directory = directory
        self.seed = seed
        schema = read_json(directory / SCHEMA.name)
        self.revision = schema["qwen_revision"]
        if seed not in schema["seeds"]:
            raise ValueError(
                "Choose one fitted measurement seed from the published schema"
            )
        if digest(projection) != schema["projection_checkpoint_sha256"]:
            raise ValueError(
                "The 48-dimensional feature projection differs from the one used for training"
            )
        raw = torch.load(projection, map_location="cpu", weights_only=True)
        if raw["revision"] != self.revision:
            raise ValueError(
                "Qwen and its experimental feature projection have different revisions"
            )
        self.projection = raw["reader"]["final.weight"].detach().float()
        self.inputs = self.projection.shape[0]
        if tuple(self.projection.shape) != (48, 2048):
            raise ValueError("Unexpected archived projection")
        self.endpoints = {}
        for k, v in schema["endpoints"].items():
            self.endpoints[k] = Endpoint(
                k,
                v["source"],
                v["likelihood"],
                tuple(v["states"]),
                torch.empty((0, v["max_positions"]), dtype=torch.long),
                v["output_width"],
                v["center"],
                v["spread"],
                tuple(v["labels"]),
                v["coordinates"],
            )
        # Construction initializes the original layer shapes. Preserve the caller's CPU RNG state.
        with torch.random.fork_rng(devices=[]):
            self.common = Readouts(self.inputs, list(self.endpoints.values())).eval()
            self.private = Private(self.common, list(self.endpoints.values())).eval()
        fit_dir = directory / f"seed-{seed}"
        receipt = read_json(fit_dir / "fit.json")
        for name, model in (("shared", self.common), ("private", self.private)):
            key = name + "-parameters.npz"
            file = fit_dir / key
            expected = schema["weights"][str(seed)]
            if (
                digest(file) != expected[name]
                or receipt["files"][key] != expected[name]
            ):
                raise ValueError(
                    "Observation-head parameters differ from their training receipt"
                )
            with np.load(file, allow_pickle=False) as blobs:
                model.load_state_dict(
                    {k: torch.from_numpy(blobs[k]) for k in blobs}, strict=True
                )
        frozen = directory / "forecast-freeze.json"
        if digest(frozen) != schema["calibration_sha256"]:
            raise ValueError("Borrowing weights differ from the held experiment")
        self.coefficients = {
            k: float(v["family_bootstrap"])
            for k, v in read_json(frozen)["seeds"][str(seed)].items()
        }

    @torch.inference_mode()
    def project_hidden(self, hidden: Tensor, *, revision: str) -> Tensor:
        """Project existing final hidden vectors. No encoding or decoding is invoked."""
        if (
            revision != self.revision
            or hidden.ndim < 2
            or hidden.shape[-1] != 2048
            or not hidden.is_floating_point()
        ):
            raise ValueError(
                "Expected final hidden states from the specified frozen Qwen3.5-2B revision"
            )
        h = hidden.detach().float()
        if not h.isfinite().all():
            raise ValueError("Final states must be finite")
        return F.linear(F.layer_norm(h, (h.shape[-1],)), self.projection.to(h.device))

    @torch.inference_mode()
    def from_article(
        self,
        endpoint: str,
        features,
        token_indices: Sequence[int],
        *,
        revision: str,
        standard_error: float | None = None,
    ) -> CategoryForecast | GaussianForecast:
        """Index one article at its actual measured token stops, without another model forward."""
        if self.endpoints[endpoint].support == "pair":
            raise ValueError(
                "Paired presentations require an explicit projected row at their registered exposures"
            )
        if not token_indices or any(
            type(i) is not int or not 0 <= i < len(features.token_ids)
            for i in token_indices
        ):
            raise ValueError(
                "Supply zero-based positions in the already encoded unmodified article"
            )
        if features.hidden.shape[0] != len(features.token_ids):
            raise ValueError("Hidden vectors and token sequence differ")
        hidden = features.hidden[list(token_indices)]
        rows = self.project_hidden(hidden, revision=revision)
        error = (
            None if standard_error is None else np.array([standard_error], dtype=float)
        )
        return self.predict_projected(endpoint, rows[None], standard_errors=error)

    @torch.inference_mode()
    def predict_projected(
        self,
        endpoint: str,
        projected: Tensor,
        *,
        available: Tensor | None = None,
        standard_errors: np.ndarray | None = None,
    ) -> CategoryForecast | GaussianForecast:
        """Consume [batch, sites, 48] archived seed projections at the endpoint's measured coordinates."""
        t = self.endpoints[endpoint]
        if (
            projected.ndim != 3
            or projected.shape[0] == 0
            or projected.shape[-1] != self.inputs
            or not projected.is_floating_point()
        ):
            raise ValueError(
                "Expected a nonempty batch of rows, measured coordinates, and 48 projected features"
            )
        nx, k, _ = projected.shape
        allowed = t.coordinates.shape[1]
        if (t.support == "trajectory" and not 1 <= k <= allowed) or (
            t.support != "trajectory" and k != allowed
        ):
            raise ValueError(
                f"{endpoint} requires the registered measurement geometry {t.support} ({allowed} sites)"
            )
        h = projected.detach().to(device="cpu", dtype=torch.float32)
        if not h.isfinite().all():
            raise ValueError("Projected states must be finite")
        h = F.normalize(h, dim=-1) * self.inputs**0.5
        if available is None:
            valid = torch.ones((nx, k), dtype=torch.bool)
        else:
            valid = available.detach().cpu()
            if (
                valid.dtype != torch.bool
                or tuple(valid.shape) != (nx, k)
                or not valid[:, 0].all()
            ):
                raise ValueError(
                    "Each row needs a nonempty Boolean mask over submitted measurement sites"
                )
        if t.support != "trajectory" and not valid.all():
            raise ValueError(
                "Every registered coordinate must be exposed for this measurement"
            )
        if k > 1 and torch.any(valid[:, 1:] & ~valid[:, :-1]):
            raise ValueError("Trajectory masks must form contiguous observed prefixes")
        if t.likelihood == "known_error_gaussian":
            if standard_errors is None:
                raise ValueError(
                    "This endpoint requires its expected measurement standard error in original response units"
                )
            se = np.asarray(standard_errors, dtype=float)
            if se.shape != (nx,) or not np.isfinite(se).all() or (se < 0).any():
                raise ValueError(
                    "Supply one finite nonnegative measurement standard error per exposure"
                )
            se = se / t.spread
        else:
            if standard_errors is not None:
                raise ValueError(
                    "Only known-error observations accept a sampling standard error"
                )
            se = 0.0
        common = self.common.measure(t, h, valid)
        private_state = torch.cat(
            [self.private.sources["source_" + t.source][s](h).tanh() for s in t.states],
            dim=-1,
        )
        personal = self.private.measurements[endpoint](private_state, valid)
        a = self.coefficients[endpoint]
        x, y = personal.double().numpy(), common.double().numpy()
        if t.likelihood in COUNT_MODELS:
            lx = np.isfinite(x)
            if not np.array_equal(lx, np.isfinite(y)):
                raise ValueError("Experts disagree on supported outcomes")
            delta = np.where(lx, y, 0.0) - np.where(lx, x, 0.0)
            logits = torch.from_numpy(x + a * delta)
            prob = logits.softmax(dim=1).numpy()
            labels = (
                t.labels
                if t.support != "trajectory"
                else (
                    *tuple(f"first_report_at_site_{i + 1}" for i in range(k)),
                    "no_report",
                )
            )
            if len(labels) != prob.shape[1]:
                raise ValueError("Output and registered labels differ")
            return CategoryForecast(endpoint, tuple(labels), prob)
        if t.likelihood == "paired_gaussian":

            def precision_and_mean(v):
                sd = v[:, 2:4]
                cross = sd[:, 0] * sd[:, 1] * v[:, 4]
                cov = np.column_stack(
                    (sd[:, 0] ** 2, cross, cross, sd[:, 1] ** 2)
                ).reshape(-1, 2, 2)
                p = np.linalg.inv(cov)
                return p, (p @ v[:, :2, None])

            p0, e0 = precision_and_mean(x)
            p1, e1 = precision_and_mean(y)
            cov = np.linalg.inv((1 - a) * p0 + a * p1)
            mu = (cov @ ((1 - a) * e0 + a * e1))[:, :, 0]
        else:
            p0 = 1 / (x[:, 1] ** 2 + se**2)
            p1 = 1 / (y[:, 1] ** 2 + se**2)
            p = (1 - a) * p0 + a * p1
            mu = (((1 - a) * p0 * x[:, 0] + a * p1 * y[:, 0]) / p)[:, None]
            cov = (1 / p)[:, None, None]
        return GaussianForecast(endpoint, mu * t.spread + t.center, cov * t.spread**2)
