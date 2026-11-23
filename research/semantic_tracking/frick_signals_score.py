"""Score frozen Frick Qwen logits and attention signals using nested story folds."""

import argparse
from collections import Counter
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from research.semantic_tracking import frick_conditional_score as conditional
from research.semantic_tracking import frick_consistency_score as base
from research.semantic_tracking import frick_context_score as contextual
from research.semantic_tracking import frick_pair_score as paired


PROTOCOL = Path(__file__).with_name("frick-signals-protocol.md")
PROTOCOL_SHA256 = "ded1537b94077bef546de999452586dc1e51454680f6f52880e96baf78b1b55f"
SOURCE_SHA256 = "1f0d2ae835429088fe6569366428226082397fb69f74660ac4ddcc0dacded46c"
PREFLIGHT_SHA256 = "b8656522e71e72e6287fddaab421c0758bd661a16b19d523a9b82acb6bb50bb9"
RAW_FEATURES = (
    "surprisal_nats",
    "predictive_entropy_nats",
    "attention_prior_mass_excess",
    "attention_peak_prior_sentence_excess",
)
FEATURES = (
    "sentence_mean_surprisal_nats",
    "sentence_max_surprisal_nats",
    "sentence_mean_predictive_entropy_nats",
    "sentence_mean_attention_prior_mass_excess",
    "sentence_mean_attention_peak_prior_sentence_excess",
)
ARMS = {"both": (0, 1, 2, 3, 4), "logits_only": (0, 1, 2), "attention_only": (3, 4)}
COMPARATORS = ("seven_feature_conditional", "global", "original_contextual")
CAUSAL_TESTS = {
    "hooked_vs_unhooked",
    "prefix_truncation",
    "replacement_suffix",
    "replacement_suffix_positive",
}


def load_signal_seal(signals, source, assay_receipt, preflight_receipt, archived_provenance):
    seal_path = signals / "seal.json"
    seal = base.read_json(seal_path)
    expected = {
        "protocol_sha256": PROTOCOL_SHA256,
        "assay_sha256": archived_provenance["preflight_sha256"],
        "preflight_sha256": PREFLIGHT_SHA256,
        "source_code_sha256": base.digest(Path(__file__).with_name("frick_signals_extract.py")),
        "feature_columns": list(RAW_FEATURES),
        "states_dtype": "float32",
        "attention_layer": 23,
        "attention_heads": 8,
        "content_forward_per_document": 1,
        "content_backbone_forward_calls": 40,
        "verification_backbone_forward_calls": 12,
        "independent_causal_checks": 12,
        **{key: archived_provenance[key] for key in (
            "candidate_id", "model", "revision", "transformers", "tokenizer_sha256",
        )},
    }
    for field, value in expected.items():
        if seal.get(field) != value:
            raise ValueError(f"Signals seal differs from the fixed assay in {field}")
    paths = ("coordinates.jsonl", "token_features.npy", "states.npy", "causality.json", "telemetry.jsonl")
    for filename in paths:
        if base.digest(signals / filename) != seal["file_sha256"][filename]:
            raise ValueError(f"Signals export failed checksum validation for {filename}")
    source_sha = base.digest(source)
    preflight = base.read_json(preflight_receipt)
    if (base.digest(assay_receipt) != expected["assay_sha256"]
            or base.digest(preflight_receipt) != PREFLIGHT_SHA256 or source_sha != SOURCE_SHA256
            or preflight["stimuli_sha256"] != source_sha or preflight["tokenizer_sha256"] != expected["tokenizer_sha256"]
            or any(seal.get(field) != value for field, value in preflight.items())):
        raise ValueError("The tokenizer receipt does not certify exactly the selected source bytes")
    checks = base.read_json(signals / "causality.json")["checks"]
    if (len(checks) != 12 or any(check["passed"] is not True for check in checks)
            or {(check["case"], check["comparison"]) for check in checks}
            != {(case, label) for case in range(1, 4) for label in CAUSAL_TESTS}):
        raise ValueError("All registered causal extraction checks must pass")
    for check in checks:
        label = check["comparison"]
        rms = check["hidden_metrics"]["max_token_relative_rms"]
        errors = check["feature_max_absolute"]
        if not isinstance(rms, (int, float)) or not isfinite(rms) or rms < 0:
            raise ValueError("Unexpected causal hidden-state comparison statistic")
        if label == "hooked_vs_unhooked":
            if rms != 0 or errors is not None:
                raise ValueError("Hooked forward pass changed the fixed-generation hidden states")
            continue
        if (type(errors) is not list or len(errors) != len(RAW_FEATURES)
                or not all(isinstance(value, (int, float)) and isfinite(value) and value >= 0 for value in errors)):
            raise ValueError("Causal signal test does not account for all four measurements")
        if label == "replacement_suffix_positive":
            if rms <= 0.02 or max(errors) <= 0.04:
                raise ValueError("The suffix replacement did not affect the future positive control")
        elif rms > (0.02 if label == "prefix_truncation" else 0.002) or any(
            error > limit for error, limit in zip(errors, (0.04, 0.04, 0.002, 0.002), strict=True)
        ):
            raise ValueError("Signal values changed at an unchanged certified causal prefix")
    for name in ("compute_dtype", "storage_dtype"):
        if seal.get(name) != "float32":
            raise ValueError(f"Expected FP32 {name} in the signal exporter")
    forward, elapsed = seal["forward_and_readout_s"], seal["elapsed_s"]
    if not isinstance(forward, (int, float)) or not isinstance(elapsed, (int, float)) or not (
        isfinite(forward) and isfinite(elapsed) and 0 < forward <= elapsed
    ):
        raise ValueError("Export runtime metadata is invalid")
    telemetry = Counter()
    for sample in base.read_rows(signals / "telemetry.jsonl"):
        if set(sample) == {"status"} and sample["status"] == "sample_unavailable":
            telemetry["unavailable_samples"] += 1
        elif (set(sample) == {"gpu_timestamp", "utilization_pct", "used_mib", "total_mib"}
              and isinstance(sample["gpu_timestamp"], str) and sample["gpu_timestamp"]
              and all(type(sample[key]) is int for key in ("utilization_pct", "used_mib", "total_mib"))
              and 0 <= sample["utilization_pct"] <= 100 and 0 <= sample["used_mib"] <= sample["total_mib"]):
            telemetry["available_samples"] += 1
        else:
            raise ValueError("Unexpected raw GPU telemetry row")
    return seal, preflight, {
        "signals_seal_sha256": base.digest(seal_path),
        "signals_stimuli_sha256": source_sha,
        "signals_assay_sha256": expected["assay_sha256"],
        "signals_preflight_sha256": expected["preflight_sha256"],
        "signals_artifacts": {name: seal["file_sha256"][name] for name in paths},
        "signals_raw_timing": {"forward_and_readout_s": forward, "elapsed_s": elapsed, **telemetry},
    }


def certified_tokens(tokenizer, text, stops):
    whole = tokenizer.encode(text, add_special_tokens=False)
    offsets = [list(span) for span in whole.offsets]
    token_stops = []
    for end in stops:
        prefix = tokenizer.encode(text[:end], add_special_tokens=False)
        count = len(prefix.ids)
        if (not count or whole.ids[:count] != prefix.ids or whole.offsets[:count] != prefix.offsets
                or prefix.offsets[-1][1] != end):
            raise ValueError("A registered sentence end is not an exact causal token prefix")
        token_stops.append(count)
    if not token_stops or token_stops != sorted(set(token_stops)) or token_stops[-1] != len(offsets):
        raise ValueError("Invalid one-based tokenizer sentence fences")
    return offsets, token_stops


def sentence_features(info, offsets, token_stops, token_values):
    """Aggregate displayed content tokens in each original sentence, excluding sentence one."""
    coordinates = np.asarray(offsets)
    if (coordinates.shape != (len(token_values), 2) or coordinates.dtype.kind not in "iu"
            or token_values.shape != (len(coordinates), len(RAW_FEATURES))):
        raise ValueError("Offsets and four signal columns must align exactly")
    if not np.isfinite(token_values).all() or np.any(token_values[:, :2] < -1e-6) or np.any(
        np.abs(token_values[:, 2:]) > 1 + 1e-6,
    ):
        raise ValueError("Signals fall outside predictive entropy or attention-mass support")
    if np.any(token_values[0, :2] != 0) or np.any(token_values[:token_stops[0], 2:] != 0):
        raise ValueError("Empty prediction history and empty previous-sentence attention must be zero")
    if len(info) != len(token_stops):
        raise ValueError("Sentence layout and tokenizer fences differ")
    rows, kept = [], []
    separators = 0
    for i, (location, begin, end) in enumerate(zip(info, (0, *token_stops[:-1]), token_stops, strict=True)):
        left, right = coordinates[begin:end].T
        past_end = info[i - 1]["stop"] if i else 0
        if np.any(left < past_end) or np.any(right > location["stop"]) or np.any(right <= left):
            raise ValueError("A token span crosses a certified sentence boundary or is empty")
        overlaps_content = right > location["start"]
        separators += int((~overlaps_content).sum())
        retained = np.arange(begin, end)[overlaps_content]
        kept.append(retained.tolist())
        if len(retained) == 0:
            raise ValueError("A displayed sentence has no aligned content tokens")
        if i == 0:
            continue
        selected = token_values[retained].astype(np.float64)
        avg_surprise, avg_entropy, avg_mass, avg_peak = selected.mean(axis=0)
        rows.append([avg_surprise, selected[:, 0].max(), avg_entropy, avg_mass, avg_peak])
    return np.asarray(rows, dtype=np.float64), kept, separators


def load_signal_features(signals, signal_stimuli, signal_assay, signal_preflight, tokenizer_path,
                         stimuli, metadata, legacy_states, legacy_provenance):
    from tokenizers import Tokenizer

    if base.digest(tokenizer_path) != legacy_provenance["tokenizer_sha256"]:
        raise ValueError("Expected the frozen Qwen Rust tokenizer bytes")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    seal, preflight, provenance = load_signal_seal(
        signals, signal_stimuli, signal_assay, signal_preflight, legacy_provenance,
    )
    selected = base.read_rows(signal_stimuli)
    receipt = {row["id"]: row for row in preflight["documents"]}
    coordinates = base.read_rows(signals / "coordinates.jsonl")
    ids = [row["id"] for row in selected]
    if (len(ids) != len(metadata) or set(ids) != metadata.keys()
            or len(preflight["documents"]) != len(receipt) or receipt.keys() != metadata.keys()
            or [row["id"] for row in coordinates] != ids
            or any(row != stimuli[row["id"]] for row in selected)):
        raise ValueError("The sealed grade-free input must be exactly the original 40 full stories in export order")
    features = np.load(signals / "token_features.npy", mmap_mode="r", allow_pickle=False)
    states = np.load(signals / "states.npy", mmap_mode="r", allow_pickle=False)
    expected_states = (sum(len(row["stops"]) for row in selected), base.WIDTH)
    expected_features = (sum(len(row["offsets"]) for row in coordinates), len(RAW_FEATURES))
    if (states.shape != expected_states or features.shape != expected_features
            or seal["states_shape"] != list(expected_states) or seal["token_features_shape"] != list(expected_features)
            or states.dtype != np.float32 or features.dtype != np.float32 or not np.isfinite(states).all()):
        raise ValueError("Unexpected number, alignment or precision of sealed float32 arrays")

    by_id = {}
    axis = []
    max_state_error = 0.0
    state_cursor = token_cursor = 0
    for document, row in zip(selected, coordinates, strict=True):
        key, text, stops = (document[field] for field in ("id", "text", "stops"))
        offsets, token_stops = certified_tokens(tokenizer, text, stops)
        if set(row) == {"id", "offsets", "stops", "token_stops"} and row["token_stops"] != token_stops:
            raise ValueError("Exported one-based token fences differ from exact Rust prefixes")
        if (set(row) not in ({"id", "offsets", "stops"}, {"id", "offsets", "stops", "token_stops"})
                or row["stops"] != stops or row["offsets"] != offsets
                or receipt[key]["stops"] != stops or receipt[key]["input_tokens"] != len(offsets)
                or receipt[key]["excluded_stops_by_reason"]):
            raise ValueError(f"Token coordinates or registered sentence endpoints differ for {key}")
        new_state = np.asarray(states[state_cursor:state_cursor + len(stops)], dtype=np.float64)
        old_state = np.asarray(legacy_states[key], dtype=np.float64)
        norm = np.linalg.norm(old_state, axis=1)
        if np.any(norm <= 0):
            raise ValueError("Unexpected zero archived sentence state")
        relative = np.linalg.norm(new_state - old_state, axis=1) / norm
        maximum = float(relative.max())
        if not maximum < base.MAX_PREFIX_ERROR:
            raise ValueError(f"New export and archived sentence states diverge for {key}: {maximum}")
        max_state_error = max(max_state_error, maximum)
        values = features[token_cursor:token_cursor + len(offsets)]
        rows, used, separators = sentence_features(metadata[key]["sentences"], offsets, token_stops, values)
        by_id[key] = rows
        axis.append({
            "id": key, "token_start": token_cursor, "state_start": state_cursor, "certified_token_stops": token_stops,
            "content_token_indices": used, "excluded_pure_separator_tokens": separators,
            "new_vs_archived_max_relative_l2": maximum,
        })
        state_cursor += len(stops)
        token_cursor += len(offsets)
    return by_id, axis, {
        **provenance, "max_relative_l2_new_vs_archived": max_state_error,
        "raw_signal_rows": token_cursor, "raw_state_rows": state_cursor,
        "excluded_pure_separator_tokens": sum(row["excluded_pure_separator_tokens"] for row in axis),
        "independent_causal_checks": seal["independent_causal_checks"],
    }


def fit_signals(arms, original, signals, normalized, families, metadata, slots, hits, totals):
    fits, predictions, checks = {}, {}, {}
    for name, columns in arms.items():
        extended = {key: np.column_stack((original[key], signals[key][:, columns])) for key in slots}
        predictions[name], fits[name], checks[name] = contextual.fit_context(
            normalized, extended, families, metadata, slots, hits, totals,
        )
    return predictions, fits, checks


def run(assay, archived_extraction, archived_packet, phase_reference, reference, signals,
        signal_stimuli, signal_assay, signal_preflight, tokenizer, output):
    if output.exists():
        raise FileExistsError(output)
    if base.digest(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("The predeclared Frick signal protocol changed")
    audit, stimuli, metadata, families, texts = base.load_layout(assay)
    archived_states, provenance = base.load_states(
        assay, archived_extraction, archived_packet, base.PROTOCOL, stimuli, metadata, families,
    )
    features, axis, signal_audit = load_signal_features(
        signals, signal_stimuli, signal_assay, signal_preflight, tokenizer,
        stimuli, metadata, archived_states, provenance,
    )
    keys = sorted(metadata)
    states = {key: paired.normalize_states(archived_states[key]) for key in keys}
    legacy = {key: paired.sentence_forms(texts[key]) for key in keys}
    ends = np.r_[0, np.cumsum([len(legacy[key]) for key in keys])]
    slots = {key: slice(start, stop) for key, start, stop in zip(keys, ends[:-1], ends[1:], strict=True)}
    comparators, phase_hash = conditional.frozen_comparators(phase_reference, slots, stimuli, provenance)
    seven, seven_hash = contextual.reference_logits(reference, slots, stimuli, provenance, comparators)
    trials, readers, hits, totals, inventory = base.load_outcomes(assay, audit, metadata, slots)
    seven_replay, _ = conditional.crossfit(states, legacy, families, metadata, slots, hits[1], totals[1], comparators[0])
    if not np.array_equal(seven_replay, seven):
        raise ValueError("Seven-feature comparators failed deterministic replay")
    predictions, fits, metadata_checks = fit_signals(
        ARMS, legacy, features, states, families, metadata, slots, hits[1], totals[1],
    )
    names = (*ARMS, *COMPARATORS)
    frozen_logits = np.vstack((*(predictions[arm] for arm in ARMS), seven, comparators[1:]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".frick-signals-score-") as tmp:
        stage = Path(tmp) / "result"
        stage.mkdir()
        scores_path = stage / "forecasts.jsonl"
        with scores_path.open("x", encoding="utf-8") as stream:
            for key, slot in slots.items():
                for stop, row in zip(stimuli[key]["stops"][1:], frozen_logits[:, slot].T, strict=True):
                    payload = {"id": key, "stop": stop, "first_logits": dict(zip(names, row.tolist(), strict=True))}
                    stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        base.write_json(stage / "fits.json", {
            "legacy_features": paired.FEATURES, "added_features": FEATURES,
            "arm_feature_indices": ARMS, "folds_by_arm": fits,
        })
        base.write_json(stage / "axis-audit.json", {
            "documents": axis, "seven_reference_bitwise_replay": True,
            "maximum_logit_change_after_removing_held_out_targets_and_counts": metadata_checks,
            "base_sentence_states": "Original archived states in all arms; repeated states used only for parity",
            **signal_audit,
        })
        files = ("fits.json", "forecasts.jsonl", "axis-audit.json")
        seal = stage / "forecast-seal.json"
        base.write_json(seal, {
            "provenance": provenance, "protocol_sha256": PROTOCOL_SHA256,
            "scorer_sha256": base.digest(Path(__file__)), "signals_seal_sha256": signal_audit["signals_seal_sha256"],
            "phase_reference_forecast_seal_sha256": phase_hash, "seven_reference_forecast_seal_sha256": seven_hash,
            "file_sha256": {name: base.digest(stage / name) for name in files},
        })
        restored = np.asarray([[row["first_logits"][k] for k in names] for row in base.read_rows(scores_path)]).T
        if not np.array_equal(restored, frozen_logits):
            raise ValueError("The archived forecasts failed exact serialization")
        scores = {}
        for i, arm in enumerate(ARMS):
            result = contextual.evaluate(restored[[i, 3, 4, 5]], slots, families, metadata, trials, readers)
            result["methods"][arm] = result["methods"].pop("context_features")
            result["comparisons"] = {
                method: {metric.replace("context_features", arm, 1): value for metric, value in metrics.items()}
                for method, metrics in result["comparisons"].items()
            }
            scores[arm] = result
        base.write_json(stage / "results.json", {
            "primary_arm": "both", "primary_pass": scores["both"]["ablation_pass"], "arms": scores,
            "inventory": inventory, "retrospective": True, "forecast_seal_sha256": base.digest(seal),
        })
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in (
        "assay", "archived-extraction", "archived-packet", "signals", "signal-stimuli",
        "signal-assay", "signal-preflight", "tokenizer", "output",
    ):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--phase-reference", type=Path, default=conditional.REFERENCE)
    parser.add_argument("--reference", type=Path, default=contextual.REFERENCE)
    run(**vars(parser.parse_args()))
