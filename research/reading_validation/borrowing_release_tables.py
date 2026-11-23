"""Export complete endpoint coverage, pairing weights, and descriptive single-expert comparisons."""

import csv
from pathlib import Path

import numpy as np

from research.backbone_transfer.extract import digest
from research.reading_validation.borrowing_data import PACKET
from research.reading_validation.borrowing_family_audit import FamilyRisks
from research.reading_validation.borrowing_family_bootstrap import OUT
from research.reading_validation.borrowing_global_families import (
    LINKED,
    grouped_summary,
)
from research.reading_validation.borrowing_prior import SEEDS, read, write
from research.union_training.observation_registry import contracts


def calibration_table(root, release, index, writer):
    geom = read(root / "geometry.json")
    bits = (
        geom["registered_by_endpoint"]
        if release == "global"
        else geom["exclusions_and_support"]
    )
    report = read(root / "scores.json")
    frozen = read(root / "forecast-freeze.json")["seeds"]
    for endpoint in sorted(index):
        spec = index[endpoint]
        group = spec["source"]
        roles = bits[endpoint]["registered_roles" if release == "global" else "rows"]
        for seed in map(str, SEEDS):
            fitted = frozen[seed][endpoint]
            held = report["by_endpoint"].get(endpoint)
            vals = (
                held["per_seed_nll" if release == "global" else "seed_nll"][seed]
                if held
                else {}
            )
            writer.writerow(
                [
                    release,
                    group,
                    endpoint,
                    spec["likelihood"],
                    spec["coordinates"],
                    "|".join(spec["states"]),
                    seed,
                    *roles,
                    held["raw_observations"] if held else 0,
                    held["families"] if held else 0,
                    fitted["families"],
                    fitted["endpoint_map"],
                    fitted["family_bootstrap"],
                    *fitted.get("mass_interval", ("", "")),
                    *(
                        vals.get(k, "")
                        for k in (
                            "private",
                            "shared",
                            "midpoint",
                            "endpoint_map",
                            "family_bootstrap",
                        )
                    ),
                ]
            )


def single_controls(root, split):
    scores = read(root / "scores.json")
    store = {}
    with np.load(root / "material-statistics.npz", allow_pickle=False) as pack:
        for k, record in scores["by_endpoint"].items():
            # Use the original descriptive expert controls from the registered third experiment.
            keys = pack[
                k + ".global_group_sha256"
                if split == "global"
                else k + ".global_family_sha256"
            ]
            t = pack[k + ".numerator"].mean(axis=0)[:, [3, 4, 2]]
            store.setdefault(record["source"], {})[k] = FamilyRisks(
                keys.tolist(), t, pack[k + ".denominator"]
            )
    connected = [[s] for s in sorted(store) if s not in LINKED] + [
        sorted(set(store) & LINKED)
    ]
    data = grouped_summary(store, connected, "fourth-unmixed-controls-" + split)
    actual = {}
    for j, (cmp, control) in enumerate(
        [("over_map", "private"), ("over_midpoint", "shared")]
    ):
        env = data["source_and_material"][cmp]
        actual[control] = {
            "point_gain": env["gain"],
            "paired_nested_nominal_ci": env["joint_source_material_nominal_ci"],
            "paired_nested_simultaneous_ci": env[
                "joint_source_material_simultaneous_ci"
            ],
            "exact_group_signflip": env["exact_signflip_p"],
            "groups": data["units"],
            "improving_groups": data["registered_source_only"]["comparisons"][
                ("endpoint_map", "midpoint")[j]
            ]["improving_sources"],
        }
    clean_rows = {s: {} for s in store if s != "frick"}
    clean_components = [[s] for s in sorted(clean_rows) if s not in LINKED] + [
        sorted(set(clean_rows) & LINKED)
    ]
    with np.load(root / "material-statistics.npz", allow_pickle=False) as pack:
        for name, record in scores["by_endpoint"].items():
            source = record["source"]
            if source == "frick":
                continue
            suffix = (
                ".global_group_sha256" if split == "global" else ".global_family_sha256"
            )
            clean_rows[source][name] = FamilyRisks(
                pack[name + suffix].tolist(),
                pack[name + ".numerator"].mean(axis=0)[:, :3],
                pack[name + ".denominator"],
            )
    clean = grouped_summary(
        clean_rows, clean_components, "remove-projection-source-" + split
    )
    return {"fully_shared_or_private": actual, "omit_every_Frick_endpoint": clean}


def main():
    spec = {v["name"]: v for v in read(PACKET / "index.json")["tasks"]}
    reg = contracts()
    if set(reg) != set(spec):
        raise ValueError("Mismatched sources and registered states")
    dst = OUT / "endpoints-calibration-and-performance.csv"
    with dst.open("w", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(
            [
                "material_split",
                "source",
                "endpoint",
                "likelihood",
                "observed_coordinates",
                "shared_states",
                "seed",
                "fit_rows",
                "tune_rows",
                "calibration_rows",
                "held_rows",
                "quarantine_rows",
                "held_response_units",
                "held_families",
                "calibration_families",
                "map_shared_weight",
                "family_bootstrap_shared_weight",
                "coefficient_draw_lower95",
                "coefficient_draw_upper95",
                "private_nll",
                "shared_nll",
                "midpoint_nll",
                "map_nll",
                "family_bootstrap_nll",
            ]
        )
        for m in ("global", "confirmation"):
            calibration_table(OUT / m, m, spec, w)
    arrays = {m: single_controls(OUT / m, m) for m in ("global", "confirmation")}
    write(
        OUT / "single-expert-controls.json",
        {
            "estimand": "Family bootstrap improvement over forced private or forced shared fitting. Secondary, descriptive, equal linked-component and material resampling.",
            "partitions": arrays,
            "endpoint_table_sha256": digest(dst),
            "program_sha256": digest(Path(__file__)),
        },
    )
    print(arrays, flush=True)


if __name__ == "__main__":
    main()
