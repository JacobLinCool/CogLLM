"""Seal locally regenerated calibration files before replaying historical scoring."""

import argparse
from pathlib import Path

from research.backbone_transfer.extract import digest
from research.reading_validation.borrowing_prior import MODES, SEEDS, read, write

HERE = Path(__file__).parent
RESULTS = Path("research/results")


def seal(name):
    root = RESULTS / ("reading-validation-borrowing-" + name)
    plan = HERE / ("borrowing-" + name + "-protocol.md")
    scoring = HERE / ("borrowing_" + name + "_score.py")
    if name == "prior":
        locks = {
            f"{mode}/seed-{seed}": root
            / mode
            / f"seed-{seed}"
            / "calibration-lock.json"
            for mode in MODES
            for seed in SEEDS
        }
        if not all(read(p)["protocol"] == digest(plan) for p in locks.values()):
            raise ValueError("Replayed calibration differs from its specified protocol")
        values = {"coefficients": {k: digest(v) for k, v in locks.items()}}
    else:
        locks = {mode: root / mode / "inputs-and-coefficients.json" for mode in MODES}
        values = {"modes": {k: digest(v) for k, v in locks.items()}}
    values.update({"protocol": digest(plan), "scoring_program": digest(scoring)})
    frozen = root / "forecast-freeze.json"
    if frozen.exists():
        original = read(frozen)
        if all(original.get(k) == v for k, v in values.items()):
            print("Existing calibration and scoring hashes verified", name)
            return
        raise ValueError(
            "Rebuilt coefficients differ from a previous seal. Use a clean reproduction tree with an empty output directory."
        )
    write(frozen, {**values, "reproduction_of_previously_scored_material": True})
    print("Sealed coefficients from this reproduction", name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", choices=("prior", "reliability"))
    seal(ap.parse_args().name)
