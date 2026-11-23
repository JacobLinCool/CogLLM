"""Refit and regrade the exact-spelling and reference-panel studies in an empty temporary directory."""

from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from research.backbone_transfer.extract import digest
from research.reading_validation import (
    population_calibration_controls,
    reading_population_calibration,
    spelling_agreement,
    spelling_agreement_score,
)
from research.reading_validation.spelling_inference import ReadingWordForecast


def compare(expected, actual, names):
    for name in names:
        if digest(actual / name) != digest(expected / name):
            raise ArithmeticError(
                f"Numerical replay differs from the published result: {name}"
            )


def replay(paragraphs, stems):
    ReadingWordForecast.load()
    results = spelling_agreement.OUT
    population = reading_population_calibration.OUT
    with TemporaryDirectory(prefix="cogllm-word-replay-") as temp:
        spelling = Path(temp) / "spelling"
        calibration = Path(temp) / "calibration"
        with (
            patch.object(spelling_agreement, "OUT", spelling),
            patch.object(spelling_agreement_score, "OUT", spelling),
        ):
            spelling_agreement.lock(paragraphs)
            spelling_agreement_score.grade()
        compare(
            results,
            spelling,
            (
                "fit-lock.json",
                "provo-forecasts.npz",
                "peelle-forecasts.npz",
                "scores.json",
            ),
        )
        with (
            patch.object(reading_population_calibration, "SPELL", spelling),
            patch.object(reading_population_calibration, "OUT", calibration),
            patch.object(population_calibration_controls, "OUT", calibration),
        ):
            reading_population_calibration.register()
            reading_population_calibration.scores()
            population_calibration_controls.evaluate(stems)
        compare(
            population,
            calibration,
            ("forecast-lock.json", "forecast.npz", "scores.json", "controls.json"),
        )
    print("All 8 fitted, forecast and evaluation files reproduced byte for byte.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("paragraph_native", type=Path)
    parser.add_argument("stem_native", type=Path)
    args = parser.parse_args()
    replay(args.paragraph_native, args.stem_native)
