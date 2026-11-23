"""Replay the two independently locked lexical forecasts without writing over their seals."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from research.reading_validation import local_jargon, local_word_readers
from research.reading_validation.local_features import OUT
from research.reading_validation.local_forecast import digest


def function_digest(path, function):
    tree = ast.parse(path.read_text())
    (node,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function]
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def run(gpu):
    report = {}
    with TemporaryDirectory() as temp:
        root = Path(temp)
        target = root / "jargon"
        local_jargon.forecast(gpu, OUT / "localization/deployment-weights.json", target)
        pred = target / "predictions.jsonl"
        primary = OUT / "jargon"
        seal = json.loads((primary / "prediction-lock.json").read_bytes())
        if digest(pred) != digest(primary / pred.name) or digest(pred) != seal["files_sha256"][pred.name]:
            raise ValueError("Contextual term forecasts changed")
        for name, module, function, file, old in (
            ("jargon", local_jargon, "forecast", pred, local_jargon.JARGON / "forecast-sealed.py"),
            (
                "word-readers",
                local_word_readers,
                "prepare",
                None,
                local_word_readers.SOURCE / "forecast-sealed.py",
            ),
        ):
            if file is None:
                dest = root / "word-readers"
                with patch.object(local_word_readers, "TARGET", dest):
                    local_word_readers.prepare()
                file = dest / "predictions.jsonl"
            primary = OUT / name
            seal = json.loads((primary / "prediction-lock.json").read_bytes())
            if digest(file) != digest(primary / "predictions.jsonl"):
                raise ValueError("Frozen score replay differs")
            code = Path(module.__file__)
            if old.exists():
                if digest(old) != seal["code_sha256"] or function_digest(old, function) != function_digest(
                    code, function
                ):
                    raise ValueError("Archived and active forecast functions differ")
            report[name] = dict(
                sealed_response_free_forecast_sha256=digest(file),
                sealed_code_sha256=seal["code_sha256"],
                active_code_sha256=digest(code),
                forecast_function_ast_sha256=function_digest(code, function),
                original_script_crosschecked=old.exists(),
                exact_replay=True,
            )
    report["outcome_parser_corrections"] = {
        "jargon": "Map distributed 0–4 category indexes to the documented survey choices 1–5.",
        "word-readers": "Exclude five DHH responses in the matched inventory with an out-of-scale raw entry; retain every valid response.",
    }
    (OUT / "forecast-replay.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=Path, required=True)
    run(parser.parse_args().gpu)
