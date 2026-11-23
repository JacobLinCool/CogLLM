"""Prepare an isolated offline dependency directory for the L4 serving verification."""

from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import ZipFile

from research.reading_validation.local_forecast import digest

WHEELS = "7b54e68acb7e7d12043f4c47264b820270d7cf6c90f5968fa85f7d877c12cca4"


def run(runtime, reader, checkpoint, input_path, output_dir):
    path = Path("nns-learner-wheels.zip")
    if sys.version_info[:2] != (3, 13) or digest(path) != WHEELS:
        raise ValueError(
            "Expected CPython 3.13 and the pinned offline Linux wheelhouse"
        )
    with tempfile.TemporaryDirectory(prefix="cogllm-learner-") as tmp:
        root = Path(tmp)
        packages = root / "packages"
        with ZipFile(path) as z:
            names = z.namelist()
            if len(names) != 13 or any(
                Path(n).name != n or not n.endswith(".whl") for n in names
            ):
                raise ValueError("Unexpected file in the sealed wheelhouse")
            for name in names:
                (root / name).write_bytes(z.read(name))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-compile",
                "--no-deps",
                "--disable-pip-version-check",
                "--target",
                str(packages),
                *[str(root / n) for n in names],
            ],
            check=True,
            timeout=120,
        )
        sys.path.insert(0, str(packages))
        try:
            import nltk

            if nltk.__version__ != "3.10.4":
                raise ValueError("Incompatible previously imported English stemmer")
            from research.reading_validation.nns_writer_worker import run as verify

            return verify(runtime, reader, checkpoint, input_path, output_dir)
        finally:
            sys.path.remove(str(packages))
