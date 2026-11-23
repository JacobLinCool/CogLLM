"""Unpack the Stage 3 payload and run the bounded experiment in an owned Colab kernel."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from time import perf_counter


root = Path("/content/cogllm-stage3")
root.mkdir()
output = root / "outputs"
output.mkdir()
started = perf_counter()
status = {"phase": "unpack", "completed": False}
try:
    with tarfile.open("/content/cogllm-stage3-inputs.tar.gz", "r:gz") as payload:
        payload.extractall(root, filter="data")
    status["phase"] = "install_transformers"
    with (output / "worker.log").open("w", encoding="utf-8") as worker_log:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "transformers==5.17.0"],
            check=True, timeout=180, stdout=worker_log, stderr=subprocess.STDOUT,
        )
    status["phase"] = "feature_extraction_and_probe"
    env = os.environ.copy()
    env.update({
        "OPENBLAS_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    })
    with (output / "worker.log").open("a", encoding="utf-8") as worker_log:
        subprocess.run(
            [
                sys.executable, "-u", "-m", "research.stage3.run",
                "--data", str(root / "data"), "--output", str(output),
                "--dtype", os.environ["COGLLM_STAGE3_COMPUTE_DTYPE"],
            ],
            cwd=root, env=env, check=True, timeout=900,
            stdout=worker_log, stderr=subprocess.STDOUT,
        )
    status.update({"phase": "complete", "completed": True})
finally:
    status["worker_elapsed_s"] = perf_counter() - started
    (output / "remote-status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    with tarfile.open("/content/cogllm-stage3-results.tar.gz", "w:gz", compresslevel=1) as bundle:
        bundle.add(output, arcname="outputs")
