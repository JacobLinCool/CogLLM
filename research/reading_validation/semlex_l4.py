"""Replay the immutable word-variant kernel on a second registered study's public inputs."""

import importlib.util
import json
from pathlib import Path

from research.reading_validation.local_forecast import digest

PACKET = "64de65fa875e4c33661a9c6a032fb9c9fbf3a249125ffc5d8c626773dbeb090a"
KERNEL = "f829ed2b6d238161aa3a28bccae8925eea21324f8b5d8259577c96d196478743"
SIZE = 5633


def run(runtime, reader, checkpoint, input_path, output_dir):
    source = Path(__file__).with_name("nns_rewrite_l4.py")
    if digest(source) != KERNEL or digest(Path(input_path)) != PACKET:
        raise ValueError("The registered extractor or external public materials changed")
    # The first study locked its input hash and size as module constants. A private
    # module instance reuses that exact implementation without mutating live workers
    # or invalidating the historical Python-source fingerprint.
    spec = importlib.util.spec_from_file_location("semlex_registered_word_kernel", source)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load the sealed word extraction implementation")
    kernel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel)
    kernel.PACKET = PACKET
    kernel.SIZE = SIZE
    receipt = kernel.run(runtime, reader, checkpoint, input_path, output_dir)
    receipt["registered_external_wrapper_sha256"] = digest(Path(__file__))
    (Path(output_dir) / "extraction.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
