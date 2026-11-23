"""Stage the trained public CogLLM model in an empty directory for Hub publication."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
RESULTS = "research/results"
LOCAL = f"{RESULTS}/reading-validation-local/serving"
CURVE = f"{RESULTS}/adaptive-joint-fixed-retention-l4/guarded-fixed-1729/curve.pt"
TRANSITION = f"{RESULTS}/reading-validation-transition-external/readouts.json"
MODEL_ID = "JacobLinCool/CogLLM-Qwen3.5-2B"
SPACE_ID = "JacobLinCool/CogLLM"
SOURCE_FILES = {
    CURVE: "c1704d4436d767fa43bdb59e9216698514e1f6bd1d45cad4b60d0f018d84abb5",
    f"{LOCAL}/weights.npz": "2f72b3b28ec5bc4f47bfe084ae76ccf2f9fdc01e4f4d61bfd4a7a38bda99bfa3",
    f"{LOCAL}/manifest.json": "c38406db17f11e803b1b46f760e5082e83de9dc29dd7b91f97d5e3e3f4fa2c7e",
    f"{LOCAL}/complex-frequencies.json.gz": "5d8259717fe0990ade0ccad8e4192b1baa87b9d287fa7ae3cf2ac64b49f36fe3",
    TRANSITION: "7cfbc7b6d89e56995e8a909294c91db619eb41aaf30fa8dd3b1a2cac8e7bbf11",
}


def stage(path, model_id=MODEL_ID, space_id=SPACE_ID):
    target = Path(path).resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("The export destination must be empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as work:
        tree = Path(work) / "model"
        for name, expected in SOURCE_FILES.items():
            original = ROOT / name
            if sha256(original.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Unrecognized source weights: {name}")
            to = tree / name
            to.parent.mkdir(parents=True, exist_ok=True)
            copyfile(original, to)

        base = json.loads((ROOT / "research/backbone_transfer/models.json").read_bytes())["models"]["qwen"]
        fit = json.loads((ROOT / CURVE).with_name("fitting.json").read_bytes())
        manifest = {
            "schema": "cogllm-writing-model-v1",
            "model_id": model_id,
            "space_id": space_id,
            "base_model": {"repo_id": base["repo"], "revision": base["revision"], "files": base["sha256"]},
            "files": SOURCE_FILES,
            "token_checkpoint": CURVE,
            "local_directory": LOCAL,
            "transition_checkpoint": TRANSITION,
            "token_training": {"sources": fit["source_names"], "objectives": fit["blocks"], "selection": fit["mode"]},
        }
        (tree / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        card = (ROOT / "release/MODEL_CARD.md").read_text()
        (tree / "README.md").write_text(card.replace(MODEL_ID, model_id).replace("spaces/" + SPACE_ID, "spaces/" + space_id))
        copyfile(ROOT / "LICENSE", tree / "LICENSE")
        tree.replace(target)
    return target


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--space-id", default=SPACE_ID)
    args = parser.parse_args()
    print(stage(args.destination, args.model_id, args.space_id))
