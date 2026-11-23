"""Create a code-only Hugging Face Space pinned to one published model commit."""

import argparse
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve().parent
# Modules used by the live CPU writer, including the optional, cached trace renderer.
# Keep the upload independent of local model caches, study materials and evaluation runs.
MODULES = {
    "research/backbone_transfer": ("extract",),
    "research/token_readout": (
        "app", "backbone", "expectation_view", "experimental_view", "extraction", "learner_view",
        "model", "physical_view", "portable_runner", "readout", "reference_view", "runner",
        "summary_view", "terminology_view", "transfer_fit", "writing_review",
    ),
    "research/reading_validation": (
        "bridge_forecast", "local_features", "local_forecast", "local_inference",
        "transition_external_forecast", "transition_inference",
    ),
    "research/stage5": ("complex_tokenization",),
    "research/stage6": ("medreadme_span_audit",),
    "release": ("medical_lexicon",),
}


def build(model_id, model_revision, output):
    if re.fullmatch(r"[\w.-]+/[\w.-]+", model_id) is None or re.fullmatch(r"[0-9a-f]{40}", model_revision) is None:
        raise ValueError("A Hub model ID and an immutable model commit are required.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose an empty destination to avoid including unrelated files in the upload.")
    config = "research/backbone_transfer/models.json"
    paths = [Path(group, f"{name}.py") for group, names in MODULES.items() for name in names]
    sources = {name: ROOT / name for name in [*paths, Path(config), Path("LICENSE")]}
    sources.update({Path("app.py"): SOURCE / "space_app.py", Path("requirements.txt"): SOURCE / "space_requirements.txt"})
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    spec = json.loads((ROOT / config).read_text())["models"]["qwen"]
    card = (SOURCE / "space-card.md").read_text()
    for key, value in {
        "__MODEL_ID__": model_id, "__MODEL_REVISION__": model_revision,
        "__BASE_FILES__": ",".join(spec["files"]), "__BASE_REVISION__": spec["revision"],
    }.items():
        card = card.replace(key, value)
    output.mkdir(parents=True, exist_ok=True)
    for name, path in sources.items():
        (output / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, output / name)
    (output / "README.md").write_text(card)
    release = {"model_id": model_id, "model_revision": model_revision}
    (output / "release.json").write_text(json.dumps(release, indent=2) + "\n")
    print(f"Staged {len(sources) + 2} public Space files under {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.model_id, args.model_revision, args.output)
