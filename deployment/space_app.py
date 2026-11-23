"""Serve the released CogLLM writer on the Hugging Face CPU Basic hardware."""

import json
from pathlib import Path

import gradio as gr
from huggingface_hub import hf_hub_download, snapshot_download

from release.medical_lexicon import load_writing_head
from research.backbone_transfer.extract import digest
from research.reading_validation.transition_inference import TransitionInference
from research.token_readout.app import CSS
from research.token_readout.portable_runner import load
from research.token_readout.writing_review import STYLE, build_app


MAX_TOKENS = 512


def load_writer(model_id, model_revision):
    manifest_path = hf_hub_download(model_id, "manifest.json", revision=model_revision, token=False)
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["schema"] != "cogllm-writing-model-v1" or manifest["model_id"] != model_id:
        raise ValueError("The release manifest does not identify the selected CogLLM writer.")
    snapshot = Path(snapshot_download(model_id, revision=model_revision, allow_patterns=list(manifest["files"]), token=False))
    for relative, expected in manifest["files"].items():
        if digest(snapshot / relative) != expected:
            raise ValueError(f"Released model file differs from its manifest: {relative}")
    base = manifest["base_model"]
    backbone = Path(snapshot_download(base["repo_id"], revision=base["revision"], allow_patterns=list(base["files"]), token=False))
    return load(
        model=backbone,
        checkpoint=snapshot / manifest["token_checkpoint"],
        english_head=load_writing_head(snapshot / manifest["local_directory"]),
        transition_head=TransitionInference.load(snapshot / manifest["transition_checkpoint"]),
        serving_device="cpu",
    )


class HostedWriter:
    def __init__(self, runner):
        self.runner = runner

    def review_english(self, text, **options):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Paste a passage or choose the example.")
        # A byte-level tokenizer can produce a single token for a long repeated run.
        # Bound tokenizer work as well as the subsequent model forward.
        if len(text) > 32_768 or len(self.runner.runtime.tokenizer.encode(text, add_special_tokens=False).ids) > MAX_TOKENS:
            raise ValueError(f"Please shorten the passage to {MAX_TOKENS} model tokens or fewer.")
        return self.runner.review_english(text, **options)


if __name__ == "__main__":
    release = json.loads(Path(__file__).with_name("release.json").read_text())
    runner = HostedWriter(load_writer(release["model_id"], release["model_revision"]))
    ui = build_app(runner, study_tools=False, max_tokens=MAX_TOKENS)
    ui.queue(default_concurrency_limit=1, max_size=12).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CSS + STYLE,
        theme=gr.themes.Soft(primary_hue="teal", neutral_hue="slate"),
    )
