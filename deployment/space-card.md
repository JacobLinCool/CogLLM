---
title: CogLLM Writing Review
emoji: 📖
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.17.3
python_version: "3.12"
app_file: app.py
suggested_hardware: cpu-basic
short_description: See which words and sentence connections may slow a reader
license: mit
models:
  - __MODEL_ID__
  - Qwen/Qwen3.5-2B
preload_from_hub:
  - Qwen/Qwen3.5-2B __BASE_FILES__ __BASE_REVISION__
---

# CogLLM Writing Review

Paste a short English passage and select **Review draft**. The background color estimates the reading ease of each completed sentence. Underlines mark wording that may need explanation. A dashed outline flags a sentence whose connection to the preceding text deserves a second look. The experimental token curves are available under the marked draft.

This Space accepts up to 512 model tokens per passage and processes one passage at a time on CPU. The first analysis after a restart may take longer. All scores for a passage share one causal forward through Qwen3.5-2B.

The reading-ease and wording heads learned from readers of biomedical texts. The connection head learned from readers marking short stories and detected fewer breaks when applied to student essays. Individual confusion, understanding and moment of resolution remain unverified. The model card describes the observed human outcomes and the held-out evaluations. [CogLLM model card](https://huggingface.co/__MODEL_ID__/blob/__MODEL_REVISION__/README.md)
