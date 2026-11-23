# CogLLM Gradio Interfaces

## Current Token-Curve Preview

The current token-curve preview uses `research.token_readout.app`. Paste an article to see the original text colored by the model's cumulative unresolved signal. The position slider selects a span and, where available, outlines associated earlier text. The expandable chart shows four token-level signals for unresolved reading, opening, resolution, and transition. Color intensity is relative to the article. These experimental estimates are evaluated against specific endpoints and observation positions. [Token-curve evidence and UI verification](../token_readout/transfer-results.md) · [Adaptive joint training and selected weights](../adaptive_joint/results.md)

On a host with the verified NVIDIA L4 environment and the pinned official Qwen3.5-2B snapshot, run the current preview from the repository root.

```sh
python -m research.token_readout.app \
  --runner research.token_readout.portable_runner:load \
  --model /path/to/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --checkpoint research/results/adaptive-joint-fixed-retention-l4/guarded-fixed-1729/curve.pt
```

Open `http://127.0.0.1:7860` on that host or through an authenticated local forward. The coordinates and traces come from one causal forward pass. Later position changes reuse the analysis. [Portable runner and reproducibility](../token_readout/transfer-results.md#reproduction-and-analysis-interface)

## Sentence- and Word-Level Research App

The separate `research.gradio_app.app` research entry point displays sentence shading and word underlines on the same text. Sentence shading shows estimated ease of reading by default and can switch to estimated self-rated comprehension. Both use a 1–5 scale. Word underlines show estimates at the end-of-word token on a 0–1 scale, where 1 is most difficult. Shading and underlines use a relative scale within the article. Warmer colors indicate higher estimated difficulty or lower estimated self-rated comprehension. Cooler colors indicate easier passages or higher estimates. Hover over a word to see its score and offsets. Expand the sentence details for zero-based, half-open Unicode character spans.

By default, sentence estimates use only the text available through the end of that sentence. The reference is the group of human readers who rated PLABA-EVAL English biomedical abstracts and their plain-language rewrites with the full text visible. Use in other domains and languages is experimental. The default full-text view uses English Punkt segmentation within paragraphs separated by blank lines. Its sentence boundaries can differ from those annotated by a dataset. [Baseline and scales](../training-protocol.md#已完成的首輪基線) · [App acceptance criteria](../execution-plan.md#完成條件)

<a id="在-apple-silicon-啟動"></a>

## Launch the Research App on Apple Silicon

Use a local terminal with Metal access, Python 3.11 or later, and a local official Qwen3.5-2B snapshot. Run the following from the repository root. The bundled default checkpoint contains English sentence-segmentation parameters and provenance records for each readout.

```sh
uv venv --python 3.11 .venv-cogllm
uv pip install --python .venv-cogllm/bin/python -r research/gradio_app/requirements.txt
.venv-cogllm/bin/python -m research.gradio_app.checkpoint verify
.venv-cogllm/bin/python -m research.gradio_app.app --model /path/to/15852e8c16360a2fea060d615a32b45270f8a8fc
```

Open `http://127.0.0.1:7860` in a browser. The app loads the candidate checkpoint and verifies the checkpoint ID, tokenizer, and projection bound to the earlier-text readout. All readouts share the same Qwen snapshot and a single forward pass on the source text. Use `--checkpoint` to select the directory containing that sealed candidate. The score provenance and scope panel shows the checkpoint ID and earlier-text readout version. [Default checkpoint manifest](../../../research/results/gradio-candidate-checkpoint/checkpoint.json) · [Earlier-text readout release](../../../research/results/arrau-bridge-l4/reference-release.json)

Use `--causal-plaba` or `--full-context-plaba` to reproduce the earlier single-readout experiments. Both diagnostic modes use the English `punkt_tab` data installed for NLTK. Across three verification articles, the largest difference between local and L4 causal sentence-end scores was 0.00321 points. [Causal weights and metrics](../../../research/results/causal-plaba-checkpoint/manifest.json) · [Local verification](../../../research/results/causal-plaba-checkpoint/local-verification.json)

<a id="實驗性逐詞與提問指向候選"></a>
<a id="實驗性逐詞與疑問片語"></a>

## Experimental Word-Level and Question-Anchor Candidates

All scores share one forward pass on the text. The ease-of-reading and self-rated comprehension views also show word underlines. To compare lexical difficulty alone, select the lexical difficulty view under the original-text annotations. Alignable words receive colored backgrounds. These switches reuse the analysis. The question-anchor view assumes a question has been posed and ranks phrases that might be its subject using the text available at the end of each paragraph. Selecting a phrase shows its span and the visibility stop used to score it. [CompLex holdout replay](../../../research/results/stage5-complex/lexical-runtime-replay.json) · [IIRC phrase holdout](../iirc-anchor-run.md)

Switch to disambiguation-cue candidates to see up to three candidates per sentence. Color marks within-sentence rank. Hover over a candidate to see its ranking score, source character span, and word-final token index. Raw scores use only the text through the candidate word. Ranks compare eligible words in the full sentence. A word receives a score only when its end and the previous word's end align precisely. The view reuses the original forward pass and retrieves cached results when selected. [Readout release](../../../research/results/blott-word-l4/word-cue-release.json)

The human-evidence scope for cue localization is Blott’s English experimental sentences. On the 2021 held-out sentence frames, the researcher-labeled cue ranked first in 69.6% of cases, compared with 39.1% for the literal baseline. The first-alert false-positive threshold was missed, and the model did not reproduce the 2023 human sentence-end comprehension difference. On general text, candidates can direct manual review. Establishing where a reader notices or resolves a question requires position-specific observations. [Word-level localization, first alerts, and human keypresses](../semantic_tracking/blott-word-results.md)

Local replay of 88 sentences and 1,073 candidates from saved representations reproduced character offsets, tokens, ranks, and raw scores exactly. The actual L4 Gradio HTTP test analyzed two articles with one Qwen forward pass each. Six subsequent view changes reused the results and added zero forward passes. The second article added a paragraph containing CRLF and Unicode. The earlier sentence retained the same 10 candidates and offsets, with a maximum raw score difference of $6.8\times10^{-7}$. A local HTTP preview also verified layout and keyboard focus at desktop width 1,280 px and mobile width 390 px. [Actual L4 highlighting preview](../../../research/results/gradio-word-cue-route-l4/cue-preview.html) · [L4 HTTP verification](../../../research/results/gradio-word-cue-route-l4/verification.json) · [Local replay and HTTP](../../../research/results/gradio-word-cue-route-l4/local-verification.json) · [Browser verification](../../../research/results/gradio-word-cue-route-l4/browser/ui-visual.json)

The two-level view passed actual Gradio HTTP verification on Colab L4 using the same candidate checkpoint. Both sentence-level views show 26 sentence marks and 425 word marks on the same source, with matching token and character offsets. One additional word lacks an exact offset alignment and remains visible as original text. Analysis used one causal forward pass, and three view changes added zero passes. Separate synthetic browser tests checked underlining and absence of horizontal overflow at 1,280 px and 390 px. [L4 verification](../../../research/results/gradio-combined-overview-l4/verification.json) · [Actual response HTML](../../../research/results/gradio-combined-overview-l4/reading-overview.html) · [Release record](../../../research/results/gradio-combined-overview-l4/run-verification.json)

CompLex human holdout annotations apply to specified words or phrases. Replaying 1,094 specified targets from saved L4 features produced a mean absolute error of 0.08065 points. Full-text word scanning and question-anchor candidates that use earlier paragraphs remain experimental. [Scope of extrapolation](../stage-5-complex-run.md#held-out-results)

To look up an earlier referent, analyze the article and expand the phrase lookup panel. Paste the phrase from the original and run the lookup. If the phrase appears several times, select the intended instance by its source excerpt and character offset. The app highlights that phrase in yellow and marks up to five earlier candidate words in blue with their ranks. The query uses a 128-dimensional projection saved during the same analysis. Starting a new analysis from the earlier-text view clears the old lookup and returns to the ease-of-reading view with sentence shading and word underlines. [Earlier-text readout and original holdout](../semantic_tracking/arrau-bridge-results.md) · [Readout release](../../../research/results/arrau-bridge-l4/reference-release.json)

Earlier-text candidates were validated on human-specified ARRAU phrases. On PEAR, top-five hit rate was 34.4% and top-one hit rate was 9.5%. The interface shows candidates in rank order. Determining when a question opens or resolves in running text requires direct observations of those events.

The new interface passed actual Gradio HTTP verification on L4 with the same Qwen3.5-2B candidate. Pasting a three-paragraph article with repeated phrases and special characters was enough to obtain a sentence-level heatmap. Looking up two instances of the same phrase and switching back to the word heatmap preserved the source and lookup offsets. Together, those subsequent actions added zero forward passes. Across 823 evaluable ARRAU cases, candidate order and scores reproduced the sealed values exactly. Browser checks at 1,280 px and 390 px used synthetic state. [L4 path verification](../../../research/results/gradio-reference-l4/verification.json) · [Readout replay and browser verification](../../../research/results/gradio-reference-l4/local-verification.json) · [Release record](../../../research/results/gradio-reference-l4/run-verification.json)

The same candidate’s word marking passed the English MedReadMe held-out localization test. Its mean within-sentence average precision was 0.713, compared with 0.504 for word length and 0.565 for inverse frequency. The dataset contains difficult words and terms marked by trained annotators. [Holdout scope and L4 results](../stage6/medreadme-lexical-results.md)

Two IIRC holdouts for evidence arrival shared an independently fitted interval readout. The difference in interval negative log-likelihood and the within-document pair discrimination rate both missed their acceptance thresholds. The app’s question-anchor view presents a ranking of source phrases at paragraph ends. [Evidence-arrival holdout results](../iirc-linked-shared-head-results.md)

Sentence details include an experimental rating distribution. It predicts the proportions of readers assigning each rating from 1 through 5 and uses weights fitted with a rewrite-effect loss. Its evidence comes from out-of-fold PLABA development comparisons. The sentence heatmap uses the original mean-score readout. [Distribution evaluation](../stage-6-holdout-plan.md#incorporating-the-effect-of-rewriting-into-the-development-of-the-control-group)

The default research candidate passed Gradio request verification with the real Qwen3.5-2B on L4. The initial analysis used one model forward pass and all three subsequent types of selection or view change reused the result. After changing later paragraphs, the 128 shared earlier words remained aligned. The maximum score difference was 0.00000043 points. Rankings for all 430 IIRC paragraphs matched the earlier L4 experiment. The maximum difference between five-category probabilities for seven test sentences and probabilities computed from the original sealed L4 states was 0.00000024. These numbers measure reproducibility of the execution path. Each holdout evaluates its own human validity question. [L4 verification](../../../research/results/gradio-candidate-l4/verification.json) · [Phrase holdout scores](../../../research/results/gradio-candidate-l4/iirc-locked-scores.json)

A four-paragraph test article of 2,496 characters and 479 tokens also went through the actual Gradio HTTP path. An article-only request produced the sentence heatmap, and the same session could switch to the word heatmap. Both views preserved the entire original text and paragraph structure. Checks matched source offsets and end tokens for 26 sentences and 425 words. One additional unaligned word remained visible. The analysis made one forward pass and the cached switch added zero. [Long article verification program](../../../research/gradio_app/verify_article_route.py) · [L4 results](../../../research/results/gradio-article-route-l4/verification.json) · [Execution and release receipts](../../../research/results/gradio-article-route-l4/run-verification.json)

Run `python -m research.gradio_app.verify_candidate_evidence` to verify hashes for the candidate weights, the actual Gradio path, and the predictions for two human holdouts. The command reads sealed files without loading the language model. [Candidate evidence manifest](../../../research/results/gradio-candidate-l4/candidate-evidence.json) · [Human holdout results](../stage-6-holdout-plan.md#the-english-held-out-results-for-the-same-gradio-candidate)

On an NVIDIA L4 with CUDA PyTorch already configured, launch the research app with `--cuda` and the same snapshot path. Install the same versions of Gradio, NLTK, NumPy, and Transformers.

<a id="核對實際模型與介面"></a>

## Verify the Research Model and Interface

The full-text legend passed actual Gradio HTTP verification in the same Qwen3.5-2B process used for the CLEAR verification batch. Each of two articles used one forward pass. Eight queries or view changes added zero passes. The same candidate’s raw sentence scores were also evaluated on an independent document-level holdout. Adding the scores to surface features improved rankings of English expository texts, with an interval above zero. The interval for literature crossed zero. This comparison measures ranking at the whole-document level. [Actual Colab highlights](../../../research/results/clear-reading-ease-l4/preview.html) · [CLEAR holdout results](../stage6/clear-reading-ease-results.md)

`verify_overview_regressions` uses the candidate tokenizer, sentence-segmentation parameters, and readout weights to check paragraph-final abbreviations, preservation of the original, score legends, and view behavior when reanalyzing from the earlier-text view. Synthetic causal states exercise Gradio callbacks without loading Qwen.

The same candidate also passed the updated Gradio HTTP path on L4. The test first analyzed an article and selected an earlier phrase. It then submitted a new article with paragraph-final abbreviations, CRLF, emoji, and combining characters. The server returned the ease-of-reading view, cleared the previous query and selection, and redrew two sentence marks and 18 word marks. Source text and offsets were checked. Each article used one forward pass, and eight subsequent queries and view changes added zero passes. The preview preserves the actual highlighting response for the second article. [Highlighting preview](../../../research/results/gradio-overview-http-l4/preview.html) · [HTTP verification](../../../research/results/gradio-overview-http-l4/verification.json) · [Execution and release receipts](../../../research/results/gradio-overview-http-l4/run-verification.json)

```sh
.venv-cogllm/bin/python -m research.gradio_app.verify_overview_regressions \
  --checkpoint research/results/gradio-candidate-checkpoint \
  --tokenizer /path/to/15852e8c16360a2fea060d615a32b45270f8a8fc/tokenizer.json
```

Use PLABA-EVAL `text_01` to reproduce the scores. The 14 scores for seven sentences across two measures match the earlier research program exactly. One local Gradio 6.17.3 API analysis request triggers one Qwen forward pass and returns both measures, the original source marks, and the selected sentence location. This check establishes implementation and interface agreement on one development article. The earlier document-pair holdout provides the generalization evidence. [Research baseline holdout results](../training-protocol.md#已完成的首輪基線)

```sh
.venv-cogllm/bin/python -m research.gradio_app.verify \
  --model /path/to/15852e8c16360a2fea060d615a32b45270f8a8fc \
  --sentences /path/to/plaba-eval/sentences.csv
```

[Original research record](../../../research/gradio_app/README.md)
