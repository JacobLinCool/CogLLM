# No resolution of the question, the continuation is a contrast

There are 12 self-generated English topics, each with three versions, totaling 36 passages. The narrative, experiment log, and assignment announcement categories each contain four topics. Each passage consists of five sentences, with a paragraph break after the third sentence. The first two sentences establish an identical unresolved question. The early-resolution version and the delayed-resolution version only swap the third and fifth sentences. the resolution sentence is verbatim, and the fourth sentence retains content related to the original question without providing an answer. The abrupt-transition version and the delayed-resolution version share the first paragraph and paragraph breaks, but the next sentence shifts to an unrelated topic. These materials are pre-designed model behavior baselines with no human evaluation. [Original text and construction](../../../research/semantic_tracking/load_curve_controls.py)

```sh
python3 -m research.semantic_tracking.load_curve_controls --output /path/to/controls
```

| File | Purpose |
| --- | --- |
| `inputs.jsonl` | Forward readout `id` with original `text`. Original contains actual segmentation. |
| `controls.jsonl` | sealed snapshot aligned post-prediction with `id`, encompassing topic, version, query description, and pre-specified character positions. |

Position is the Python Unicode character index. Phrase uses half-open interval `[start, end)`. `opening` denotes the phrase introducing a question, `cue` provides clues for answering, `shift` is a hint to another topic, and the latter two may be empty depending on the version. `common_prefix_end` marks the end of the second sentence, `paragraph_end` the end of the third sentence, `continuation_start` the position before the first character of the next paragraph, including the preceding two newlines. `before_cue_end` stops at the last non-whitespace character before the clue. `sentence_ends` preserves the five sentence ends, and `late_comparison_end` equals the end of the fifth sentence. Verified against the fixed Qwen tokenizer, all sentence ends, paragraph breaks, positions immediately before and after clues, and phrase boundaries align precisely. The phrase start may cut into a token containing left-side whitespace, and the prefix reading stop must use the verified character endpoint. [Score comparison](../../../research/semantic_tracking/load_curve_control_score.py)

## Comparison definition

Report the load score for every token across all methods. If a continuation quality with a preceding question mark has already appeared in the same prefix, report it collectively with `cohort`. The overall high or low can be verified by comparing the scores after the transition. To interpret this as the original question remaining unresolved, a fixed question group `cohort` result is required. The token ID and position on the same prefix must be identical, with values allowing an absolute difference `1e-6` and a relative difference `1e-5`. Each topic compares the early resolution against the first two sentences of the other two versions, and the delayed resolution against the entire first paragraph and line break of the abrupt version. If the technical check has not passed, the threshold for the effect of this method is considered not passed.

| Comparison | Pre-specified difference |
| --- | --- |
| Is the end of the section still unresolved? | The delayed-answer variant at the end of the third sentence is subtracted from the early-answer variant. |
| Subsequent answers and abrupt transitions | The abrupt transition at the end of the fifth sentence, featuring a delayed response variant. |
| Does the decline appear in the solution trace? | The delayed-answer variant exhibits a reduction in the fifth-sentence answer cue, comparable to the multi-token reduction at the end of the fourth sentence. The fourth sentence remains centered on the original query. |
| When encountering abrupt scene transitions, do we still retain unresolved questions? | The minimum value from the end of the fourth sentence's self-turning phrase to the sentence end, reduced by 80% of the third sentence's end value. The end value of the third sentence for each topic must be greater than zero. |

Pairwise differences were matched within the topic, and resampling with equal weighting was performed 10,000 times per topic. The lower bound of the 95% percentile interval must exceed zero. An 80% threshold was used for this control. Additionally, the position of the first pair of curves exceeding the allowable error margin relative to the anchor span was reported. Adjacent tokens following the edit in the two continuations may differ in word, and this distance describes when the divergence occurs. This set of stimuli and resampling measures behavior only on self-generated topics. [Implementation comparison](../../../research/semantic_tracking/load_curve_control_score.py)

## The scope where human data is available for verification

- **Onea et al.** The original workbook's opening and resolution were original lines verified by different annotators after reading. Readers' word-by-word uncertainty was measured separately, with multiple lines visible on the same screen. Coordinates can be used to verify parallelism between questions. The model reweighting for the story translation remains to be confirmed. [Research §§3–4](https://doi.org/10.1017/langcog.2025.10057) · [Data](https://osf.io/m524v/)
- **Frick and Schüler.** A single reader evaluates whether the current sentence aligns with the earlier text sentence by sentence. Responses indicating agreement are recorded only for the current sentence. The full trajectory can be cross-checked against the initial sentence-level report, the spreadsheet, and the conditions for stimulus reuse. [Experiment 1](https://doi.org/10.3758/s13421-026-01900-7) · [Source Family Verification](frick-consistency-source.md)
- **Blott et al.** Materials under CC BY 4.0 are answerable for word-sense disambiguation at the end of sentences. Answer cues include original coordinates. humans were not rated before or after, nor were there persistent item curves. These materials were previously used in this project's evaluation. [Experiment 1](https://doi.org/10.1080/23273798.2022.2159992) · [Existing Source Verification](blott-format-source.md)

[Original research record](../../../research/semantic_tracking/load-curve-controls.md)
