# The English word-guess readout failed to reproduce the first correct response curve of the German ACT

On 20 German ACT word lists and 1,999 evaluable historical trials, the fixed English readout Qwen3.5-2B model achieved a 75.84% trial accuracy when reading to the sixth cue, compared to 17.76% for human readers. This represents a +58.09 percentage point difference, with a word list paired 95% confidence interval of [51.07, 64.55], exceeding the ±5 percentage point predetermined allowance. This external test used original experiment IDs to identify the cue word for each exposure. [Public Materials and Alignment](act-source.md) · [Pre-fixed Analysis](act-transfer-protocol.md) · [Full Scoring](../../../research/results/act-transfer-l4/results.json)

## held-out results

| Analysis | Step count NLL only | Add literal features NLL | Add Qwen NLL | Qwen's improvement relative to steps and vocabulary matching 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Six steps, fully following English readout. | 1.40299 | 1.61601 | 1.53646 | −0.13347 [−0.14848, −0.11741] |
| In fifteen steps, the ACT re-evaluates the baseline rate for each word in the other vocabulary. | 2.40275 | 2.40277 | 2.40315 | −0.00040 [−0.00373, +0.00270] |
| As a six-step distribution, the truncated fifteen-step fold-out distribution. | 0.76001 | 0.76001 | 0.75984 | +0.00017 [−0.00082, +0.00127] |

NLL was computed with equal weighting over the vocabulary, in units of nats/trial, where lower values indicate higher human outcome probabilities. In the main analysis, adding Qwen yielded a relative readout improvement of 0.07955 nats/trial [0.07082, 0.08857]. performance was worse than the no-issue baseline using only step counts and failed to meet the threshold requiring improvement in both metrics. The cumulative accuracy at steps 1, 3, and 6 was overestimated by 8.42, 34.34, and 58.09 percentage points, respectively. all three failed the magnitude acceptance test. [Six-step complete interval](../../../research/results/act-transfer-l4/results.json)

In the supplementary analysis, each fold used only the other 16 word lists to estimate the baseline probability of a correct response at each step and retained the text-feature coefficients fitted to English. Across fifteen steps, the intervals for both NLL comparisons with Qwen included zero. A retrospective analysis truncated the same out-of-fold predictions at the sixth step and added the probability mass from the seventh step onward to the right-censored outcome. Qwen's estimate at the sixth step fell to 18.80%, a difference of +1.04 percentage points from the human rate. The interval [−6.27, +7.68] still extended beyond the prespecified tolerance. With exposure matched, adding text representations still yielded a small gain whose interval included zero. [Diagnostic over the same six steps](../../../research/results/act-transfer-l4/matched-six-diagnostic.json)

These results support the calibration of item-level accuracy and the ranking across time points between vocabularies. ACT can verify the point at which a reader first answers correctly after a known cue. In general, questions in articles can be formed or disappear without an explicit response, an event not observed in ACT.

## Reproducibility and resourcefulness

Rebuild the model vocabulary with `act_prepare stimuli --horizon 15` and verify 300 causal reading stops with `beukers_extract preflight`. Run a forward pass through each vocabulary with the same Gradio candidate. 20 vocabularies yield 1,186 tokens. The L4 forward and readout take 0.59 seconds, with peak GPU retention of approximately 7.51 GiB, and configuration to release takes 109.92 seconds. All 12 causal controls pass. Use `act_transfer_score forecast` to first perform a sealed snapshot prediction, then combine with `score` to merge local historical human results. Re-run the sealed snapshot and scoring bit-for-bit, and verify intercepts, probability mass, NLL, and cumulative bias using a scalar algorithm. [Scoring Program](../../../research/semantic_tracking/act_transfer_score.py) · [Execution and Release Log](../../../research/results/act-transfer-l4/run-verification.json)

[Original research record](../../../research/semantic_tracking/act-transfer-results.md)
