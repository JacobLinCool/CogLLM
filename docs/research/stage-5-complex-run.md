# Stage 5 · CompLex context and training data comparisons

**Readout error on target words or phrases was lowest across all three training data sizes in the held-out set.** As readout increased to full data, error continued to decrease. For the same target across different sentences, the two contextual readouts showed 51.0% and 52.1% agreement in ranking under full data. [held-out results](../../research/results/stage5-complex/scores.json)

We fixed Qwen3.5-2B on Colab L4 to compare three input conditions, estimating the vocabulary difficulty of a target word or phrase after readers process the entire sentence. Published scores are the mean human ratings mapped to a 0–1 scale. All three conditions used this annotation, with empty reader backgrounds. [Original measurement](https://aclanthology.org/2021.semeval-1.1.pdf#page=3) · [Stage 5 design](stage-5-scope.md)

## Pre-comparison in the results

| Conditions | Representations read out from frozen models |
| --- | --- |
| Target word or phrase only | Given a specified word or phrase as the sole input, take the final token state. Forward pass is shared across identical source characters. |
| Read until the target. | In the causal forward pass of the entire sentence, extract the target final token state. Include only cases where the tokenization is fully consistent with truncation to the target end. |
| Read the full sentence. | Extract the target final state and the sentence-final state from the same forward pass. |

The first two conditions each apply L2 normalization to the 2,048-dimensional representation. The combined conditions normalize both states separately and concatenate them into a 4,096-dimensional vector, then divide by $\sqrt{2}$. All three groups use ridge regression with an unconstrained intercept, with predictions clipped to the 0–1 range. If there are $n$ labeled samples, the minimization objective is the sum of squared residuals plus $(n/8667)\lVert w\rVert_2^2$. The penalty coefficient is scaled according to the number of samples so that the three data sizes have identical normalized intensity under mean squared loss. The adjustment set score is for diagnostic purposes only.

After sorting by sentence group, nested sampling was performed with seed 0–4. 25%, 50%, and 100% utilized 1,901, 3,802, and 7,603 training sentence groups, respectively, sharing the same data under the same seed. The 100% training data was identical across all sub-samples, with each condition fitted once. Training, adjustment, and test sentence groups were mutually exclusive. All three readouts excluded 15 tokens encountering out-of-bound non-whitespace characters, leaving 8,667 training, 503 adjustment, and 1,094 test samples. Test scoring was retained locally and merged with frozen predictions after download. [Data and Boundary Audit](stage-5-scope.md#first-word-and-phrase-experiment)

The primary metric is the mean absolute error (MAE) on a 0–1 scale, where lower is better. Scores are first computed per seed and then averaged. We report overall, word-level, and phrase-level scores, as well as the proportion of target forms seen or unseen in the full training set. We evaluate the impact of poor conditions and limited data by performing 10,000 paired resamples on held-out sentence groups, fixing the sampling seed and the fitted model. Additionally, we examine the proportion of model rankings that align with human judgments for pairs of identical target strings across different sentences, where human scores vary. Ties are counted as half-points. This measurement preserves the original data's intact sentence structure.

## held-out results

The test set comprises 1,067 sentence groups and 1,094 target scores. The 25% and 50% training subsamples contain 2,148–2,203 and 4,294–4,368 scores, respectively. The table reports the mean absolute error on a 0–1 difficulty scale, where 0 indicates perfect agreement with the human mean. [Subsampling by seed](../../research/results/stage5-complex/replay.json) · [All scores](../../research/results/stage5-complex/scores.json)

| Visible content | 25% | 50% | 100% |
| --- | ---: | ---: | ---: |
| Target word or phrase only | **0.07286** | **0.07101** | **0.06988** |
| Read until the target. | 0.08804 | 0.08353 | 0.08065 |
| Read the full sentence. | 0.08910 | 0.08467 | 0.08281 |

With full data, the MAE for readout and complete sentence were each 0.01077 and 0.01293 higher, respectively, than using only the target. The matched 95% sentence bootstrap confidence intervals were [0.00743, 0.01415] and [0.00948, 0.01643]. The MAE differences for readout increasing from 50% to 100% were sequentially −0.00112, −0.00287, and −0.00186, with all matched intervals below 0. The three differences from increasing from 25% to 50% were also all below 0. The intervals reflect the variance of fixed readout on held-out material. [Matched difference](../../research/results/stage5-complex/scores.json)

Among the 124 targets seen in full training, the three MAEs were 0.07009, 0.07035, and 0.07048. The remaining 970 targets sequentially had MAEs of 0.06986, 0.08197, and 0.08438, with the primary source of the discrepancy in the two readout errors coming from this group. Whether a target has been seen is determined by word or phrase type and normalized target string. [Grouped Scoring](../../research/results/stage5-complex/scores.json)

219 pairs of identical source strings formed 1,050 human score contexts with varying ranks. Additionally, there were 17 human ties. Considering only the readout scores, which were identical across all pairs, assigning half a point for ties yielded 50%. The concordance rates for the readout at 25%, 50%, and 100% thresholds were 51.10%, 51.65%, and 51.00%, respectively, while the full sentence concordance rates were 51.35%, 51.68%, and 52.10%, respectively. This describes the existing sentence pairs. [Sentence Pairs and Ranking](../../research/results/stage5-complex/scores.json)

We read out the complete sentence and append the post-text state, employing a larger feature space. thus, we compare these three specific readout designs. Both the target-only and read-target groups estimate the same rating for human observation of the complete sentence. CompLex's validation unit is a word or phrase pre-specified in the data. The semantic question of whether a token is triggered or extinguished requires corresponding human observation at that specific time point.

## L4 Empirical Testing and Validation

Fixed BF16 weights with FP32 forward pass, linear readout fitted to FP64. 15 causal and positive control checks passed, where the maximum relative RMS for comparisons maintaining identical state is $2.29\times10^{-6}$. A total of 9,166 sentences and 5,437 independent target strings were processed, summing to 333,423 non-padding tokens. Batching by length, with a maximum of 8 inputs per batch. Forward and readout took 197.60 seconds, averaging 1,687 tokens/second. The extraction phase including load, check, and write took 262.55 seconds, with a CUDA peak retention of 7.51 GiB. 33 fit stages took 10.44 seconds, and the solve loop took 5.70 seconds. [Environment](../../research/results/stage5-complex/features/environment.json) · [Causal Check](../../research/results/stage5-complex/features/causality.json) · [Extraction](../../research/results/stage5-complex/features/extraction.json) · [Fit](../../research/results/stage5-complex/fit/fitting.json)

The initial startup failed with a disconnection when the working directory was set via CLI, and the experiment was not yet submitted. The reconfigured L4 completed its formal execution. the configuration response time to stop request was 362.69 seconds. Both work stages were confirmed released. Account sampling was recorded before the first configuration, between the two configurations, and after formal execution release as 768.60, 768.56, and 768.47 CU, respectively, with an execution rate of 1.94 CU/hour displayed. The balance represents the account readout. [Formal Execution and Release](../../research/results/stage5-complex/manifest.json) · [First Release](../../research/results/stage5-complex/failed-start-manifest.json) · [Account Sampling](../../research/results/stage5-complex/usage.jsonl)

Reconstructed input and recalculated scores match byte-for-byte. Independently validated nested sampling and ridge solutions against training labels, then replayed 49,230 held-out predictions from saved features and weights. Max absolute difference $4.44\times10^{-16}$. [Input hash](../../research/results/stage5-complex/input-audit.json) · [Replay](../../research/results/stage5-complex/replay.json)

## Reproduce the entry point

1. Read the fixed version with `research.stage5.complex_prepare` to generate locatable stimuli and observations. Reconstruct the common input and the masked test score `design.jsonl` of the three conditions using `research.stage5.complex_inputs` and the fixed version's `tokenizer.json`. [CompLex fixed version](https://github.com/MMU-TDMLab/CompLex/tree/4b31c2bd34da4c1265f9db3d727d95ee23099f14) · [Qwen3.5-2B fixed revision](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)
2. In L4, use `research.stage5.complex_extract` to read `stimuli.jsonl` and `targets.jsonl`, output `features.npz`. Use `research.stage5.complex_fit` to read `design.jsonl` and `features.npz`, output readout, no-issue, prefix, source family, random seed, matching, nats, held-out, anchor span, response distribution, documentation and retention constraints.
3. On this machine, verify the data sampling and post-download weights using `research.stage5.complex_verify`. Use `research.stage5.complex_score` to read `design.jsonl`, the raw `observations.jsonl`, and the `predictions.npz` of L4, and recalculate the held-out scores.

[Original research record](../../research/stage-5-complex-run.md)
