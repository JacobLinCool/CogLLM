# Stage 3 · Feature ablations for sentence-level difficulty

Freeze Qwen3.5-2B and perform a fixed five-fold PLABA-EVAL comparison of three feature types on readout gains. The sequential Brier scores for end-state length are "no-issue" **0.2009** and "ambiguous terminology" **0.1987**. After adding end-layer surprisal, both point estimates decrease by approximately 0.0023, with paired resampling intervals crossing zero. Subsequent comparisons follow the end-state length approach. [Full metrics](../../research/results/stage3-features/metrics.json)

## Observation and Methodology

Utilized structured selections from 108 existing readers across 19 article families, totaling 286 sentences and 858 structured selections. Of these, 452 reports indicated at least one difficulty, 406 explicitly reported no difficulty, and 335 selections were flagged as unclear terminology. The terminology endpoint recorded whether the item was selected. The data reading configuration allowed subjects to consult the full text. model input was the body text with no reader background. Training and validation folds were partitioned by Stage 2 reader connectivity groups, ensuring no overlap between article families and readers. [Data Specifications](stage-2-data.md) · [held-out sentence-level prediction](../../research/results/stage3-features/predictions.jsonl)

Each article obtains the state and attention at each position via a single causal backbone forward pass without per-token generation. The end-of-sentence baseline uses the L2-normalized 2,048-dimensional final layer state, augmented with the log-transformed character count and whitespace token count. The three appended feature groups are reduced to the sentence level. [Feature Definition](../../research/stage3/feature_signals.py)

| Feature | sentence-level value |
| --- | --- |
| Last layer surprisal | Compute the negative log-likelihood of the current token conditioned on the previous position, then calculate the sentence-level average and maximum. |
| Layer 12 logits | Project the layer states through the final layer normalization and shared vocabulary projection, then compute the in-sentence average surprisal and average next-token distribution entropy. |
| Attention summarization | We compute the average across attention heads and layers of the end-of-sequence query's earlier text weights, recent eight-token weights, and prefix-length-normalized entropy. |

Each condition used a binary ridge logistic model with the L2 coefficient fixed at 1 and an unpenalized intercept. Length and additional features were standardized using training-fold means and standard deviations. For conditions with sentence-final representations, length and each additional feature block were divided by the square root of the block's dimensionality. The primary measure was the mean squared error between individual human selections and held-out probabilities (Brier score, where lower is better). We also report AUROC with equal weighting across five folds, and pairwise ranking for sentences in the same passage with differing observed selection levels. Ties count as half. Brier differences use 10,000 paired resamples of the 13 groups linked by shared readers that contain checklist data, with held-out predictions fixed. The following are descriptive 95% intervals without a multiple-comparison adjustment. [Scoring implementation](../../research/stage3/issue_probe.py) · [Fold results and paired intervals](../../research/results/stage3-features/metrics.json)

## Results

| Readout features | There is a difficult Brier ↓ | Ambiguous terminology Brier ↓ |
| --- | ---: | ---: |
| Length | 0.2262 | 0.2213 |
| Length + Original/vernacular version tag | 0.2125 | 0.2008 |
| End-of-sentence state + length | 0.2009 | 0.1987 |
| Baseline + final layer surprisal | 0.1986 | 0.1965 |
| Baseline + 12th layer logits | 0.2028 | 0.2032 |
| Baseline + 12th layer logits + final layer surprisal | 0.2008 | 0.2011 |
| Baseline + Attention Summarization | 0.2016 | 0.2000 |
| Baseline + all three feature groups | 0.2019 | 0.2023 |

The Brier score with surprisal was questionable at −0.00229 [−0.00542, +0.00196] and unclear at −0.00225 [−0.00491, +0.00116]. The point estimates for sentence-pair ranking increased from 72.8% to 74.9% and from 70.2% to 70.8%, respectively. Current evidence is insufficient to confirm that the lower Brier score replicates in other splits of the same corpus. [Main comparison](../../research/results/stage3-features/metrics.json)

The attention weights for the prefix context and recent tokens, relative to uniform weights over the same prefix, are 0.94 and 0.81, respectively. Calculated using the zero-based prefix token index `s` and suffix index `e`, the uniform weights are `s / (e + 1)` and `min(8, e + 1) / (e + 1)`. Additional analysis incorporates both conditions with sentence order, first-sentence flag, intra-sentence token count, visible prefix token count, and both uniform weights. With attention included, the Brier score is ambiguous −0.00041 [−0.00535, +0.00489] and unclear +0.00020 [−0.00377, +0.00388]. The conclusion covers the three attention summaries from this test round. [Position control results](../../research/results/stage3-features/position-sensitivity.json)

In this round, the labels align to the scoring sentence. Selecting results that support sentence-level uncertainty analysis. The first occurrence of cross-sentence questions, the targeted text, and the timing of resolution each require corresponding positional or temporal prior human observations.

## L4 measurement and re-run

We ran extraction on an NVIDIA L4 with a pinned Qwen revision, Transformers 5.17.0, FP32 computation, and eager attention. At three prefix lengths, we tested equal-length suffix replacement, truncation, and right padding. All nine checks passed. The maximum relative RMS differences for prefix states in the final layer and layer 12 were 4.00×10⁻⁶ and 4.21×10⁻⁶, respectively. The maximum absolute difference for attention was 5.01×10⁻⁶. The full extraction covered 78 articles, 609 sentences, and 17,140 non-padding tokens. The 20 batches took 18.61 seconds in total (about 921 tokens/s), with 7.29 GiB of peak allocated CUDA memory. The reported timing covers backbone computation, feature readout, and synchronization and excludes model loading, prefix checks, and cache writes. [Prefix checks](../../research/results/stage3-features/causality.json) · [Environment](../../research/results/stage3-features/environment.json) · [Extraction measurements](../../research/results/stage3-features/extraction.json)

After the original cloud entry is completed and extracted, it proceeds to scoring. The simplified upload data lacks the `processing_ease` fields required by the shared loader, causing the cloud scoring to abort. On the local machine, the complete features downloaded are combined with Stage 2 data for imputation. the independently rerun metrics align with sentence-level predictions. L4 transitions from configuration completion to sending a stop request in approximately 130 seconds, and resource release is confirmed post-stop. During the sampling period, the account shows a usage of 768.80 CU with a precision of 0.01 CU. a single charge cannot be identified by the account breakpoint. [Validation and Hash](../../research/results/stage3-features/verification.json) · [L4 Work Log](../../research/results/stage3-features/manifest.json) · [Usage](../../research/results/stage3-features/usage.jsonl)

Currently, the entry at L4 extracts only body features, requiring `stimuli.jsonl`. After downloading `features.npz`, perform scoring locally while retaining the complete Stage 2 PLABA-EVAL dataset. Cloud work records should retain the hash of the entry version at the time.

Execute at L4

```bash
python -m research.stage3.ablation_run \
  --data /content/cogllm/data \
  --output /content/cogllm/outputs
```

Score on the full dataset on this machine

```bash
python -m research.stage3.issue_probe \
  --data .cache/cogllm/stage2/plaba \
  --features /path/to/downloaded/features.npz \
  --output /path/to/results
```

[Original research record](../../research/stage-3-feature-ablation.md)
