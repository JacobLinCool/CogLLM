# Stage 3 initial Colab L4 experiment

**The initial run is complete.** Qwen3.5-2B passed the FP32 prefix check on Colab L4 and extracted each passage's features in one causal forward pass. In five-fold PLABA-EVAL evaluation, sentence-level self-rated comprehension MAE fell from **0.732** for the length baseline to **0.616**. Article families and readers had no cross-split overlap within each validation fold. [L4 metrics](../../research/results/stage3-l4/float32/metrics.json) · [Independent replay](../../research/results/stage3-l4/float32/verification.json)

## sentence-level self-assessment

Freeze the official Qwen3.5-2B fixed weights. The model input consists solely of the full text, with each article processed exactly once. Focus states are extracted from the final layer at the end of each sentence and the end of the document, concatenated into a 4,096-dimensional vector, and then L2 normalized. These two states are used to simulate reader ratings after reviewing the full text. Evidence sentences, reader IDs, ground truth labels, and version tags are excluded from the model. Additionally, a length + version baseline is provided with known original/vernacular labels to measure the variance explained by these labels. [readout implementation](../../research/stage3/run.py) · [fixed model version](https://huggingface.co/Qwen/Qwen3.5-2B/tree/15852e8c16360a2fea060d615a32b45270f8a8fc)

We aggregated 1,826 ratings across 609 sentences and used the revised five-fold splits from Stage 2. The 39 article families contain 78 versions and 220 readers. Each fold used its training data to estimate the mean, normalize length, and fit the ridge weights with fixed `λ=1`. We report sentence-level mean absolute error on the 1–5 scale. Lower values are better. [Folds and scores](../../research/results/stage3-l4/float32/metrics.json) · [Held-out predictions by sentence](../../research/results/stage3-l4/float32/predictions.jsonl)

| Input features | Ease of handling MAE ↓ | Understand self-assessed MAE ↓ |
| --- | ---: | ---: |
| Training mean fold | 0.737 | 0.813 |
| Length | 0.632 | 0.732 |
| Length + known version | 0.613 | 0.687 |
| Freeze Qwen | 0.660 | 0.658 |
| Freeze Qwen + length | **0.593** | **0.616** |

After fixing the held-out predictions, perform 10,000 paired bootstrap resamples across 25 shared reader-connected article groups. The MAE difference of Qwen+ relative to Length is Ease of Processing −0.039 (95% CI [−0.072, −0.005]) and Understanding Self-Assessment −0.116 ([−0.152, −0.080]). Relative to Length+Known Version, Understanding Self-Assessment remains 0.071 points lower (difference −0.071, CI [−0.110, −0.030]). The confidence interval for the Ease of Processing difference crosses zero. Resampling uses the fixed predictions without retraining each fold. Features downloaded from Colab are refitted locally. the maximum difference in all MAEs is 1.22×10⁻¹⁵ points. [Resampling and Verification](../../research/results/stage3-l4/float32/verification.json)

Across 265 sentence pairs of original and vernacular text alignments, the mean human self-assessed understanding difference was higher by **0.723** points for the vernacular version. The Qwen + length held-out prediction estimate was higher by **0.529** points. The 95% confidence interval for the difference between the predicted difference and the human difference was [−0.431, +0.006] points, which is insufficient to confirm calibration of the rewriting magnitude. This comparison first averaged the one-to-many vernacular sentences and then aligned positions with equal weighting. These two endpoints address sentence-level scores and the rewriting difference from the same source. [Rewriting difference recalculated](../../research/results/stage3-l4/float32/verification.json)

## Prefix testing and resources

Three prefix groups, each consisting of the first 32, 65, and 97 tokens of a shared 216-token test prefix. Compare post-suffixes via same-length replacement, truncation, and batch right-padding. The metric is the maximum per-token relative RMS difference across the shared prefix, with thresholds of 0.002, 0.02, and 0.02 respectively. Conduct a positive control on same-length replacement post-suffixes to confirm that the new content genuinely alters the post-suffix state. Additionally, using a fixed tokenizer, validate the 609 suffix prefixes and body offsets at sentence boundaries.

| Computational accuracy | Subsequent replacement | Truncate the maximum difference | Right-padded to the maximum difference | Disposal |
| --- | ---: | ---: | ---: | --- |
| BF16 | 0 | 0.02416 | 0.02174 | Stop at a value exceeding the predetermined threshold before formal sampling. |
| FP32 | 0 | 4.24×10⁻⁶ | 2.66×10⁻⁶ | Through the same threshold, complete the extraction and five-fold cross-validation. |

Two runs utilized NVIDIA L4 (approximately 22.0 GiB), PyTorch 2.14.1 plus CUDA 13.0, Transformers 5.17.0, and SDPA. FP32 converts official BF16 weights to FP32 computation. matrix operations adopt the highest FP32 precision, with cuDNN TF32 disabled. The reduction in discrepancy aligns with shape-dependent numerical errors under low precision. Causal validation applies to the prefix backbone state, while sentence-level readout additionally employs the state after full completion. [BF16 original validation](../../research/results/stage3-l4/bfloat16/causality.json) · [FP32 original validation](../../research/results/stage3-l4/float32/causality.json) · [computation environment](../../research/results/stage3-l4/float32/environment.json)

FP32 processes up to four documents per batch, performing 20 forward passes over 78 documents, totaling 17,140 non-padding tokens. Synchronous extraction and feature retrieval take 7.33 seconds, approximately **2,339 token/s**, excluding installation, download, and periodic checkpointing. CUDA peak memory usage is **7.23 GiB** with a retention of **7.60 GiB**. The time from configuration completion to sending a stop request is 74 seconds and 83 seconds for the two runs, both confirming release. Colab account usage shows 768.80 CU before and after both runs. the sampled single active allocation rate is 1.94 CU/hour, and a single charge cannot be distinguished from the account checkpoint. [Extraction metrics](../../research/results/stage3-l4/float32/extraction.json) · [BF16 usage](../../research/results/stage3-l4/bfloat16/usage.jsonl) · [FP32 usage](../../research/results/stage3-l4/float32/usage.jsonl) · [FP32 release log](../../research/results/stage3-l4/float32/manifest.json)

## Retraining and subsequent validation scope

Executed at L4 using PyTorch and Transformers 5.17.0. Prepare [Stage 2 PLABA-EVAL data](stage-2-data.md), with remote data defaults to project-assigned local disk paths. The following can be directly executed for extraction and readout. the Colab workspace package entry is also provided. [Package entry](../../research/stage3/colab_entry.py) · [Trained readout weights](../../research/results/stage3-l4/float32/head.npz)

```bash
python -m research.stage3.run \
  --data /content/cogllm-stage3/data \
  --output /content/cogllm-stage3/outputs \
  --dtype float32
```

In this round, support for the same English biomedical abstract corpus showed no article family, no reader full-text sentence-level self-assessment. Parallel data verification found ROAMM's key report and backtracking selection, with the endpoint being self-attention interruption. Questions remain regarding the semantic localization of cross-sentence pointing and resolution, requiring corresponding human observation. [Parallel data verification](stage-1/data-selection.md#stage-3-parallel-verification-reading-report-data)

## Subsequent experiments on cache features

Fixed the same development fold, using the same L4 cache: take the last 2,048 dimensions and the first 2,048 dimensions. Apply existing L2 normalization and ridge evaluation individually or combined. standardize the length features across each fold.

| The Qwen features used are all length-inclusive. | Ease of handling MAE ↓ | Understand self-assessed MAE ↓ |
| --- | ---: | ---: |
| End of sentence | 0.577 | 0.604 |
| End of passage | 0.623 | 0.664 |
| End-of-sentence and end-of-document | 0.593 | 0.616 |

The end-of-sentence estimate is lower. The end-of-sentence reduction in merged self-assessed MAE error is −0.011, with a 95% confidence interval from resampling across 25 reader groups of [−0.027, +0.002]. Within the same article, when there is a difference in human self-assessment between two sentences, the proportion of sentence-end + length yielding the same ranking is 69.0%, while using only length yields 63.0%. Each prediction is counted as half a point, with a denominator of 1,829 sentence pairs. These results make the end-of-sentence feature a viable starting point for the next round. [End-of-sentence vs. End-of-document comparison](../../research/results/stage3-l4/float32/feature-ablation.json)

** sentence-level uncertain report.** Additionally, 858 structured reports (452 uncertain, 406 certain, from 286 sentences) were fitted with a binary logistic ridge model, with each human report receiving equal weighting, an L2 coefficient of 1, and an intercept that was not penalized. The 968 free-text interface records from structured reports that were not tested did not enter this endpoint. Using the same five-fold, the average AUROC within the folds at sentence end plus length was 0.757, with a Brier score of 0.201. the corresponding length baseline was 0.685 and 0.226. When sentence end and document end were combined, the results were 0.758 and 0.199. The Brier score represents the mean squared error of the cumulative probability and human binary reports, where lower is better. This is the same corpus development result used for the App sentence-level uncertain report experiment. [Uncertain report experiment](../../research/results/stage3-l4/float32/issue-diagnostic.json)

Subsequent analysis used the same split to compare surprisal, intermediate layer logits, and attention for "no-issue" and "ambiguous terminology." The scope of formal release requires independent validation. [Feature comparison report](stage-3-feature-ablation.md)

[Original research record](../../research/stage-3-first-run.md)
