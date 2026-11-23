# Stage 4: Supervised effect of Chinese sentence-initial pronouns

Weak supervision reduced the held-out material interaction effect error by 5.21 percentage points, but only replicated a small gap. The human two-material shift difference was +24.46 percentage points, with readout changing from −4.09 to +1.12. The sequential Brier difference was +0.00104, with material resampling intervals crossing zero. [Full results](../../research/results/stage4-zhan-weak/metrics.json)

This round of validation tests whether weak supervision enables readout to capture distinct reference shift patterns in two Chinese corpora following the appearance of a subject pronoun at the sentence initial position. We fixed the Stage 3 weights of Qwen3.5-2B and used five-fold cross-validation. For each material, within a single causal forward pass, we extracted the final layer state from the position immediately preceding the period at the end of the previous sentence and the position immediately following the pronoun in the next sentence, then computed the readout probability of selecting the previous sentence's subject as the continuation. We reused the same 36 material sets and 1,651 existing trials across the same 50 readers for all folds. [Stage 3 baseline](stage-3-zhan-landmark.md) · [Original study Fig. 1](https://doi.org/10.1371/journal.pone.0237012.g001)

## Pre-fixed comparison results

For each fold, 4 directly labeled instances were obtained per training material and prompt condition. A fixed label sampling of 12 was generated using seeds 0–11. A small number of direct and weakly supervised groups shared the same sequential labels. The weakly supervised group additionally used the remaining training responses from that fold to compute a **single aggregated effect**. Full direct supervision utilized all valid responses from that fold. All three groups employed the same input, a shared logistic readout, and identical normalization strength. Responses from the held-out material were used solely for final scoring.

The difference between the two conditions within each material is the subject re-mention rate with pronouns minus the subject re-mention rate with free cues. Differences across verb classes are averaged using material equal weighting. The single main effect $I$ is the shift of IC-2 minus the shift of IC-1. Model probabilities $\hat p$ are calculated following the same rule to yield $\hat I$. Verb types are used to aggregate this difference. The original study included both hierarchical models and reader-aggregated results. the current round targets the recalculated effect using material equal weighting. [Original study Experiment 1, Interpretation biases](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0237012#sec008) · [effect loss specification](mathematical-formulation.md#從文獻效應回傳梯度)

For each fold, read out the L2-normalized final-layer vectors at each position and minimize

$$
\mathcal L
= \frac{\mathrm{NLL}_{\mathrm{direct}}}{N_{\mathrm{direct}}}
+ \frac{\lVert w\rVert_2^2}{2 N_{\mathrm{full,train}}}
+ \frac{\lambda}{2}(\hat I-I)^2 .
$$

Intercept no penalty. Weakly supervised fixed $\lambda=1$, two groups directly supervised $\lambda=0$. Same denominator $N_{\mathrm{full,train}}$ fixed, three groups' average likelihood normalized for comparison. Using Colab L4 official FP32 representation, then training and verifying convergence with double precision readout. Qwen frozen throughout.

The sequential Brier score and the absolute error of the primary interaction effect serve as two common criteria. additionally, report the mean absolute errors for the Brier score under each condition, the two types of bias, and the material-specific bias. First, compute the score for each resample, then average across the 12 resamples. The resampling procedure retains two prompts, verb stratification, the same held-out set for prediction, and the reader sample. The primary comparison assesses whether weak supervision reduces the interaction effect error and how the sequential probability error changes. This constitutes a held-out material control from the same study.

## held-out results

| Supervised conditions | Sequential Brier ↓ | IC-1 shift | IC-2 shift | IC-2 − IC-1 | The absolute error of the interaction effect decreased. |
| --- | ---: | ---: | ---: | ---: | ---: |
| human observation |  | +8.68 pp | +33.14 pp | +24.46 pp |  |
| Small sequential | 0.21085 | +24.94 pp | +20.85 pp | −4.09 pp | 28.55 pp |
| Small-step + cumulative effect | 0.21189 | +22.63 pp | +23.75 pp | +1.12 pp | 23.34 pp |
| Complete sequential | 0.20714 | +25.25 pp | +21.07 pp | −4.18 pp | 28.64 pp |

For each fold, a small number of samples used 224–232, while full conditions used 1,288–1,329. For the same set of small labels, the absolute difference in effect for weak supervision was −5.21 percentage points (paired materials 95% interval [−6.38, −4.02] percentage points). Across 12 sampling runs, weak supervision showed a positive interaction effect in 8, while small number sequential showed 1. The sequential Brier increase for weak supervision was 0.00104 ([−0.00059, +0.00277]), and the sequential material offset MAE decreased from 17.14 to 15.75 percentage points. Full sequential labels had the lowest sequential Brier across three groups, yet the average condition effect remained biased toward human. [Sampling results and resampling intervals](../../research/results/stage4-zhan-weak/metrics.json) · [Sequential material held-out probability](../../research/results/stage4-zhan-weak/predictions.jsonl)

The weakly supervised in-fold effect average was +24.48 percentage points. the fitted in-fold effect average was +16.21, and after merging held-out material, only +1.12 remained. This round's support condition can shift toward the human direction via aggregated effects, with only modest improvement observed on unseen material. The next study family must retain independence to examine transfer. This batch of existing data measures reference selection under two prompts, and questions regarding emergence and resolution require corresponding observation. [Training Diagnosis](../../research/results/stage4-zhan-weak/fitting.json)

## Resources and Verification

Colab NVIDIA L4 utilizes PyTorch 2.14.1, Transformers 5.17.0, FP32 backbone, and SDPA. 10 major batches process 36 groups totaling 297 tokens, with forward and readout latency approximately 5.15 seconds. peak CUDA memory configuration is approximately 7.06 GiB. 36 groups of truncated controls all pass, with maximum relative RMS difference 4.35×10⁻⁶. 125 FP64 readouts approximate 36.13 seconds. peak CUDA memory configuration is approximately 20.2 MiB. Configuration to stop generation request takes approximately 114.7 seconds, after which release is confirmed. Active account shows 1.94 CU/hour, with usage checkpoints at 768.80 CU. single charge cannot be determined from these checkpoints. [Resource Log](../../research/results/stage4-zhan-weak/manifest.json) · [Extracted Results](../../research/results/stage4-zhan-weak/extraction.json) · [Usage](../../research/results/stage4-zhan-weak/usage.jsonl)

Invert all held-out answers for each fold while keeping the fold-wise label counts and effect targets unchanged. After downloading, recompute gradients, effects, and held-out probabilities across all 125 weight sets using NumPy. the maximum probability difference is 2.22×10⁻¹⁶. Re-running both score outputs yields byte-identical results. [Verification and Hash](../../research/results/stage4-zhan-weak/verification.json)

## Re-run

On this machine, reconstruct the cumulative effect of sequential label counts and remaining responses using [Stage 2 data](stage-2-data.md), retaining the complete sequential responses. Transfer `design.json` and `stimuli.jsonl` to the L4 data directory.

```bash
python -m research.stage4.zhan_weak prepare \
  --data .cache/cogllm/stage2/zhan \
  --output .cache/cogllm/stage4/design.json
```

First extract the state at L4, then train the readout.

```bash
python -m research.stage4.zhan_features \
  --data /content/cogllm/data \
  --output /content/cogllm/outputs
python -m research.stage4.zhan_weak fit \
  --design /content/cogllm/data/design.json \
  --features /content/cogllm/outputs/features.npz \
  --output /content/cogllm/outputs --device cuda
```

Download `heads.npz`, using full human reaction scores.

```bash
python -m research.stage4.zhan_weak_score \
  --data .cache/cogllm/stage2/zhan \
  --heads /path/to/downloaded/heads.npz \
  --output /path/to/results
```

[Original research record](../../research/stage-4-zhan-weak-supervision.md)
