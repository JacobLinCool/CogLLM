# Joint training with supervision from published effects

**With the addition of two literature effects, the shared model aligns more closely with sentence-level understanding self-assessments and sequential reference selection. however, the improvement in the held-out effect magnitude remains unestablished.** In the linear contrast, the absolute error of Zhan held-out effect decreased by 4.53 percentage points. The linear model still underestimates the human effect magnitude. [All five-fold results](../../research/results/joint-effects-run/scores.json)

In this round, we address one question: after fitting the model to cross-condition human offsets, will the absence of material offsets bring the predictions closer to human responses, and how will this alter the per-sample predictions? We reuse the L4 cache from the previous Qwen3.5-2B run. We compare two frozen-embedding linear heads and a 32-dimensional shared residual adapter, each split into two groups: one using only a small number of sequential annotations and the other using the same number of annotations plus an effect term. The first comparison involves two shared models. The linear condition is used to assess whether the effect can be improved through task-specific parameters. [Previous three-task results](joint-first-run.md)

## Post-hoc fixed-effects design

We followed PLABA, Zhan, and OneStop across three datasets, five held-out folds, and a shared feature set. Each fold repeated five pre-specified random seeds. PLABA sampled one human response per sentence, retaining both the 1–5 Likert scale and the checkbox when selected. Zhan used its existing definition of sampling four responses per prompt. OneStop utilized all training responses from the fold, with both groups employing only sequential loss. For each fold, retained materials were scored using all known human responses. [Data Source and Split](stage-2-data.md)

For groups including the effect, only the training reaction estimates the target without sequential loss. Supervised reactions are split per item. participants may provide reactions across different materials. PLABA first applies equal weighting to the original or vernacular corresponding sentences at the rewrite alignment position, then averages by position. The target is the vernacular minus the original comprehension self-rating difference. Zhan's target is the subject mention rate shift after adding pronouns, calculated by material with equal weighting as IC-2 minus IC-1. The effect coefficient is non-zero only for training folds. The effect target is recalculated by fold and seed. [Rewrite effect](stage-1-observations.md#e01-plaba-eval-paraphrasing-and-sentence-by-sentence-self-assessment) · [Zhan effect definition](stage-4-zhan-weak-supervision.md)

Let $L_{\rm trial}$ denote the equal-weighted average of the sequential losses of the three previous tasks. The new group minimizes

$$
L=L_{\rm trial}+\frac12\left[\left(\frac{\widehat\Delta_{\rm PLABA}-\Delta_{\rm PLABA}}{4}\right)^2+
\left(\widehat\Delta_{\rm Zhan}-\Delta_{\rm Zhan}\right)^2\right].
$$

PLABA normalizes the 1–5 inter-group scale width by 4. Zhan computes in probability units of 0–1. Two additional weights are fixed, sharing seeds, data, and 160 FP32 AdamW updates with the sequential supervised group of the same architecture. The batch-wise contrast differentiable output is routed to the readout and applicable shared adapter.

## Acceptance

We primarily report the PLABA and Zhan held-out effect differences with and without the effect, and evaluate the sequential error at each endpoint. PLABA additionally reports the sentence-level average self-assessment MAE. OneStop has complete direct annotations but lacks its own effect term. we observe how changes to shared parameters affect the literature effect in other tasks. We first score five random seeds independently, then average the scores. We perform paired resampling to retain both model predictions and human responses, matched by material group. The Zhan effect is stratified by two verb types. The estimated scope covers unseen materials within existing studies, while formal cross-study held-out evaluation is planned separately.

## Sequential held-out results

All groups used the same sequential responses. For each fold, PLABA had 480–491 self-assessments with 190–253 selected, Zhan had 224–232 continuations, and OneStop had 1,545–1,562 responses. The supervisor additionally used 945–978 remaining comprehension scores from 207–220 alignment positions for PLABA, and 1,064–1,097 remaining responses for Zhan. [split, denominator, and effect target](../../research/results/joint-effects-run/effect-preparation.json)

| Lower endpoint values are preferable. | Shared + Sequential | Shared + sequential with effect | Difference 95% paired interval |
| --- | ---: | ---: | ---: |
| PLABA self-assessment, sentence-level MAE | 0.7556 | 0.6963 | [−0.0825, −0.0394] |
| Zhan reference selection, Brier | 0.2616 | 0.2474 | [−0.0200, −0.0087] |
| OneStop human option, NLL | 0.9480 | 0.9559 | [−0.0216, +0.0384] |

Paired intervals for the other trial-level endpoints in the shared group crossed zero. The linear group's MAE for self-rated comprehension in PLABA fell from 0.6421 to 0.6301, a difference of −0.0120 with interval [−0.0229, −0.0007]. OneStop predictions were bitwise identical. After effect supervision was added, the shared readout still had higher trial-level error than the linear readout on every endpoint in this round. The linear baseline remains the lower-error reference. [Endpoints, seeds, and paired intervals](../../research/results/joint-effects-run/scores.json)

## The magnitude of the publication bias effect

| held-out effect | human | Linear sequential | Linear additive effect | Shared sequential | Shared effect |
| --- | ---: | ---: | ---: | ---: | ---: |
| PLABA's vernacular understanding self-evaluation is poor, and the division is | +0.723 | +0.649 | +0.673 | +0.678 | +0.626 |
| Zhan reference interaction effect, percentage points | +24.46 | −2.37 | +2.17 | +5.46 | +8.45 |

For shared models, the absolute error change in the PLABA effect was +0.043 points ([−0.053, +0.076]), with Zhan at −3.00 percentage points ([−6.30, +1.57]). Both intervals crossed zero. After adding the effect to the linear model, the absolute error in Zhan decreased by an average of 4.53 percentage points (difference interval [−6.18, −2.98]). With fixed five seeds and reader samples in material pairing, the linear model's PLABA effect difference interval crossed zero. These intervals were computed based on each seed's absolute error. [ effect, absolute error, and interval ](../../research/results/joint-effects-run/scores.json)

The shared-add effect group's training Zhan interaction effect averaged +24.32 percentage points, with a target average of +23.93. the combined model after excluding materials showed +8.45. Gains from sequential prediction and held-out effect replication must each be validated. This round checked held-out materials within two existing studies. Training employed fixed-scale normalization, and effect weights were not adjusted based on publication estimation precision. [Training Diagnostics](../../research/results/joint-effects-run/fitting.json)

## L4 Empirical Measurement and Replay

100 fit reuses three cache states, taking 85.85 seconds, with a CUDA peak configuration of approximately 120.5 MiB. L4 configuration to send-stop request takes approximately 121.18 seconds, confirmed released. Account sampling activity rate is 1.94 CU/hour. all four cutpoints show 768.60 CU, and a single deduction fails to identify this. [Execution and Release](../../research/results/joint-effects-run/manifest.json) · [Account Sampling](../../research/results/joint-effects-run/usage.jsonl)

The effect and held-out scores are byte-identical after reconstruction. In each fold, the held-out response values are replaced while the fold's training supervision and effect target remain unchanged. After downloading and replaying all 100 weight sets, the maximum absolute difference in held-out scores is 8.37×10⁻⁶, and the maximum difference in training effect is 4.89×10⁻⁷. The two conditions of the OneStop linear group yield identical predictions. [Reconstruction and Data Validation](../../research/results/joint-effects-run/verification.json)

## Reproduce the entry point

This machine uses `research.joint.effect_design` to read fixed Stage 2 and prior sequential training designs, establishing a five-fold effect input. L4 is trained with `research.joint.effect_fit`, using a single cache for four comparison groups. After download, `research.joint.effect_score` re-rates based on complete held-out responses and the PLABA alignment table. [ effect data](../../research/joint/effect_design.py) · [training](../../research/joint/effect_fit.py) · [scoring](../../research/joint/effect_score.py)

[Original research record](../../research/joint-effects-run.md)
