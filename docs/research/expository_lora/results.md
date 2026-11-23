# Frozen Backbone and LoRA for Sentence-Level Expository Reading Responses

Across 609 out-of-fold PLABA-EVAL sentence ends, frozen Qwen3.5-2B predicted the distributions of human reading ease and self-rated comprehension more accurately. Across 12 sets of Kerwer abstracts with the same body text with and without subheadings, the frozen readout's mean effect was close to zero, the final-layer rank-4 LoRA estimate was negative, and the human effect was positive. This round selected the frozen readout as the sentence-level candidate. [Full scores](../../../research/results/expository-lora-l4/scores.json) · [Prespecified data and losses](protocol.md)

<a id="逐句反應與配對改寫"></a>

## Sentence Responses and Paired Rewrites

The five folds used the same 39 material families and 25 connected material-and-reader groups. Both arms used the same readout for two five-level sentence rating distributions and the squared error for rewrite effects within the training fold. Negative log likelihood (NLL), in nats, measures the probability the model assigns to the human-selected rating. The ranked probability score (RPS) measures squared errors of cumulative probabilities across four rating thresholds and ranges from 0 to 1. Both metrics weight each sentence equally, with a minimum of 0. The 95% intervals for differences come from 10,000 resamples of the 25 material-and-reader groups. [Scoring script](../../../research/expository_lora/score.py) · [PLABA-EVAL study](https://lrec.elra.info/lrec2026-ws-readixtsar-04)

| Out-of-fold endpoint | Frozen | LoRA | LoRA minus frozen, with 95% interval |
| --- | ---: | ---: | ---: |
| Reading ease NLL | 1.329 | 1.911 | +0.582 [0.506, 0.671] |
| Self-rated comprehension NLL | 1.305 | 1.787 | +0.482 [0.400, 0.575] |
| Reading ease RPS | 0.1421 | 0.1807 | +0.0386 [0.0310, 0.0465] |
| Self-rated comprehension RPS | 0.1422 | 0.1843 | +0.0421 [0.0324, 0.0518] |

Across 265 author-aligned locations pairing original sentences and plain-language sentence groups, the LoRA point estimate of the mean rewrite effect was closer to the observed human effect. Mean gains in reading ease were +0.537 points for humans, +0.415 for the frozen model, and +0.505 for LoRA. For self-rated comprehension these were +0.723, +0.558, and +0.688, respectively. In both ratings, intervals for the improvement in absolute error of the mean effect crossed zero. Pairwise absolute error in self-rated comprehension increased from 0.755 to 0.854 points, a LoRA-minus-frozen difference of +0.099 points with 95% interval [+0.047, +0.156]. All scores use the original 1–5 scales. LoRA reached a lower training objective in every fold. Together with the held-out distributions and paired outcomes, this is consistent with overfitting under this configuration and sample size. [Rewrite pairs and convergence records](../../../research/results/expository-lora-l4/scores.json) · [Objective by fold](../../../research/results/expository-lora-l4/convergence.json)

<a id="同正文小標的外部對照"></a>

## External Contrast for Subheadings on Identical Body Text

The Kerwer study has 1,283 valid ratings in the two arms from 166 readers. Each arm contains 181 verbatim-aligned body-text sentences. The headings remained in the causal prefix. Mean self-rated comprehension at body-text stops was mapped to the study's 1–8 scale, then the subheadings-minus-no-subheadings contrast was taken within each of 12 DOIs. The human postreading rating effect was +0.359 points, with 95% interval [+0.062, +0.676]. The frozen readout estimated −0.00017 points. LoRA estimated −0.067 points, with 95% interval [−0.118, −0.013]. LoRA increased the mean effect's absolute error relative to humans by 0.067 points, with paired 95% interval [+0.009, +0.123]. The interval for the difference in DOI-level absolute errors crossed zero. The 90% intervals for differences from the human effect extended beyond the prespecified ±0.5-point tolerance for both models. [External effect and results by DOI](../../../research/results/expository-lora-l4/scores.json) · [Kerwer study](https://doi.org/10.1525/collabra.18898) · [Matched body-text coordinates and study protocol](../stage6/kerwer-article-holdout.md)

PLABA-EVAL participants had the full text available when rating sentences. Kerwer collected ratings after each passage. These observations cover sentence-end rating distributions, plain-language rewrite magnitude, and postreading effects of subheadings. Validating locations and intensity of within-sentence load requires matching human observations. Kerwer had previously been used to evaluate the frozen readout and served here as a structure contrast outside PLABA training.

<a id="收斂單次前向與資源"></a>

## Convergence, Single-Forward Inference, and Resources

All six frozen fits and twelve LoRA initializations passed convergence across five folds and one full-data fit. The maximum final absolute gradient on the original parameters was 2.88×10⁻⁷, the largest L2 gradient norm was 1.28×10⁻⁶, and the largest relative objective change between consecutive checkpoints was 2.42×10⁻¹⁰. Their thresholds were 3×10⁻⁷, 10⁻⁵, and 10⁻⁹, respectively. The first six L4 runs each stopped after at least one initialization missed a threshold. Once both solvers were fixed using training-side convergence records, the formal run passed every check. All human Kerwer ratings were joined locally after model predictions were sealed. [Convergence traces for each initialization](../../../research/results/expository-lora-l4/convergence.json) · [Retries, hashes, and release receipts](../../../research/results/expository-lora-l4/run-verification.json)

Formal inference ran a frozen model or a model with LoRA attached. Each arm used one causal forward pass per article over 102 articles and 1,039 sentence ends. The largest probability difference between cached fitting and live model predictions was 5.74×10⁻⁶. Each arm passed all 12 independent tests with truncated or replaced continuations. The largest relative RMS difference over the shared prefix was 4.64×10⁻⁶. [Forward-pass verification](../../../research/results/expository-lora-l4/causality.json) · [Sealed predictions](../../../research/results/expository-lora-l4/predictions.jsonl) · [Readout and LoRA parameters](../../../research/results/expository-lora-l4/checkpoints.npz)

In the successful run, model loading, caching, fitting, and validation took 412.18 seconds. Caching frozen features for 29,158 tokens took 16.77 seconds, about 1,739 tokens per second. Peak CUDA allocation was 7.52 GiB and peak reservation was 7.64 GiB. Colab took 562.81 seconds from confirmed allocation to confirmed release, an estimated 0.30 compute units at the single-GPU snapshot rate. All seven allocations have confirmed releases and total approximately 1.36 compute units by the same method. These rate-based usage figures are estimates. Colab billing records contain the complete metering. The two scoring runs produced byte-identical outputs, and independent recomputation from the raw ratings gave the same point estimates. [Run verification and usage snapshot](../../../research/results/expository-lora-l4/run-verification.json) · [Sealed software and model](../../../research/results/expository-lora-l4/seal.json)

[Original research record](../../../research/expository_lora/results.md)
