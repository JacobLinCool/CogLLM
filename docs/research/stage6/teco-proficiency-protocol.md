# TECO English reading ability and discourse difficulty alignment

**Research hypothesis.** Adding pre-reading English reading ability improves the prediction of the 1–9 level discourse difficulty distribution on unseen readers and unseen articles. We also examine when this background alters the article difficulty ranking. This study design uses 41 Japanese native English learners, 30 EIKEN expository texts, and 1,208 immediate post-reading scores as the response distribution. Reading ability is measured by scores from an independent 19-item test. [Original study](https://link.springer.com/article/10.1007/s11145-026-10888-0) · [Source and cross-version verification](../nahatame-teco-data-audit.md)

## Model visible data versus held-out

Infer only the readout from the article body and the selected 0–19 reading ability scores. For each article, perform a single causal forward pass using Qwen3.5-2B, using the FP32 state at the final reading stop token as the discourse representation. The sentence-level and per-token output scales remain unchanged from their existing baselines. Since human scoring occurs after segment-wise display and interleaved yes/no questions, the discourse scores estimated here under this experimental exposure are expected to generalize to a single-pass full-article reading scenario only upon validation. [Experimental exposure method](https://tsukuba.repo.nii.ac.jp/record/2011041/files/RMAL_3-2-100123.pdf#page=5)

All available readers and articles were each split into five folds using fixed hashing. Articles were partitioned equally within each EIKEN level. Each cell in the cross-tabulation was fitted using trials where both the reader fold and article fold were different. Ratings for the same article from the same participant must remain within the same article fold. article and reader identifiers were used solely for splitting and scoring. The actual presentation order, pre-test items, subject identifiers, and any response performance were excluded from the readout. Additionally, it was verified that no ID overlaps occurred across the two dimensions for each cell in the cross-tabulation.

## Readout, acceptance, and scale

Fix four readouts, estimating scale, weights, and response distribution from each training fold using identical nine-level probability outputs and L2-regularized negative log-likelihood, with no penalty on the intercept.

| Readout | Input |
| --- | --- |
| Text baseline | The number of words, characters, and English readability features that are reproducible. |
| Freeze text state | Identical textual features and causal end-state |
| Prereading ability | main effect of identical state and pre-reading ability |
| Capability to interact with text | Identical main effects and pre-reading ability multiplicative causal states |

We primarily compare the sequential NLL difference of readout relative to the frozen text state. After resampling readers and articles in groups, the 95% confidence interval upper bound must be less than zero. Interactions are compared using the same method for the ability main effect model and reported independently. The baseline is computed as the marginal average of the trained ability distribution over the trained conditional probability, and is compared against a text-only model using the same held-out sequential NLL.

For observed trials within each article, also compare readers with prereading scores at or below the training-cell first quartile against readers at or above the third quartile. Prediction and observation both use the actual trial count in each group as the scoring denominator. Report the mean difficulty contrast between the low- and high-score groups for each article, grouped intervals, and the difference between estimates and observed values. For articles with both groups in the holdout, use a prespecified ±0.5-point equivalence margin, equal to half a rating category, and test aggregate magnitude error with a grouped 90% interval. The comparison measures an association in the observed groups. Article-specific prior knowledge, general readers, and word-level questions require separate validation data.

This file fixes the experimental problem and statistical criteria. A sealed snapshot must be exact split, executed, and readable feature version and input hash before entering model extraction and fitting.

[Original research record](../../../research/stage6/teco-proficiency-protocol.md)
