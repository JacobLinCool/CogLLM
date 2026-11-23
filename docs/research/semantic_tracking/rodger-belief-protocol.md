# Rodger Study 1: Two-reading stop judgment update fixed comparison

Verify whether the frozen page embeddings of Qwen3.5-2B can reproduce the change in readers' validity ratings for the same policy after exposure to policy evidence. Condition 4 represents the no-issue case, and Condition 1 represents the original study's control condition. The main contrast is the difference in average validity scores (post minus pre) between the two conditions. Use raw sequential data. scores are integers ranging from 0 to 100. [Original Study 1 and Figures 1–2](https://doi.org/10.1017/bpp.2025.10028) · [Construction and Scale Audit](rodger-belief-source.md)

## Material and visible reading stop

Group by policy and perform six-fold cross-validation with held-out sets. In each fold, hold out one policy along with its six versions of evidence, training on the remaining five policies. Do not model the reader. A total of 301 readers appear across folds, so this design tests the transfer of new policy materials within the same experiment. All Qwen inputs contain only the QSF baseline page and visible evidence pages, excluding responses, condition labels, or reader fields. The research procedure includes a credibility question page. post-reading rating measures are taken on this page, followed by constructing text retention up to the end of the evidence page. [Original procedure and source](rodger-belief-source.md)

Use a single frozen Gradio candidate to perform causal forward passes for 36 policy-evidence pairs and 6 independent baselines at L4. The Qwen prefix tokenization for both reading stops must align precisely. Retain the 128-dimensional projection to verify consistency between the baseline reading stop and independent baseline for the same policy across six paired inputs, with a maximum L2 difference upper bound of 0.002. Feature scoring uses the 2,048-dimensional final state from the end of the baseline page and evidence page. both are normalized via L2 independently. The complete state for identical baseline pages is sourced from the independent baseline input, and all subsequent pages under the same policy share this value. [Frozen Candidate](../../../research/results/gradio-candidate-checkpoint/checkpoint.json)

## Readout and comparison

| Method | Training and visible features |
| --- | --- |
| Only the literal text of the body. | Jointly fit pre- and post-read. Baseline page: log1p word count, whether the evidence page is fully read, the page's log1p word count, and the continuous number of field counts. Pre-read evidence features are 0. |
| Body text only with Qwen | Same surface features, plus baseline state vector and post-read minus pre-read state vector. The pre-read difference vector is 0. |
| Incorporate empirical priors | Perform literal and literal-plus-Qwen separately. Fit only post-test scores, jointly using the pre-test validity scores from the same trial. When computing changes, use the actual pre-test scores directly. |
| Conditional Information Contrast | The two prior modes are each augmented with literal features and six classes of evidence pages encoded as one-hot vectors, used solely to distinguish how much signal the study assignment itself provides. |

Let the original score be $y\in\{0,\ldots,100\}$. Using the empirical distribution $q(y)$ of the training scores as the base, add 0.5 per level. Readout is conditioned on $s(X)=b+X\theta$ configuration, $p(y\mid X)\propto q(y)\exp[s(X)(y-50)/50]$, with its expectation serving as the scalar readout. Training minimizes the average NLL per observation with equal weighting plus $0.10\lVert\theta\rVert_2^2/2$, with no penalty on the intercept. Literal and prior fields use the mean and standard deviation from the training data. Each vector block is normalized by subtracting the training mean and dividing by the square root of the weighted L2 norm of the block after centering. Pre-read and post-read observations are each included, with equal weighting applied to the loss. Priors and scales within a fold must not be adjusted via held-out results.

## Scoring and thresholds

The primary analysis employed the body-only mode. Per-item post-read NLL was evaluated, with a minimum of 0 nats, where lower values indicate higher model probability of the actual rating. Expected value MAE was also reported, in units of the original scale. For each held-out policy, the average post-read minus pre-read score under condition 4 was calculated relative to condition 1, and the difference was taken. Six policies were equal-weighted, with the model and human using identical trials and denominators. Other controlled differences and comparisons with empirical priors were also reported, all constituting secondary endpoints.

For each policy and reader, we resampled with replacement from the original population 10,000 times using seed 735972. We paired the retained model with human ratings under identical conditions. First, we retained trials masked in the held-out training set. intervals were reweighted using predictions from a sealed snapshot, estimating the impact of these six policy materials and the six-fold readout constraints already fitted. Additionally, we listed the effect error for each held-out policy and the mean absolute error (MAE) of the six errors, contrasted against the condition of five policies. This condition-only contrast yields the same overall effect when averaged across the six folds, while per-policy errors reveal its performance on material differences. We fixed the six materials and re-sampled only the reader intervals as a supplement. The claim "replicating this effect in held-out materials" simultaneously requires the following three items.

1. The literal post-read NLL difference relative to literal Qwen is 95% confidence interval upper bound below 0.
2. The difference in change between conditions 4 and 1 is negative for both model and human responses, and the upper bound of the model's interval is below 0.
3. The model's reduction in human effect difference had a 95% confidence interval entirely within the pre-specified ±5 range.

This comparison validates the update of specified policy judgments before and after evidence pages, assessing the distribution of changes after reading the evidence. Readers may raise doubts or resolve questions at different tokens within a sentence, requiring distinct baseline measurements.

[Original research record](../../../research/semantic_tracking/rodger-belief-protocol.md)
