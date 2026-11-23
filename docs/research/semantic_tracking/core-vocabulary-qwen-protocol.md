# Qwen prefix representations and the response distribution of the group achieving first correct answers per sentence

The public trial table for Core Vocabulary records the trial number of first correct response, lacking the actual stimulus ID observed by each reader. This experiment uses 768 pre-sealed snapshot possible paths to test whether freezing the Qwen3.5-2B end-of-turn state improves the first-correct-response distribution for the historical reader group. First, using the author's four conditions, a fixed set of 64 training words and 32 held-out words was selected. Each word was assigned eight distinct paths via a pre-sealed snapshot seed. Since the same reader may have responded to materials in both splits, the primary generalization unit is the new target word. [Sampling and Endpoint](core-vocabulary-marginal-protocol.md) · [Fixed Split](../../../research/results/core-vocabulary-marginal-pilot/split.json)

## Feature and fit

Each path provides only the author's masked text, using a single Qwen causal forward pass. For each Gradio candidate, the IIRC projection yields six suffix states, each 128-dimensional. A fixed tokenizer is used to verify the original text coordinates against the six prefixes. The readout observes only the current and prior cues. The path's hidden targets, conditions, human guesses, and their correctness are not sent to the GPU.

| Readout | Visible features |
| --- | --- |
| Number of steps | Estimate the conditional probability of six steps based on the empirical distribution of the training objective. |
| Literal | The six-step intercept, in-sentence, and cumulative blank token count `log1p`, the proportion of in-sentence lemmas that appeared in the preceding context. |
| Literal addition of Qwen | Identical surface features, augmented with the current 128-dimensional projection state at the sentence end. |

Unicode English-Chinese units after NFKC and casefold. The first sentence has an overlap proportion of zero. The literal variable estimates mean and standard deviation on the training path and reading stop. Qwen vectors are first column-wise unit-normalized, then centered by training mean, and the entire block is scaled by the root-mean-square of the centered vector's L2 norm. Two logistic hazards are fitted with training target equal weighting, six-step intercepts are unpenalized, and all feature coefficients are ridge-penalized with a fixed 0.10. For each path, six first-correct probabilities and the probability of not being correct after the sixth sentence are computed, then averaged across eight paths according to the target. The loss is the negative log-likelihood of the human seven-class result in this mixture distribution. Target word combinations, coefficients, and hyperparameters are fixed before the held-out annotation is joined. [Definition of mixture distribution](core-vocabulary-marginal-protocol.md#distribution-and-score)

## held-out threshold

The primary criterion requires lower held-out NLL for the lexical-plus-Qwen readout than for both the step-only and lexical readouts. The lower bound of the paired 95% interval must be above zero for each improvement. We also check cumulative first-correct response rates after the first, third, and sixth sentences. Each 95% interval for the model rate minus the human rate must fall entirely within ±5 percentage points to conclude that the model matches the magnitude of the human group-level distribution within the practical tolerance. The same cluster resamples are used for each difference. We resample the held-out targets 10,000 times, stratifying by the authors' four conditions, with seed 735972. We also report intervals for resampling readers and targets if that analysis can be reproduced using the published de-identified reader IDs. Target-only intervals are conditional on the readers in this sample.

This experiment uses eight possible paths for each target to approximate the authors' sampling distribution. It can validate the population-level curve for first correct responses across these paths. Individual human paths are unavailable, so the probability of resolution of any single source sentence remains unobserved. The additional one to six sentences are part of the fixed cueing task, which measured no onset for spontaneous questions. [Data fields and limitations](core-vocabulary-source.md)

[Original research record](../../../research/semantic_tracking/core-vocabulary-qwen-protocol.md)
