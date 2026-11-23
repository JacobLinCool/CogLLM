# Human selection-based fitting improves WikiNews ranking. the additional gain from CompLex remains unestablished

In the 7 held-out source families of WikiNews, the fused readout of word length, word frequency, and existing CompLex scores increased the vocabulary ordering concordance from **64.35%** to **77.28%**. Fitting only word length and word frequency also achieved **77.19%**. The difference in relative literal readout was +0.09 percentage points, with a matched 95% confidence interval [−1.12, +1.21], and the difference interval for per-ticket NLL also contained zero. Neither piece of evidence met the predetermined CompLex gain threshold. [Specified Criteria](cwi-lexical-readout-protocol.md) · [Results and Intervals](../../../research/results/cwi-lexical-readout-l4/results.json)

## Materials and Comparisons

Following the prior WikiNews family split, exclude sentences spanning multiple HITs and candidates that cannot map to a single complete interface token. Retain 591 original sentences and 5,723 candidates. Among these, the training set contains 338 sentences, 21 families, and 3,205 candidates, with 3,204 completed alignments used for fitting. Development and test sets each comprise 7 distinct families. Under the normalization criterion of full sentences or 25% five-word fragments, the training set shows no valid overlap with the two held-out sets and professional news. Each sentence directly utilizes the current Gradio checkpoint's single causal forward pass and actual display accuracy. After freezing the original interface scores, fit three scores using only the human votes from the original training set. [Data Category, Exclusion, and Denominator](../../../research/results/cwi-lexical-readout-l4/assay.json) · [Frozen Weights](../../../research/results/cwi-lexical-readout-l4/fit.json)

| Evaluation materials | Original interface | Unfitted word length and word frequency | Fitted word length and word frequency | Fitting fusion |
| --- | ---: | ---: | ---: | ---: |
| WikiNews development | 64.91% | 77.39% | 78.82% | 79.22% |
| WikiNews test | 64.35% | 76.48% | 77.19% | 77.28% |
| Professional news retrieval | 65.67% | 75.25% | 77.39% | 77.00% |

The WikiNews test covers 1,358 candidate pairs and 5,224 human-voted pairs with identical sentences. Fusion relative to the baseline improves by +12.92 percentage points, with a confidence interval of [+8.96, +17.10]. Per-ticket NLL is 0.2637. the fitted literal readout is 0.2653, yielding a difference of −0.00157 nats [−0.00370, +0.00058]. In professional news backtracking, the ranking difference between fusion and fitted literal readout is −0.39 percentage points [−1.55, +0.66]. This result constrains claims of gain from existing CompLex single-score metrics. Directly using Qwen's suffix representation to fit the new readout remains a distinct experimental condition. [Pairwise statistics and coverage](../../../research/results/cwi-lexical-readout-l4/results.json)

## Reproducibility and Resources

Reconstruct the original sentence and the three sets of separated scores using `cwi_wikinews_prepare`, adhering to the Gradio vocabulary exporter. `cwi_lexical_readout predict` reads only the training scores, followed by a sealed snapshot of weights, predictions, and scoring program hashes, then executes `score`. The sealed snapshot results are rerun bit-for-bit consistent, independently computing verification of all denominators, rankings, NLL, and vocabulary scores. The L4 remote forward export phase takes 78.26 seconds, configured to verification release at 140.32 seconds. Local fitting and two predictions take approximately 0.18 seconds, with the combined rerun peak RSS at approximately 174 MiB. [Preparation](../../../research/core_features/cwi_wikinews_prepare.py) · [Fitting and Scoring](../../../research/core_features/cwi_lexical_readout.py) · [Execution and Release](../../../research/results/cwi-lexical-readout-l4/run-verification.json) · [Independent Verification](../../../research/results/cwi-lexical-readout-l4/independent-check.json)

[Original research record](../../../research/core_features/cwi-lexical-readout-results.md)
