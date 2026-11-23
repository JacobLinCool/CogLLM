# Rodger Study 1 material held-out judgment update results

When only the text is provided, Qwen's readout reproduced the direction of the score decline but underestimated the magnitude by 6.65 points. Across 1,806 trials in six policy held-out folds, the human score change for negative cases relative to controls was −7.86 points. Adding Qwen yielded a total of −1.21 points, with a difference interval of [−0.70, 14.10] points exceeding the pre-specified ±5 points. After adding Qwen, the NLL difference for per-trial readout scores was −0.0109 nats, with an interval of [−0.0378, 0.0085], which also failed to pass. [Fixed specification](rodger-belief-protocol.md) · [Itemized scoring and matching intervals](../../../research/results/rodger-belief-l4/results.json)

| Readout | Post-readout NLL, nats ↓ | Negative case relative control, split | 95% confidence interval of the effect, split | By policy effect MAE, split ↓ |
| --- | ---: | ---: | ---: | ---: |
| human | — | −7.86 | [−15.15, −0.66] | — |
| Literal | 4.183 | −0.17 | [−0.47, −0.01] | 7.69 |
| Literal and Qwen | 4.172 | −1.21 | [−1.53, −0.86] | 6.80 |
| Literal and research conditions | 4.177 | −5.17 | [−5.74, −4.58] | 5.67 |
| The five training policy conditions were all significantly worse. | — | −7.86 | — | 6.53 |
| Literal and empirical pre-test readout scores | 3.904 | −1.67 | [−5.09, 1.84] | 6.21 |
| Literal, Qwen, and observed pretest scores | 3.902 | −2.78 | [−6.21, 0.73] | 5.46 |
| Literal, study conditions, and observed pretest scores | 3.899 | −6.56 | [−9.79, −3.23] | 4.23 |

The original scale ranges from 0 to 100. Within each policy, the effect is first compared across conditions and then averaged across the six policies using equal weighting. NLL represents the negative log-likelihood of the probability assigned by the model to human post-reading ratings, measured in nats, with a minimum value of 0. The policy effect MAE is the average of the absolute differences between the model's effect and the human effect for each of the six policies, with a minimum of 0 points. The "Study Conditions" column uses experimental assignment, while other readout features are derived from the text and listed background information. Intervals are constructed via cross-validation over the same reader and policy, with coefficients fixed. [Full Score](../../../research/results/rodger-belief-l4/results.json)

Policy-level bias has a substantive impact. For example, the human effect of the e-cigarette ban is −17.26 points, while the Qwen readout based solely on the body text is −0.84 points. Across five training policies, the condition of all pairwise comparisons is inferior. the six-fold average can reproduce the overall mean difference, yet the policy-level error remains at 6.53 points. After incorporating same-trial measured pre-test scores, Qwen's mean difference becomes −2.78 points, and the magnitude difference interval between the model and human remains exceeding ±5 points. [Policy-level estimates and errors](../../../research/results/rodger-belief-l4/results.json)

## Reproducibility and Scope of Application

The same frozen candidate performed forward passes across 36 policy–evidence pairs and 6 baselines in L4, totaling 8,893 model tokens. All 78 input reading stops and 12 independent causal controls passed. The maximum L2 difference between the 36 readout projections and independent inputs was 0.0000928, with a predicted upper bound of 0.002. The forward and readout phases consumed 4.32 seconds, while the entire extraction stage took 43.4 seconds, with content-batch GPU peak retention of approximately 8.44 GiB. L4 has been released. Two single-threaded native evaluations each took about 1.1 seconds, with peaks around 431 MiB. coefficients, probabilities, and scores matched bit-for-bit. We recomputed the post-readout NLL, MAE, and six policy effects under both prior modes using sealed snapshots, and validated the training score histograms for the 36 fits. [Extraction & Resources](../../../research/results/rodger-belief-l4/extraction.json) · [Release manifest](../../../research/results/rodger-belief-l4/run-verification.json) · [Replay verification](../../../research/results/rodger-belief-l4/local-verification.json)

The same cohort of 301 readers was crossed across six folds. The post-test involved completing full-page evidence and credibility questions followed by explicit ratings of a designated policy. The original study drafted two types of case stimuli with ChatGPT and had them reviewed by the authors. The scope of this experiment is limited to these six policies and measurement procedures. there is no original observation regarding the point at which intra-sentence doubt or resolution occurs. [Original Study 1, Materials, Procedure, and Results](https://www.cambridge.org/core/journals/behavioural-public-policy/article/negative-anecdotes-reduce-policy-support-evidence-from-three-experimental-studies-on-communicating-policy-ineffectiveness/9B673E6D29B154BE39D68BFBBFA5FDAA) · [Data and Scope](rodger-belief-source.md)

[Original research record](../../../research/semantic_tracking/rodger-belief-results.md)
