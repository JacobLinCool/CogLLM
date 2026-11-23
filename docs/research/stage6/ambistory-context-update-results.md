# AmbiStory's held-out results for semantic update

Using human scores from 378 training story families, we fit the semantic rationality readout of the frozen Qwen3.5-2B model. Both methods share identical weights, reading the last reading stop after a common prefix and the last reading stop after appending different suffixes. The formal test includes 155 unseen families, 4,663 human scores, and 310 sets of same-semantic suffix comparisons. The model uses only training scores. The first fit reaches the 300-step limit. With original data and validation rules retained, convergence is achieved at step 310 after increasing the iteration limit for contextual readout. Two L4 releases have been verified. [Original Specification](ambistory-context-update.md) · [Numerical Correction](ambistory-numerical-amendment.md) · [Error Manifest of First Fit](../../../research/results/ambistory-fit-first-attempt.json) · [Completed L4 Manifest](../../../research/results/ambistory-fit-l4/run-verification.json)

| Readout | Per-vote NLL (nats) | Ending change MAE (pts) | Consistency of the direction of change at the end | Mean Absolute Change (points) |
| --- | ---: | ---: | ---: | ---: |
| Morpheme–lemma correspondence | 1.56521 | 1.15849 | 50.0% | 0.000 |
| Shared prefix reading stop | 1.57555 | 1.15849 | 50.0% | 0.000 |
| Actual ending reading stop | 1.56819 | 1.13509 | 64.0% | 0.120 |
| human ratings |  |  |  | 1.158 |

Compared to the shared beginning, reading to the end reduced per-ticket NLL by 0.00736 nats, 95% interval [−0.01309, −0.00201]. The same-lemma ending change reduced MAE by 0.02340 points, 95% interval [−0.04118, −0.00536]. Both original improvement conditions were met. Per-ticket NLL across all versions was slightly lower in the form-lemma alignment, which is a different comparison.

The predicted magnitude of change averages only about one-tenth of that of humans. The model's mean absolute change difference relative to humans is −1.038 points, with a 90% confidence interval [−1.149, −0.926], exceeding the ±0.5 point equivalence threshold. After excluding votes marked as nonsignificant, the determination of both the improvement and the magnitude threshold remains consistent. This supports the hypothesis that the model representation contains partial signals capable of distinguishing word meanings supported by the ending, yet the current readout has not yet achieved full magnitude acceptance. Since each human rater observed only one version, the measured change reflects the group-level shift between versions. [Corrected scores from the pre-sealed snapshot](../../../research/results/ambistory-score-protocol-v2.json) · [Full scores and intervals](../../../research/results/ambistory-fit-l4/scores.json)

[Original research record](../../../research/stage6/ambistory-context-update-results.md)
