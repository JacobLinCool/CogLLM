# AmbiStory's supervised held-out results show a significantly worse ending performance

Retain the frozen Qwen3.5-2B features and perform five-fold cross-validated word-group selection on the original training set. Weight coefficients meeting the threshold are 1 and 4. the MAE on held-out folds is updated to select 4 based on predetermined criteria. Although higher values of 16 and 64 reduce the held-out MAE, the per-ticket NLL increases by 0.034 and 0.470 nats, respectively, exceeding the 0.01 nats threshold. The CPU fitting stage on the Colab L4 instance took 52.8 seconds. Scoring occurs after both weight sets and predictions are sealed into a snapshot, confirming instance release. [Pre-sealed snapshot specifications](ambistory-effect-supervision.md) · [Training fold diagnostics](../../../research/results/ambistory-effect-l4/train-diagnostics.json) · [Execution and release manifest](../../../research/results/ambistory-effect-l4/run-verification.json)

The held-out set comprises 98 story families, 54 homograph clusters, 2,952 votes, and 196 synonym group end-to-end pairs from the authors' validation set. Both end-to-end readouts share the same causal features, differing only in whether the human end-to-end penalty is added to the original training distribution loss. [ sealed snapshot score ](../../../research/results/ambistory-effect-l4/scores.json)

| Readout | Per-vote NLL (nats) | Symbolic update MAE (pts) | Mean Absolute Update (MAU) |
| --- | ---: | ---: | ---: |
| No-difference supervised, end | 1.59278 | 1.24198 | 0.118 |
| Difference-aware supervision, shared prefix | 1.59174 | 1.26088 | 0.000 |
| Differential supervision, end | 1.58802 | 1.22015 | 0.164 |
| human ratings |  |  | 1.261 |

The difference in supervised relative to the same feature's original readout updates the MAE difference to −0.02183 points, 95% interval [−0.03275, −0.01071]. The per-ticket NLL difference is −0.00476 nats, 95% interval [−0.01152, +0.00163], passing the +0.01 nats no-issue threshold. The candidate's NLL difference interval relative to its own common prefix still spans zero. The candidate's absolute update magnitude is 1.097 points less than human, 90% interval [−1.234, −0.952], exceeding the ±0.5 point equivalence bound. Five fixed conditions pass three, and the overall acceptance fails.

This experiment supports the hypothesis that direct supervised difference magnitude reduces update error between two specified endings. The average magnitude remains significantly underestimated, and existing weights serve as a baseline for the next round of readout comparison. Both the original test and validation human results have been evaluated. If the model is adjusted based on these results, a new confirmatory test requires an additional independent held-out set. [Model vs. human comparison](../../../research/results/ambistory-effect-l4/scores.json)

[Original research record](../../../research/stage6/ambistory-effect-supervision-results.md)
