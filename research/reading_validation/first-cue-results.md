# Can an article predict the sentence that first makes a reader stop?

In an independent third cohort, 233 readers made 1,398 story readings in twelve families. When an earlier premise was inconsistent with an unchanged target sentence, first reports at the target increased by **31.7 percentage points [20.4, 42.7]**. An input-only model that explicitly conserved the probability of encountering the first cue predicted a 4.7-point increase. Adding direct supervision on the source condition effect raised this to 7.1 points [0.2, 14.0]. Its discrepancy from readers was −24.6 points [−35.8, −12.8]. These estimates use a new reader cohort and leave out the target story family from each source fit. The families had previously been studied and appear across cohorts. [Original recording and design](memory-protocol.md) · [First-cue estimation](first-cue-protocol.md) · [Cohort-level and family results](../results/reading-validation-first-cue/scores.json)

| Article-only estimator | Full first-report path, nats ↓ | Change in first reports at the target |
|---|---:|---:|
| Previously fitted directional risk-set model | 1.502 | +7.7 pp |
| Previous unbounded effect-supervised model | 3.487 | +19.0 pp |
| First-cue position only | 1.842 | 0.0 pp |
| Conserved first-cue model with text | 1.570 | +4.7 pp |
| Conserved first-cue model plus paired-effect supervision | 1.662 | +7.1 pp |
| First-cue model with transferred contextual reader-click probabilities | 1.682 | −0.2 pp |
| Same transfer with paired-effect supervision | 1.858 | +0.1 pp |
| Diagnostic supplied the true location and condition | 1.303 | +32.6 pp |

The bounded effect-supervised model reduced path loss by 1.825 nats [.858, 3.101] relative to its unbounded predecessor. The directional risk-set model remained better on the point estimate. A prospective transfer of the noun pointer supplied a direct test of mechanism. The source pointer had successfully recovered the earlier words chosen by game players. We used its article-only probabilities to weight candidate noun comparisons at each sentence in 55 experimental narratives. All 10,730 word representations came from one unmarked pass per narrative. On the independent cohort, the transferred contextual pointer increased path loss by .112 nats [.028, .201] relative to the first-cue model. In this task, an explicit decision about noun reference did not identify what would make the same person stop and report a contradiction. The supplied-state diagnostic suggests a bottleneck in locating and classifying the relevant event from text. [Noun-transfer design](nominal-report-transfer-protocol.md) · [Source and held response paths](../results/reading-validation-nominal-reports/scores.json) · [Earlier risk-set calibrations](taxonomic-premise-results.md)
