# IIRC shared-readout results

Fixed the L4 causal state of Qwen3.5-2B, fitting the readout interval with 3,240 training paths. This fit excluded 33 article families from the same-text pairing. The official 452 held-out and 114 held-out (same-text pairing) share a single sealed snapshot weights. When replaying the 114 paths, the maximum difference between the per reading stop probability and the original fit prediction is \(7.22\times10^{-16}\). The Colab L4 instance used SciPy CPU solver to complete the fit, with outputs verified against the release. [Predefined Acceptance Criteria](iirc-linked-shared-head.md) · [Weights and Replay Manifest](../../research/results/iirc-shared-head-l4/official-seal.json) · [Colab Execution Log](../../research/results/iirc-shared-head-l4/run-verification.json)

| held-out control | Mean difference (nats) | 95% paired interval | Acceptance |
| --- | ---: | --- | --- |
| Official dev, explicit anchor NLL minus positional NLL | −0.03692 | [−0.05466, −0.01981] | Upper bound not reached −0.02 |
| Official dev, explicit anchor NLL minus in-path body NLL | −0.00611 | [−0.01601, +0.00372] | Interval crosses zero |
| Same-text matching, explicit anchor swap interval: NLL minus original interval NLL | +0.00436 | [−0.01982, +0.03131] | Interval crosses zero |
| Same-text matching improves alignment of explicit anchors relative to the body text. | +0.00344 | [−0.01729, +0.02415] | Interval crosses zero |

On official dev, the cumulative Brier score met the noninferiority threshold against the position baseline. Of 37 same-document pairs, each of the three readouts had 0 pairs in which both paths favored their own annotated intervals. Neither holdout met all its criteria. These weights remain a baseline for later experiments, and the evidence-arrival interval does not yet have a deployable readout. The endpoint measures evidence in the linked article that answerers marked. Aligned observations of when the original question arose or was resolved remain unavailable. [Scores for both holdouts](../../research/results/iirc-shared-head-l4/scores.json)

Subsequent backward-looking comparisons identified 15 pairs of parallel texts where one had complete evidence while the other lacked complete evidence until the end of the observation window. After incorporating the ranking loss, the original held-out reading stop hits increased from 21/37 to 23/37. the 95% confidence interval for the difference crossed zero, and the NLL of the original annotated intervals did not improve. This comparison still supports retaining the original readout as the baseline. [Right-censored ranking comparison](iirc-censored-contrast-results.md)

[Original research record](../../research/iirc-linked-shared-head-results.md)
