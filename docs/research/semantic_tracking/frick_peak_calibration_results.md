# Frick causal peak calibration's held-out results

Calibrate the peak score of the same prefix using attention readout, achieving a **+16.32 percentage point** improvement in the same target sentence report contradiction compared to the original attention readout of **+20.01** and human **+48.27**. Relative to the original attention readout, the absolute error increase for the peak calibration is **+3.68 percentage points** with a 95% confidence interval [1.83, 5.84]. Relative to affine calibration, the error increase is **+0.11 percentage points** with an interval [−0.15, 0.38]. The primary improvement threshold was not met. [Fixed Spec](frick_peak_calibration_protocol.md) · [Full Score](../../../research/results/frick-peak-calibration/results.json)

| Readout | The first report of the target sentence (percentage points) | The entire NLL (nats/essay) | For readers whose inconsistent-target sentence has not yet been reported, their in-sentence hazard |
| --- | ---: | ---: | ---: |
| human | +48.27 | — | 59.14% |
| Original seven features | +18.95 | 1.59387 | 30.15% |
| Original attention | +20.01 | 1.56399 | 30.73% |
| Affine calibration | +16.43 | 1.53904 | 31.17% |
| Peak calibration | +16.32 | 1.54239 | 30.95% |

Lower NLL indicates a higher probability that the model assigns the exact first position or no report at the end of the document. The peak relative NLL difference from original attention is −0.02161 nats, with a 95% confidence interval [−0.07762, +0.02846], passing the +0.05 non-inferior threshold. The early report bias for consistent and inconsistent versions are +13.66 [8.04, 18.97] and +13.28 [6.88, 19.77] percentage points, respectively, both exceeding the ±10 equivalent range. The target sentence conditional report probability for the consistent version also increased from 6.76% readout to 8.65%, with human at 3.31%. This calibration maintains the underestimation for inconsistent target sentences while boosting the cumulative report rate from prefix to target for both versions. [Conditional and control results](../../../research/results/frick-peak-calibration/results.json)

For each of the 20 folds, one story family was held-out from two versions, totaling 1,800 human trajectories and 584 end-of-sentence reading stops. The calibrated readout inputs derived from inner-layer logits that had not seen the held-out family nor the outer held-out family. The outer-layer attention readout from the sealed snapshot was fully reproducible and consistent. All 1,088 prefix checks against the unread corpus were consistent value-for-value. Two CPU runs took 6.35 and 6.48 seconds, respectively, with a peak memory usage of approximately 233 MiB, and all five primary outputs were identical bit-for-bit. Independent recomputation of the NLL, target magnitude, and conditional probability for each method yielded a maximum difference of $4.44\times 10^{-16}$. [Program](../../../research/semantic_tracking/frick_peak_calibration.py) · [sealed snapshot prediction](../../../research/results/frick-peak-calibration/forecast-seal.json) · [Independent verification](../../../research/results/frick-peak-calibration/independent-check.json)

This study continues to employ the previously analyzed Frick story, readers, and frozen features. Evidence is limited to the first post-hoc contradiction report and the effects of two prior conditions. no corresponding observation was found for the first intra-sentence detection and the resolution of the same question. [Original Study and Endpoint](frick-consistency-source.md)

[Original research record](../../../research/semantic_tracking/frick_peak_calibration_results.md)
