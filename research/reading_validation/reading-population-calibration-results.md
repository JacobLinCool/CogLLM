# How much does a small reference sample help in a new reading task?

On Peelle's sentence endings, the UCL-trained observer systematically predicted that readers would disagree more often than they actually did. A single fitted shift on the log-odds scale, learned from 760 public responses in a disjoint group of 32 people, brought the mean forecast from 84.85% to 67.49%. Readers in the test group disagreed on 68.34% of pairs. Every text-dependent coefficient came from another study. Each new user still supplies only their passage. This calibration targets the original sentence-ending completion task. Applying its intercept to ordinary paragraphs would require separate evidence. [Source-study model](spelling-agreement-results.md) · [Reading task and original release](https://link.springer.com/article/10.3758/s13428-020-01351-1) · [Calibration procedure](reading-population-calibration-protocol.md)

All 64 reference stems chosen by hash had at least five responding reference readers. The fitted log-odds shift was −1.078. The remaining 201 readers and 3,021 stems contributed 255,514 usable, previously recorded, test responses. No respondent or sentence crossed between fitted calibration cells and test cells. Numerical predictions were locked before opening the corresponding test answers in this procedure. The larger study's outcomes were already familiar from previous models. The calibrated observer has the same ordering of sentences as the source-only observer (Spearman .347). The new intercept changes the level of its predictions. [Role assignments and fitted intercept](../results/reading-validation-population-calibration/forecast-lock.json) · [All test scores](../results/reading-validation-population-calibration/scores.json)

| Forecast for a new person and stem | Error in full-spelling disagreement, points ↓ |
| --- | ---: |
| UCL conditional spelling observer, no adjustment | 18.99 |
| UCL source mean | 18.03 |
| Mean of the new reference sample | 18.74 |
| Median of the same new reference sample | 17.80 |
| Raw Qwen with the same type of study-specific adjustment | 17.81 |
| UCL conditional observer with one study-specific adjustment | **16.99** |

Against the new reference mean, the reduction was 1.75 points with a simultaneous 95% interval [.13, 3.37]. Against the unadjusted UCL observer it was 2.00 points [−.26, 4.26]. The median is a stronger constant for scoring absolute error, and native probabilities provide a stronger text-dependent control. Relative to them, gains were .81 points [−1.50, 3.12] and .83 points [−.63, 2.28]. These sensitivity intervals do not resolve an advantage. The estimation resampled whole readers and stems in both the calibration and test sets, refitting the offset and the controls on each draw. The change in mean level is large, but the residual average prediction error remains substantial. [Comparator specification](population-calibration-controls.md) · [Complete controls, including squared errors](../results/reading-validation-population-calibration/controls.json) · [Independent human-group variability](spelling-agreement-human-precision.md)
