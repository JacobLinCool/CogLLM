# Comparison of evidence intervals after symmetric calibration

After calibrating with two parameters, the per-token readout achieved a lower per-token readout interval NLL than reading only the stem by **0.1171 nats**, with a 95% article matching interval of [−0.1700, −0.0631]. The difference relative to relative paragraph interactive readout was −0.0151 nats [−0.0207, −0.0097]. The position-calibrated readout had the lowest average NLL but assigned identical ranks to different questions on the same article. The per-token readout ranked 435/698 pairs where evidence appeared earlier and more completely higher. [Fixed analysis specification](evidence-calibration-comparison.md) · [Full results](../../../research/results/evidence-calibration-comparison.json)

| Readout | Original interval NLL | Calibrated interval NLL | Calibrated cumulative Brier | Tongwen tong reading stop wins, draw counts as half. |
| --- | ---: | ---: | ---: | ---: |
| Position | 5.9489 | 3.0486 | 0.1816 | 50.0% |
| Stem | 3.1963 | 3.1931 | 0.1917 | 54.0% |
| Paragraph interaction | 5.1161 | 3.0911 | 0.1844 | 62.6% |
| per-token interaction | 5.0903 | 3.0760 | 0.1836 | 62.3% |

Interval NLL measures the probability assigned by the model to the reference interval, while cumulative Brier measures the cumulative probability error at each reading stop. lower values are preferable. The per-token NLL difference relative to position is +0.0274 nats [−0.0034, +0.0587]. This batch of data supports the hypothesis that corpus features distinguish in-text problems. Measured by absolute interval scores, the advantage of corpus features over the position baseline remains to be confirmed. Four conversion groups retain the full ranking of cumulative intensities for all 698 reading stops. [readout of sealed snapshot and comparison](expmrc-transfer-results.md) · [interval and matching complete results](../../../research/results/evidence-calibration-comparison.json)

Retain ExpMRC's 165 articles, 555 questions, and two equally weighted references, with five-fold calibration by article. Four parameter sets fit on the complete data were sealed for independent scoring on other corpora. This round is a retrospective out-of-fold comparison on existing human annotations. The intervals measure stops at which all text supporting the specified question is present. Scoring reran bitwise identically, taking about 0.65 seconds per run and about 83 MiB in peak process memory. [Code](../../../research/semantic_tracking/evidence_calibration_comparison.py)

[Original research record](../../../research/semantic_tracking/evidence-calibration-comparison-results.md)
