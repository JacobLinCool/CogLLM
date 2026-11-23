# Does a stable tendency to report difficulty explain the missed contradiction?

Moving one earlier premise into conflict with an unchanged sentence increased the proportion of readers making their **first inconsistency report at that sentence by 31.7 percentage points** [20.3, 42.8]. A model allowing people to differ consistently in how readily they report a problem predicted a 4.1-point increase [−4.5, 13.6]. Its discrepancy from the human effect was −27.6 points [−40.9, −14.0], outside the prespecified ±10-point equivalence range. Accounting for stable differences in reporting improved the fit of its matched model while leaving the experimental effect substantially underestimated. [Numerical results](../results/reading-validation-reporting-propensity/scores.json) · [Study design](reporting-propensity-protocol.md)

In Frick and Schüler’s third English-language experiment, 233 readers read six stories each. The analysis retained 1,398 first-report paths across 12 story families. First reports at the target rose from 22.0% in the neutral condition to 53.7% in the inconsistent condition. Counting every target judgment, including those made after an earlier report, gives a different increase of 35.2 points [22.1, 47.8]. The distinction matters because a person who has already reported a problem has no further opportunity to make a first report. [Original study](https://doi.org/10.3758/s13421-026-01900-7) · [Observation and exclusion rules](memory-protocol.md)

| Forecast | Loss per complete first-report path, nats ↓ | Change in first reports at the unchanged target |
|---|---:|---:|
| Measured reader response | — | +31.7 pp |
| Shared reporting tendency fixed across people | 2.133 | +4.9 pp |
| Stable, varying reporting tendency integrated over new readers | 2.038 | +4.1 pp |
| Earlier article-only directional risk-set model | 1.502 | +7.7 pp |
| Diagnostic supplied the true target position and experimental condition | 1.303 | +32.6 pp |

Allowing reporting propensity to vary reduced loss by 0.095 nats per path [0.026, 0.178] compared with the same model with variation fixed at zero. Its loss was 0.536 nats higher [0.130, 1.090] than the earlier directional model. All 12 source fits for each matched model converged. The estimated between-reader standard deviation ranged from 1.01 to 1.18 on the log-odds scale. Each forecast excluded the target family from source fitting and integrated over possible reporting propensities for an unidentified new reader. Intervals use 10,000 crossed reader and story-family resamples. [Verified source fits](../results/reading-validation-reporting-propensity/fit-lock.json) · [Fit and score](reporting_propensity.py) · [Resampling implementation](memory_score.py)

The third cohort's responses had been examined in earlier comparisons. This follow-up fixed its new source-only fits before grading them on that same cohort. A keypress locates the first *reported* inconsistency at sentence resolution. It leaves the onset of unreported difficulty and its position within the sentence unmeasured. [Earlier first-cue evaluation](first-cue-results.md)
