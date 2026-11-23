# Calibrating the size of a local change in readers’ expectations

A gate with eight coefficients reduced error in the *amount* by which readers' expectations changed. The gate redistributes probability among the same 8,193 word categories, letting some prefixes receive a sharper distribution and others a broader one. It trained on 135,905 original UCL choices, using source probabilities predicted with each sentence left out of its base training fold. No Provo choices set any coefficient. Both gate fits converged, with maximum derivative below 4 × 10⁻⁹. Each forecast uses the same single backbone pass as the original word observer. [Fitting objective and observation units](word-population-dispersion-protocol.md) · [Source coefficients, gradients and sealed predictions](../results/reading-validation-word-dispersion/forecast-lock.json)

| Fixed forecast on Provo | Loss per actual next-word response, nats ↓ | Absolute error in change, percentage points ↓ | Correct direction | Absolute error in disagreement level, points ↓ |
| --- | ---: | ---: | ---: | ---: |
| Native Qwen probabilities | 3.8601 | 20.57 | 73.6% | 14.65 |
| Original UCL population observer | 3.6592 | 18.68 | **74.2%** | 14.32 |
| Gate supervised by word choices | **3.6066** | **16.57** | 73.8% | **13.55** |
| Same gate supervised by word choices and adjacent changes | 3.6094 | 16.72 | 73.2% | 14.10 |

The joint gate improved the original observer's word-choice likelihood by .0497 nats, with a simultaneous 95% interval of [.0298, .0697], and reduced change error by 1.96 percentage points [.78, 3.14]. Its direct comparison with the gate fitted solely to word choices gave −.0029 nats [−.0063, .0006] and −.15 percentage points [−.69, .39] of improvement respectively. The data support learning how concentrated readers' answers are. The additional adjacent-response objective has not shown an incremental benefit. The original observer still had the highest point estimate for direction. The study used 71,647 directly recorded choices at 1,791 verified positions, with 1,566 adjacent updates. Provo had already been evaluated with the original estimator, so this is a source-only improvement tested on a familiar benchmark. [Every paired contrast and score](../results/reading-validation-word-dispersion/scores.json) · [Original external study and denominator audit](cloze-population-results.md)

## Changes away from sentence boundaries

Restricting the test to two consecutive lowercase alphabetic forms removed sentence-initial capitals and terminal punctuation. On 1,226 nonzero changes inside sentences across 45 paragraphs, the original observer recovered direction at 76.3% [73.3, 79.1], native Qwen at 76.0%, and a lexical form model refitted in other Provo paragraphs at 70.0%. Adding the contextual curve to an equally trained model that already had both native and calibrated curves changed direction by −0.08 points [−0.82, 0.71]. The lexical effect persists within sentences. Its advantage over a fully informed text baseline depends on the outcome and comparator. [Specified controls](word-change-specificity-protocol.md) · [All and restricted positions](../results/reading-validation-cloze-population/form-specificity.json)

## Fixed-parameter transfer to a second laboratory

The same eight coefficients, fitted entirely on UCL, improved word-choice likelihood on 297,927 actual responses from 233 people in a separate sentence-completion study. On each of 3,085 sentence beginnings, Qwen received just the printed stem. The order and native logits of the new run matched the original capture bit for bit. The full set of forecasts was sealed before opening individual answers for this comparison. The material and cohort had previously been examined for a different reader-history hypothesis. [Original independent study](https://doi.org/10.3758/s13428-020-01351-1) · [Transfer protocol](dispersion-transfer-protocol.md) · [Forecast seal](../results/reading-validation-word-dispersion/transfer-lock.json)

| Fixed UCL observer | Loss per response, nats ↓ | Gain over original, simultaneous 95% interval | Mean absolute error in disagreement level, points ↓ |
| --- | ---: | ---: | ---: |
| Original population observer | 3.4556 | — | 30.12 |
| Gate trained on response identity | 3.4331 | .02249 [.01852, .02646] | 29.96 |
| Gate trained on response identity and adjacent changes | 3.4163 | .03934 [.03382, .04486] | 29.80 |

All 3,085 stems had at least two distinct measured respondents. The archived transfer protocol used “nine-word” as shorthand. The original presented stems actually range from six to thirteen whitespace-delimited words, and 1,521 contained nine. The forecast used every complete original stem. We excluded 1,498 empty answers and scored 75,328 valid answers in the residual spelling category. Both the odds of supplying an unlisted word and the distribution among the named alternatives improved. Intervals resample both readers and stems. The two external studies favor the same direction of improvement in word-choice likelihood, despite different stimuli and rankings of the two gates. Error in the absolute disagreement level remained about 30 percentage points on these stems. Their single published response position cannot test when expectations change inside a passage. [All estimates and source hashes](../results/reading-validation-word-dispersion/transfer-results.json) · [Independent reader design](reader-history-results.md)
