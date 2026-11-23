# The shared word embedding coefficient did not improve held-out prediction for both tasks

Each study independently estimated the per-step correct baseline rate, then shared the Qwen3.5-2B representation coefficients. The English Core Vocabulary held-out NLL increased from **1.86004** to **1.86217 nats / trial**. The German ACT decreased from 0.76723 to 0.76657, but the 95% confidence interval for the difference included zero. The joint readout did not pass the pre-specified gain threshold for both studies. [Analysis Specifications](joint-cue-hazard-protocol.md) · [Complete held-out results](../../../research/results/joint-cue-hazard/results.json)

| Research | Number of steps | Literal | Qwen individual fit | Qwen joint fitting | Joint improvement in individual NLL and material matching 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| English, 32 targets, 1,603 trials | 1.86763 | 1.86402 | 1.86004 | 1.86217 | −0.00213 [−0.00352, −0.00095] |
| German, 20 wordlists, 1,999 trials | 0.75975 | 0.76581 | 0.76723 | 0.76657 | +0.00067 [−0.00034, +0.00183] |

NLL with material equal weighting, in units of nats/trial, where lower values indicate higher prediction probabilities for human outcomes. English joint readout outperformed literal readout by 0.00185 [0.00104, 0.00278], but underperformed English individual fitting with the same representation. German step readout point estimation was optimal, with joint readout relative to literal improving by −0.00075 [−0.00209, +0.00044]. When both readers and materials were resampled, the difference intervals for individual readout across both studies were similarly entirely negative and crossed zero. [Pairing results](../../../research/results/joint-cue-hazard/results.json)

Post-hoc verification of the individually fitted 128-dimensional slopes from both studies yielded five-fold cosine similarities ranging from −0.036 to +0.019. values near zero indicate that vector directions are nearly orthogonal. This describes the slopes already fitted under a shared projection. The five-fold approach shares the majority of the training vocabulary, and this range is not a statistical confidence interval. [ sealed snapshot coefficient](../../../research/results/joint-cue-hazard/models.json)

## Cumulative accuracy

| Research and steps | human | Joint readout | The model reduced human and material matching by 95% (percentiles). |
| --- | ---: | ---: | ---: |
| English, sentence 1 | 9.35% | 7.62% | −1.73 [−4.48, +0.70] |
| English, sentence 3 | 35.82% | 34.50% | −1.32 [−5.88, +2.90] |
| In English, sentence 6 | 69.89% | 71.55% | +1.66 [−2.47, +5.98] |
| German, 1st word | 0.40% | 0.46% | +0.06 [−0.55, +0.46] |
| German, the 3rd word | 3.95% | 4.24% | +0.29 [−2.67, +2.51] |
| German, the 6th word | 17.76% | 18.41% | +0.66 [−6.90, +7.58] |

The prespecified equivalence range is ±5 percentage points. The model’s group mean at the sixth step in German was close to the observed human mean, but the interval still extended beyond the tolerance. Estimating the baseline rate separately for each study substantially corrected the earlier cross-study bias. This round provided no evidence that shared Qwen coefficients consistently improve prediction of when readers answer correctly across materials. [Fixed English readout and transfer to German](act-transfer-results.md)

## Reproducibility and Scope

`joint_cue_hazard_study forecast` First, seal the snapshot with four readout parameters and seven prediction classes. `score` then retrieve the held-out trial. For German, each word list uses training data only from folds other than its own. For English, the two comparisons use the original 64 training targets, newly sampled 32 held-out target paths, and the same frozen candidate. The held-out data from both datasets were previously used for other analyses. the results presented here constitute a retrospective comparison. The entire pipeline adopts the representation from the publicly available sealed snapshot, with no new model forward pass, local Qwen loading, or GPU configuration. Fitting and scoring each take approximately 0.24 seconds, totaling a maximum of 122 MiB of local resident memory. [Training and scoring script](../../../research/semantic_tracking/joint_cue_hazard_study.py) · [Resources and sealed snapshot hash](../../../research/results/joint-cue-hazard/run-verification.json)

[Original research record](../../../research/semantic_tracking/joint-cue-hazard-results.md)
