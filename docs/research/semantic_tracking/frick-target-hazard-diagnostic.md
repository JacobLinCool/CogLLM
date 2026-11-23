# Frick first reported that the primary difference stems from the target sentence at hand

Fixed four calibration readout probabilities for the target sentence, even allowing arbitrary changes to the pre-target no-report rate. the model's first-report conditional difference is at most **28.61 percentage points**. Human is **48.27**, with a difference of **19.66**. the cross-validation 95% confidence interval is **[7.91, 31.17]**. The primary room for improvement in existing readouts lies in distinguishing consistent from inconsistent cases when reading the same target sentence, based on context. [ sealed snapshot prediction](../../../research/results/frick-mass-calibration/forecast-seal.json) · [this decomposition and interval](../../../research/results/frick-target-hazard-diagnostic.json)

## Target sentence decomposition

For story $s$, condition $c$, and the researcher-defined target sentence $t^*$, let $T$ be the order of the first report. Let $S_{sc}=P(T\ge t^*)$ be the proportion of readers who did not report before the target sentence, and $h_{sc}=P(T=t^*\mid T\ge t^*)$ be the reporting probability for this group at the target sentence. The overall first-report probability for the target sentence is $S_{sc}h_{sc}$. The sealed snapshot model computes the hazard for all preceding sentences as $\widehat S_{sc}=\prod_{t<t^*}(1-\widehat h_{sct})$. The baseline model begins calculation from the second sentence, treating non-report throughout the text as an observation censoring point. Each effect is first computed as the difference between inconsistent and consistent within the story, then averaged across 20 stories with equal weighting. [First Event and Denominator](frick-pair-protocol.md) · [Raw Data Alignment](frick-consistency-source.md)

| The conditional probability of the target sentence | Consistent with the preceding text. | Inconsistent prior |
| --- | ---: | ---: |
| human | 3.31% | 59.14% |
| Four calibration readout types | 7.74% | 28.61% |

Fix the four types of readout $\widehat h_{sc}$. Since each $S_{sc}$ lies between 0 and 1, $\operatorname{mean}_s(S_{s,02}\widehat h_{s,02}-S_{s,01}\widehat h_{s,01})\le\operatorname{mean}_s\widehat h_{s,02}$. This upper bound even allows all inconsistent readers to be retained and all consistent readers to report before the target sentence. [Computing and Numerical](../../../research/results/frick-target-hazard-diagnostic.json)

| Substitution quantity | First, the difference, percentage points, and 95% confidence interval in the target sentence report. |
| --- | ---: |
| The unreported rate of human and the reported probability of the target sentence. | +48.27 [39.10, 57.26] |
| The model's unreported rate versus the target sentence report probability. | +17.15 [11.39, 23.94] |
| Report only the pre-submission rate for the human target. | +17.99 [11.72, 25.60] |
| Report only the probability of the target sentence being substituted by human. | +46.25 [37.74, 54.56] |
| The mathematical upper bound of the probability of the fixed model target sentence report. | +28.61 [20.17, 38.14] |

Substituting only the human probability of reaching the target with no earlier report improved the effect by +0.83 percentage points, with an interval of [−1.13, 3.07]. Both substitutions were retrospective diagnostics using the same held-out human responses to estimate the magnitude of the possible effect. The original readout stayed frozen. We ran 10,000 crossed resamples that varied reader and story weights together, retaining seed 735972 and using the same draws for all differences. We verified all 1,800 human trajectories and 584 sealed reading stops. We also used exact fractions to recalculate the across-story human proportion whose first inconsistency report occurred exactly at the target sentence. The maximum difference from the sealed condition scores was $5.56\times10^{-17}$. [Recalculation code](../../../research/semantic_tracking/frick_target_hazard_diagnostic.py) · [Original calibration scores](../../../research/results/frick-mass-calibration/results.json)

## The next pre-specified control

First, fix a seven-feature readout paired-condition likelihood ratio test to assess whether the report difference for the same target sentence can be improved under two different contexts. Use the existing four-class calibration model to provide the cross-fitted logit base, augmented with a small causal feature readout. During training, only pair versions that share the same story, the same sentence order, and have identical character-level text at the current sentence. Include identical sentences outside the target sentence to evaluate whether the effect diffuses to earlier or later positions. For the two groups that have reached the sentence but have never reported it, the count is $R_{01},R_{02}$. the total count of first reports for the sentence is $K$. The count of inconsistent reports is $Y_{02}$. If the two groups within this pairing follow a binomial distribution based on their respective group hazards, the paired-condition likelihood ratio fixed at $K$ is

\[
P(Y_{02}=y\mid K,R_{01},R_{02})
=
\frac{\binom{R_{02}}{y}\binom{R_{01}}{K-y}\exp[y(\eta_{02}-\eta_{01})]}
{\sum_{j=L}^{U}\binom{R_{02}}{j}\binom{R_{01}}{K-j}\exp[j(\eta_{02}-\eta_{01})]},
\quad L=\max(0,K-R_{01}),\quad U=\min(K,R_{02}).
\]

Among them, $\eta$ represents the base logit adjusted by causal features. During computation, log-sum-exp is used for normalization. The conditional likelihood constrains only the difference between the two logits, while retaining the original complete first-event and four-position likelihood estimates of absolute probabilities. It describes the risk set up to the point where the sentence has not yet been reported, without directly estimating the causal effect for all readers. Cross-story matching may involve identical readers. inference continues via reader-story cross-resampling. Target positions, prior context, and another version all belong to training and scoring data. Formal prediction reads only the causal features of a single story body.

The outer layer still groups held-out story families. Each inner-layer validation fold reselects base readouts and feature directions from its training family. During fit correction, the base logit is fitted using within-group cross-validation on the training stories, and the human responses of these training stories do not participate in generating their own base logit. Candidate weights are pre-fixed to $0, 0.25, 1, 4$, selected by prioritizing the inner-layer's first report NLL against the target sentence difference, and the full specification must be sealed in a snapshot prior to the new fit. The outer layer still verifies ±10 percentage points of the target sentence magnitude, ±10 early report bias for each condition, and a non-inferiority bound of +0.05 nats per story relative to the original control. This is a new back-readout of the same set of analyzed materials. independent experiments require additional unseen reading materials. [Existing thresholds and splits](frick-mass-calibration-protocol.md)

[Original research record](../../../research/semantic_tracking/frick-target-hazard-diagnostic.md)
