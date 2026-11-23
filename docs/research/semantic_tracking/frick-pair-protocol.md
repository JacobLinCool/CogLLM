# Calibrate Frick's sentence-by-sentence report using the paired-context effect

Previously, relying solely on sentence-wise report fitting, the model showed almost no ability to replicate the human's conditional difference on the same target sentence. Using the remaining stories to estimate the direction of the difference in the paired diagnosis, 18 out of 20 held-out story sets placed the maximum difference on the target sentence. This round reuses the same frozen model and 20-fold story held-out to check whether incorporating the human paired effect from the training stories into the loss improves the report probability for individual texts. Having already reviewed the initial scoring and paired diagnosis of this material, this round serves as a retrospective comparison. [Previous round results and difference vector diagnosis](frick-consistency-results.md)

## Single-document usable features

For each fold, the direction of state estimation for the target sentence of 19 training stories is used. First, the end-of-sentence states $h$ from both versions are normalized to unit vectors, and then the difference vector between the inconsistent and consistent versions is normalized. The training difference vectors are averaged across stories and then normalized into a single direction $w$. For any article, $s_t=w^Th_t$ and $s_t-s_{t-1}$ are computed at each end-of-sentence, where $h_t$ is also a normalized state. These two projections are combined with five fixed literal features. The first sentence provides only the adjacent features of the second sentence. These seven features can be computed in a single causal forward-backward pass per article, without reading the other version, target sentence position, experimental condition, or human button. The state, original text, and reading stop are extracted using the same frozen candidate sealed snapshot. [reading stop and literal features](frick-consistency-protocol.md) · [original extraction](../../../research/results/frick-consistency-l4/extraction.json)

## Three training pairs with the same feature

For each fold, the seven pure quantities are estimated using all valid observations from the training set. The intercept is unpenalized, while the remaining coefficients are penalized identically to $0.10\lVert\theta\rVert^2/2$. When the sentence report and first report are trained separately. The first report applies a within-risk-set sentence Bernoulli likelihood, then multiplies the hazard by the no-event probability of the preceding reading stop to obtain the overall first-report probability for all readers.

| Training approach | The observed |
| --- | --- |
| All observations equal weighting | Use only the Bernoulli likelihood and penalty from the previous experiment. |
| Target sentence weighting | Fitted with the same likelihood. The likelihood weights of all observations in the training target sentence sum to 1/2, while the remaining sentences sum to 1/2. |
| Add matching effect | Use the likelihood of all observations with equal weighting as the baseline. Additionally, add $4\,\mathrm{mean}_{s\in\mathrm{train}}(\widehat\Delta_s-\Delta_s)^2$. $\Delta_s$ represents the difference in human inconsistency probability between training stories and the target sentence. The first report uses the unconditional first probability difference. |

All three groups used identical seven-item inputs and held-out groups. Directions were fitted to all coefficients using only training stories, and the held-out versions were composed of differences after each predicted all sentences individually.

## Comparison and Scope of Application

Adopt the successive report and first-event endpoints. Retain the four categories of target sentence effect (target pre/target sentence/target post/end-of-document no report) from both versions, along with the NLL of the full-document probability distribution. Aggregate each metric by first averaging within each story version, then applying equal weighting across stories and conditions. Use identical 10,000 cross-validation resamples for readers and stories with seed 735972 and fixed coefficients. Additionally report the interval for fixed stories with reader resampling only. [Original endpoints and denominator](frick-consistency-protocol.md)

Compare the magnitude of error and NLL of the paired effect relative to two same-feature controls. This round of checks requires the 95% confidence interval lower bound of the direction difference to be greater than 0, the interval of the model minus human effect difference to be fully contained within ±10 percentage points, and the upper bound of the interval of the relative difference between all equal weighting readouts and the prior prefix Qwen NLL to be below the non-inferiority limit. The non-inferiority limit is pre-specified as +0.005 nats/sentence for the sentence level and +0.05 nats/document for the document level. Both magnitude and NLL are scored simultaneously, with each endpoint reported separately. These determinations are used for the held-out material of the same experiment, while experimental validation on independent corpora requires fixation.

[Original research record](../../../research/semantic_tracking/frick-pair-protocol.md)
