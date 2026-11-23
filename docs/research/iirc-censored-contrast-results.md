# IIRC right-censored ranking results

Adding 15 right-censored training controls, with the earlier evidence reading stop fixed across the existing 37 same-document pairs, increased the number of correctly ordered responses from 21 to 23. This represents a +5.41 percentage point increase, with a 95% article group paired confidence interval of [−19.44, +29.73]. The interval crosses zero. [Scheduled pruning and loss](iirc-censored-contrast.md) · [sealed snapshot summary](../../research/results/iirc-censored-contrast.json)

| Metrics | Unsorted | Right-censored ranking | Differences and 95% paired confidence intervals |
| --- | ---: | ---: | ---: |
| The number of hits at the reading stop rank. | 21/37 | 23/37 | +5.41 percentage points [−19.44, +29.73] |
| Exchange-reduced interval NLL (nats) | +0.00436 | +0.01481 | +0.01045 [−0.02349, +0.04467] |
| Original interval NLL (nats) | 3.21991 | 3.25076 | +0.03085 [−0.00925, +0.06944] |
| Both paths prefer pairing within their respective intervals. | 0/37 | 0/37 | 0 |

The optimal NLL for the baseline interval is 0. A higher exchange reduction in baseline interval NLL indicates greater model preference for human-annotated evidence. The ranking NLL of original training pairs decreased from 0.7084 to 0.4448, demonstrating that readout can fit this set of controls. Among 13 training article groups, 15 pairs had 11 comparison points at the first two reading stops, with a median character distance of 29,700 between the two evidence endpoints. These controls primarily provide evidence for early emergence or differences outside the observation window. Ranking on the held-out set still lacks stable gains compared to baseline interval results, and endpoints only cover text selected by respondents post-hoc. [Data and scoring program](../../research/semantic_tracking/iirc_censored_contrast.py)

First, reconstruct 15 sets of the original held-out sealed snapshot training keys using the programmatic `prepare` from the original IIRC training release and the paired experiment's sealed snapshot, followed by executing `forecast` on a fixed 128-dimensional cache. After applying each method's weights and the predicted hash to the sealed snapshot, only then read the 37 original held-out annotations (`score`). Using the published weights, the probability is reproducible to a maximum difference of 7.22 × 10⁻¹⁶, and the maximum difference for zero-sorted weight re-fitting is 4.97 × 10⁻¹³. Single-threaded CPU streaming verification, re-fitting, and writing takes approximately 0.28 seconds, with peak resident memory at approximately 238 MiB. Streaming and full-array re-runs yield identical predictions at the bit level. [Original weights and scoring specification](iirc-linked-shared-head.md) · [This round's hashing and scoring](../../research/results/iirc-censored-contrast.json)

[Original research record](../../research/iirc-censored-contrast-results.md)
