# Choosing the next unit in a piece of news

People can write several coherent accounts from the same source sentences. Using 639 observed choices from 94 participant-event arrangements, three-study learning improved held-event normalized choice loss from **1.0165 to 0.9624** relative to an identical native model trained solely on the order choices. The paired gain was 0.0541 (two-comparison simultaneous interval 0.0107–0.1050, exact event-family sign-flip Bonferroni P = 0.0430). A model using text form and word probability scored 0.9556, and one adding the older source readouts scored 0.9528. The gain supports borrowing between these measured responses within a matched native architecture. [Registered choice likelihood](columbia-choice-training.md) · [Cross-fitted outcomes](../results/reading-validation-columbia-order/crossfit-scores.json)

## Observed arrangements

Barzilay, Elhadad and McKeown asked participants to order randomly presented, authentic news units about one event so that the resulting summary would make sense. We used the public multidocument material and the listed complete arrangements. The project overview describes ten participants. Six original topic tables contain ten rows each and four contain eleven each, totaling **104 participant-event arrangements**. The source file identifiers do not establish 104 different people. We reserved one independent ordering for each event before reading any outcome. The other 94 orders supplied 639 choices with at least two remaining candidates and 2,582 alternatives. Two El Salvador sets share five twelve-word windows and were kept in one analysis family. We grouped whole events in uncertainty estimates and resampled the nine independent event families. [Original task and data](https://www.cs.columbia.edu/~noemie/ordering/coll/) · [Original research](https://www.cs.columbia.edu/~gmw/candidacy/Barzilayetal02.pdf) · [Public participant tables and pinned file identities](../results/reading-validation-columbia-order/observed-orders.json)

Of the 88 original units, some contain multiple sentences. Each of the 706 possible ordered unit pairs was presented unchanged to Qwen3.5-2B, with a newline at the unit boundary. We used the first completed printed sentence of the next unit for the boundary score. Every estimate was fixed before reading the ten actual-order tables. The native extraction encoded every candidate in 59 batches. Five source observers reproduced 1,645 previously archived boundary scores with maximum numerical disagreement 2.23 × 10⁻¹⁶. A candidate's average score after other possible predecessors was subtracted to remove inherent target ease. The evaluation asks whether the chosen continuation ranked above an alternative from the identical remaining set. [Predeclared design](columbia-order-protocol.md) · [Forecast freeze and parity](../results/reading-validation-columbia-order/forecast-seal.json) · [Historical exposure audit](../results/reading-validation-columbia-order/historical-exposure.json)

## Frozen cross-study forecasts

| Candidate selector | Chosen unit ranks above an unchosen unit | Correct top choice |
|---|---:|---:|
| Uniform assignment | 50.0% | 25.2% |
| Previous bilingual reading observer | 56.0% [50.4, 60.4] | 32.4% |
| Same observer with training on original versus reordered Italian passages | 57.7% [51.8, 62.2] | 32.3% |
| Native cosine similarity to the previous unit | 56.8% [51.3, 62.0] | 33.1% |
| TF–IDF content cosine | 62.9% [58.2, 68.4] | 38.7% |
| Native contextual word surprisal | 63.2% [59.2, 66.7] | 40.7% |
| One independent human writer's order | 60.5% [57.0, 63.5] | 41.1% |

Adding Italian source-order training yielded **+1.69 percentage points** in pairwise choice agreement, with a three-comparison simultaneous 95% interval **+0.60 to +3.00** and Bonferroni-adjusted exact P = 0.00586. Top-one accuracy was almost identical. The previous bilingual observer's difference from chance was +5.99 points (simultaneous −0.86 to +11.09, adjusted P = 0.123). Its difference from contextual word surprisal was −7.25 points (simultaneous −11.72 to −3.20). An unchosen candidate can also be a sensible continuation. The human reference uses the directly observed order from an independent participant-event row, and the native observer receives the previous unit and next candidate only. [All frozen scores, held-human choices and event estimates](../results/reading-validation-columbia-order/scores.json)

## Learning from actual next-unit selections

A conditional softmax over the still-available candidates uses one native representation for each unmarked unit pair. Its target is the unit the participant actually chose. Normalize each choice loss by the logarithm of the number of available candidates, so uniform choice has loss one. Hold whole event families out of supervision and hyperparameter fitting. Enrich one model with an original English boundary-mark likelihood and a distinct Italian 1–5 paragraph-grade likelihood. All arms use the same native features and real source split. The outcome establishes a choice among currently available units, with limited context from the immediately preceding unit. [Conditional estimator](columbia_choice_training.py) · [Synthetic gradient and held-topic preflight](../results/reading-validation-columbia-order/synthetic-preflight.json)

| Cross-fitted model | Normalized choice log loss | Pairwise choice | Top choice |
|---|---:|---:|---:|
| Uniform | 1.0000 | 50.0% | 25.2% |
| Form and word probability | 0.9556 | 61.5% | 37.0% |
| These features with fixed cross-study scores | 0.9528 | 62.3% | 38.4% |
| Native, actual order choices only | 1.0165 | 58.1% | 33.1% |
| Same native architecture, three human observation types | 0.9624 | 60.4% | 35.6% |

Adding fixed reading features to the form and word-probability model improved log loss numerically by 0.00282 (simultaneous −0.0144 to +0.0137). Sharing conditional-choice features with the two earlier reading studies improved the matched native model across the prespecified test. The strongest simple selector still ranked more individual choices correctly. Eleven numerical outcomes, including four normalized held-choice log losses, were independently reconstructed from participant orders and fixed predictions. Maximum absolute difference was 2.23 × 10⁻¹⁶. [Verification](../results/reading-validation-columbia-order/public-replay.json) · [Reproduction commands](news-order-reproduction.md) · [Study comparison figure](../results/reading-validation-summeval/figure/human-news-order.pdf)
