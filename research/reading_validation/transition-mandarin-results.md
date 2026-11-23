# Human referent choices in Mandarin completions

Mandarin participants read a short clause with two people and a verb. Some supplied a free continuation. Others continued after an overt pronoun. CogLLM trained one distribution over the person and the referring form. Free continuations supervised all four outcomes. Overt-pronoun continuations supervised which person was meant, conditional on that form. The held cohort contained 87 people, 1,504 overt choices and 1,237 free continuations in 72 condition cells. Each source fit excluded the corresponding one of 36 verb/name families. [Protocol and original experiments](transition-mandarin-protocol.md)

| Prediction of the held overt choice | Mean log loss, lower is better |
|---|---:|
| Text form and character cues, coupled observations | .649 |
| Qwen reading states, coupled observations | .643 |
| Same Qwen states, overt responses only | .644 |
| Same Qwen states, pooled response modes | .654 |
| Human-rated semantic class and overt responses | .619 |
| Human Bayesian estimate from the same items' separate free continuations | .686 |
| Same-item mirror estimate | .650 |

Qwen improved on the matched surface representation by −.0054 log loss [−.0100, −.0011]. Joint use of the free responses yielded −.0010 [−.0038, +.0017] versus the same native model using only overt choices. Comparison with pooled observation types gave −.0110 [−.0459, +.0261]. Both differences permit zero. The independently rated semantic-class predictor performed better than the native model by .0245 [.0002, .0473]. All 252 source fits reached maximum absolute gradient below 1.4 × 10⁻⁸. [Held predictions, source stationarity and estimates](../results/reading-validation-transition-mandarin/scores.json)

Participants chose the first referent 26.5 percentage points more often with IC-1 than IC-2 verbs in active sentences, and 23.7 points more often in 把 sentences. The native observer reproduced the direction in both constructions, estimating 5.8 points and 1.6 points, respectively. Its effect sizes substantially underestimate the human shift. The semantic-category benchmark estimated 21.9 points in each construction. These observations motivate learning an explicit verb-specific semantic prior from independent norms.

One frozen forward per original stimulus covered 220 strings and all 110 changed-suffix checks. Each target decision was a referent choice in the original continuation task. The data identify the chosen referent after a short clause. A word-level time at which a reader experienced effort remains unobserved. [Source intake](../results/reading-validation-transition-mandarin/source-seal.json) · [Human continuation data](https://osf.io/t87my/) · [Original publication](https://doi.org/10.1371/journal.pone.0237012)
