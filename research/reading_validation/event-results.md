# A later clue can fit more than one earlier action

Workers read a complete five-sentence story, answered questions about particular actions and indicated which sentences helped them. We withheld the questions and let Qwen3.5-2B score all earlier–later sentence pairs. The most informative test held the later clue fixed and asked which of two earlier actions it supported. On 1,133 official test stories, consistently scored choices of this kind were sparse. The contextual readout did not improve over a supervised word-and-position comparator. [Original human study](https://aclanthology.org/2021.findings-acl.53/) · [Protocol](event-protocol.md) · [All held results](../results/reading-validation-events/scores.json)

| Interpretation of the released support indices | Comparisons / stories | Correct earlier action, contextual model | Lexical model | Nearest action |
| --- | ---: | ---: | ---: | ---: |
| Published one-based convention | 39 / 33 | 53.0% [36.4, 69.7] | 62.1% | 60.6% |
| Zero-based convention | 73 / 57 | 71.3% | 77.8% | 75.1% |
| Support independently traceable in a written answer | 9 / 7 | 100% | 100% | 100% |

The two indexing conventions change both eligibility and the apparent rate of success. In the training split, 9,394 of 11,885 uniquely traceable answer phrases agreed with zero-based indices and 1,043 agreed with the published one-based description. We registered the two interpretations and a third independently traceable subset before evaluating held forecasts. No test story met the simultaneous two-way selectivity criterion under any convention. The dataset therefore supplies limited evidence about which earlier action a later clue supports and provides no time stamp for the reader's own moment of resolution. [Author documentation](https://stonybrooknlp.github.io/tellmewhy/) · [Coordinate audit](../results/reading-validation-events/coordinate-audit.json) · [Decision before test scoring](event-coordinate-amendment.md)
