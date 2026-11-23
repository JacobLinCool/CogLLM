# Extending a predicted word to the phrase a reader actually selected

A small English boundary model learned the exact extent of 8,283 earlier phrases selected in the source reading game. On 366 held-out pronouns with 2,196 reader clicks, the original single-word pointer matched the **entire** phrase 27.2% of the time. Predicting both boundaries raised exact match to **63.7% [53.4, 73.6]**. Mean intersection over union increased by 0.222 [0.162, 0.284]. This measure gives partial credit when the predicted span overlaps the clicked text. The comparison includes pronouns where the original model pointed to the wrong word. [Registered measurement](transition-pointer-phrase-protocol.md) · [Held-out word and phrase results](../results/reading-validation-transition-pointer/phrases/results.json)

| Frozen word location with… | Game mean overlap | Independent GAP name mean overlap |
|---|---:|---:|
| No expansion | 0.476 | 0.534 |
| Source span-length prior | 0.476 | 0.534 |
| Fixed grammatical noun-phrase chunker | 0.660 | 0.605 |
| Learned left and right boundaries | 0.697 | 0.604 |

On 1,107 independently annotated Wikipedia name targets, expansion improved mean overlap by 0.070 [0.055, 0.085] and increased exact extent matches from 43.6% to 55.7%. Its difference from the fixed noun-phrase chunker was −0.001 [−0.009, 0.007] on these names and +0.038 [−0.006, 0.092] for the game clicks. Ordinary noun-phrase syntax accounts for much of the benefit. Among the game's 191 high-confidence pronouns, complete phrase accuracy reached 78.4% [68.2, 87.5]. High-confidence GAP names had already often been single words, and that subgroup's mean overlap change was −0.006 [−0.033, 0.021]. [Full subgroup denominators and intervals](../results/reading-validation-transition-pointer/phrases/results.json) · [Independent name benchmark](https://aclanthology.org/Q18-1042/)

Each decision to extend observes only words and punctuation before the pronoun. The readout learns from true clicks using the source word range (up to 34 words). Every 366 game and 1,107 GAP measured extents replayed at the same original character offsets through the reusable English writer function. Unicode and changed-future controls passed with zero additional backbone calls. This component answers which phrase surrounds an offered earlier word. Whether ordinary text warrants offering a pronoun link requires its separately trained and tested action model. [Verified runtime replay](../results/reading-validation-transition-pointer/phrases/serving-replay.json) · [English inference code](reference_inference.py)
