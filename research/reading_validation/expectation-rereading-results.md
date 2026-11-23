# Do readers look back where their expectations change?

In Provo, 84 readers contributed 70,291 first-pass word fixations at which a regression could be observed. They looked back 12,150 times. Another panel had supplied the next-word guesses from the same prefixes. At 1,530 original words in 45 paragraphs, both the current and the previous word had certified spelling, offsets and native surprisal. A reader and a paragraph both had to be absent from the fitted model for their combination to contribute to the test. [Original corpus](https://osf.io/sjefs/) · [Analysis protocol and opportunity definition](expectation-rereading-protocol.md)

| Available evidence at the current word | Held log loss per regression opportunity ↓ | Brier score ↓ | Paired improvement over lexical control, simultaneous 95% interval |
| --- | ---: | ---: | --- |
| Length, frequency, punctuation, position and Qwen surprisal | .457404 | .142163 | Reference |
| Control plus estimated next-word disagreement and its change | .457478 | .142179 | −.000073 [−.000526, .000380] |
| Control plus observed next-word disagreement and its change | .457374 | .142144 | +.000031 [−.000401, .000462] |

Independently measured expectations and model estimates both produced negligible changes in held rereading loss once the lexical controls were available. Conditioning additionally on current first-pass duration lowered the control loss to .434431. Neither expectation measure improved that diagnostic. The reported intervals resample readers and passages and account for the two primary comparisons. A skipped word provides no opportunity for a first-pass regression from that word. Each chance to look back is scored once. [Scores, bootstrap intervals and inclusion counts](../results/reading-validation-expectation-rereading/scores.json) · [Reproduction code](expectation_rereading.py)

The writing app therefore describes this curve as the chance that two readers supply different next words. Detecting a narrowing of that distribution can succeed at a location where predicting a look back remains difficult. The study did not directly measure confusion or an unresolved conceptual question. [Actual word choices and held curve](cloze-population-results.md)
