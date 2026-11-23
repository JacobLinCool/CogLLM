# Predicting whether a pronoun links to earlier text

Phrase Detectives players could link a pronoun to an earlier phrase or select one of three other discourse actions. Training a single distribution over these possibilities improved the estimated probabilities on independent articles. The model combined each possible earlier-word utility with three null-action utilities and their log normalizer. Source data comprised 24,243 choices at 2,758 pronouns in 230 works. The 243-parameter native output reached a maximum absolute likelihood gradient of 3.4 × 10⁻⁹. [Registered outcome definitions](transition-pointer-null-protocol.md) · [Source decisions](../results/reading-validation-transition-pointer/joint-actions/source-decision.json)

| Held reader observations, equal weight per work | Native four-action log loss | Independent gate | Paired difference, 95% work bootstrap |
|---|---:|---:|---:|
| Earlier gold, 4,880 choices at 631 pronouns, 39 works | .842 | .885 | −.042 [−.069, −.016] |
| Newly evaluated articles, 28,393 choices at 3,320 pronouns, 82 works | .941 | .992 | −.052 [−.089, −.018] |
| Entirely new works, 2,613 choices at 357 pronouns, 80 works | .943 | .996 | −.052 [−.089, −.019] |

Most additional pronouns come from two already known works. The 80-work restriction requires a different title and rechecks the improvement. The separate panel of previously unseen players on new articles supplied only 519 eligible actions across nine scoreable works. Their estimate of the same difference was −.072 [−.476, +.253]. Source and reader overlap restrictions answer different questions. [Independent panels, denominators and full scores](../results/reading-validation-transition-pointer/joint-actions/scores.json)

Native and lexical four-action losses were similar. Native representations added substantially more value for identifying the actual earlier region when a player linked. On 80 new works, native versus lexical reduced the combined action-and-click-region log loss by .298 [.219, .384]. Neither observer qualified for automatic ordinary-English marking under the required source-calibrated 70% precision rule. The production system retains manually selected backward-reference inspection. Its 49-input native replay reproduced 366,887 earlier-word scores at 757 pronouns, including Unicode and changed-suffix checks. [Manual pointer replay](../results/reading-validation-transition-pointer/serving/l4-replay.json) · [Scope of manual inspection](transition-pointer-abstention-results.md)
