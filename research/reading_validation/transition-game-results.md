# Direct training on disagreements did not generalize to the independent readers

Training with silver game responses did not reliably identify which gold pronouns would make two readers choose different entities. Across 373 pronouns in 37 texts and 35 releases, the observed disagreement probability for two different eligible players was 4.05% on the pronoun-weighted Brier endpoint. Only 38 pronouns exhibited any disagreement and only 13 texts identified within-text rank ordering. The source-development forecast of the mean rate was 7.97%. [Registration](transition-game-supervision-protocol.md) · [Original human observations](https://aclanthology.org/N19-1176/) · [External scores](../results/reading-validation-transition-game/scores.json)

| Source-supervised prediction | Gold mean risk | Expected reader-pair Brier ↓ | Within-text concordance |
|---|---:|---:|---:|
| Source development rate | .0797 | .04036 | .500 |
| Pronoun form, lexical cues and native surprisal | .0759 | .0411 | .573 [.363, .771] |
| Full frozen native word states plus controls | .0803 | .0431 | .460 [.285, .638] |
| Joint game and three narrative-mark objectives | .0790 | .0428 | .499 [.315, .672] |
| Narrative-mark observer alone | — | — | .563 [.405, .722] |

The joint model's Brier loss was .00247 worse than the source constant [.00067, .00353]. Its difference from the direct native model was −.00028 [−.00053, .00006]. The estimated room between the constant and a hypothetical observer that knew each pronoun's true underlying disagreement rate was .00883 [.00368, .02057]. None of the classifiers established improved reader-pair prediction on this limited gold population. Giving the three Grimm excerpts one work's weight yielded concordance .444 [.248, .644] for the direct observer and .490 [.285, .690] for the joint observer. [Attainable-signal analysis](transition-game-reliability.md) · [Distinct-work sensitivity](../results/reading-validation-transition-game/distinct-work-pronoun-sensitivity.json)

The first shared source fit stopped above the registered stationarity tolerance. The corrected run kept the same logistic and smooth squared nuclear-norm objective, using analytic singular-value gradients and an exact spectral proximal update. The three fitted observers reached maximum absolute source gradients 1.69 × 10⁻⁸ (surface), 8.89 × 10⁻⁹ (direct), and 3.51 × 10⁻⁷ (joint). It processed 374 complete source/gold articles in 269 seconds on one L4. The absence of useful transfer persisted after source convergence. Direct word-location learning from the same kinds of reader choices produced a substantially different result. [Solver and convergence trace](../results/reading-validation-transition-game/receipt.json) · [Prior-word results](transition-pointer-results.md)
