# AmbiStory's end-difference supervised

This experiment tests whether adding human suffix variance to the five-level distribution supervision of sense rationality improves the prediction update magnitude of the same sense under two contexts. It adopts the Qwen3.5-2B features, form control, five-level ordered readout, and reading stop from the AmbiStory v2 sealed snapshot. The same-context readout is taken from the common prefix and suffix distributions. Form control reads only the target word, sense description, and common prefix, with identical outputs under both suffixes. The original test's 155 story families provided only the motivation for this study. This time, the 98 families and 2,952 votes from the author's publicly released validation split are reserved as project held-out sets until their annotations are loaded after the prediction sealed snapshot. The author's SemEval development stage has already made these annotations public to participants. [Original research §§2.3, 3.1](https://aclanthology.org/2026.semeval-1.448.pdf#page=3) · [AmbiStory data construction](../../../research/results/ambistory-preparation.json) · [Fixed v2 readout](../../../research/results/ambistory-fit-protocol-v2.json)

## Supervised objective and model selection

For each pair \(j\), fix a story family and a sense, then sort the two ending versions by stimulus ID. Let \(d_j\) be the difference in human average rationality scores between the second version and the first version, and let \(\hat d_j\) be the difference in the same direction obtained from the expected scores of the two five-level prediction distributions. Each pair uses the same context readout \(W\), with input retaining the existing causal representations of each version at the specified reading stop. Story endings or sense strengthening by the original author are not used as model features. Let \(N\) be the number of training rating items, \(M\) be the number of training pairs, and \(\mathrm{CE}_i\) be the original five-level distribution cross-entropy for the \(i\)-th rating item. The fixed target is

\[
\mathcal L_\beta(W)=
\frac{\sum_{i=1}^{N}\mathrm{CE}_i(W)+\tfrac12\lVert W_{\mathrm{nonint}}\rVert_F^2}{N}
+\frac{\beta}{2M}\sum_{j=1}^{M}\left(\frac{\hat d_j(W)-d_j}{4}\right)^2.
\]

The divisor 4 represents the maximum magnitude of unidirectional difference on a 1–5 Likert scale. Numerical fitting employs \(N\mathcal L_\beta\), with the intercept following the v2 handling. \(\beta=0\) is used directly `ambistory_fit.fit_context`. Each \(\beta>0\) uses the same folded \(\beta=0\) weight as its unique initial value, following the same optimization method and convergence criteria, with an iteration limit of 3,000.

Use only the author training split to select \(\beta\). After casefolding homographs, sort them by SHA-256 hash and assign them to five folds in round-robin order, ensuring that all story families, senses, and versions of the same homograph reside in the same fold. Use only the training data within each fold to estimate feature scaling. Fix the candidate set to \(\{0,1,4,16,64\}\). A candidate is eligible only if, when all five folds converge, the merged out-of-fold MAE with symbol updates is strictly below \(\beta=0\), and the out-of-fold per-ticket NLL increase does not exceed 0.01 nats. Among eligible candidates, select the one with the minimum out-of-fold MAE, breaking ties with the smaller \(\beta\). If no eligible candidate exists, the model selection fails. After selection, fit both \(\beta=0\) and the selected candidate using the full author training split, and generate sealed snapshots for two sets of validation predictions. [Existing training and readout specifications](ambistory-context-update.md)

## held-out validation

Using 10,000 paired resamples with casefold homographs as sampling units and random seed 43, retain the two suffixes, identical senses, and shared prefix within each homograph. Compute differences as candidate minus control. Update magnitude bias as the candidate mean \(|\hat d_j|\) minus the human mean \(|d_j|\). Update MAE for each group as \(|\hat d_j - d_j|\) for two fixed-order versions. Acceptance requires satisfying all conditions simultaneously.

| Contrast | Metrics | threshold |
| --- | --- | --- |
| Candidate suffixes sharing the same common prefix with the same candidate | Per-ticket NLL difference, signed-update MAE difference | Both 95% confidence interval upper bounds are less than 0. |
| Candidate endings versus \(\beta=0\) endings | Symbolic update MAE difference | The upper bound of the 95% interval is less than 0. |
| Candidate endings versus \(\beta=0\) endings | Per-ticket NLL difference | The upper bound of the 95% interval is less than 0.01 nats. |
| Candidates vs. human | Mean Absolute Update Magnitude Difference | 90% interval falls entirely within ±0.5 levels |

Per-ticket NLL retains full scoring. Shared prefixes yield identical distributions across both ending variants for the same sense, resulting in a zero difference for their ending deltas. These contrasts allow assessment of sense update magnitude and per-ticket prediction on author validation. Applicability refers to the group rationality score of the sense definition, excluding individual query onset or resolution time. [Shared reading stop across endpoints and conditions](ambistory-context-update.md)

## Execution and data separation

The following commands first seal the rules, then prepare the data and fit the readout. `AMBISTORY_EXTRACTION_DIR` points to the existing, verified v2 feature directory. Preparation keeps the dev and test inputs unlabeled, and fitting receives only train ratings. Formal scoring gains access to validation annotations after writing `fit-seal.json`.

```sh
python -m research.stage6.ambistory_effect_fit preseal \
  --output research/results/ambistory-effect-fit-protocol.json

python -m research.stage6.ambistory_effect_fit stage-inputs \
  --prepared .cache/cogllm/ambistory/causal-v3 \
  --preparation research/results/ambistory-preparation.json \
  --output .cache/cogllm/ambistory/effect-inputs

python -m research.stage6.ambistory_effect_fit fit \
  --staged .cache/cogllm/ambistory/effect-inputs \
  --extraction "$AMBISTORY_EXTRACTION_DIR" \
  --protocol research/results/ambistory-effect-fit-protocol.json \
  --output .cache/cogllm/ambistory/effect-fit
```

After selecting candidates, the outputs are fitted to `baseline-predictions.jsonl`, `effect-predictions.jsonl`, `heads.npz`, `fit-seal.json`, and `train-diagnostics.json`. Both prediction sets retain the `id`, `sense`, `form_control`, `common_prefix`, and `ending` fields from AmbiStory, with the latter three containing probabilities ranked 1–5. If no candidates are selected, only a sealed snapshot diagnosis and a manifest with status `no_candidate` are generated. no held-out scoring is performed. [Original prediction and scoring format](../../../research/stage6/ambistory_fit.py)

[Original research record](../../../research/stage6/ambistory-effect-supervision.md)
