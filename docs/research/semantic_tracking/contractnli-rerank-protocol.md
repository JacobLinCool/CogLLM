# Vocabulary and Qwen proposition reordering evidence

In the previous round, analysis of 47 EDGAR files in the official test split of ContractNLI revealed that BM25 excels at pushing evidence into the top of the ranking. After subtracting scores shared across multiple propositions within an article, Qwen demonstrates greater discrimination between propositions across two sets of provided evidence. In this round, a lightweight readout was trained on the aforementioned 47 analyzed files, and a new held-out set of 23 EDGAR files from the official validation split was used. This latter set was reserved for model scoring prior to this experiment. The objective is to verify whether combining vocabulary matching with Qwen proposition signals can improve end-to-end evidence localization for new articles. [Previous Round Results](contractnli-evidence-results.md) · [Original Split and Evidence Definition](https://aclanthology.org/2021.findings-emnlp.164.pdf#page=2)

## Data and causal conditions

Both groups utilized the same frozen Qwen3.5-2B snapshot and prior QASPER per-token readout. Each article was sampled once for causal state. All 17 official claims were independently sampled once and shared. Data retained full original text and source candidate sentences or list items. Verification with the specified tokenizer enabled precise mapping back to the prefix candidate reading stop. Only alignable candidates were evaluated. positive examples required alignment with the entire set of human evidence. Both groups' identifiers, SEC source locations, and verbatim and normalized bodies were audited. Additionally, cross-group similarity was checked using the Jaccard coefficient on 5-character and 5-word sets of the normalized body. any pair reaching 0.80 was excluded from cross-group matching. All 70 materials passed under the above rules. Model scores were sealed against the held-out evidence annotation prior to the snapshot. [Original data and indexing specifications](https://stanfordnlp.github.io/contract-nli/#dataset-specification)

Vocabulary features are computed using BM25 with a fixed training candidate set, employing IDF and average candidate length, with fixed parameters \(k_1=1.2\) and \(b=0.75\). Vocabulary tokens are normalized to Unicode English-Chinese characters via NFKC and casefold, with lemma deduplication. For each reading stop, only the candidate from its start to the visible end is read. trailing invisible whitespace is removed, and \(\log(1+\mathrm{BM25})\) is calculated. The Qwen feature comprises the original per-token readout hazard logit and the difference between this and the average of 17 known lemma logits within the same reading stop. Vocabulary statistics and the three features' means and scales are estimated solely from the training file. During readout at each candidate position, only the current context and already-read text are used, while lemma and vocabulary statistics are provided prior to reading.

## Readout and validation

Three groups of trainable readouts use two features and identical regularization. For each training hypothesis with complete alignment evidence, record candidate scores \(s_j\) and evidence set \(A\). The loss is \(\log\sum_j \exp(s_j)-\sum_{j\in A}s_j/|A|\), ensuring each piece of human evidence enters the comparison. Apply group equal weighting, within-group file equal weighting, and same-hypothesis equal weighting. Average loss plus ridge with \(0.05\lVert w\rVert^2/2\). Additive intercepts have no effect on ranking, so only coefficients are used. All candidate outputs are ranking scores.

| Readout | Input |
| --- | --- |
| BM25 | Vocabulary scores using only a fixed training candidate set |
| Qwen original score / subtract shared score | Use the original logit and the logit with the shared score subtracted. |
| Fit using only Qwen | Original logit and logit after subtracting shared score |
| Vocabulary addition to Qwen's baseline score | Vocabulary score and original logit |
| Vocabulary + Qwen - shared score | Vocabulary score and shared logit score |

The final arm is the primary candidate. The prespecified requirement is that the lower bounds of the paired 95% intervals for all three differences exceed 0. Full-document evidence AP must exceed both BM25 and the fitted Qwen-only readout. Swap ranking for the two nonoverlapping support and refutation evidence sets within the same document must also outperform BM25. AP and the swap score range from 0 to 1, with higher scores better. Swap ranking compares each complete evidence set's mean raw score across hypotheses. Average the two directions and assign 0.5 for ties. Resample documents jointly across all methods 10,000 times with seed 735972. Also report differences against lexical plus raw Qwen scores, separate support and refutation results, data exclusions, and prefix checks. [First-round ranking criteria](contractnli-evidence-protocol.md#ranking-and-effect)

This test employs fixed propositions and author-provided candidate spans. Model scores can verify the evidence location for a specified proposition and validate segment reordering induced by proposition variation. The elicitation or resolution latency of open propositions and spontaneous questions requires separate observation and held-out.

[Original research record](../../../research/semantic_tracking/contractnli-rerank-protocol.md)
