# Cross-sentence prefix readout matching

The existing readout achieved a top-ranked hit rate of 6.9% on the GUMBridge test set and 0% on GENTLE. This comparison asks whether replacing a single cosine similarity with paired features for each dimension improves the ranking of earlier text using the same frozen Qwen3.5-2B representations. It is a retrospective comparison on previously evaluated material. We fit the new coefficients using only the official GUM training split and score the development set, test set, and GENTLE using the original reading stops, candidate word ends, and exclusion criteria. [Existing GUM results](gum-bridge-results.md) · [Existing GENTLE results](../stage6/external-projection-results.md)

Human annotation provides new reference phrases and acceptable preceding entity contexts. Queries are issued at the point of reading the new reference phrase, ranking from the end words of all eligible terms in the previously read main text. A positive example is any end word annotated with a preceding phrase. Each candidate contains words from the current segment's read portion. Text features are readable only up to the query position. Qwen vectors follow causal extraction from two sealed snapshots. [Annotation and Candidate Definitions](gum-bridge-protocol.md) · [Frozen Extraction](../../../research/results/gum-bridge-l4/extraction.json) · [GENTLE Extraction](../../../research/results/external-projection-l4/extraction.json)

| Readout | It can be seen that the input |
| --- | --- |
| Literal translation | The six literal features and coefficients of the sealed snapshot. |
| Original cosine | The literal features of the sealed snapshot, Qwen cosine similarity, and coefficient. |
| Expand the literal | Original literal features and five positive explicit features, including function words, containing numbers, post-word punctuation, preceding word is a preposition, and co-occurring with the query in the same segment. |
| Expand literal and cosine | Expand the literal and the same cosine |
| Dimension-wise matching | Expand the literal, cosine, and per-dimension normalized vector product by subtracting the column mean. |
| Manual part-of-speech tagging | Pairwise matching against the UPOS category in the source CoNLL-U, with missing values forming their own category. |

For each new readout, we use article weights and intra-article label weights to estimate the training scale and marginal negative log-likelihood, with L2 regularization fixed at 0.05. All candidates and all positive examples are jointly normalized, with no subsampling of negative examples. Part-of-speech tags are used solely for the final diagnostic group. Additionally, the original cosine ranking is restricted to nouns, proper nouns, pronouns, and numerals. all queries lacking valid positive examples are retained and counted as non-matches to evaluate how many artificial targets may be missed by this restriction. Part-of-speech missing values also do not pass this restriction.

The NLL of report article weights (lower is better, minimum 0 nats), first-choice and top-five hit rates (higher is better, range 0–100%), and mean reciprocal rank (higher is better, range 0–1), with ties resolved by expected values under uniform ranking. Perform 10,000 paired article resamples using a fixed random seed, reporting the NLL and top-five hit rate difference for pairwise relative expansion cosine. Model superiority requires both GUM testing and GENTLE to show NLL improvement and increased top-five hit rates, with the 95% confidence intervals for each difference excluding zero. Validation on independent corpora must be conducted separately with a fixed seed.

This endpoint provides conditional prefix localization following a known new reference. Where human confusion begins, the intensity of confusion, and which subsequent word resolves the same query all require corresponding readout observations.

After completing the first-round scoring, we add a probability decomposition that preserves the coefficients and primary metrics. When all positive examples occur at an earlier segment $E$, the NLL of the positive example set $G$ equals $-\log p(E) - \log p(G\mid E)$. The first term quantifies the total probability across segment candidates, while the second term quantifies the probability assigned to those candidates within human positions. This post-hoc diagnostic is used to identify the contribution of explicit cross-segment design to the results.

[Original research record](../../../research/semantic_tracking/bridge-rerank-protocol.md)
