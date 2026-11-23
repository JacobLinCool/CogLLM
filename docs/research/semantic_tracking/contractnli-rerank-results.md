# Evidence ranking through vocabulary fusion with Qwen is validated via held-out testing

On 23 EDGAR documents held out from fitting, vocabulary-fused Qwen readouts passed three predetermined thresholds. The overall evidence AP was **0.502**, higher than BM25's 0.440 and Qwen-fitted-only 0.245. When the article remained identical but the hypothesis changed, the paired scores for both human evidence sets rose from BM25's 0.701 to **0.809**. This batch of measurements specifies the textual evidence corresponding to the hypothesis. [Fixed Specification](contractnli-rerank-protocol.md) · [sealed snapshot held-out results](../../../research/results/contractnli-rerank-l4/results.json)

## Fitting and held-out

We trained the readout on 47 EDGAR documents scored in the previous round. Each group estimated two coefficients using the same training scale, equal weighting document family, and ridge regression. The held-out validation set fixed the BM25 vocabulary scores and added per-token Qwen scores, subtracting shared scores for 17 pre-specified propositions within the same document and same reading stop. The two coefficients were 1.724 and 1.120 on the training standardized scale. With fixed vocabulary statistics and frozen weights, all candidate model features could be obtained in parallel from a single Qwen causal forward pass per document. [Training materials](contractnli-evidence-results.md) · [Coefficients and training vocabulary statistics](../../../research/results/contractnli-rerank-l4/fitted.json)

The new held-out set consists of 23 files from the official dev split and 391 article×proposition combinations. The identifiers, raw text, and normalized text for 47 training samples and 23 held-out samples do not overlap. The cross-group character and vocabulary 5-shingle Jaccard maximums are approximately 0.796 and 0.704, respectively, both below the fixed 0.80 threshold, suggesting potential shared writing templates. After removing 54 pure blank candidates and 1 misaligned candidate, 1,684 evaluable positions remain. Since a single raw annotation includes spaces in the complete evidence set, 234 of the 235 propositions with evidence are included in AP, while the held-out raw manual fields are fully preserved. [Cross-group material audit](../../../research/results/contractnli-rerank-l4/source-audit.json) · [tokenizer pre-check](../../../research/results/contractnli-rerank-l4/preflight.json)

## held-out results

AP measures whether artificial evidence concentrates in the top-ranked segment, while the paired score assesses whether higher-ranked positions within the same document shift with the hypothesis. Both are scored 0–1, with higher values being better. The paired score is computed across 308 support/rebuttal hypothesis pairs and 616 comparison directions across 20 documents. Values within the table are first averaged within documents, then weighted by document family using equal weighting. [Per-item statistics](../../../research/results/contractnli-rerank-l4/results.json)

| Readout | Full-text evidence AP | Same-document item matching |
| --- | ---: | ---: |
| BM25 | 0.440 | 0.701 |
| Qwen original score | 0.168 | 0.530 |
| Qwen minus shared score | 0.227 | 0.863 |
| Fit using only Qwen | 0.245 | 0.844 |
| Vocabulary addition to Qwen's baseline score | 0.435 | 0.700 |
| **Vocabulary + Qwen minus shared score** | **0.502** | **0.809** |

Fusion AP increased relative to BM25 by **0.062**, with a 95% confidence interval [0.034, 0.093]. fusion AP increased relative to using only Qwen fitting by **0.256** [0.212, 0.303]. For same-document query matching, fusion AP increased relative to BM25 by **0.108** [0.068, 0.150]. Compared to the baseline score of Qwen with equal vocabulary size, the fusion AP difference was +0.066 [0.040, 0.097]. The matching score for pairs with only shared scores was higher, and the difference interval for fused pairs included zero. These results support using two signals for whole-document candidate ranking while retaining requirements for readout combinations at different endpoints. [Full comparison and intervals](../../../research/results/contractnli-rerank-l4/results.json)

This readout validation consists of 17 fixed propositions and English contracts with original candidate positions. It outputs ranking scores without estimating absolute understanding probabilities in this experiment. Cross-domain, open propositions, in-text spontaneous questions, and the "where to stop" reading criterion each require their own held-out evidence. [Specifications and Scope](contractnli-rerank-protocol.md)

## Reproducibility and Resources

Run `research.semantic_tracking.contractnli_rerank_prepare` to prepare held-out material for `research.semantic_tracking.contractnli_rerank_forecast` certification of reading stops and extraction of frozen scores. `research.semantic_tracking.contractnli_rerank`'s `fit`, `predict`, and `score` sequentially seal snapshot weights, predictions, and results. Each of the three steps is rerun once, yielding identical bit-by-bit results. A separate program independently reconstructs 391 rank orders from raw vocabulary counts, Qwen scores, and sealed snapshot coefficients, with a maximum absolute difference of approximately \(1.38\times10^{-14}\). The maximum difference in confidence intervals is \(1.11\times10^{-16}\). In the eight cache data change-after-text comparisons, all 6,120 shared prefix feature vectors remain unaffected by the changes. [Code](../../../research/semantic_tracking/contractnli_rerank.py) · [Verify manifest](../../../research/results/contractnli-rerank-l4/run-verification.json)

Colab L4 processed 53,121 body and hypothesis tokens, 1,724 certified reading stops, and 12 causal checks passed. Pre-fetch and readout latency were 24.10 seconds, remote latency was 81.38 seconds, and GPU peak retention was approximately 9.07 GiB. L4 has been confirmed released. Local fit, prediction, and scoring took approximately 0.41, 0.30, and 0.26 seconds, respectively, with peak RSS at approximately 119 MiB. [GPU fetch](../../../research/results/contractnli-rerank-l4/extraction.json) · [Resource snapshot](../../../research/results/contractnli-rerank-l4/usage.jsonl)

[Original research record](../../../research/semantic_tracking/contractnli-rerank-results.md)
