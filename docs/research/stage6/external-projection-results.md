# Cross-domain and cross-lingual held-out results

The same Gradio candidate completed GENTLE cross-sentence context localization and Turkish word-for-word sentence validation. The primary effect estimates for both groups trended toward improvement, with pre-specified 95% confidence intervals both crossing zero. [Fixed specification](external-projection-protocol.md) · [Model and L4 execution manifest](../../../research/results/external-projection-l4/run-verification.json)

| held-out endpoint | Literal or surface baseline NLL | NLL with Qwen or App scores | Matched difference and 95% confidence interval |
| --- | ---: | ---: | --- |
| GENTLE's artificial prefix position | 4.4232 | 4.2622 | −0.1609 [−0.5906, +0.2739] |
| Turkish human subjects are more likely to select sentences. | 0.68658 | 0.68608 | −0.00050 [−0.00294, +0.00228] |

The NLL unit is nats. lower values indicate higher probability for human-annotated or selected sentences.

## Cross-span localization of GENTLE

We replicate the distance, literal, and literal-plus-Qwen readout groups from GUMBridge's original sealed snapshot. Among GENTLE's 26 officially tested articles, 14 contain 34 annotations that can be mapped to different natural segments. after excluding 3 for subword and word-end conditioning, we retain **13 articles, 31 annotations**. Each annotation has a median of 188 evaluable word ends, including those within the current segment. The prior article count threshold has been met, but the article pairing interval has not passed. Among the top three groups, all first-place hits are zero. Post-hoc, recalculating the top 5 hits with the same article weights yields a literal hit rate of 2.6% and a literal-plus-Qwen hit rate of 15.4%. These results can inform the design of more precise source candidates, but current evidence remains insufficient to treat model ranking as a reliable localization. [Source Audit](../../../research/results/gum-bridge-gentle-audit.json) · [Fixed Readout held-out](../../../research/results/external-projection-l4/gentle.json) · [Top 5 Recalculation](../../../research/results/external-projection-l4/local-verification.json)

The endpoint is evaluated at the end of the manually specified bridging phrase, assessing the common reference position in the read article. It encompasses word-final ranking, the first question of the preceding phrase, subjective effort, and the time to resolution of awareness, without corresponding sequential reader observation. [GUMBridge annotation scheme](https://aclanthology.org/2026.lrec-1.543.pdf#page=4)

## Turkish word-segmented sentence pairs

Using the App to fix sentence-level readability, we convert the difference in scores between two sentences into the probability that the earlier-presented sentence is selected as easier. The original App's word and sentence segmentation can be recovered for 778 out of 779 sentences. Using only fit materials for calibration, the held-out set contains **616 readers, 700 choices, 393 sentence pairs, and 9 connected components**. Splitting the original sentences at both ends and separating the absolute word count level, the source records include 502 readers across two groups. After filtering by author quality, 554 held-out choices are retained. the surface baseline and NLL with the App score are 0.68473 and 0.68407, respectively, with the direction consistent with the primary comparison. [Fixed data and split](../../../research/results/external-projection-l4/turkish-assay.json) · [Calibration and held-out](../../../research/results/external-projection-l4/turkish-scores.json)

The NLL of App scores is 0.68303, with an ordered baseline of 0.69260. the post-hoc difference interval is [−0.02631, +0.00637] nats. Word count is the author's calculation. Surface baseline also controls for the number of original characters and whitespace splits visible to App. punctuation differences between two sentences are still reflected in these counts. This result constrains the scope to currently measured languages and endpoints. Turkish readability remains to be independently supported. [Independent recalculation](../../../research/results/external-projection-l4/local-verification.json) · [Author's public fields](https://zenodo.org/records/22775520/files/README.md?download=1)

## Shared execution and reproducibility

The initial Colab launch did not obtain a job scheduler confirmation. the L4 instance was released. After extending the launch confirmation time, the test packet and remote job were completed with byte-identical verification. Across 805 documents, there were 46,501 Qwen tokens and 18,199 certified readout reading stops. GPU performed a single forward pass per document, with 12 causal control checks, all passing. Forward and readout took approximately 23.9 seconds. extraction phase took approximately 70.9 seconds. GPU peak memory retention was 8.45 GiB. Both configurations were confirmed released. CPU replay and independent formula verification matched. replay total was approximately 1.31 seconds, with program peak at approximately 168 MiB. [GPU summary](../../../research/results/external-projection-l4/extraction.json) · [Local replay](../../../research/results/external-projection-l4/local-verification.json) · [Release records for both runs](../../../research/results/external-projection-l4/run-verification.json)

[Original research record](../../../research/stage6/external-projection-results.md)
