# GUMBridge cross-sentence prefix positioning results

Across 19 scorable articles and 89 annotated cross-paragraph references in the official test set, adding causal Qwen representations to the lexical model reduced the negative log likelihood at the human-annotated locations **from 4.732 to 4.197 nats**. The difference was −0.535 nats, with an article-paired 95% bootstrap interval of [−0.759, −0.317]. The top-ranked candidate still hit only **6.9%** of the time. The experiment prespecified a minimum of 20 articles, so it did not meet the full acceptance criteria. [Prespecified protocol](gum-bridge-protocol.md) · [Holdout scores](../../../research/results/gum-bridge-l4/scores.json)

| Readout | Test NLL, where lower is better. | The top-ranked hit. | Mean rank, higher is better |
| --- | ---: | ---: | ---: |
| Distance | 5.212 | 0.5% | 0.036 |
| Literal | 4.732 | 3.6% | 0.076 |
| Literal and Qwen | 4.197 | 6.9% | 0.172 |

Each candidate is the end of the readout body text, including words within the current segment, with a median of 380. The ground truth manual annotations provide new references and prior context entities, evaluating from the end of the new reference phrase. Qwen's advantage supports conditional manual prior context ordering. Spontaneous difficulties, how long a question persists, where the answer is found, and whether any word requires anaphora resolution, remain to be observed by corresponding human annotators. The current app accesses segment-level phrase readouts validated via IIRC held-out testing. [Manual annotation definitions and disagreements](https://aclanthology.org/2026.lrec-1.543.pdf#page=4) · [Current app localization scope](../../../research/gradio_app/combined.py#L114)

## Materials and Reproducibility

UD English GUM r2.19 fixed-source removal cross-cut shared URL, retaining 177 training, 28 development, 30 official test. Cross-span and tokenization conditions yielded actual evaluations of 530, 72, 89, respectively from 114, 18, 19 documents. One item in the official test possessed cross-span annotation and was excluded when aligned at the end of the candidate word and tokenization. The two readout NLL fits for the development set were 4.825 and 4.159 nats, with a difference interval of [−1.025, −0.360]. Same-document equal weighting scoring, held-out annotations were integrated only after the training weights sealed snapshot. [Source and Exclusion](../../../research/results/gum-bridge-l4/assay.json) · [Independent Probability Recalculation and Bitwise Re-run](../../../research/results/gum-bridge-l4/local-verification.json)

The same frozen Qwen3.5-2B Gradio checkpoint processed 235 texts and 254,686 model tokens on one L4. It yielded 226,396 coordinates that passed the prefix-tokenization check. Each passage required one FP32 causal forward pass. The independent diagnostics also tested truncated prefixes, replaced suffixes, and padding. All 12 checks passed. Forward passes and representation extraction took 109.6 seconds. The entire extraction run took 227.8 seconds and peaked at 9.05 GiB of reserved GPU memory. CPU fitting and scoring took about two seconds, and the rerun peaked at about 156 MiB. The model weights were loaded on the L4, and we verified that the runtime was released. [Causal checks](../../../research/results/gum-bridge-l4/causality.json) · [L4 extraction](../../../research/results/gum-bridge-l4/extraction.json) · [Release receipt](../../../research/results/gum-bridge-l4/run-verification.json)

GENTLE cross-domain corpora under the same scheme have undergone source auditing. Among 26 documents, 14 contain 34 cross-paragraph reading stops matching natural paragraph boundaries, with no intersection in file ID, normalized full text, or source URL with GUM's three splits. This batch has not yet been used to extract model features. [GENTLE corpus audit](../../../research/results/gum-bridge-gentle-audit.json) · [Official cross-domain test set](https://aclanthology.org/2023.law-1.17/)

[Original research record](../../../research/semantic_tracking/gum-bridge-results.md)
