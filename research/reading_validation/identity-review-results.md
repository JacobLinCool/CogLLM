# Addressing reader-reported concerns in scientific revisions

Given two authentic reviewer comments about the same paper and two revised passages, the Qwen3.5-2B readout correctly assigned **both revisions in 53.7%** of comparisons (95% paper bootstrap interval 35.1–71.7%). The 65 pairs came from 18 independent papers in the official manually annotated ARIES test set. Each edit addressed one concern, and an annotator had explicitly marked it as unrelated to the other. Papers received equal weight. The model saw the comments and revision text and had been trained beforehand on separate reading-comprehension corpora. [ARIES study](https://aclanthology.org/2024.acl-long.377/) · [Protocol](identity-review-protocol.md) · [Scores](../results/reading-validation-identity-review/scores.json)

| Model | Both assignments correct | One assignment correct, averaged across passages |
| --- | ---: | ---: |
| Fixed position, fractional tie credit | 25.0% | 50.0% |
| Released joint readout | 11.6% | 55.8% |
| BM25 | 33.4% | 64.5% |
| Trained lexical readout | 51.9% | 76.0% |
| Trained contextual readout | 53.7% | 75.1% |

The contextual readout gained 42.1 percentage points over the released joint readout (paired 95% interval 25.6–58.8) and 20.3 over BM25 (0.3–41.2). Its gain over trained lexical matching was 1.7 points (−19.4 to 23.2). This last comparison was required for the registered superiority criterion, which the study missed. Only 20 pairs across eight papers also had both original passages available. Their mean within-passage change in normalized evidence for the targeted concern after revision was 0.00019 (−0.00119 to 0.00170). Assigning the edited passage to a supplied concern and establishing that the revision actually resolves that concern for the reader require different measurements. ARIES supplies the former. [Item results](../results/reading-validation-identity-review/pair-scores.json) · [Input and exclusions](../results/reading-validation-identity-review/input-seal.json)
