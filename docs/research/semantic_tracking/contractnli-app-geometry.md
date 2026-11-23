# How much of the manual evidence segment is retained when pasting the body into the App?

Fixed-hypothesis evidence ranking in English confidentiality agreements, previously validated on **23** EDGAR held-out materials. The result utilizes sentence-level data and item boundary annotations. When switching to the current Gradio pure-text automatic sentence segmentation, among the **354** certified segments in the same batch that were previously manually circled, **only 72 (20.3%)** fully reproduced their start and end positions. Additionally, **195 (55.1%)** could be fully encompassed by a single App-level sentence, including the aforementioned 72. All segment edges were uniformly trimmed prior to comparison. [Original ranking and scope](contractnli-rerank-results.md) · [Boundary verification](../../../research/results/contractnli-app-geometry.json)

| Evaluation unit | Certifiable denominator | Consistent start and end | The same causal sentence end | Covered by a single App sentence |
| --- | ---: | ---: | ---: | ---: |
| All original candidates | 1,684 | 309 | 593 | 988 |
| The only position that is at least marked as evidence by one proposition | 354 | 72 | 150 | 195 |
| Compute the evidence labeling for each hypothesis separately. | 508 | 102 | 208 | 280 |

Of 234 hypotheses with verifiable complete evidence, 58 had every evidence span reproduced exactly. The remaining 1 of the original 235 evidence-bearing hypotheses had an annotation containing whitespace that could not be scored. A total of 19 documents met the current app's whole-text character and token limits. In those 19 documents, 72 of 254 human evidence spans (28.3%) were reproduced exactly. Four documents exceeded the limits and would be rejected by the plain-text route. They remain in the table denominator above. The check reconstructed coordinates with the frozen tokenizer, the app's paragraph and sentence splitting, and its sentence-end alignment method. Two CPU runs were bitwise identical. No model ranking was recomputed. [Coordinate reproduction code](../../../research/semantic_tracking/contractnli_app_geometry.py) · [Sentence and list-item segmentation in the original study](https://aclanthology.org/2021.findings-emnlp.164.pdf#page=3)

The previously validated evidence ordering remains fixed at 17 propositions and the provided English confidentiality boundary. Arbitrary posts automatically generate precise answer intervals, which must be trained and validated using the same segmentation scheme, or the user-provided segmentation must be explicitly specified. The manually selected answer segments correspond to predefined propositions, and the data does not query when a single reader confirms the resolution of a query during reading. [Original readout boundaries and matching results](contractnli-rerank-results.md)

[Original research record](../../../research/semantic_tracking/contractnli-app-geometry.md)
