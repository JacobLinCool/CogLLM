# Gradio word-by-word color-coded MedReadMe held-out results

Using the same CompLex suffix weights as the current Gradio candidate checkpoint, perform a single forward pass on each of the 1,109 held-out original sentences on a single L4, sending only the original sentences to the model. Scores are reported as two decimal places and original character coordinates. There are 23,286 evaluable tokens, with 23,284 precisely aligned, yielding a coverage of 99.991%. Among these, 666 sentences contain both annotated and unannotated tokens, enabling intra-sentence ranking comparisons across 15 source families. [Specifications of the pre-sealed snapshot](medreadme-lexical-holdout.md) · [L4 and release manifest](../../../research/results/medreadme-lexical-l4/run-verification.json) · [Actual scoring](../../../research/results/medreadme-lexical-l4/scores.json)

| Word order alignment | Average Per Sentence AP | 95% paired confidence interval for the candidate reduction control |
| --- | ---: | ---: |
| Gradio suffix readout | 0.713 |  |
| Longer | 0.504 | [+0.125, +0.289] |
| CompLex fit corpus negative frequency | 0.565 | [+0.091, +0.206] |

Candidate passes the predetermined coverage and ranking thresholds. Results apply to MedReadMe English sentences, specifically the difficult words and medical terms selected by trained annotators. The same model score is currently available for ranking. however, the cognitive load intensity in individual reading and the word-level validity of continuous discourse require corresponding human data. [Same Gradio candidate acceptance manifest](../../../research/results/gradio-candidate-l4/candidate-evidence.json)

[Original research record](../../../research/stage6/medreadme-lexical-results.md)
