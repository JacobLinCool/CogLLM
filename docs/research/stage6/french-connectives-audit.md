# French conjunction selection and reader background

Tskhovrebova et al. administered three rounds of French connective cloze items. Readers saw text on both sides of the blank and selected a connective for that position. The study measured author recognition after round one. Before the second cloze test, readers received either active practice or instruction through reading the rules. The third test followed four weeks later. Items changed each round, and both groups received instruction. Responses can test the association between judgments of cross-sentence relations and print exposure. The response timing and item design do not establish the first token at which a reader formed or resolved a question. [Original study, Materials and Procedure](https://doi.org/10.1017/S0142716424000481)

The long table contains 30,620 binary responses. Combining the author's `order`, `item`, and `session`, all 28,557 ratings in the wide table match perfectly. Of these, 28,542 have associated selected conjunctions, allowing item-by-item recalculation of the correct response code against the target conjunction. Additionally, 15 responses are missing selected options and were originally recorded as incorrect. The long table also includes 2,063 responses that did not enter the wide table. For adolescent data, there are two `id` values each corresponding to multiple `order` values. therefore, cross-round matching uses `order` as the reader key. Stimulus files contain 120 fill-in-the-blank passages, with 40 item numbers across rounds being reused. Before model evaluation, rounds must be verified against the original items, options, and item numbers. [Long Table](https://osf.io/c8tpx/) · [Wide Table](https://osf.io/s5pqa/) · [Stimulus](https://osf.io/z7t3m/) · [Statistical Recalculation](../../../research/results/french-connectives-audit.json)

| Observation | Adolescents | Adults |
| --- | ---: | ---: |
| Deduplicated reader key count | 205 | 60 |
| The number of readers with scores of 40 on all three scales. | 182 | 55 |
| The complete reader's second round average accuracy minus the first round average accuracy. | −0.03 percentage points | +2.27 percentage points |
| The average accuracy of the complete reader in the third round minus that in the first round. | −1.35 percentage points | −0.68 percentage points |
| Pearson (r) correlation between the second-round accuracy and author identification score | 0.404 (195 individuals) | 0.245 (60 people) |

The correlation coefficient ranges from −1 to 1. Positive values indicate that readers with higher author recognition scores typically have higher correct-answer rates. This is the raw descriptive statistic at the reader level. The original paper additionally estimated reading exposure correlation using item-level mixed-effects models. There are no between-item differences, no untreated control group, and the between-item differences in the table only describe the three responses of completers. [Original study Analysis strategy, Results](https://doi.org/10.1017/S0142716424000481) · [Reproducible code](../../../research/stage6/french_connectives_audit.py)

Source files were verified against the SHA-256 hash of author OSF file metadata. In this instance, only reader group summaries were retained. individual reader records, stimuli, and options were not included in the model. The paper is licensed under CC BY 4.0, and the OSF data project does not specify individual reuse permissions. [Author Reproducibility Project](https://osf.io/9fx6u/)

[Original research record](../../../research/stage6/french-connectives-audit.md)
