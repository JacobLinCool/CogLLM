# Comparison of nCloze options within the same article and the same span

This analysis compares the accuracy of nCloze-m relative to CDGP and T5-multi. All three methods share the same masked article and space positions. A total of 92 spaces, fixed by prior audit, are included, covering all 12 presentation units. Each space has four distinct option strings under each of the three conditions, excluding the author-programmed index `9`. Correctness is determined by matching the published option letter with the scoring key. [Source and Alignment Audit](ncloze-source.md) · [Original Paper §4.3–4.6](https://aclanthology.org/2024.naacl-long.220.pdf#page=7)

The primary analysis utilizes all submitted A–D responses. For each blank, the proportion of correct responses is first calculated within each condition, followed by the difference between the nCloze-m condition and the comparison condition. These proportions are then averaged with equal weighting within each article unit, and subsequently averaged with equal weighting across article units. If no responses are available for inclusion in any condition for a given blank, both conditions jointly exclude that blank, and the missing condition and affected number of article units are recorded. Two paired 95% percentile intervals are resampled using the same set of article units, with 10,000 resamples and a fixed seed of 735972. These intervals estimate uncertainty between article units, conditional on this batch of observed responses.

Sensitivity analysis employs the publication parser's correlation inclusion rules. The response of the experimental index `9` must conform to the publication scoring key. the accuracy of the teacher version's other empty spaces must be at least 25%. The publication's interface data is insufficient to verify the two attention check options required by the paper. therefore, this analysis relies on the publication program and scoring key. Responses to the control condition come from different submitters, with each submitter seeing only one experimental version. The full text is readable, and the observed value is the final selection for that space. [Publication Program lines 82–109, 117–139](https://github.com/ondovb/nCloze/blob/10260b8e2dd20aa93b5f6fdcdb83ba2bb8154ceb/Data/parseResults.py#L82-L139) · [Original Paper §4.5–4.6, Figure 2](https://aclanthology.org/2024.naacl-long.220.pdf#page=8)

<!-- Results follow -->

## Results

Of the 92 primary analysis cells, each had responses across three conditions, totaling 1,159 observations. After applying equal weighting to article units, the point estimates of accuracy for nCloze-m were lower than those for CDGP and T5-multi. Both paired confidence intervals crossed zero, and the direction of the difference remains uncertain. [Full counts and intervals](../../../research/results/ncloze-choice-contrast.json)

| Analysis | nCloze-m accuracy | CDGP accuracy | T5-multi accuracy | nCloze-m reduction in CDGP (percentage points) | nCloze-m minus T5-multi (percentage points) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Submit all, 12 text units, 92 spaces | 53.77% | 64.42% | 56.95% | −10.65 [−25.56, +4.32] | −3.18 [−14.67, +8.04] |
| Published parser filtering, 11 article units, 87 spaces | 64.28% | 72.73% | 69.32% | −8.45 [−23.85, +5.99] | −5.04 [−14.20, +4.95] |

After incorporating the parser inclusion rule, all five items in an article unit lacked nCloze-m responses, thereby excluding the entire unit from the two-item analysis. The remaining 11 units yielded 744 responses. The point estimate direction of the sensitivity analysis matched that of the primary analysis, and the confidence interval crossed zero. The statistical unit for both analyses was the article presentation unit. [Number of items included per condition](../../../research/results/ncloze-choice-contrast.json)

Analysis of the published A–D response keys and scoring rubric required no steps involving the host model. A second execution reconstructed an identical bit-for-bit results file. Additionally, independent recalculation of the three accuracy groups, pairwise comparisons, and bootstrap confidence intervals yielded differences from the results file all smaller than \(10^{-14}\). [Reproduction code](../../../research/semantic_tracking/ncloze_choice_contrast.py)

[Original research record](../../../research/semantic_tracking/ncloze-choice-contrast.md)
