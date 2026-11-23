# Anglin Study 1 · Repeated belief judgments and confidence

**Public data: 312 participants rated the same given proposition four times.** The four group sizes and 16 belief means from the original paper's Table 1 can be recalculated. The recalculated belief condition × time interaction is F(9, 924) = 11.2468, consistent with the original report of 11.25. The same cohort of 312 individuals also has four certainty ratings available for independent exploration. [Original paper Study 1 results](https://online.ucpress.edu/collabra/article/11/1/133914/209762/Predictors-and-Persistence-of-Belief-Change-in) · [Author's publicly available raw data](https://osf.io/download/v6nkd/) · [Recalculated results](../../research/results/anglin-study1-audit.json)

## Reconstruction of samples and scales

Participants rated their belief and certainty before reading (T1), after the first study material and evaluation prompt (T2), after the second study material and evaluation prompt (T3), and in the next-day survey (T4). Belief was rated from 1 to 8, with higher scores indicating a stronger belief that the punishment deters the behavior. Certainty was also rated from 1 to 8 and measured how sure participants were about the position they selected on that occasion. A participant could report a certainty of 6 on both occasions even after changing their position. The researchers randomized the order of the two study types and the direction of the results. [Original paper, Study 1 methods](https://doi.org/10.1525/collabra.133914) · [First-day survey](https://osf.io/download/juam5/) · [Next-day survey](https://osf.io/download/nyrgb/)

The original SPSS dataset contained 452 variables, with 315 having T4 belief scores. Table 1 included the 312 participants who met the assignment condition and had values for all four beliefs. The reconstructed analysis was recalculated by assignment condition using author analysis codes `StudyA`, `StudyB`, and `Study_order`, retaining only participants who failed the attention check. Additionally, 345 participants with complete T1–T3 belief scores and certainty ratings, and with assignment conditions recorded, were included, retaining immediate responses regardless of whether they responded the following day. This group of 345 participants, defined by item availability, differs from the denominator of 343 first-day questionnaire completers reported in the original text. [SPSS analysis codes](https://osf.io/download/zwfqc/) · [Sample and eight presentation order denominators](../../research/results/anglin-study1-audit.json)

## Recalculable endpoints

Table 1 presents the matched-pair mean change for the complete sample, measured in original scale scores. Positive values indicate an increase. the interval represents the 99% matched-pair t interval. The belief analysis follows the endpoints used in the original paper. The confidence level reflects the exploratory findings of this data audit. [Two-item response complete distribution and matched difference](../../research/results/anglin-study1-audit.json)

| Evidence condition | number | Belief T2 − T1 | Belief T3 − T2 | Degree of certainty T2 − T1 | Degree of certainty T3 − T2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Both findings oppose the deterrent effect. | 68 | +0.01 [−0.52, +0.55] | −0.29 [−0.72, +0.13] | −0.13 [−0.60, +0.33] | +0.01 [−0.37, +0.40] |
| Both findings support the deterrent effect. | 93 | +1.01 [+0.55, +1.47] | +0.03 [−0.20, +0.27] | +0.26 [−0.15, +0.67] | +0.14 [−0.09, +0.37] |
| First oppose, then support. | 72 | −0.29 [−0.72, +0.14] | +0.53 [+0.15, +0.90] | +0.10 [−0.37, +0.57] | +0.04 [−0.33, +0.41] |
| First support, then oppose | 79 | +0.70 [+0.32, +1.07] | −0.47 [−0.80, −0.14] | +0.24 [−0.14, +0.62] | −0.01 [−0.33, +0.30] |

Of the 16 belief standard deviations in the original table, 14 were rounded to two decimal places. The remaining two were recalculated from the original values: the "twice-supported" T4 was approximately 2.04496 (original table: 2.05), and the "first-supported then opposed" T3 was approximately 1.95486 (original table: 1.96). [Cell-wise difference](../../research/results/anglin-study1-audit.json)

## The model employs a boundary

**Study 1 supports tracking belief in a specified proposition within the full experimental setting.** The detailed material for each combination of study type and result direction includes a raster image containing a numerical table, a bar chart, and an explanation. The full results for the two longitudinal versions appear on PDF pages 5 and 7. The two cross-sectional versions appear on pages 12 and 13. For example, in the New Jersey stimulus, figures for the year before and the year after the ban are 4.0→4.9 in the supporting version and 4.9→4.0 in the opposing version. The numbers are embedded in the images and are omitted by text-only PDF extraction. A transcription can preserve the values. Preserving the original visual presentation requires retaining the images. [First-day questionnaire, pp. 5, 7, 12, and 13](https://osf.io/download/juam5/) · [Original methods and materials](https://doi.org/10.1525/collabra.133914)

The eight presentation orders in Study 1 both reused two identical source family direction variants. Splitting participants allowed measuring new readers on the same stimuli. Splitting by complete source family still failed to maintain independent same-type training and held-out sets, as each reader had viewed both families before T3. T4 was measured via an independent questionnaire the following day. These observations can be used to separately inspect belief and confidence segment updates, with data granularity stopping at the four experimental questionnaire time points. [Original paper study design](https://doi.org/10.1525/collabra.133914) · [Next-day questionnaire](https://osf.io/download/nyrgb/)

## Reproduction method

The OSF project lists a CC BY 4.0 license. Save the [Study 1 raw data](https://osf.io/download/v6nkd/), [SPSS analysis code](https://osf.io/download/zwfqc/), [first-day questionnaire](https://osf.io/download/juam5/), and [next-day questionnaire](https://osf.io/download/nyrgb/) as `study1.sav`, `study1.sps`, `stimulus.pdf`, and `followup.pdf`, respectively. The reconstruction script pins the hashes of all four files and reads only the required fields. Its output retains each group's distribution, within-participant differences, and stimulus locations. Two runs produced bit-for-bit identical JSON. [OSF license field](https://api.osf.io/v2/nodes/gfp48/) · [License terms](https://creativecommons.org/licenses/by/4.0/) · [Reconstruction code](../../research/semantic_tracking/anglin_audit.py)

```sh
python -m research.semantic_tracking.anglin_audit \
  --source-dir .cache/cogllm/anglin \
  --output research/results/anglin-study1-audit.json
```

This study was reconstructed using Python 3.14, NumPy 2.5.5, SciPy 1.17.1, and pyreadstat 1.3.8.

[Original research record](../../research/anglin-study1-audit.md)
